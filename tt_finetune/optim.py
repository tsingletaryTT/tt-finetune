"""Blackhole-aware optimizer implementations."""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch


class TilePackedAdamW(torch.optim.AdamW):
    """AdamW over bounded 32x32-aligned buffers with parameter views.

    LoRA ranks below 32 leave most of every isolated TT tile unused. This
    optimizer streams flattened parameter segments into bounded tile-aligned
    buffers and performs AdamW over those buffers. Parameters contained in one
    pack become storage-sharing views, so the model observes updates without a
    copyback graph. Only parameters spanning multiple packs use exact slice
    copyback. Forward/backward and PEFT checkpoint shapes remain unchanged.

    A parameter may span several packs, and unrelated parameter shapes may
    share one pack. This removes the old shape-group padding and allows large
    LoRA tensors to use the same bounded optimizer path.
    """

    tile_elements = 32 * 32
    packed_width = 32
    default_max_pack_bytes = 1024 * 1024

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        max_pack_bytes: int = default_max_pack_bytes,
        packing_strategy: str = "contiguous",
        profile_phases: bool = False,
        split_phases: bool = False,
        share_parameter_views: bool = True,
        learning_rate_schedule_values: list[torch.Tensor] | None = None,
        initial_schedule_step: int = 0,
        **kwargs: Any,
    ):
        supplied = list(parameters)
        if supplied and isinstance(supplied[0], dict):
            source_groups = []
            self.model_parameters = []
            self.model_parameter_group_indices = []
            for group_index, supplied_group in enumerate(supplied):
                if not isinstance(supplied_group, dict):
                    raise TypeError(
                        "TilePackedAdamW parameters must be all tensors or all groups"
                    )
                group = dict(supplied_group)
                group_parameters = list(group.pop("params"))
                if not group_parameters:
                    raise ValueError("optimizer parameter groups cannot be empty")
                source_groups.append(group)
                self.model_parameters.extend(group_parameters)
                self.model_parameter_group_indices.extend(
                    [group_index] * len(group_parameters)
                )
        else:
            if any(isinstance(value, dict) for value in supplied):
                raise TypeError(
                    "TilePackedAdamW parameters must be all tensors or all groups"
                )
            self.model_parameters = supplied
            self.model_parameter_group_indices = [0] * len(supplied)
            source_groups = [{}]
        if not self.model_parameters:
            raise ValueError("TilePackedAdamW requires trainable parameters")
        devices = {parameter.device for parameter in self.model_parameters}
        dtypes = {parameter.dtype for parameter in self.model_parameters}
        if len(devices) != 1 or len(dtypes) != 1:
            raise ValueError(
                "TilePackedAdamW requires one device and dtype per group"
            )
        if max_pack_bytes <= 0:
            raise ValueError("max_pack_bytes must be positive")
        if packing_strategy not in {"contiguous", "shape_grouped"}:
            raise ValueError(
                "packing_strategy must be contiguous or shape_grouped"
            )
        self.max_pack_bytes = max_pack_bytes
        self.packing_strategy = packing_strategy
        self.profile_phases = profile_phases
        self.split_phases = split_phases
        self.share_parameter_views = share_parameter_views
        self.phase_seconds: dict[str, list[float]] = {}
        self._packed_gradients_ready = False
        if initial_schedule_step < 0:
            raise ValueError("initial_schedule_step must be non-negative")
        if learning_rate_schedule_values is not None:
            if not learning_rate_schedule_values:
                raise ValueError(
                    "learning_rate_schedule_values must not be empty"
                )
            if initial_schedule_step >= len(learning_rate_schedule_values):
                raise ValueError(
                    "initial_schedule_step must be inside the schedule"
                )
            for value in learning_rate_schedule_values:
                if value.numel() != 1 and value.shape != (32, 32):
                    raise ValueError(
                        "each scheduled learning rate must be scalar or one tile"
                    )
                if value.device not in devices:
                    raise ValueError(
                        "scheduled learning rates must use the parameter device"
                    )
        self.learning_rate_schedule_values = learning_rate_schedule_values
        self.learning_rate_schedule_buffer: torch.Tensor | None = None
        self.host_learning_rate_schedule: list[float] | None = None
        self._schedule_host_step = initial_schedule_step

        self.logical_elements = sum(
            parameter.numel() for parameter in self.model_parameters
        )
        element_size = self.model_parameters[0].element_size()
        max_pack_elements = max_pack_bytes // element_size
        max_pack_elements = (
            max_pack_elements // self.tile_elements
        ) * self.tile_elements
        if max_pack_elements < self.tile_elements:
            raise ValueError(
                "max_pack_bytes must fit at least one 32x32 tile"
            )

        self.packs: list[dict[str, Any]] = []
        self.packed_parameters: list[torch.nn.Parameter] = []
        pending_segments: list[dict[str, int]] = []
        pending_pieces: list[torch.Tensor] = []
        pending_elements = 0
        pending_optimizer_group_index: int | None = None

        def finish_pack() -> None:
            nonlocal pending_segments, pending_pieces, pending_elements
            nonlocal pending_optimizer_group_index
            if not pending_segments:
                return
            padded = (
                math.ceil(pending_elements / self.tile_elements)
                * self.tile_elements
            )
            padding = padded - pending_elements
            gradient_padding = None
            if padding:
                gradient_padding = torch.zeros(
                    padding,
                    dtype=self.model_parameters[0].dtype,
                    device=self.model_parameters[0].device,
                )
                pending_pieces.append(gradient_padding)
            packed = torch.nn.Parameter(
                torch.cat(pending_pieces)
                .reshape(-1, self.packed_width)
                .detach(),
                requires_grad=True,
            )
            indices = list(
                dict.fromkeys(
                    segment["parameter_index"]
                    for segment in pending_segments
                )
            )
            self.packed_parameters.append(packed)
            self.packs.append(
                {
                    "indices": indices,
                    "segments": pending_segments,
                    "logical_elements": pending_elements,
                    "padded_elements": padded,
                    "gradient_padding": gradient_padding,
                    "optimizer_group_index": pending_optimizer_group_index,
                }
            )
            pending_segments = []
            pending_pieces = []
            pending_elements = 0
            pending_optimizer_group_index = None

        optimizer_parameter_groups = [
            [
                index
                for index, assigned_group in enumerate(
                    self.model_parameter_group_indices
                )
                if assigned_group == optimizer_group_index
            ]
            for optimizer_group_index in range(len(source_groups))
        ]

        for optimizer_group_index, optimizer_parameter_group in enumerate(
            optimizer_parameter_groups
        ):
            if packing_strategy == "contiguous":
                packing_groups = [optimizer_parameter_group]
            else:
                indices_by_shape: dict[tuple[int, ...], list[int]] = {}
                for index in optimizer_parameter_group:
                    indices_by_shape.setdefault(
                        tuple(self.model_parameters[index].shape), []
                    ).append(index)
                packing_groups = [
                    indices_by_shape[shape]
                    for shape in sorted(indices_by_shape)
                ]
            for parameter_group in packing_groups:
                pending_optimizer_group_index = optimizer_group_index
                for parameter_index in parameter_group:
                    parameter = self.model_parameters[parameter_index]
                    flat = parameter.detach().reshape(-1)
                    if (
                        packing_strategy == "shape_grouped"
                        and flat.numel() <= max_pack_elements
                        and pending_elements
                        and flat.numel()
                        > max_pack_elements - pending_elements
                    ):
                        # Reproduce the historical shape-group planner for
                        # checkpoint/performance comparisons: do not split a
                        # tensor that can fit whole in the next pack.
                        finish_pack()
                        pending_optimizer_group_index = optimizer_group_index
                    parameter_offset = 0
                    while parameter_offset < flat.numel():
                        available = max_pack_elements - pending_elements
                        length = min(
                            available, flat.numel() - parameter_offset
                        )
                        pending_segments.append(
                            {
                                "parameter_index": parameter_index,
                                "parameter_offset": parameter_offset,
                                "pack_offset": pending_elements,
                                "length": length,
                            }
                        )
                        pending_pieces.append(
                            flat[
                                parameter_offset : parameter_offset + length
                            ]
                        )
                        parameter_offset += length
                        pending_elements += length
                        if pending_elements == max_pack_elements:
                            finish_pack()
                            pending_optimizer_group_index = optimizer_group_index
                if packing_strategy == "shape_grouped":
                    finish_pack()
            # Never mix distinct optimizer hyperparameter/LR groups in one
            # physical pack.
            finish_pack()

        self.padded_elements = sum(
            pack["padded_elements"] for pack in self.packs
        )
        segments_by_parameter: list[
            list[tuple[torch.nn.Parameter, dict[str, int]]]
        ] = [[] for _ in self.model_parameters]
        for packed, pack in zip(self.packed_parameters, self.packs):
            for segment in pack["segments"]:
                segments_by_parameter[
                    segment["parameter_index"]
                ].append((packed, segment))

        # A model parameter wholly contained in one pack can share that
        # pack's storage. Torch/XLA preserves this alias across an in-place
        # optimizer update, eliminating hundreds of copyback graph nodes for
        # common LoRA layouts. Parameters that straddle pack boundaries keep
        # the exact copyback path below.
        self.shared_view_indices: set[int] = set()
        for parameter_index, parameter_segments in enumerate(
            segments_by_parameter
        ):
            if not self.share_parameter_views:
                continue
            if len(parameter_segments) != 1:
                continue
            packed, segment = parameter_segments[0]
            start = segment["pack_offset"]
            stop = start + segment["length"]
            parameter = self.model_parameters[parameter_index]
            parameter.data = (
                packed.reshape(-1)[start:stop]
                .reshape_as(parameter)
                .data
            )
            self.shared_view_indices.add(parameter_index)
        base_lr = float(kwargs.get("lr", 1e-3))
        packed_by_group: list[list[torch.nn.Parameter]] = [
            [] for _ in source_groups
        ]
        for packed, pack in zip(self.packed_parameters, self.packs):
            packed_by_group[pack["optimizer_group_index"]].append(packed)
        packed_groups = []
        for group_index, group_options in enumerate(source_groups):
            packed_group = dict(group_options)
            group_lr = float(packed_group.get("lr", base_lr))
            packed_group["params"] = packed_by_group[group_index]
            packed_group["_tt_lr_multiplier"] = group_lr / base_lr
            packed_groups.append(packed_group)
        super().__init__(packed_groups, **kwargs)

    def logical_state_tensors_from_state_dict(
        self,
        optimizer_state_dict: dict[str, Any],
    ) -> list[tuple[str, torch.Tensor]]:
        """Reconstruct Adam state in logical model-parameter order.

        ``torch.optim`` checkpoints describe the physical packed parameters.
        Their tensor count, padding and integer state IDs legitimately change
        with pack size/strategy, so hashing that representation would reject
        numerically identical autotuner candidates.  This method removes only
        those physical-layout details: moment slices are joined back into the
        original parameter shapes and scalar state is represented once per
        optimizer group.

        The returned tensors are suitable for exact digests and finiteness
        checks.  No numerical conversion is performed.
        """

        state = optimizer_state_dict.get("state", {})
        groups = optimizer_state_dict.get("param_groups", [])
        physical_ids = [
            state_id
            for group in groups
            for state_id in group.get("params", [])
        ]
        if len(physical_ids) != len(self.packs):
            raise ValueError(
                "packed optimizer checkpoint does not match the live pack "
                f"descriptor: {len(physical_ids)} states for "
                f"{len(self.packs)} packs"
            )

        def values_for(pack_index: int) -> dict[str, Any]:
            state_id = physical_ids[pack_index]
            values = state.get(state_id)
            if values is None:
                # JSON-like checkpoint transforms can stringify integer keys.
                values = state.get(str(state_id))
            if values is None:
                raise ValueError(
                    f"optimizer checkpoint is missing packed state {state_id}"
                )
            return values

        output: list[tuple[str, torch.Tensor]] = []
        # Scalar Adam state (normally ``step``) is identical for every pack in
        # one logical optimizer group.  Recording it once makes the digest
        # independent of physical fragmentation while preserving group state.
        for group_index in range(len(groups)):
            pack_indices = [
                index
                for index, pack in enumerate(self.packs)
                if pack["optimizer_group_index"] == group_index
            ]
            if not pack_indices:
                continue
            values = values_for(pack_indices[0])
            for name, value in sorted(values.items()):
                if torch.is_tensor(value) and value.numel() == 1:
                    output.append((f"group.{group_index}.{name}", value))

        segments_by_parameter: list[list[tuple[int, dict[str, int]]]] = [
            [] for _ in self.model_parameters
        ]
        for pack_index, pack in enumerate(self.packs):
            for segment in pack["segments"]:
                segments_by_parameter[segment["parameter_index"]].append(
                    (pack_index, segment)
                )

        for parameter_index, segments in enumerate(segments_by_parameter):
            if not segments:
                raise ValueError(
                    f"logical optimizer parameter {parameter_index} has no pack segment"
                )
            tensor_state_names: set[str] = set()
            for pack_index, _ in segments:
                for name, value in values_for(pack_index).items():
                    if torch.is_tensor(value) and value.numel() > 1:
                        tensor_state_names.add(name)
            for name in sorted(tensor_state_names):
                pieces = []
                for pack_index, segment in segments:
                    value = values_for(pack_index).get(name)
                    if not torch.is_tensor(value) or value.numel() <= 1:
                        raise ValueError(
                            f"packed state {name!r} is incomplete for logical "
                            f"parameter {parameter_index}"
                        )
                    start = segment["pack_offset"]
                    stop = start + segment["length"]
                    pieces.append(value.reshape(-1)[start:stop])
                logical = torch.cat(pieces).reshape_as(
                    self.model_parameters[parameter_index]
                )
                output.append((f"parameter.{parameter_index}.{name}", logical))
        return output

    def logical_gradient_tensors(self) -> list[torch.Tensor]:
        """Reconstruct current packed gradients in logical parameter order.

        This is a qualification/debug boundary.  It deliberately returns
        views/joins without changing dtype or arithmetic so an evolving-state
        oracle can compare the exact gradients consumed by packed AdamW.
        """

        output: list[torch.Tensor] = []
        segments_by_parameter: list[list[tuple[int, dict[str, int]]]] = [
            [] for _ in self.model_parameters
        ]
        for pack_index, pack in enumerate(self.packs):
            for segment in pack["segments"]:
                segments_by_parameter[segment["parameter_index"]].append(
                    (pack_index, segment)
                )
        for parameter, segments in zip(
            self.model_parameters, segments_by_parameter, strict=True
        ):
            pieces = []
            for pack_index, segment in segments:
                gradient = self.packed_parameters[pack_index].grad
                if gradient is None:
                    raise RuntimeError(
                        "packed logical-gradient audit ran before gradients were packed"
                    )
                start = segment["pack_offset"]
                stop = start + segment["length"]
                pieces.append(gradient.reshape(-1)[start:stop])
            output.append(torch.cat(pieces).reshape_as(parameter))
        return output

    def configure_learning_rate_schedule(
        self,
        values: list[torch.Tensor],
        *,
        host_values: list[float],
        initial_step: int = 0,
    ) -> None:
        """Install pre-materialized scalar LRs without value-specializing XLA.

        Each scalar is a device-data input, so a different value does not
        change the AdamW graph hash. ``host_values`` includes the initial LR
        plus every post-step scheduler value and is used for metrics without
        synchronizing a device scalar back to the host.
        """

        if not values:
            raise ValueError("learning-rate schedule must not be empty")
        parameter_device = self.model_parameters[0].device
        device_buffer_mode = parameter_device.type == "xla"
        expected_updates = len(host_values) - 1
        if expected_updates <= 0:
            raise ValueError(
                "host_values must contain update and post-final values"
            )
        if not device_buffer_mode and len(host_values) != len(values) + 1:
            raise ValueError(
                "host_values must contain one post-final scheduler value"
            )
        if device_buffer_mode and len(values) != 1:
            raise ValueError(
                "TT schedule configuration requires one reusable LR tile"
            )
        if not 0 <= initial_step < expected_updates:
            raise ValueError("initial_step must be inside the schedule")
        for value in values:
            if (
                value.numel() != 1
                and value.shape != (32, 32)
            ) or value.device != parameter_device:
                raise ValueError(
                    "scheduled learning rates must be scalar/tile parameter-device "
                    "tensors"
                )
        if device_buffer_mode:
            if values[0].shape != (32, 32):
                raise ValueError("TT schedule buffer must be one 32x32 tile")
            self.learning_rate_schedule_values = None
            self.learning_rate_schedule_buffer = values[0]
        else:
            self.learning_rate_schedule_values = values
            self.learning_rate_schedule_buffer = None
        self.host_learning_rate_schedule = list(host_values)
        self._schedule_host_step = initial_step

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        self._packed_gradients_ready = False
        for parameter in self.model_parameters:
            parameter.grad = None

    @torch.no_grad()
    def pack_gradients(self) -> None:
        """Materialize model gradients into optimizer packs once."""

        from time import perf_counter

        missing = [
            index
            for index, parameter in enumerate(self.model_parameters)
            if parameter.grad is None
        ]
        if missing:
            raise RuntimeError(
                "TilePackedAdamW requires a gradient for every packed "
                f"parameter; missing indices: {missing[:8]}"
            )
        started = perf_counter()
        for packed, pack in zip(self.packed_parameters, self.packs):
            gradients = [
                self.model_parameters[
                    segment["parameter_index"]
                ].grad.detach().reshape(-1)[
                    segment["parameter_offset"] :
                    segment["parameter_offset"] + segment["length"]
                ]
                for segment in pack["segments"]
            ]
            if pack["gradient_padding"] is not None:
                gradients.append(pack["gradient_padding"])
            packed.grad = torch.cat(gradients).reshape_as(packed)
        self._finish_profile_phase("gradient_pack", started)
        self._packed_gradients_ready = True

    @torch.no_grad()
    def step(self, closure=None):
        from time import perf_counter

        if not self._packed_gradients_ready:
            self.pack_gradients()
        started = perf_counter()
        scheduled_lr = None
        if self.learning_rate_schedule_buffer is not None:
            assert self.host_learning_rate_schedule is not None
            if self._schedule_host_step >= (
                len(self.host_learning_rate_schedule) - 1
            ):
                raise RuntimeError("learning-rate schedule is exhausted")
            host_value = self.host_learning_rate_schedule[
                self._schedule_host_step
            ]
            source = torch.full(
                (32, 32),
                host_value,
                dtype=self.learning_rate_schedule_buffer.dtype,
            ).to(self.learning_rate_schedule_buffer.device)
            self.learning_rate_schedule_buffer.copy_(source)
            scheduled_lr = self.learning_rate_schedule_buffer
        elif self.learning_rate_schedule_values is not None:
            if self._schedule_host_step >= len(
                self.learning_rate_schedule_values
            ):
                raise RuntimeError("learning-rate schedule is exhausted")
            scheduled_lr = self.learning_rate_schedule_values[
                self._schedule_host_step
            ]
        if scheduled_lr is None:
            loss = super().step(closure=closure)
        elif self.model_parameters[0].device.type != "xla":
            base_value = (
                self.host_learning_rate_schedule[self._schedule_host_step]
                if self.host_learning_rate_schedule is not None
                else float(scheduled_lr.detach().cpu().item())
            )
            for group in self.param_groups:
                group["lr"] = base_value * group["_tt_lr_multiplier"]
            loss = super().step(closure=closure)
        else:
            loss = self._scheduled_adamw_step(
                scheduled_lr, closure=closure
            )
        if (
            self.learning_rate_schedule_values is not None
            or self.learning_rate_schedule_buffer is not None
        ):
            self._schedule_host_step += 1
        self._packed_gradients_ready = False
        self._finish_profile_phase("adamw_core", started)

        started = perf_counter()
        updated_pieces: list[list[torch.Tensor]] = [
            [] for _ in self.model_parameters
        ]
        for packed, pack in zip(self.packed_parameters, self.packs):
            packed_flat = packed.reshape(-1)
            for segment in pack["segments"]:
                if (
                    segment["parameter_index"]
                    in self.shared_view_indices
                ):
                    continue
                start = segment["pack_offset"]
                stop = start + segment["length"]
                updated_pieces[segment["parameter_index"]].append(
                    packed_flat[start:stop]
                )
        for parameter_index, (parameter, pieces) in enumerate(
            zip(self.model_parameters, updated_pieces)
        ):
            if parameter_index in self.shared_view_indices:
                continue
            # Only parameters that straddle a pack reach this path.
            updated = (
                pieces[0] if len(pieces) == 1 else torch.cat(pieces)
            )
            parameter.copy_(updated.reshape_as(parameter))
        self._finish_profile_phase("parameter_copyback", started)
        return loss

    def _scheduled_adamw_step(
        self, learning_rate: torch.Tensor, closure=None
    ):
        """AdamW with a non-scalar device-data learning-rate input.

        TT-XLA 0.12 corrupts capturable AdamW when a lazy 0-D learning-rate
        tensor replaces the Python float captured by the first executable.
        Keeping the single runtime value as a 1x1 device tensor avoids that
        scalar lowering path while preserving the AdamW operation order.  It
        also keeps one executable across every scheduler value.
        """

        loss = closure() if closure is not None else None
        if learning_rate.shape != (32, 32):
            raise ValueError(
                "scheduled TT AdamW requires one 32x32 learning-rate tile"
            )
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            epsilon = group["eps"]
            weight_decay = group["weight_decay"]
            group_learning_rate = (
                learning_rate * group["_tt_lr_multiplier"]
            )
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["step"] = torch.zeros(
                        (), dtype=torch.float32, device=parameter.device
                    )
                    state["exp_avg"] = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                if parameter.ndim != 2 or parameter.shape[1] != 32:
                    raise ValueError(
                        "scheduled TT AdamW requires packed [tile_rows, 32] state"
                    )
                learning_rate_field = group_learning_rate.repeat(
                    parameter.shape[0] // 32, 1
                )
                step.add_(1)
                if weight_decay != 0:
                    parameter.mul_(1 - learning_rate_field * weight_decay)
                exp_avg.lerp_(gradient, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    gradient, gradient, value=1 - beta2
                )
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                normalized_update = (
                    exp_avg / bias_correction1
                ) / ((exp_avg_sq / bias_correction2).sqrt() + epsilon)
                parameter.add_(
                    learning_rate_field * normalized_update,
                    alpha=-1.0,
                )
        return loss

    def scheduled_learning_rate(self) -> float | None:
        """Return the host-visible LR for the next optimizer update."""

        if (
            self.learning_rate_schedule_values is None
            and self.learning_rate_schedule_buffer is None
        ):
            return None
        if (
            self.learning_rate_schedule_values is not None
            or self.learning_rate_schedule_buffer is not None
        ):
            index = min(
                self._schedule_host_step,
                (
                    len(self.host_learning_rate_schedule) - 1
                    if self.host_learning_rate_schedule is not None
                    else len(self.learning_rate_schedule_values) - 1
                ),
            )
            if self.host_learning_rate_schedule is not None:
                return self.host_learning_rate_schedule[
                    min(
                        self._schedule_host_step,
                        len(self.host_learning_rate_schedule) - 1,
                    )
                ]
            assert self.learning_rate_schedule_values is not None
            return float(
                self.learning_rate_schedule_values[index].detach().cpu().item()
            )

    def _finish_profile_phase(self, name: str, started: float) -> None:
        if not self.profile_phases and not self.split_phases:
            return
        if self.model_parameters[0].device.type == "xla":
            import torch_xla

            torch_xla.sync(wait=True)
        if self.profile_phases:
            from time import perf_counter

            self.phase_seconds.setdefault(name, []).append(
                perf_counter() - started
            )
