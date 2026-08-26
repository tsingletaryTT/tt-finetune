"""Torch/XLA implementation of the backend-neutral training operation contract."""

from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter
from typing import Any


def _tt_xla_clip_grad_norm_(
    torch: Any,
    parameters: list[Any],
    maximum: float,
    gradient_unscale_factor: float = 1.0,
) -> tuple[float, float, float, list[dict[str, Any]]]:
    """Clip dense gradients with an exact host-controlled TT-XLA fallback.

    PyTorch's generic implementation takes a vector norm of every gradient,
    stacks the scalar results, and takes a second vector norm.  The resulting
    TT-XLA graph currently reaches an invalid reshape in TT-Metal for the two
    large token-row gradients used by foundation warm-up.  A device-side
    squared reduction has the same backend issue when its scalar is broadcast
    back over those gradients.

    The training loop has already synchronized the backward pass before this
    method runs.  Copying only the trainable gradients to the host therefore
    adds a small bounded transfer (2 MiB for the Router foundation), computes
    the exact combined FP32 L2 norm without entering the TT graph, and sends a
    plain Python scalar multiply back into the optimizer graph.
    """

    gradients = [
        parameter.grad
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return 0.0, 1.0, 1.0, []

    with torch.no_grad():
        total_squared = 0.0
        diagnostics = []
        for index, gradient in enumerate(gradients):
            host_gradient = gradient.detach().cpu().float()
            finite = torch.isfinite(host_gradient)
            finite_count = int(finite.sum().item())
            element_count = host_gradient.numel()
            finite_values = host_gradient[finite]
            diagnostics.append(
                {
                    "index": index,
                    "shape": list(host_gradient.shape),
                    "elements": element_count,
                    "nonfinite": element_count - finite_count,
                    "max_abs_finite": (
                        float(finite_values.abs().max().item())
                        if finite_count
                        else None
                    ),
                }
            )
            if finite_count != element_count:
                raise FloatingPointError(
                    "Non-finite TT gradient before clipping: "
                    f"{diagnostics[-1]}"
                )
            # Accumulate in FP64 on the host. Squaring a large but finite BF16
            # gradient in FP32 can overflow and falsely report an infinite
            # combined norm.
            host_gradient64 = host_gradient.double()
            total_squared += float(
                torch.sum(host_gradient64 * host_gradient64).item()
            )
        scaled_norm = total_squared**0.5
        total_norm = scaled_norm * gradient_unscale_factor
        coefficient = min(float(maximum) / (total_norm + 1e-6), 1.0)
        applied_multiplier = gradient_unscale_factor * coefficient
        if applied_multiplier != 1.0:
            for gradient in gradients:
                gradient.mul_(applied_multiplier)
    return total_norm, coefficient, applied_multiplier, diagnostics


class TorchXlaTrainingOps:
    """Connect a compiled causal LM to :class:`TrainingLoop`.

    Imports stay inside construction so the host-side notebook package remains
    importable without installing the TT runtime into the host interpreter.
    """

    def __init__(
        self,
        *,
        model: Any,
        optimizer: Any,
        device_manager: Any,
        scheduler: Any = None,
        ignored_index: int = -100,
        tt_clip_mode: str = "native",
        audit_device_inputs: bool = False,
        record_batch_signatures: bool = True,
        backward_loss_scale: float = 1.0,
        right_padding_safe_attention: bool = False,
        host_master_parameters: list[Any] | None = None,
        profile_phases: bool = False,
        profile_ir_dir: str | Path | None = None,
        ttnn_forward_graph_path: str | Path | None = None,
        ttnn_forward_graph_capture_after: int = 0,
        split_clip_optimizer: bool = False,
        split_after_forward: bool = False,
        split_after_loss: bool = False,
        audit_gradients: bool = False,
        training_dropout: Any = None,
    ):
        import torch
        import torch.nn.functional as functional

        self.torch = torch
        self.functional = functional
        self.model = model
        self.optimizer = optimizer
        self.device_manager = device_manager
        self.scheduler = scheduler
        self.ignored_index = ignored_index
        if tt_clip_mode not in {"native", "host", "packed_exact_norm"}:
            raise ValueError(
                "tt_clip_mode must be native, host, or packed_exact_norm"
            )
        self.tt_clip_mode = tt_clip_mode
        self.audit_device_inputs = audit_device_inputs
        self.record_batch_signatures = record_batch_signatures
        if backward_loss_scale < 1.0:
            raise ValueError("backward_loss_scale must be at least 1")
        self.backward_loss_scale = backward_loss_scale
        self.right_padding_safe_attention = right_padding_safe_attention
        self.profile_phases = profile_phases
        self.profile_ir_dir = (
            Path(profile_ir_dir) if profile_ir_dir is not None else None
        )
        self._captured_ir_phases: set[str] = set()
        self.ttnn_forward_graph_path = (
            Path(ttnn_forward_graph_path)
            if ttnn_forward_graph_path is not None
            else None
        )
        if ttnn_forward_graph_capture_after < 0:
            raise ValueError(
                "ttnn_forward_graph_capture_after must be non-negative"
            )
        self.ttnn_forward_graph_capture_after = int(
            ttnn_forward_graph_capture_after
        )
        self.ttnn_forward_graph_captured = False
        self._forward_invocations = 0
        self.split_clip_optimizer = split_clip_optimizer
        self.split_after_forward = split_after_forward
        self.split_after_loss = split_after_loss
        self.audit_gradients = audit_gradients
        self.gradient_audit: dict[str, Any] | None = None
        self.training_dropout = training_dropout
        self.phase_seconds: dict[str, list[float]] = {}
        self.trainable_named_parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self.trainable_parameters = [
            parameter for _name, parameter in self.trainable_named_parameters
        ]
        self.host_master_parameters = host_master_parameters
        if (
            self.host_master_parameters is not None
            and len(self.host_master_parameters)
            != len(self.trainable_parameters)
        ):
            raise ValueError(
                "host_master_parameters must match trainable parameters"
            )
        self.last_gradient_norm: float | None = None
        self.last_clip_coefficient: float | None = None
        self.last_gradient_multiplier: float | None = None
        self.last_gradient_diagnostics: list[dict[str, Any]] = []
        self.last_host_batch_signature: str | None = None
        self.last_device_batch_signature: str | None = None

    def zero_grad(self) -> None:
        started = perf_counter()
        self.optimizer.zero_grad()
        if self.host_master_parameters is not None:
            for parameter in self.trainable_parameters:
                parameter.grad = None
        self._record_phase("zero_grad", started)

    def forward_loss(self, batch: Any) -> Any:
        if self.record_batch_signatures or self.audit_device_inputs:
            self.last_host_batch_signature = self._batch_signature(batch)
        started = perf_counter()
        batch = self.device_manager.prepare_batch(batch)
        self._record_phase("prepare_batch", started)
        if self.audit_device_inputs:
            roundtrip = {
                name: tensor.detach().cpu()
                for name, tensor in batch.items()
                if name in {"input_ids", "labels"}
            }
            self.last_device_batch_signature = self._batch_signature(
                roundtrip
            )
            if (
                self.last_host_batch_signature
                != self.last_device_batch_signature
            ):
                raise RuntimeError(
                    "TT input round-trip changed the training batch: "
                    f"host={self.last_host_batch_signature}, "
                    f"device={self.last_device_batch_signature}"
                )
        started = perf_counter()
        self.device_manager.shard_model(self.model)
        attention_mask = batch["attention_mask"]
        if self.right_padding_safe_attention:
            # Every causal dataset handled by this worker is right padded.
            # Real positions cannot attend to later padding under the causal
            # mask, so marking pad queries valid leaves supervised logits
            # unchanged. It does prevent all-masked padded query rows, whose
            # TT backward currently propagates NaNs through zero loss grads.
            attention_mask = self.torch.ones_like(attention_mask)
        training = bool(self.model.training)
        if self.training_dropout is not None:
            dropout_started = perf_counter()
            self.training_dropout.prepare(training=training)
            if self.profile_phases and self.device_manager.config.use_tt:
                import torch_xla

                torch_xla.sync(wait=True)
            self._record_phase("prepare_dropout", dropout_started)
        capture_ttnn_graph = bool(
            self.ttnn_forward_graph_path is not None
            and not self.ttnn_forward_graph_captured
            and self._forward_invocations
            >= self.ttnn_forward_graph_capture_after
        )
        self._forward_invocations += 1
        if capture_ttnn_graph:
            # TTNN graph capture is intentionally started after model/device
            # creation and after the requested warmup forwards.  It records
            # the compiler-selected steady-state runtime operations without
            # enabling per-op blocking or requiring a Tracy-enabled build.
            import ttnn

            self.ttnn_forward_graph_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        output = self.model(
            input_ids=batch["input_ids"],
            attention_mask=attention_mask,
        )
        if self.training_dropout is not None:
            self.training_dropout.finish(training=training)
        self._capture_ir("forward", [output.logits])
        if (
            (
                self.profile_phases
                or self.split_after_forward
                or capture_ttnn_graph
            )
            and self.device_manager.config.use_tt
        ):
            import torch_xla

            torch_xla.sync(wait=True)
        if capture_ttnn_graph:
            ttnn.graph.end_graph_capture_to_file(
                str(self.ttnn_forward_graph_path)
            )
            self.ttnn_forward_graph_captured = True
        self._record_phase("forward", started)
        shift_logits = output.logits[:, :-1, :].contiguous()
        started = perf_counter()
        loss = self.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            batch["labels"].reshape(-1),
            ignore_index=self.ignored_index,
        )
        # The public worker intentionally computes causal CE itself. Include
        # installed MoE router regularization here as part of that same device
        # loss graph; otherwise auxiliary/z-loss semantics would be silently
        # dropped even though expert routing itself executed correctly.
        from tt_finetune.integration.sparse_moe import (
            sparse_moe_regularization_loss,
        )
        router_regularization = sparse_moe_regularization_loss(
            self.model,
            attention_mask=batch.get("attention_mask"),
        )
        if router_regularization is not None:
            loss = loss + router_regularization
        self._capture_ir("cross_entropy", [loss])
        if (
            (self.profile_phases or self.split_after_loss)
            and self.device_manager.config.use_tt
        ):
            import torch_xla

            torch_xla.sync(wait=True)
        self._record_phase("cross_entropy", started)
        return loss

    @staticmethod
    def _batch_signature(batch: Any) -> str:
        digest = hashlib.sha256()
        for name in ("input_ids", "labels"):
            tensor = batch.get(name)
            if tensor is None:
                continue
            host = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(host.shape)).encode("utf-8"))
            digest.update(host.numpy().tobytes())
        return digest.hexdigest()[:20]

    def backward(self, loss: Any, divisor: int) -> None:
        started = perf_counter()
        (
            loss / (divisor * self.backward_loss_scale)
        ).backward()
        self._capture_ir(
            "backward",
            [
                parameter.grad
                for parameter in self.trainable_parameters
                if parameter.grad is not None
            ],
        )
        self._profile_barrier()
        self._record_phase("backward", started)

    def sync(self) -> None:
        started = perf_counter()
        if self.device_manager.config.use_tt:
            import torch_xla

            torch_xla.sync(wait=True)
        self._record_phase("post_backward_sync", started)

    def audit_gradient_state(self) -> None:
        """Capture the first accumulated pre-clipping gradient authority."""

        if not self.audit_gradients or self.gradient_audit is not None:
            return
        named_gradients = [
            (name, parameter.grad.detach())
            for name, parameter in self.trainable_named_parameters
            if parameter.grad is not None
        ]
        gradients = [gradient for _name, gradient in named_gradients]
        # Do not transfer full XLA gradient buffers for this audit.  Large
        # strided/autograd views can be materialized through a different
        # layout on the host and have produced false all-zero/Inf reports even
        # while the packed optimizer moments remain finite.  Reduce the facts
        # on device and transfer only scalar authorities.
        finite_scalars = [
            self.torch.isfinite(gradient).all() for gradient in gradients
        ]
        nonzero_scalars = [
            gradient.count_nonzero() > 0 for gradient in gradients
        ]
        finite_max_scalars = [
            self.torch.where(
                self.torch.isfinite(gradient),
                gradient.abs(),
                self.torch.zeros((), dtype=gradient.dtype, device=gradient.device),
            ).max()
            for gradient in gradients
        ]
        finite_sum_scalars = [
            self.torch.where(
                self.torch.isfinite(gradient),
                gradient.abs(),
                self.torch.zeros((), dtype=gradient.dtype, device=gradient.device),
            ).sum()
            for gradient in gradients
        ]
        scalar_authorities = (
            finite_scalars
            + nonzero_scalars
            + finite_max_scalars
            + finite_sum_scalars
        )
        if scalar_authorities and self.device_manager.config.use_tt:
            import torch_xla

            torch_xla._XLAC._xla_sync_multi(
                scalar_authorities,
                devices=[],
                wait=True,
                sync_xla_data=True,
            )
        finite = [bool(value.cpu().item()) for value in finite_scalars]
        nonzero = [bool(value.cpu().item()) for value in nonzero_scalars]
        maxima = [float(value.cpu().item()) for value in finite_max_scalars]
        sums = [float(value.cpu().item()) for value in finite_sum_scalars]
        nonfinite_names = [
            name
            for (name, _gradient), is_finite in zip(named_gradients, finite)
            if not is_finite
        ]
        zero_names = [
            name for (name, _gradient), has_values in zip(named_gradients, nonzero)
            if not has_values
        ]
        self.gradient_audit = {
            "boundary": "accumulated_pre_clip",
            "tensor_count": len(gradients),
            "nonzero_tensor_count": sum(
                nonzero
            ),
            "nonfinite_tensor_count": sum(
                not value for value in finite
            ),
            "nonfinite_tensor_names": nonfinite_names,
            "zero_tensor_count": len(zero_names),
            "zero_tensor_names": zero_names,
            "max_abs": max(
                maxima,
                default=0.0,
            ),
            "sum_abs": sum(sums),
        }

    def clip_grad_norm(self, maximum: float) -> None:
        started = perf_counter()
        if (
            self.device_manager.config.use_tt
            and self.tt_clip_mode == "host"
        ):
            (
                self.last_gradient_norm,
                self.last_clip_coefficient,
                self.last_gradient_multiplier,
                self.last_gradient_diagnostics,
            ) = _tt_xla_clip_grad_norm_(
                self.torch,
                self.trainable_parameters,
                maximum,
                gradient_unscale_factor=self.backward_loss_scale,
            )
            self._record_phase("clip_grad_norm", started)
            return
        if self.tt_clip_mode == "packed_exact_norm":
            if self.backward_loss_scale != 1.0:
                with self.torch.no_grad():
                    for parameter in self.trainable_parameters:
                        if parameter.grad is not None:
                            parameter.grad.mul_(self.backward_loss_scale)
            gradients = [
                parameter.grad
                for parameter in self.trainable_parameters
                if parameter.grad is not None
            ]
            total_norm = self.torch.nn.utils.get_total_norm(gradients)
            self.optimizer.pack_gradients()
            self.torch.nn.utils.clip_grads_with_norm_(
                self.optimizer.packed_parameters,
                maximum,
                total_norm,
            )
            if (
                (self.profile_phases or self.split_clip_optimizer)
                and self.device_manager.config.use_tt
            ):
                import torch_xla

                torch_xla.sync(wait=True)
            self._record_phase("clip_grad_norm", started)
            return
        if self.backward_loss_scale != 1.0:
            with self.torch.no_grad():
                for parameter in self.trainable_parameters:
                    if parameter.grad is not None:
                        parameter.grad.mul_(self.backward_loss_scale)
        self.torch.nn.utils.clip_grad_norm_(
            self.trainable_parameters,
            max_norm=maximum,
        )
        if (
            (self.profile_phases or self.split_clip_optimizer)
            and self.device_manager.config.use_tt
        ):
            import torch_xla

            torch_xla.sync(wait=True)
        self._record_phase("clip_grad_norm", started)

    def optimizer_step(self) -> None:
        started = perf_counter()
        # Fused-loop variants can skip the ordinary pre-optimizer boundary;
        # keep the same audit contract as a safe fallback in that case.
        self.audit_gradient_state()
        if self.host_master_parameters is not None:
            for device_parameter, master_parameter in zip(
                self.trainable_parameters,
                self.host_master_parameters,
            ):
                if device_parameter.grad is None:
                    master_parameter.grad = None
                else:
                    master_parameter.grad = (
                        device_parameter.grad.detach().cpu().float()
                    )
            self.optimizer.step()
            with self.torch.no_grad():
                for device_parameter, master_parameter in zip(
                    self.trainable_parameters,
                    self.host_master_parameters,
                ):
                    device_parameter.copy_(
                        master_parameter.detach().to(
                            device=device_parameter.device,
                            dtype=device_parameter.dtype,
                        )
                    )
            if self.device_manager.config.use_tt:
                import torch_xla

                torch_xla.sync(wait=True)
            self._record_phase("optimizer_step", started)
            return
        self.device_manager.optimizer_step(self.optimizer)
        self._record_phase("optimizer_step", started)

    def scheduler_step(self) -> None:
        if self.scheduler is not None:
            self.scheduler.step()

    def loss_value(self, loss: Any) -> float:
        started = perf_counter()
        value = float(loss.item())
        self._record_phase("loss_read", started)
        return value

    def learning_rate(self) -> float | None:
        scheduled_learning_rate = getattr(
            self.optimizer,
            "scheduled_learning_rate",
            None,
        )
        if scheduled_learning_rate is not None:
            value = scheduled_learning_rate()
            if value is not None:
                return value
        value = self.optimizer.param_groups[0]["lr"]
        if isinstance(value, self.torch.Tensor):
            return float(value.item())
        return float(value)

    def _profile_barrier(self) -> None:
        if self.profile_phases and self.device_manager.config.use_tt:
            import torch_xla

            torch_xla.sync(wait=True)

    def _record_phase(self, name: str, started: float) -> None:
        if self.profile_phases:
            self.phase_seconds.setdefault(name, []).append(
                perf_counter() - started
            )

    def _capture_ir(self, name: str, tensors: list[Any]) -> None:
        if (
            self.profile_ir_dir is None
            or name in self._captured_ir_phases
            or not self.device_manager.config.use_tt
        ):
            return
        from torch_xla.core import xla_model as xm

        self.profile_ir_dir.mkdir(parents=True, exist_ok=True)
        stablehlo = xm.get_stablehlo(tensors)
        (self.profile_ir_dir / f"{name}.stablehlo.mlir").write_text(
            stablehlo,
            encoding="utf-8",
        )
        self._captured_ir_phases.add(name)
