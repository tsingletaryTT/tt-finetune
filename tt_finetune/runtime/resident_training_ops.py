"""Public training-loop adapter for the packed resident causal-LM executor."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import torch

from .execution_runtime import ResidentCausalLMExecutor
from .resident_engine import AdamWScalars


def _full_tile(
    value: float, *, dtype: torch.dtype, device: torch.device,
) -> torch.Tensor:
    # A host-backed XLA input keeps the scalar as runtime data rather than a
    # value-specialized executable constant.
    return torch.full((32, 32), value, dtype=dtype).to(device)


def _adamw_scalars(
    *,
    step: torch.Tensor,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    weight_decay: float,
    device: torch.device,
) -> AdamWScalars:
    """Match the accepted capturable torch.optim.AdamW scalar spelling."""

    correction1 = 1 - beta1**step
    correction2 = 1 - beta2**step
    negative_step_size = -(learning_rate / correction1)
    denominator_scale = correction2.sqrt() * negative_step_size
    scaled_epsilon = epsilon / negative_step_size
    scalar_tile = torch.ones((32, 32), dtype=torch.float32, device=device)
    return AdamWScalars(
        _full_tile(
            1.0 - learning_rate * weight_decay,
            dtype=torch.bfloat16,
            device=device,
        ),
        scalar_tile * denominator_scale,
        scalar_tile * scaled_epsilon,
    )


def _normalize_resident_batch(
    batch: Any,
    *,
    physical_batch: int,
    physical_sequence: int,
    ignored_index: int,
) -> tuple[dict[str, Any], int]:
    """Map a compatible logical token bucket onto one physical executable.

    The qualified resident owners are shape-static.  A smaller right-padded
    causal-text bucket can nevertheless share the largest executable without
    changing the useful-token computation: the suffix has ignored labels and
    real causal rows cannot attend to later padded tokens.  Keeping this
    normalization explicit makes the logical/physical boundary independently
    testable and prevents an accidental executable rebuild per data bucket.
    """

    normalized = dict(batch)
    input_ids = normalized.get("input_ids")
    labels = normalized.get("labels")
    if (
        input_ids is None
        or labels is None
        or input_ids.ndim != 2
        or labels.ndim != 2
    ):
        raise ValueError(
            "resident execution requires rank-2 input_ids and labels"
        )
    logical_batch, logical_sequence = map(int, input_ids.shape)
    if logical_batch != physical_batch:
        raise ValueError(
            "resident batch size differs from its executable: "
            f"{logical_batch} != {physical_batch}"
        )
    if (
        logical_sequence <= 0
        or logical_sequence > physical_sequence
        or logical_sequence % 32
    ):
        raise ValueError(
            "resident logical sequence bucket must be a positive 32-token "
            f"multiple no larger than {physical_sequence}; got {logical_sequence}"
        )

    position_ids = normalized.get("position_ids")
    if position_ids is not None:
        if tuple(position_ids.shape) != (physical_batch, logical_sequence):
            raise ValueError(
                "resident position_ids must match the logical input geometry"
            )
        expected_positions = torch.arange(
            logical_sequence,
            device=position_ids.device,
            dtype=position_ids.dtype,
        ).reshape(1, -1).expand(physical_batch, -1)
        if not bool(torch.equal(position_ids, expected_positions)):
            raise ValueError(
                "resident execution currently requires canonical position_ids"
            )

    attention_mask = normalized.get("attention_mask")
    if attention_mask is not None and tuple(attention_mask.shape) != (
        physical_batch,
        logical_sequence,
    ):
        raise ValueError(
            "resident attention_mask must match the logical input geometry"
        )

    # Some internal exact-reference fixtures carry already-shifted S-1
    # targets.  Reconstruct the canonical unshifted label row before applying
    # the same right-padding rule used by ordinary dataset batches.
    if tuple(labels.shape) == (physical_batch, logical_sequence - 1):
        labels = torch.nn.functional.pad(
            labels, (1, 0), value=int(ignored_index)
        )
    elif tuple(labels.shape) != (physical_batch, logical_sequence):
        raise ValueError(
            "resident labels differ from the logical sequence bucket: "
            f"labels={tuple(labels.shape)}, input={tuple(input_ids.shape)}"
        )

    suffix = physical_sequence - logical_sequence
    if suffix:
        normalized["input_ids"] = torch.nn.functional.pad(
            input_ids, (0, suffix), value=0
        )
        labels = torch.nn.functional.pad(
            labels, (0, suffix), value=int(ignored_index)
        )
        if attention_mask is not None:
            normalized["attention_mask"] = torch.nn.functional.pad(
                attention_mask, (0, suffix), value=0
            )
        if position_ids is not None:
            normalized["position_ids"] = torch.arange(
                physical_sequence,
                device=position_ids.device,
                dtype=position_ids.dtype,
            ).reshape(1, -1).expand(physical_batch, -1)
    normalized["labels"] = labels
    return normalized, logical_sequence


class ResidentOptimizerCheckpoint:
    """Small optimizer-shaped object consumed by the existing checkpoint API."""

    def __init__(self, operations: "ResidentTrainingOps") -> None:
        self.operations = operations
        self.param_groups = [{"lr": operations.current_learning_rate}]

    def state_dict(self) -> dict[str, Any]:
        return {
            **self.operations.executor.packed_state_dict(),
            "optimizer_step": self.operations.optimizer_steps,
            "learning_rate_index": self.operations.optimizer_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.operations.executor.load_packed_state_dict(state)
        step = int(state.get("optimizer_step", 0))
        self.operations.set_optimizer_step(step)

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        return None


class ResidentTrainingOps:
    """Expose the frozen resident owner through :class:`TrainingLoop`.

    The resident engine already owns backward, clipping and AdamW.  This class
    maps the public loop's phase methods onto that owner without materializing
    logical gradients or parameters between steps.
    """

    def __init__(
        self,
        *,
        executor: ResidentCausalLMExecutor,
        device_manager: Any,
        learning_rates: Sequence[float],
        beta1: float,
        beta2: float,
        epsilon: float,
        weight_decay: float,
        initial_step: int = 0,
        profile_phases: bool = False,
        audit_gradients: bool = False,
        audit_device_inputs: bool = False,
    ) -> None:
        if not learning_rates:
            raise ValueError("resident training requires at least one learning rate")
        self.executor = executor
        self.device_manager = device_manager
        self.learning_rates = tuple(float(value) for value in learning_rates)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.weight_decay = float(weight_decay)
        self.profile_phases = bool(profile_phases)
        self.audit_gradients = bool(audit_gradients)
        self.audit_device_inputs = bool(audit_device_inputs)
        self.phase_seconds: dict[str, list[float]] = {}
        self.optimizer_steps = 0
        self.current_learning_rate = self.learning_rates[0]
        self.set_optimizer_step(initial_step)
        self.optimizer = ResidentOptimizerCheckpoint(self)

        self._initial_hidden: torch.Tensor | None = None
        self._top_gradient: torch.Tensor | None = None
        self._raw_gradients: torch.Tensor | None = None
        self._pending_loss: torch.Tensor | None = None
        self._clip_requested = False
        self.gradient_audit: dict[str, Any] | None = None
        self.evolving_audit: list[dict[str, Any]] = []
        self._audited_loss_value: float | None = None
        self._audited_loss_identity: int | None = None
        self.last_gradient_norm: float | None = None
        self.last_clip_coefficient: float | None = None
        self.last_gradient_multiplier: float | None = None
        self.last_gradient_diagnostics: list[dict[str, Any]] = []
        self.last_host_batch_signature: str | None = None
        self.last_device_batch_signature: str | None = None
        self.logical_sequence_lengths_seen: set[int] = set()
        self.tt_clip_mode = "resident_packed_exact_norm"

        # Preserve the accepted accelerator-side scalar evaluation.  The
        # small runtime tiles are prepared once and reused; no per-step host
        # scalar graph or value-specialized compile is introduced.
        device = self.executor.engine.device
        step_value = torch.tensor(0.0, dtype=torch.float32, device=device)
        self._scalar_schedule: list[AdamWScalars] = []
        for index, learning_rate in enumerate(self.learning_rates, 1):
            step_value = step_value + 1.0
            self._scalar_schedule.append(_adamw_scalars(
                step=step_value,
                learning_rate=learning_rate,
                beta1=self.beta1,
                beta2=self.beta2,
                epsilon=self.epsilon,
                weight_decay=self.weight_decay,
                device=device,
            ))
        import torch_xla

        torch_xla.sync(wait=True)

    def set_optimizer_step(self, step: int) -> None:
        if step < 0 or step >= len(self.learning_rates):
            raise ValueError(
                f"resident optimizer step {step} is outside the scheduler horizon"
            )
        self.optimizer_steps = int(step)
        self.current_learning_rate = self.learning_rates[step]
        if hasattr(self, "optimizer"):
            self.optimizer.param_groups[0]["lr"] = self.current_learning_rate

    def _record(self, name: str, started: float) -> None:
        if self.profile_phases:
            self.phase_seconds.setdefault(name, []).append(
                perf_counter() - started
            )

    @staticmethod
    def _device_barrier() -> None:
        """Submit one accepted resident phase and wait for completion.

        The qualified five-call schedule is materially faster on the current
        PJRT/TT-Metal stack than asking XLA to co-lower all resident owners and
        the generic causal edge into one large executable.  These are device
        execution boundaries, not host tensor round-trips; packed parameters,
        checkpoints and gradients remain device resident throughout.
        """

        import torch_xla

        torch_xla.sync(wait=True)

    @staticmethod
    def _signature(batch: Any) -> str:
        digest = hashlib.sha256()
        for name in ("input_ids", "labels"):
            value = batch.get(name)
            if value is None:
                continue
            host = value.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tuple(host.shape)).encode())
            digest.update(host.numpy().tobytes())
        return digest.hexdigest()[:20]

    @staticmethod
    def _tensor_hash(value: torch.Tensor) -> str:
        host = value.detach().cpu().contiguous()
        return hashlib.sha256(
            host.reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest()

    @classmethod
    def _logical_hashes(
        cls, values: dict[str, torch.Tensor],
    ) -> dict[str, str]:
        return {
            name: cls._tensor_hash(value)
            for name, value in sorted(values.items())
        }

    def zero_grad(self) -> None:
        started = perf_counter()
        if any(value is not None for value in (
            self._initial_hidden,
            self._top_gradient,
            self._raw_gradients,
            self._pending_loss,
        )):
            raise RuntimeError("resident step state was not fully consumed")
        self._clip_requested = False
        self._record("zero_grad", started)

    def forward_loss(self, batch: Any) -> torch.Tensor:
        started = perf_counter()
        physical_batch = self.executor.plan.batch_size
        physical_sequence = self.executor.plan.sequence_length
        batch, logical_sequence = _normalize_resident_batch(
            batch,
            physical_batch=physical_batch,
            physical_sequence=physical_sequence,
            ignored_index=int(self.executor.edges.ignored_index),
        )
        self.logical_sequence_lengths_seen.add(logical_sequence)
        self.last_host_batch_signature = self._signature(batch)
        prepared = self.device_manager.prepare_batch(batch)
        if self.audit_device_inputs:
            # Device auditing is intentionally opt-in: copying the real batch
            # back to the host is a synchronization boundary and must never be
            # hidden in the ordinary steady-state measurement.
            self.last_device_batch_signature = self._signature(prepared)
            if self.last_host_batch_signature != self.last_device_batch_signature:
                raise RuntimeError(
                    "TT input transfer changed the resident training batch"
                )
        else:
            self.last_device_batch_signature = None
        input_ids = prepared["input_ids"]
        labels = prepared["labels"]
        expected = (
            self.executor.plan.batch_size,
            self.executor.plan.sequence_length,
        )
        if tuple(input_ids.shape) != expected or tuple(labels.shape) != expected:
            raise ValueError(
                "resident batch geometry differs from its executable: "
                f"input={tuple(input_ids.shape)}, labels={tuple(labels.shape)}, "
                f"expected={expected}"
            )
        phase_started = perf_counter()
        initial_hidden = self.executor.edges.embed(input_ids)
        self._device_barrier()
        self._record("embedding", phase_started)

        phase_started = perf_counter()
        final_hidden = self.executor.engine.forward_hidden(initial_hidden)
        self._device_barrier()
        self._record("resident_forward", phase_started)

        diagnostic_path = os.environ.get(
            "TT_FINETUNE_DEBUG_FORWARD_CHECKPOINT_PATH"
        )
        if diagnostic_path:
            path = Path(diagnostic_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "initial_hidden": initial_hidden.detach().cpu(),
                    "checkpoint_hidden": final_hidden.detach().cpu(),
                    "checkpoint_arena": (
                        self.executor.engine.checkpoints[0].detach().cpu()
                    ),
                    "checkpoint_layout": (
                        self.executor.engine.diagnostic_forward_layout
                    ),
                    "logical_sequence_length": logical_sequence,
                },
                path,
            )
            raise RuntimeError(
                f"resident forward checkpoint diagnostic captured at {path}"
            )

        phase_started = perf_counter()
        loss, top_gradient = self.executor.edges.loss_and_top_gradient(
            final_hidden, labels
        )
        self._device_barrier()
        self._record("causal_loss_and_top_gradient", phase_started)
        self._initial_hidden = initial_hidden
        self._top_gradient = top_gradient
        self._pending_loss = loss
        self._record("resident_forward_and_loss", started)
        return loss

    def backward(self, loss: torch.Tensor, divisor: int) -> None:
        started = perf_counter()
        if divisor != 1:
            raise ValueError(
                "resident execution does not yet support gradient accumulation"
            )
        if loss is not self._pending_loss:
            raise RuntimeError("resident backward received an unrelated loss")
        if self._initial_hidden is None or self._top_gradient is None:
            raise RuntimeError("resident backward requires a preceding forward")
        gradients, _ = self.executor.engine.backward_raw(
            self._initial_hidden, self._top_gradient
        )
        self._device_barrier()
        self._raw_gradients = gradients
        self._top_gradient = None
        self._record("resident_backward", started)

    def sync(self) -> None:
        started = perf_counter()
        self._device_barrier()
        self._record("resident_sync", started)

    def audit_gradient_state(self) -> None:
        if not self.audit_gradients or self.gradient_audit is not None:
            return
        if self._raw_gradients is None:
            raise RuntimeError("resident gradient audit has no packed gradient")
        finite = torch.isfinite(self._raw_gradients.float()).all()
        nonzero = self._raw_gradients.count_nonzero() > 0
        import torch_xla

        torch_xla._XLAC._xla_sync_multi(
            [finite, nonzero], devices=[], wait=True, sync_xla_data=True
        )
        self.gradient_audit = {
            "boundary": "resident_packed_pre_clip",
            "logical_tensor_count": sum(
                region.parameter_name is not None
                for region in self.executor.packed.regions
            ),
            "physical_slot_count": len(self.executor.packed.regions),
            "finite": bool(finite.cpu().item()),
            "nonzero": bool(nonzero.cpu().item()),
        }

    def clip_grad_norm(self, maximum: float) -> None:
        if maximum != 1.0:
            raise ValueError(
                "qualified resident clipping requires max_grad_norm=1.0"
            )
        self._clip_requested = True

    def optimizer_step(self) -> None:
        started = perf_counter()
        if self._raw_gradients is None:
            raise RuntimeError("resident optimizer has no packed gradients")
        if not self._clip_requested:
            raise RuntimeError(
                "resident optimizer requires its exact packed clipping phase"
            )
        if self.optimizer_steps >= len(self._scalar_schedule):
            raise RuntimeError("resident optimizer exceeded its compiled schedule")
        self.executor.engine.clip_adamw(
            self._raw_gradients,
            self._scalar_schedule[self.optimizer_steps],
        )
        import torch_xla

        torch_xla.sync(wait=True)
        if self.audit_gradients:
            raw_host = self._raw_gradients.detach().cpu()
            clipped_host = (
                self.executor.engine.clipped_gradient_output.detach().cpu()
            )
            state_host = self.executor.engine.optimizer_state.detach().cpu()
            rows = self.executor.packed.trainable_pages.shape[0]
            loss_value = float(self._pending_loss.detach().cpu().item())
            self._audited_loss_value = loss_value
            self._audited_loss_identity = id(self._pending_loss)
            self.evolving_audit.append({
                "step": self.optimizer_steps + 1,
                "loss": self._tensor_hash(self._pending_loss),
                "loss_value": loss_value,
                "raw_gradients": self._logical_hashes(
                    self.executor.packed.logical_gradients(raw_host)
                ),
                "clipped_gradients": self._logical_hashes(
                    self.executor.packed.logical_gradients(clipped_host)
                ),
                "parameters": self._logical_hashes(
                    self.executor.packed.logical_state(state_host[:rows])
                ),
                "first_moment": self._logical_hashes(
                    self.executor.packed.logical_state(
                        state_host[rows:2 * rows]
                    )
                ),
                "second_moment": self._logical_hashes(
                    self.executor.packed.logical_state(state_host[2 * rows:])
                ),
            })
        self.optimizer_steps += 1
        self._initial_hidden = None
        self._raw_gradients = None
        self._pending_loss = None
        self._record("resident_clip_adamw", started)

    def scheduler_step(self) -> None:
        if self.optimizer_steps < len(self.learning_rates):
            self.current_learning_rate = self.learning_rates[self.optimizer_steps]
            self.optimizer.param_groups[0]["lr"] = self.current_learning_rate

    def loss_value(self, loss: torch.Tensor) -> float:
        if self._audited_loss_identity == id(loss):
            assert self._audited_loss_value is not None
            return self._audited_loss_value
        return float(loss.detach().cpu().item())

    def learning_rate(self) -> float:
        return self.current_learning_rate

    def sync_model_for_checkpoint(self) -> None:
        self.executor.sync_logical_model()

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state)


__all__ = ["ResidentOptimizerCheckpoint", "ResidentTrainingOps"]
