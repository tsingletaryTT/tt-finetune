"""Trainer-owned optimizer state machine with no framework import dependency."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Protocol, Sized


@dataclass(frozen=True)
class LoopConfig:
    num_epochs: int
    gradient_accumulation_steps: int = 1
    max_steps: int | None = None
    max_grad_norm: float | None = None
    fuse_backward_optimizer: bool = False
    loss_read_interval: int = 1

    def __post_init__(self) -> None:
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.loss_read_interval <= 0:
            raise ValueError("loss_read_interval must be positive")


@dataclass
class LoopState:
    epoch: int = 0
    optimizer_step: int = 0
    micro_step: int = 0
    batch_in_epoch: int = 0
    stopped: bool = False


@dataclass(frozen=True)
class OptimizerStep:
    step: int
    epoch: int
    micro_batches: int
    loss: float | None
    seconds: float
    learning_rate: float | None
    useful_tokens: int = 0
    padded_tokens: int = 0
    sequence_lengths: tuple[int, ...] = ()
    data_wait_seconds: float = 0.0


class TrainingOps(Protocol):
    """Small execution contract implemented by Torch/XLA or another backend."""

    def zero_grad(self) -> None: ...

    def forward_loss(self, batch: Any) -> Any: ...

    def backward(self, loss: Any, divisor: int) -> None: ...

    def sync(self) -> None: ...

    def audit_gradient_state(self) -> None: ...

    def clip_grad_norm(self, maximum: float) -> None: ...

    def optimizer_step(self) -> None: ...

    def scheduler_step(self) -> None: ...

    def loss_value(self, loss: Any) -> float: ...

    def learning_rate(self) -> float | None: ...


class TrainingLoop:
    """Exact optimizer-step loop reusable by every execution backend.

    A dataloader factory is used instead of a one-shot iterator so ``max_steps``
    may cross epoch boundaries. Short final accumulation windows are normalized
    by their actual micro-batch count.
    """

    def __init__(
        self,
        config: LoopConfig,
        ops: TrainingOps,
        *,
        on_step: Callable[[OptimizerStep, LoopState], None] | None = None,
    ):
        self.config = config
        self.ops = ops
        self.on_step = on_step

    def run(
        self,
        dataloader_factory: Callable[[], Iterable[Any]],
        *,
        state: LoopState | None = None,
    ) -> tuple[LoopState, list[OptimizerStep]]:
        state = state or LoopState()
        records: list[OptimizerStep] = []
        target_epochs = self._epochs_required(dataloader_factory, state)
        initial_epoch = state.epoch

        for epoch in range(state.epoch, target_epochs):
            if self._at_step_limit(state):
                break
            dataloader = self._dataloader(dataloader_factory, epoch)
            if not isinstance(dataloader, Sized):
                raise TypeError(
                    "the Blackhole training loop requires a sized dataloader "
                    "to compile exact accumulation windows"
                )
            batches_in_epoch = len(dataloader)
            if batches_in_epoch == 0:
                raise ValueError("training dataloader has zero batches")
            state.epoch = epoch
            accumulation_step = 0
            accumulation_target = 0
            accumulated_loss = 0.0
            pending_losses: list[Any] = []
            window_started = 0.0
            useful_tokens = 0
            padded_tokens = 0
            sequence_lengths: list[int] = []
            data_wait_seconds = 0.0
            self.ops.zero_grad()

            iterator = iter(dataloader)
            start_batch = state.batch_in_epoch if epoch == initial_epoch else 0
            if start_batch > batches_in_epoch:
                raise ValueError("resume batch position exceeds the dataloader")
            for _ in range(start_batch):
                next(iterator)

            batch_index = start_batch
            while batch_index < batches_in_epoch:
                wait_started = perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                batch_wait = perf_counter() - wait_started
                if self._at_step_limit(state):
                    break
                if accumulation_step == 0:
                    accumulation_target = min(
                        self.config.gradient_accumulation_steps,
                        batches_in_epoch - batch_index,
                    )
                    accumulated_loss = 0.0
                    pending_losses = []
                    # Include exposed input-pipeline wait in useful end-to-end
                    # throughput.  DataLoader workers can overlap preparation
                    # with the previous device step; only the remaining wait
                    # is charged here.
                    window_started = wait_started
                    useful_tokens = 0
                    padded_tokens = 0
                    sequence_lengths = []
                    data_wait_seconds = 0.0
                    next_optimizer_step = state.optimizer_step + 1
                    read_window_loss = (
                        next_optimizer_step
                        % self.config.loss_read_interval
                        == 0
                        or (
                            self.config.max_steps is not None
                            and next_optimizer_step
                            == self.config.max_steps
                        )
                    )

                batch_useful, batch_padded = self._batch_token_counts(batch)
                useful_tokens += batch_useful
                padded_tokens += batch_padded
                sequence_length = self._batch_sequence_length(batch)
                if sequence_length is not None:
                    sequence_lengths.append(sequence_length)
                data_wait_seconds += batch_wait
                loss = self.ops.forward_loss(batch)
                self.ops.backward(loss, accumulation_target)
                if self.config.fuse_backward_optimizer:
                    if read_window_loss:
                        pending_losses.append(loss)
                else:
                    self.ops.sync()
                    if read_window_loss:
                        accumulated_loss += self.ops.loss_value(loss)
                accumulation_step += 1
                state.micro_step += 1
                state.batch_in_epoch = batch_index + 1
                batch_index += 1

                if accumulation_step < accumulation_target:
                    continue
                # Qualification reads the unmodified accumulated gradients at
                # the exact boundary before clipping.  Auditing from inside
                # optimizer_step observes an XLA in-place clipping alias on
                # some graphs rather than the values consumed by the packed
                # optimizer, which can incorrectly report an all-zero set.
                self.ops.audit_gradient_state()
                if self.config.max_grad_norm is not None:
                    self.ops.clip_grad_norm(self.config.max_grad_norm)
                self.ops.optimizer_step()
                # optimizer_step is the TT-XLA execution barrier. Deferring
                # the loss read until after it lets backward, clipping, and
                # AdamW compile as one graph instead of forcing two graphs.
                if self.config.fuse_backward_optimizer:
                    if read_window_loss:
                        accumulated_loss = sum(
                            self.ops.loss_value(item)
                            for item in pending_losses
                        )
                self.ops.scheduler_step()
                self.ops.zero_grad()
                state.optimizer_step += 1
                record = OptimizerStep(
                    step=state.optimizer_step,
                    epoch=epoch,
                    micro_batches=accumulation_target,
                    loss=(
                        accumulated_loss / accumulation_target
                        if read_window_loss
                        else None
                    ),
                    seconds=perf_counter() - window_started,
                    learning_rate=self.ops.learning_rate(),
                    useful_tokens=useful_tokens,
                    padded_tokens=padded_tokens,
                    sequence_lengths=tuple(sequence_lengths),
                    data_wait_seconds=data_wait_seconds,
                )
                records.append(record)
                if self.on_step is not None:
                    self.on_step(record, state)
                accumulation_step = 0

            state.epoch = epoch + 1
            if state.batch_in_epoch >= batches_in_epoch:
                state.batch_in_epoch = 0

        state.stopped = self._at_step_limit(state) or state.epoch >= target_epochs
        return state, records

    def _epochs_required(
        self,
        dataloader_factory: Callable[[], Iterable[Any]],
        state: LoopState,
    ) -> int:
        if self.config.max_steps is None:
            return self.config.num_epochs
        dataloader = self._dataloader(dataloader_factory, state.epoch)
        if not isinstance(dataloader, Sized):
            raise TypeError("dataloader must be sized")
        steps_per_epoch = (
            len(dataloader) + self.config.gradient_accumulation_steps - 1
        ) // self.config.gradient_accumulation_steps
        if steps_per_epoch == 0:
            raise ValueError("training dataloader has zero batches")
        remaining = max(0, self.config.max_steps - state.optimizer_step)
        if state.batch_in_epoch:
            remaining_batches = max(0, len(dataloader) - state.batch_in_epoch)
            current_epoch_steps = (
                remaining_batches + self.config.gradient_accumulation_steps - 1
            ) // self.config.gradient_accumulation_steps
            remaining_after_current = max(0, remaining - current_epoch_steps)
            extra_epochs = 1 + (
                remaining_after_current + steps_per_epoch - 1
            ) // steps_per_epoch
        else:
            extra_epochs = (remaining + steps_per_epoch - 1) // steps_per_epoch
        return max(self.config.num_epochs, state.epoch + extra_epochs)

    def _at_step_limit(self, state: LoopState) -> bool:
        return (
            self.config.max_steps is not None
            and state.optimizer_step >= self.config.max_steps
        )

    @staticmethod
    def _dataloader(
        factory: Callable[..., Iterable[Any]],
        epoch: int,
    ) -> Iterable[Any]:
        """Use an epoch-aware deterministic factory when one is provided."""

        try:
            signature = inspect.signature(factory)
        except (TypeError, ValueError):
            # Builtins and some extension callables do not expose a signature.
            # The historical factory contract is zero-argument in that case.
            return factory()
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ]
        has_varargs = any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            for parameter in signature.parameters.values()
        )
        return factory(epoch) if positional or has_varargs else factory()

    @staticmethod
    def _batch_token_counts(batch: Any) -> tuple[int, int]:
        """Return non-padding and physical token counts without Torch imports."""

        if not isinstance(batch, Mapping):
            return 0, 0
        mask = batch.get("attention_mask")
        source = mask if mask is not None else batch.get("input_ids")
        if source is None:
            return 0, 0

        def scalar(value: Any) -> int:
            if hasattr(value, "item"):
                value = value.item()
            return int(value)

        if hasattr(source, "numel"):
            padded = int(source.numel())
            if mask is None:
                return padded, padded
            total = source.sum()
            return scalar(total), padded

        def flatten(values: Any):
            if isinstance(values, (list, tuple)):
                for item in values:
                    yield from flatten(item)
            else:
                yield values

        values = list(flatten(source))
        padded = len(values)
        useful = padded if mask is None else sum(scalar(value) for value in values)
        return useful, padded

    @staticmethod
    def _batch_sequence_length(batch: Any) -> int | None:
        """Return the compiled sequence dimension without framework imports."""

        if not isinstance(batch, Mapping):
            return None
        source = batch.get("input_ids")
        if source is None:
            source = batch.get("attention_mask")
        if source is None:
            return None
        shape = getattr(source, "shape", None)
        if shape is not None and len(shape):
            return int(shape[-1])
        if isinstance(source, (list, tuple)) and source:
            first = source[0]
            if isinstance(first, (list, tuple)):
                return len(first)
            return len(source)
        return None
