"""Real causal-LM edges for the experimental resident runtime resident stack.

The resident runtime core owns the homogeneous decoder-layer stack.  This module owns
the deliberately small model-facing boundary that was previously represented
by a synthetic ``dHidden`` seed:

``token embedding -> resident layers -> final RMSNorm -> LM head -> loss``.

Only the gradient with respect to the final hidden state is requested from
autograd.  Frozen embedding, normalization and vocabulary weights therefore
do not acquire parameter gradients or duplicate optimizer state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _decoder(model: nn.Module) -> nn.Module:
    candidates = (
        "base_model.model.model",
        "model.model",
        "model",
    )
    for path in candidates:
        value: object = model
        try:
            for component in path.split("."):
                value = getattr(value, component)
        except AttributeError:
            continue
        if isinstance(value, nn.Module) and hasattr(value, "layers"):
            return value
    raise ValueError("could not locate the decoder model")


@dataclass(frozen=True)
class CausalLMHostEdges:
    """Detached host representation of the frozen model edges."""

    embedding_weight: torch.Tensor
    final_norm_weight: torch.Tensor
    lm_head_weight: torch.Tensor
    lm_head_weight_t: torch.Tensor
    final_norm_epsilon: float
    ignored_index: int = -100

    @classmethod
    def from_model(
        cls, model: nn.Module, *, ignored_index: int = -100,
    ) -> "CausalLMHostEdges":
        decoder = _decoder(model)
        input_embedding = model.get_input_embeddings()
        output_embedding = model.get_output_embeddings()
        final_norm = getattr(decoder, "norm", None)
        if not isinstance(input_embedding, nn.Module):
            raise ValueError("model has no input embedding module")
        if not isinstance(output_embedding, nn.Module):
            raise ValueError("model has no output embedding module")
        if not isinstance(final_norm, nn.Module):
            raise ValueError("decoder has no final normalization module")
        try:
            embedding_weight = input_embedding.weight.detach()
            lm_head_weight = output_embedding.weight.detach()
            final_norm_weight = final_norm.weight.detach()
            epsilon = float(final_norm.variance_epsilon)
        except AttributeError as exc:
            raise ValueError("unsupported causal-LM edge modules") from exc
        if embedding_weight.ndim != 2 or lm_head_weight.ndim != 2:
            raise ValueError("embedding and LM-head weights must be matrices")
        if embedding_weight.shape[1] != lm_head_weight.shape[1]:
            raise ValueError("input embedding and LM head hidden sizes differ")
        if final_norm_weight.shape != (embedding_weight.shape[1],):
            raise ValueError("final RMSNorm weight has the wrong hidden size")
        dtype = embedding_weight.dtype
        if dtype != torch.bfloat16:
            raise ValueError("resident runtime causal-LM edges require BF16 weights")
        if lm_head_weight.dtype != dtype or final_norm_weight.dtype != dtype:
            raise ValueError("resident runtime causal-LM edge dtypes differ")
        embedding_weight = embedding_weight.cpu().contiguous()
        lm_head_weight = lm_head_weight.cpu().contiguous()
        return cls(
            embedding_weight=embedding_weight,
            final_norm_weight=final_norm_weight.cpu().contiguous(),
            lm_head_weight=lm_head_weight,
            lm_head_weight_t=lm_head_weight.transpose(0, 1).contiguous(),
            final_norm_epsilon=epsilon,
            ignored_index=ignored_index,
        )

    def to(self, device: torch.device) -> "CausalLMDeviceEdges":
        # Keep one copy when input embeddings and the LM head are tied.  The
        # identity is detected before either host tensor is transferred.
        tied = (
            self.embedding_weight.untyped_storage().data_ptr()
            == self.lm_head_weight.untyped_storage().data_ptr()
        )
        embedding = self.embedding_weight.to(device)
        lm_head = embedding if tied else self.lm_head_weight.to(device)
        return CausalLMDeviceEdges(
            embedding_weight=embedding,
            final_norm_weight=self.final_norm_weight.to(device),
            lm_head_weight=lm_head,
            lm_head_weight_t=self.lm_head_weight_t.to(device),
            final_norm_epsilon=self.final_norm_epsilon,
            ignored_index=self.ignored_index,
        )


@dataclass(frozen=True)
class CausalLMDeviceEdges:
    """Frozen device tensors and the exact Hugging Face causal-loss spelling."""

    embedding_weight: torch.Tensor
    final_norm_weight: torch.Tensor
    lm_head_weight: torch.Tensor
    lm_head_weight_t: torch.Tensor
    final_norm_epsilon: float
    ignored_index: int = -100

    @property
    def vocabulary_size(self) -> int:
        return int(self.lm_head_weight.shape[0])

    @property
    def hidden_features(self) -> int:
        return int(self.lm_head_weight.shape[1])

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.embedding_weight)

    def final_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # This is the literal Qwen2/SmolLM2 RMSNorm order in the installed
        # Transformers version.  In particular, the reduction and rsqrt are
        # FP32 and the result is rounded back to BF16 before multiplication by
        # the BF16 gamma vector.
        input_dtype = hidden_states.dtype
        value = hidden_states.to(torch.float32)
        variance = value.pow(2).mean(-1, keepdim=True)
        value = value * torch.rsqrt(variance + self.final_norm_epsilon)
        return self.final_norm_weight * value.to(input_dtype)

    def loss_and_top_gradient(
        self,
        final_hidden: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if final_hidden.shape[-1] != self.hidden_features:
            raise ValueError("final hidden size does not match the LM head")
        if labels.shape != final_hidden.shape[:-1]:
            raise ValueError("labels must match the hidden token dimensions")

        # The resident core has a manually owned backward, so this is an
        # intentional local autograd leaf.  It builds only the final
        # RMSNorm/head/loss -> dHidden graph and never traverses decoder layers.
        hidden_leaf = final_hidden.detach().requires_grad_(True)
        normalized = self.final_norm(hidden_leaf)
        logits = F.linear(normalized, self.lm_head_weight)

        # Match transformers.loss.loss_utils.ForCausalLMLoss exactly: logits
        # are upcast, labels are padded on the right, and then shifted left.
        logits_f32 = logits.float().reshape(-1, self.vocabulary_size)
        padded_labels = F.pad(labels, (0, 1), value=self.ignored_index)
        shifted_labels = padded_labels[..., 1:].contiguous().reshape(-1)
        loss = F.cross_entropy(
            logits_f32,
            shifted_labels,
            ignore_index=self.ignored_index,
            reduction="mean",
        )
        top_gradient, = torch.autograd.grad(loss, hidden_leaf)
        return loss, top_gradient

    def streamed_loss_and_top_gradient(
        self,
        final_hidden: torch.Tensor,
        labels: torch.Tensor,
        *,
        token_block_size: int = 128,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact deterministic causal loss without a full ``T x V`` tensor.

        Each ascending token block owns its final RMSNorm, frozen LM-head
        projection, FP32 cross entropy, and dHidden production.  Only block
        loss scalars and the final hidden gradient survive the block boundary.
        The accumulation order and final mean division define the scalable
        native geometry numerical contract.
        """

        if final_hidden.shape[-1] != self.hidden_features:
            raise ValueError("final hidden size does not match the LM head")
        if labels.shape != final_hidden.shape[:-1]:
            raise ValueError("labels must match the hidden token dimensions")
        if token_block_size <= 0 or token_block_size % 32:
            raise ValueError("loss token block must be a positive tile multiple")

        flat_hidden = final_hidden.detach().reshape(-1, self.hidden_features)
        padded_labels = F.pad(labels, (0, 1), value=self.ignored_index)
        shifted_labels = padded_labels[..., 1:].contiguous().reshape(-1)
        valid_count = (shifted_labels != self.ignored_index).sum().float()

        block_losses: list[torch.Tensor] = []
        block_gradients: list[torch.Tensor] = []
        for start in range(0, flat_hidden.shape[0], token_block_size):
            end = min(flat_hidden.shape[0], start + token_block_size)
            hidden_block = flat_hidden[start:end].requires_grad_(True)
            normalized = self.final_norm(hidden_block)
            logits = F.linear(normalized, self.lm_head_weight)
            block_loss = F.cross_entropy(
                logits.float(),
                shifted_labels[start:end],
                ignore_index=self.ignored_index,
                reduction="sum",
            )
            block_gradient, = torch.autograd.grad(block_loss, hidden_block)
            if final_hidden.device.type == "xla":
                # A lazy-XLA graph otherwise keeps every token-block logits
                # allocation live until the outer training-step marker.  At
                # Qwen B2/S4096 that recreates the T*V footprint as dozens of
                # separate 77.8 MB buffers and fragments device DRAM.  This is
                # the exact safe/reference boundary: only one loss block is
                # materialized at a time, and no tensor is copied to the host.
                import torch_xla
                import torch_xla.core.xla_model as xm

                xm.mark_step()
                torch_xla.sync(wait=True)
            block_losses.append(block_loss)
            block_gradients.append(block_gradient)

        loss_sum = block_losses[0]
        for block_loss in block_losses[1:]:
            loss_sum = loss_sum + block_loss
        loss = loss_sum / valid_count
        top_gradient = torch.cat(block_gradients, dim=0) / valid_count
        return loss, top_gradient.reshape_as(final_hidden)

    def allocate_native_workspace(
        self,
        sequence_length: int,
        *,
        forward_reduction_tiles_per_phase: int = 2,
        backward_reduction_tiles_per_phase: int = 1,
    ) -> "CausalLMNativeWorkspace":
        """Allocate the explicit DPS buffers for the bounded native bridge."""

        dimensions = (
            sequence_length, self.hidden_features, self.vocabulary_size,
        )
        if any(value <= 0 or value % 32 for value in dimensions):
            raise ValueError("native causal-LM bridge requires 32-aligned shapes")
        return CausalLMNativeWorkspace(
            logits=self.lm_head_weight.new_zeros(
                (sequence_length, self.vocabulary_size)
            ),
            normalized_gradient=self.lm_head_weight.new_zeros(
                (sequence_length, self.hidden_features)
            ),
            forward_reduction_tiles_per_phase=(
                forward_reduction_tiles_per_phase
            ),
            backward_reduction_tiles_per_phase=(
                backward_reduction_tiles_per_phase
            ),
        )

    def native_bounded_loss_and_top_gradient(
        self,
        final_hidden: torch.Tensor,
        labels: torch.Tensor,
        workspace: "CausalLMNativeWorkspace",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compile-bounded exact bridge while the fully resident edge lands.

        The two vocabulary matmuls are native bounded custom operations.  The
        accepted cross-entropy and RMSNorm spellings remain ordinary device
        operations for now, so logits/dLogits still materialize in DRAM.  This
        method is deliberately separate from the safe default and is used to
        unblock matched full-model resident runtime qualification; the final owner will
        replace these materialized tensors with distributed L1 shards.
        """

        from tt_finetune.kernels.tt_lang_frozen_linear_backward import (
            build_tt_lang_frozen_linear_backward_dx,
        )
        from tt_finetune.kernels.tt_lang_frozen_linear_forward import (
            build_tt_lang_frozen_linear_forward,
        )

        if final_hidden.shape[-1] != self.hidden_features:
            raise ValueError("final hidden size does not match the LM head")
        if labels.shape != final_hidden.shape[:-1]:
            raise ValueError("labels must match the hidden token dimensions")
        tokens = final_hidden.numel() // self.hidden_features
        if tuple(workspace.logits.shape) != (tokens, self.vocabulary_size):
            raise ValueError("native logits workspace has the wrong shape")
        if tuple(workspace.normalized_gradient.shape) != (
            tokens, self.hidden_features,
        ):
            raise ValueError("native dNormalized workspace has the wrong shape")

        hidden_leaf = final_hidden.detach().requires_grad_(True)
        normalized = self.final_norm(hidden_leaf)
        forward = build_tt_lang_frozen_linear_forward(
            tokens=tokens,
            input_features=self.hidden_features,
            output_features=self.vocabulary_size,
            reduction_tiles_per_phase=(
                workspace.forward_reduction_tiles_per_phase
            ),
        )
        logits = forward(
            normalized.reshape(tokens, self.hidden_features),
            self.lm_head_weight_t,
            workspace.logits,
        )

        # Treat the exact native logits as the explicit top-loss leaf.  The
        # vocabulary-weight path is frozen and dHidden is owned below.
        logits_leaf = logits.detach().requires_grad_(True)
        logits_f32 = logits_leaf.float().reshape(-1, self.vocabulary_size)
        padded_labels = F.pad(labels, (0, 1), value=self.ignored_index)
        shifted_labels = padded_labels[..., 1:].contiguous().reshape(-1)
        loss = F.cross_entropy(
            logits_f32,
            shifted_labels,
            ignore_index=self.ignored_index,
            reduction="mean",
        )
        dlogits, = torch.autograd.grad(loss, logits_leaf)

        backward = build_tt_lang_frozen_linear_backward_dx(
            tokens=tokens,
            input_features=self.hidden_features,
            output_features=self.vocabulary_size,
            reduction_tiles_per_phase=(
                workspace.backward_reduction_tiles_per_phase
            ),
        )
        dnormalized = backward(
            dlogits.reshape(tokens, self.vocabulary_size),
            self.lm_head_weight,
            workspace.normalized_gradient,
        )
        top_gradient, = torch.autograd.grad(
            normalized,
            hidden_leaf,
            grad_outputs=dnormalized.reshape_as(normalized),
        )
        return loss, top_gradient


@dataclass(frozen=True)
class CausalLMNativeWorkspace:
    logits: torch.Tensor
    normalized_gradient: torch.Tensor
    forward_reduction_tiles_per_phase: int
    backward_reduction_tiles_per_phase: int


__all__ = [
    "CausalLMDeviceEdges",
    "CausalLMHostEdges",
    "CausalLMNativeWorkspace",
]
