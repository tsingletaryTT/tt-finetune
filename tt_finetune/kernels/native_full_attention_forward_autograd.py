"""Autograd owner for the exact resident attention forward and v90 backward.

The forward is one TT-Metal custom call from hidden states through the O
projection.  Its single BF16 result is a private page arena whose first region
is the module output and whose remaining regions are exact v90 checkpoints.

Backward consumes the private forward arena directly.  Q/K/V, head output,
rank checkpoints and the two-page raw-FP32 probability representation never
become framework tensors and never cross a PJRT/DRAM boundary between owners.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .tt_lang_attention_forward_qkv_resident import (
    build_tt_lang_attention_forward_qkv_resident,
)
from .tt_lang_resident_attention_forward_owner import (
    build_tt_lang_resident_attention_forward_owner,
)
from .tt_lang_lora import pad_lora_weights
from .tt_lang_output_attention_resident_backward import (
    build_tt_lang_output_attention_resident_backward,
    output_attention_projection_arena_shape,
)


def _pack_qwen_grouped_qkv_lora(
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    k_a: torch.Tensor,
    k_b: torch.Tensor,
    v_a: torch.Tensor,
    v_b: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Pack Qwen's logical rank-48 reference contract into rank-64 tiles.

    Q/K/V A share one [Q|K|V] operand.  B remains projection-specific, but
    each member is placed at its corresponding position in the grouped rank
    axis so the native delta matmul can consume one selected grouped-H tile.
    """

    rank = int(q_a.shape[0])
    if rank != 16 or tuple(k_a.shape) != tuple(q_a.shape) or tuple(
        v_a.shape
    ) != tuple(q_a.shape):
        raise ValueError("grouped QKV LoRA currently requires three rank-16 A tensors")
    physical_rank = 64
    # Keep A at its logical rank-48 shape. This deliberately selects TTNN's
    # same 2-D multicast matmul family as the untouched Qwen graph; padding it
    # to 64 in Python before lowering can select a different numerical path.
    grouped_a = torch.cat((q_a, k_a, v_a), dim=0)

    def pack_b(value: torch.Tensor, offset: int) -> torch.Tensor:
        packed = value.new_zeros((value.shape[0], physical_rank))
        packed[:, offset : offset + rank].copy_(value)
        return packed

    return (
        grouped_a,
        pack_b(q_b, 0),
        pack_b(k_b, rank),
        pack_b(v_b, 2 * rank),
    )


def _pack_smollm_grouped_qvk_lora(
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    k_a: torch.Tensor,
    k_b: torch.Tensor,
    v_a: torch.Tensor,
    v_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack SmolLM2's untouched logical rank-24 ``[Q|V|K]`` contract."""

    rank = int(q_a.shape[0])
    if rank != 8 or tuple(k_a.shape) != tuple(q_a.shape) or tuple(
        v_a.shape
    ) != tuple(q_a.shape):
        raise ValueError("Smol grouped Q/V/K LoRA requires three rank-8 A tensors")
    grouped_a = torch.cat((q_a, v_a, k_a), dim=0)

    def pack_b(value: torch.Tensor, offset: int) -> torch.Tensor:
        packed = value.new_zeros((value.shape[0], 32))
        packed[:, offset : offset + rank].copy_(value)
        return packed

    return (
        grouped_a,
        pack_b(q_b, 0),
        pack_b(k_b, 2 * rank),
        pack_b(v_b, rank),
    )


class NativeFullAttentionForwardFunction(torch.autograd.Function):
    """Own exact attention forward and route its private arena to v90."""

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        q_weight, q_weight_t, q_bias_tiled, q_a, q_b,
        k_weight, k_weight_t, k_bias_tiled, k_a, k_b,
        v_weight, v_weight_t, v_bias_tiled, v_a, v_b,
        o_weight, o_weight_t, o_a, o_b,
        cosine, sine, mask,
        forward_arena, backward_arena,
        q_scale, k_scale, v_scale, o_scale, attention_scale,
        query_heads, kv_heads, head_dim, has_bias,
        o_lora_k_tiles_per_phase, dedicated_o_rank_producer,
        canonical_native_o_path, exact_attention_tail,
        canonical_bf16_attention,
    ):
        if not (q_scale == k_scale == v_scale == o_scale):
            raise ValueError("full resident attention requires one LoRA scale")
        hidden_shape = tuple(hidden_states.shape)
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        tokens, hidden_features = flat_hidden.shape
        kv_features = int(k_weight.shape[0])
        q_a32, q_b32 = pad_lora_weights(q_a, q_b)
        k_a32, k_b32 = pad_lora_weights(k_a, k_b)
        v_a32, v_b32 = pad_lora_weights(v_a, v_b)
        o_a32, o_b32 = pad_lora_weights(o_a, o_b)
        grouped_qkv_lora = bool(
            has_bias
            and int(q_a.shape[0]) == 16
            and int(k_a.shape[0]) == 16
            and int(v_a.shape[0]) == 16
        )
        canonical_smol_qvk = bool(
            canonical_native_o_path
            and not has_bias
            and int(q_a.shape[0]) == 8
            and int(k_a.shape[0]) == 8
            and int(v_a.shape[0]) == 8
        )
        if canonical_smol_qvk:
            grouped_a, q_b32, k_b32, v_b32 = _pack_smollm_grouped_qvk_lora(
                q_a, q_b, k_a, k_b, v_a, v_b
            )
            # The outer artifact consumes this logical A tensor in the same
            # native TTNN matmul selected by the untouched graph, retains its
            # rank-24 result in compiler-private L1, and remaps all three dead
            # A slots to that result before entering the resident program.
            q_a32 = k_a32 = v_a32 = grouped_a
        elif grouped_qkv_lora:
            grouped_a, q_b64, k_b64, v_b64 = _pack_qwen_grouped_qkv_lora(
                q_a, q_b, k_a, k_b, v_a, v_b
            )
            grouped_h = flat_hidden @ grouped_a.transpose(0, 1)
            grouped_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
            grouped_bias = torch.cat(
                (
                    q_bias_tiled[0, : q_weight.shape[0]],
                    k_bias_tiled[0, : k_weight.shape[0]],
                    v_bias_tiled[0, : v_weight.shape[0]],
                ),
                dim=0,
            )
            grouped_base = F.linear(
                flat_hidden, grouped_weight, grouped_bias
            )
            # The resident reader consumes Q/K from physical tile 0 and V
            # from physical tile 1. Reuse the three dead A operand slots to
            # preserve the established custom-call ABI while this exact 2-D
            # producer is being absorbed into the same TT-Metal invocation.
            q_a32 = k_a32 = v_a32 = grouped_h
            q_weight_t = k_weight_t = v_weight_t = grouped_base
            q_b32, k_b32, v_b32 = q_b64, k_b64, v_b64
        operation_builder = (
            build_tt_lang_resident_attention_forward_owner
            if canonical_native_o_path
            else build_tt_lang_attention_forward_qkv_resident
        )
        operation_arguments = {
            "sequence_length": tokens,
            "hidden_features": hidden_features,
            "kv_features": kv_features,
            "query_heads": int(query_heads),
            "kv_heads": int(kv_heads),
            "head_dim": int(head_dim),
            "physical_rank": 32,
            "lora_scale": float(q_scale),
            "attention_scale": float(attention_scale),
            # Qwen's full fused graph may select a two-tile frozen-base
            # reduction epoch even though the independently materialized
            # projection qualifier selects one.  Keep this shape-specific
            # while full-model exactness determines the production contract.
            "base_k_tiles_per_phase": 1,
            "has_bias": bool(has_bias),
            "grouped_qkv_lora": grouped_qkv_lora,
            "o_lora_k_tiles_per_phase": o_lora_k_tiles_per_phase,
        }
        if canonical_native_o_path:
            operation_arguments.update(
                logical_rank=int(o_a.shape[0]),
                canonical_rank_k_tiles_per_phase=2,
                canonical_grouped_qkv_rank=canonical_smol_qvk,
                exact_attention_tail=bool(exact_attention_tail),
                canonical_bf16_attention=bool(canonical_bf16_attention),
            )
        else:
            operation_arguments["dedicated_o_rank_producer"] = bool(
                dedicated_o_rank_producer
            )
        operation = operation_builder(**operation_arguments)
        arena = operation(
            flat_hidden,
            q_weight_t, q_bias_tiled, q_a32, q_b32,
            k_weight_t, k_bias_tiled, k_a32, k_b32,
            v_weight_t, v_bias_tiled, v_a32, v_b32,
            o_weight_t, o_a32, o_b32,
            cosine.reshape(-1, int(head_dim)),
            sine.reshape(-1, int(head_dim)),
            mask.reshape(tokens, tokens),
            # The accepted attention-only owner does not consume this slot.
            # It keeps a stable ABI with the opt-in attention-block owner,
            # where the actual input RMSNorm gamma is passed here.
            q_bias_tiled,
            q_bias_tiled,
            forward_arena,
        )
        ctx.hidden_shape = hidden_shape
        ctx.tokens = tokens
        ctx.hidden_features = hidden_features
        ctx.kv_features = kv_features
        ctx.query_heads = int(query_heads)
        ctx.kv_heads = int(kv_heads)
        ctx.head_dim = int(head_dim)
        ctx.scale = float(q_scale)
        ctx.attention_scale = float(attention_scale)
        ctx.save_for_backward(
            arena,
            flat_hidden,
            q_weight, q_a, q_b,
            k_weight, k_a, k_b,
            v_weight, v_a, v_b,
            o_weight, o_a, o_b,
            cosine, sine,
            backward_arena,
        )
        # The output region is first and row-aligned by construction.
        return arena[:tokens, :hidden_features].reshape(hidden_shape)

    @staticmethod
    def backward(ctx, output_gradient):
        (
            forward_arena,
            hidden,
            q_weight, q_a, q_b,
            k_weight, k_a, k_b,
            v_weight, v_a, v_b,
            o_weight, o_a, o_b,
            cosine, sine,
            backward_arena,
        ) = ctx.saved_tensors
        tokens = ctx.tokens
        hidden_features = ctx.hidden_features
        kv_features = ctx.kv_features
        q_a32, q_b32 = pad_lora_weights(q_a, q_b)
        k_a32, k_b32 = pad_lora_weights(k_a, k_b)
        v_a32, v_b32 = pad_lora_weights(v_a, v_b)
        o_a32, o_b32 = pad_lora_weights(o_a, o_b)
        operation = build_tt_lang_output_attention_resident_backward(
            sequence_length=tokens,
            hidden_features=hidden_features,
            query_heads=ctx.query_heads,
            kv_heads=ctx.kv_heads,
            head_dim=ctx.head_dim,
            physical_rank=32,
            lora_scale=ctx.scale,
            attention_scale=ctx.attention_scale,
            resident_qkv_projection=True,
            forward_checkpoint_arena=True,
        )
        packed = operation(
            output_gradient.reshape(tokens, hidden_features),
            o_weight, o_b32, o_a32,
            cosine.reshape(-1, ctx.head_dim),
            sine.reshape(-1, ctx.head_dim),
            hidden,
            q_weight, q_b32, q_a32,
            k_weight, k_b32, k_a32,
            v_weight, v_b32, v_a32,
            forward_arena,
            backward_arena,
        )
        rank = int(q_a.shape[0])
        projection_rows, _ = output_attention_projection_arena_shape(
            tokens=tokens, hidden_features=hidden_features
        )
        input_gradient = packed[:tokens, :hidden_features].reshape(
            ctx.hidden_shape
        )
        o_a_gradient = packed[
            tokens : tokens + 32, :hidden_features
        ][:rank]
        o_b_gradient = packed[
            tokens + 32 : tokens + 32 + hidden_features, :32
        ][:, :rank]
        q_a_gradient = packed[
            projection_rows : projection_rows + 32, :hidden_features
        ][:rank]
        k_a_gradient = packed[
            projection_rows + 32 : projection_rows + 64, :hidden_features
        ][:rank]
        v_a_gradient = packed[
            projection_rows + 64 : projection_rows + 96, :hidden_features
        ][:rank]
        q_b_start = projection_rows + 96
        q_b_gradient = packed[
            q_b_start : q_b_start + hidden_features, :32
        ][:, :rank]
        k_b_start = q_b_start + hidden_features
        k_b_gradient = packed[
            k_b_start : k_b_start + kv_features, :32
        ][:, :rank]
        v_b_gradient = packed[
            k_b_start + kv_features : k_b_start + 2 * kv_features, :32
        ][:, :rank]

        gradients = [None] * 39
        gradients[0] = input_gradient
        gradients[4], gradients[5] = q_a_gradient, q_b_gradient
        gradients[9], gradients[10] = k_a_gradient, k_b_gradient
        gradients[14], gradients[15] = v_a_gradient, v_b_gradient
        gradients[18], gradients[19] = o_a_gradient, o_b_gradient
        return tuple(gradients)


def resident_full_attention_forward(
    hidden_states: torch.Tensor,
    *,
    q_parts, k_parts, v_parts, o_parts,
    q_weight_t: torch.Tensor, k_weight_t: torch.Tensor,
    v_weight_t: torch.Tensor, o_weight_t: torch.Tensor,
    q_bias_tiled: torch.Tensor, k_bias_tiled: torch.Tensor,
    v_bias_tiled: torch.Tensor,
    cosine: torch.Tensor, sine: torch.Tensor, mask: torch.Tensor,
    forward_arena: torch.Tensor, backward_arena: torch.Tensor,
    query_heads: int, kv_heads: int, head_dim: int,
    attention_scale: float, has_bias: bool,
    o_lora_k_tiles_per_phase: int | None = None,
    dedicated_o_rank_producer: bool = False,
    canonical_native_o_path: bool = False,
    exact_attention_tail: bool = False,
    canonical_bf16_attention: bool = False,
) -> torch.Tensor:
    return NativeFullAttentionForwardFunction.apply(
        hidden_states,
        q_parts[0], q_weight_t, q_bias_tiled, q_parts[1], q_parts[2],
        k_parts[0], k_weight_t, k_bias_tiled, k_parts[1], k_parts[2],
        v_parts[0], v_weight_t, v_bias_tiled, v_parts[1], v_parts[2],
        o_parts[0], o_weight_t, o_parts[1], o_parts[2],
        cosine, sine, mask,
        forward_arena, backward_arena,
        float(q_parts[3]), float(k_parts[3]),
        float(v_parts[3]), float(o_parts[3]), float(attention_scale),
        int(query_heads), int(kv_heads), int(head_dim), bool(has_bias),
        o_lora_k_tiles_per_phase, bool(dedicated_o_rank_producer),
        bool(canonical_native_o_path),
        bool(exact_attention_tail),
        bool(canonical_bf16_attention),
    )


resident_full_attention_forward.experimental_backend = (
    "full_attention_forward_v1_exact_opt_in"
)
