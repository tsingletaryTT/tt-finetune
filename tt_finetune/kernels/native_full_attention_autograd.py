"""Full Llama-family attention backward ownership boundary.

This is the integration boundary for the eventual one-program TT-Metal
implementation.  The first accepted stage composes the exact resident
attention core with explicit PEFT projection and RoPE schedules while hiding
the original framework backward graph.  Subsequent native phases can replace
the internals without changing Transformers or checkpoint integration.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .full_attention_block_planner import plan_full_attention_block
from .full_attention_exact_math import (
    accumulate_qkv_input_gradients_exact,
    frozen_lora_backward_exact,
    inverse_rope_gradient_exact,
)
from .native_attention_core import native_attention_core_backward
from .native_frozen_lora_autograd import exact_base_dx_k_phase
from .tt_lang_grouped2_resident_backward import (
    build_tt_lang_grouped2_resident_backward,
    resident_backward_support_error,
)
from .tt_lang_output_attention_resident_backward import (
    build_tt_lang_output_attention_resident_backward,
    full_attention_backward_arena_shape,
    output_attention_arena_shape,
    output_attention_projection_arena_shape,
    output_attention_support_error,
)
from .tt_lang_qkv_resident_backward import (
    build_tt_lang_qkv_resident_backward,
    qkv_resident_backward_arena_shape,
    qkv_resident_backward_support_error,
)


def full_attention_kv_arena_shape(
    *, tokens: int, hidden_features: int, kv_features: int
) -> tuple[int, int]:
    """Physical output arena for the exact V-then-K resident projection."""

    rows = tokens + 2 * 32 + 2 * kv_features
    return rows, hidden_features


def full_attention_kv_projection_support_error(
    *, tokens: int, hidden_features: int, kv_features: int
) -> str | None:
    """Return why the stage-2 paired projection cannot run, if anything."""

    phase = exact_base_dx_k_phase(hidden_features, kv_features)
    if phase is None:
        return (
            "no exact base dX schedule for attention K/V projection "
            f"{hidden_features}->{kv_features}"
        )
    return resident_backward_support_error(
        tokens=tokens,
        input_features=hidden_features,
        member_output_features=kv_features,
        physical_rank=32,
        base_reduction_tiles_per_phase=phase,
    )


def full_attention_qkv_projection_support_error(
    *, tokens: int, hidden_features: int, kv_features: int
) -> str | None:
    """Return why the exact one-program Q/K/V projection cannot run."""

    return qkv_resident_backward_support_error(
        tokens=tokens,
        hidden_features=hidden_features,
        kv_features=kv_features,
        physical_rank=32,
    )


def _pad_rank32(
    a: torch.Tensor, b: torch.Tensor, hidden: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rank = int(a.shape[0])
    if not 0 < rank <= 32 or b.shape[1] != rank or hidden.shape[1] != rank:
        raise ValueError("invalid logical LoRA rank for physical rank-32 path")
    padding = 32 - rank
    return (
        F.pad(a, (0, 0, 0, padding)),
        F.pad(b, (0, padding, 0, 0)),
        F.pad(hidden, (0, padding, 0, 0)),
    )


class NativeFullAttentionBackwardFunction(torch.autograd.Function):
    """Own output projection, core, inverse RoPE, and Q/K/V backward."""

    @staticmethod
    def forward(
        ctx,
        exact_output,
        hidden_states,
        probability,
        query,
        key,
        value,
        cosine,
        sine,
        head_output,
        o_hidden,
        q_hidden,
        k_hidden,
        v_hidden,
        q_weight,
        q_a,
        q_b,
        k_weight,
        k_a,
        k_b,
        v_weight,
        v_a,
        v_b,
        o_weight,
        o_a,
        o_b,
        q_scale,
        k_scale,
        v_scale,
        o_scale,
        attention_scale,
        gradient_arena,
        resident_core,
        kv_projection_arena,
        resident_kv_projection,
        output_attention_arena,
        resident_output_attention,
    ):
        ctx.hidden_shape = tuple(hidden_states.shape)
        ctx.query_shape = tuple(query.shape)
        ctx.key_shape = tuple(key.shape)
        ctx.output_shape = tuple(exact_output.shape)
        ctx.save_for_backward(
            hidden_states.detach().reshape(-1, hidden_states.shape[-1]).clone(),
            probability.detach().reshape(-1, probability.shape[-1]).clone(),
            query.detach().reshape(-1, query.shape[-1]).clone(),
            key.detach().reshape(-1, key.shape[-1]).clone(),
            value.detach().reshape(-1, value.shape[-1]).clone(),
            cosine.detach().clone(),
            sine.detach().clone(),
            head_output.detach().reshape(-1, head_output.shape[-1]).clone(),
            o_hidden.detach().reshape(-1, o_hidden.shape[-1]).clone(),
            q_hidden.detach().reshape(-1, q_hidden.shape[-1]).clone(),
            k_hidden.detach().reshape(-1, k_hidden.shape[-1]).clone(),
            v_hidden.detach().reshape(-1, v_hidden.shape[-1]).clone(),
            q_weight,
            q_a,
            q_b,
            k_weight,
            k_a,
            k_b,
            v_weight,
            v_a,
            v_b,
            o_weight,
            o_a,
            o_b,
            gradient_arena,
            kv_projection_arena,
            output_attention_arena,
        )
        ctx.scales = (
            float(q_scale), float(k_scale), float(v_scale), float(o_scale)
        )
        ctx.attention_scale = float(attention_scale)
        ctx.resident_core = bool(resident_core)
        ctx.resident_kv_projection = bool(resident_kv_projection)
        ctx.resident_output_attention = bool(resident_output_attention)
        return exact_output

    @staticmethod
    def backward(ctx, output_gradient):
        (
            hidden_states,
            probability,
            query,
            key,
            value,
            cosine,
            sine,
            head_output,
            o_hidden,
            q_hidden,
            k_hidden,
            v_hidden,
            q_weight,
            q_a,
            q_b,
            k_weight,
            k_a,
            k_b,
            v_weight,
            v_a,
            v_b,
            o_weight,
            o_a,
            o_b,
            gradient_arena,
            kv_projection_arena,
            output_attention_arena,
        ) = ctx.saved_tensors
        q_scale, k_scale, v_scale, o_scale = ctx.scales
        flat_output_gradient = output_gradient.reshape(
            -1, output_gradient.shape[-1]
        )

        query = query.reshape(ctx.query_shape)
        key = key.reshape(ctx.key_shape)
        value = value.reshape(ctx.key_shape)
        probability = probability.reshape(
            ctx.query_shape[0], ctx.query_shape[1],
            ctx.query_shape[2], ctx.query_shape[2],
        )
        full_qkv_owned = bool(
            ctx.resident_output_attention and ctx.resident_kv_projection
        )
        if ctx.resident_output_attention:
            # Phase 0+1: one TT-Metal program owns output projection backward
            # and the complete attention core. The output-projection dX is
            # sent directly from the 50-core projection island to the 60-core
            # attention island through L1/NoC and never becomes a PJRT value.
            o_a32, o_b32, o_h32 = _pad_rank32(o_a, o_b, o_hidden)
            if full_qkv_owned:
                if q_scale != k_scale or k_scale != v_scale:
                    raise RuntimeError(
                        "resident Q/K/V owner requires identical adapter scales"
                    )
                q_a32, q_b32, q_h32 = _pad_rank32(q_a, q_b, q_hidden)
                k_a32, k_b32, k_h32 = _pad_rank32(k_a, k_b, k_hidden)
                v_a32, v_b32, v_h32 = _pad_rank32(v_a, v_b, v_hidden)
            else:
                # These operands are absent from the non-QKV kernel tensor
                # maps; keeping real tensors preserves one stable Python ABI.
                q_a32, q_b32, q_h32 = q_a, q_b, q_hidden
                k_a32, k_b32, k_h32 = k_a, k_b, k_hidden
                v_a32, v_b32, v_h32 = v_a, v_b, v_hidden
            operation = build_tt_lang_output_attention_resident_backward(
                sequence_length=ctx.query_shape[2],
                hidden_features=int(o_weight.shape[0]),
                query_heads=ctx.query_shape[1],
                kv_heads=ctx.key_shape[1],
                head_dim=ctx.query_shape[3],
                physical_rank=32,
                lora_scale=o_scale,
                attention_scale=ctx.attention_scale,
                resident_qkv_projection=full_qkv_owned,
            )
            packed = operation(
                flat_output_gradient,
                o_weight,
                o_b32,
                o_h32,
                o_a32,
                head_output,
                probability.reshape(-1, ctx.query_shape[2]),
                query.reshape(-1, ctx.query_shape[3]),
                key.reshape(-1, ctx.key_shape[3]),
                value.reshape(-1, ctx.key_shape[3]),
                cosine.reshape(-1, ctx.query_shape[3]),
                sine.reshape(-1, ctx.query_shape[3]),
                hidden_states,
                q_weight, q_b32, q_h32, q_a32,
                k_weight, k_b32, k_h32, k_a32,
                v_weight, v_b32, v_h32, v_a32,
                output_attention_arena,
                output_attention_arena,
            )
            tokens = flat_output_gradient.shape[0]
            hidden_features = int(o_weight.shape[0])
            rank = int(o_a.shape[0])
            o_a_gradient = packed[
                tokens : tokens + 32, :hidden_features
            ][:rank]
            o_b_gradient = packed[
                tokens + 32 : tokens + 32 + hidden_features, :32
            ][:, :rank]
            projection_rows, _ = output_attention_projection_arena_shape(
                tokens=tokens, hidden_features=hidden_features
            )
            if full_qkv_owned:
                kv_features = int(k_weight.shape[0])
                input_gradient = packed[:tokens, :hidden_features]
                q_a_gradient = packed[
                    projection_rows : projection_rows + 32,
                    :hidden_features,
                ][:rank]
                k_a_gradient = packed[
                    projection_rows + 32 : projection_rows + 64,
                    :hidden_features,
                ][:rank]
                v_a_gradient = packed[
                    projection_rows + 64 : projection_rows + 96,
                    :hidden_features,
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
                    k_b_start + kv_features :
                    k_b_start + 2 * kv_features,
                    :32,
                ][:, :rank]
            else:
                q_rows = ctx.query_shape[1] * ctx.query_shape[2]
                kv_rows = ctx.key_shape[1] * ctx.key_shape[2]
                dq_rotated = packed[
                    projection_rows : projection_rows + q_rows,
                    : ctx.query_shape[3],
                ].reshape(ctx.query_shape)
                dk_rotated = packed[
                    projection_rows + q_rows :
                    projection_rows + q_rows + kv_rows,
                    : ctx.key_shape[3],
                ].reshape(ctx.key_shape)
                dv = packed[
                    projection_rows + q_rows + kv_rows :
                    projection_rows + q_rows + 2 * kv_rows,
                    : ctx.key_shape[3],
                ].reshape(ctx.key_shape)
        else:
            # Safe path: retain the separately qualified output projection
            # and attention-core boundaries.
            o_backward = frozen_lora_backward_exact(
                head_output,
                flat_output_gradient,
                o_weight,
                o_a,
                o_b,
                o_scale,
            )
            o_a_gradient = o_backward.a_gradient
            o_b_gradient = o_backward.b_gradient
            core_output_gradient = (
                o_backward.input_gradient.reshape(
                    ctx.query_shape[0], ctx.query_shape[2],
                    ctx.query_shape[1], ctx.query_shape[3],
                )
                .transpose(1, 2)
                .contiguous()
            )
        if (
            not full_qkv_owned
            and not ctx.resident_output_attention
            and ctx.resident_core
        ):
            dq_rotated, dk_rotated, dv = native_attention_core_backward(
                probability,
                core_output_gradient,
                query,
                key,
                value,
                scale=ctx.attention_scale,
                arena=gradient_arena,
                resident_owner=True,
            )
        elif not full_qkv_owned and not ctx.resident_output_attention:
            # Diagnostic/reference contract. Production installation always
            # selects the resident core after its shape gate succeeds.
            group = query.shape[1] // key.shape[1]
            expanded_key = key.repeat_interleave(group, dim=1)
            expanded_value = value.repeat_interleave(group, dim=1)
            probability_bf16 = probability.to(query.dtype)
            dv_heads = (
                probability_bf16.transpose(-2, -1)
                @ core_output_gradient
            )
            dp = (
                core_output_gradient
                @ expanded_value.transpose(-2, -1)
            ).float()
            ds = (
                probability
                * (dp - (dp * probability).sum(dim=-1, keepdim=True))
                * ctx.attention_scale
            ).to(query.dtype)
            dq_rotated = ds @ expanded_key
            dk_heads = ds.transpose(-2, -1) @ query
            dv = dv_heads.reshape(
                key.shape[0], key.shape[1], group,
                key.shape[2], key.shape[3],
            ).sum(dim=2)
            dk_rotated = dk_heads.reshape(
                key.shape[0], key.shape[1], group,
                key.shape[2], key.shape[3],
            ).sum(dim=2)

        # Phase 2: inverse RoPE. The fused output-attention owner performs
        # the two exact BF16 products and final add/sub while dQ/dK remain in
        # the attention island. The safe path retains the framework-equivalent
        # helper here.
        if full_qkv_owned:
            pass
        elif ctx.resident_output_attention:
            dq, dk = dq_rotated, dk_rotated
        else:
            cosine = cosine.unsqueeze(1) if cosine.ndim == 3 else cosine
            sine = sine.unsqueeze(1) if sine.ndim == 3 else sine
            dq = inverse_rope_gradient_exact(dq_rotated, cosine, sine)
            dk = inverse_rope_gradient_exact(dk_rotated, cosine, sine)

        if not full_qkv_owned:
            dq = dq.transpose(1, 2).contiguous().reshape(
                -1, q_weight.shape[0]
            )
            dk = dk.transpose(1, 2).contiguous().reshape(
                -1, k_weight.shape[0]
            )
            dv = dv.transpose(1, 2).contiguous().reshape(
                -1, v_weight.shape[0]
            )

        # Phase 3: frozen Q/K/V projections and their LoRA tails. The input
        # accumulation order is an explicit part of the exact contract.
        if full_qkv_owned:
            pass
        elif ctx.resident_kv_projection:
            if q_scale != k_scale or k_scale != v_scale:
                raise RuntimeError(
                    "resident Q/K/V owner requires identical adapter scales"
                )
            q_a32, q_b32, q_h32 = _pad_rank32(q_a, q_b, q_hidden)
            v_a32, v_b32, v_h32 = _pad_rank32(v_a, v_b, v_hidden)
            k_a32, k_b32, k_h32 = _pad_rank32(k_a, k_b, k_hidden)
            tokens, hidden_features = hidden_states.shape
            kv_features = int(k_weight.shape[0])
            error = full_attention_qkv_projection_support_error(
                tokens=tokens,
                hidden_features=hidden_features,
                kv_features=kv_features,
            )
            if error is not None:
                raise RuntimeError(
                    f"validated Q/K/V projection schedule vanished: {error}"
                )
            operation = build_tt_lang_qkv_resident_backward(
                tokens=tokens,
                hidden_features=hidden_features,
                kv_features=kv_features,
                physical_rank=32,
                scale=k_scale,
                schedule="parallel_kv_q",
            )
            packed = operation(
                hidden_states,
                dq, q_weight, q_b32, q_h32, q_a32,
                dk, k_weight, k_b32, k_h32, k_a32,
                dv, v_weight, v_b32, v_h32, v_a32,
                kv_projection_arena,
            )
            rank = int(k_a.shape[0])
            input_gradient = packed[:tokens, :hidden_features]
            q_a_gradient = packed[
                tokens : tokens + 32, :hidden_features
            ][:rank]
            k_a_gradient = packed[
                tokens + 32 : tokens + 64, :hidden_features
            ][:rank]
            v_a_gradient = packed[
                tokens + 64 : tokens + 96, :hidden_features
            ][:rank]
            q_b_start = tokens + 96
            q_b_gradient = packed[
                q_b_start : q_b_start + hidden_features, :32
            ][:, :rank]
            k_b_start = q_b_start + hidden_features
            k_b_gradient = packed[
                k_b_start : k_b_start + kv_features, :32
            ][:, :rank]
            v_b_gradient = packed[
                k_b_start + kv_features : k_b_start + 2 * kv_features,
                :32,
            ][:, :rank]
        else:
            q_backward = frozen_lora_backward_exact(
                hidden_states, dq, q_weight, q_a, q_b, q_scale
            )
            k_backward = frozen_lora_backward_exact(
                hidden_states, dk, k_weight, k_a, k_b, k_scale
            )
            v_backward = frozen_lora_backward_exact(
                hidden_states, dv, v_weight, v_a, v_b, v_scale
            )
            input_gradient = accumulate_qkv_input_gradients_exact(
                q_backward,
                k_backward,
                v_backward,
            )
            k_a_gradient = k_backward.a_gradient
            k_b_gradient = k_backward.b_gradient
            v_a_gradient = v_backward.a_gradient
            v_b_gradient = v_backward.b_gradient
            q_a_gradient = q_backward.a_gradient
            q_b_gradient = q_backward.b_gradient
        input_gradient = input_gradient.reshape(ctx.hidden_shape)

        return (
            None,                         # exact_output graph is replaced
            input_gradient,
            None, None, None, None,       # P/Q/K/V checkpoints
            None, None, None,             # cos/sin/head output
            None, None, None, None,       # saved O/Q/K/V LoRA H
            None, q_a_gradient, q_b_gradient,
            None, k_a_gradient, k_b_gradient,
            None, v_a_gradient, v_b_gradient,
            None, o_a_gradient, o_b_gradient,
            None, None, None, None, None, # scalar scales
            None,                         # gradient arena
            None,                         # resident-core flag
            None,                         # K/V projection arena
            None,                         # resident K/V projection flag
            None,                         # output+attention arena
            None,                         # resident output+attention flag
        )


def attach_full_attention_backward(
    exact_output: torch.Tensor,
    hidden_states: torch.Tensor,
    probability: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    head_output: torch.Tensor,
    o_hidden: torch.Tensor,
    q_hidden: torch.Tensor,
    k_hidden: torch.Tensor,
    v_hidden: torch.Tensor,
    *,
    q_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor, float],
    k_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor, float],
    v_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor, float],
    o_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor, float],
    attention_scale: float,
    gradient_arena: torch.Tensor,
    resident_core: bool = True,
    kv_projection_arena: torch.Tensor | None = None,
    resident_kv_projection: bool = False,
    output_attention_arena: torch.Tensor | None = None,
    resident_output_attention: bool = False,
) -> torch.Tensor:
    """Replace the whole attention-module backward with one owner."""

    plan_full_attention_block(
        batch_size=query.shape[0],
        sequence_length=query.shape[2],
        hidden_features=hidden_states.shape[-1],
        query_heads=query.shape[1],
        kv_heads=key.shape[1],
        head_dim=query.shape[3],
        logical_rank=q_parts[1].shape[0],
    )
    ranks = {parts[1].shape[0] for parts in (q_parts, k_parts, v_parts, o_parts)}
    if len(ranks) != 1:
        raise ValueError("full attention owner requires one LoRA rank")
    tokens = hidden_states.numel() // hidden_states.shape[-1]
    hidden_features = int(hidden_states.shape[-1])
    kv_features = int(k_parts[0].shape[0])
    if resident_output_attention:
        if not resident_core:
            raise ValueError(
                "resident output+attention requires the resident core"
            )
        error = output_attention_support_error(
            sequence_length=query.shape[2],
            hidden_features=hidden_features,
            query_heads=query.shape[1],
            kv_heads=key.shape[1],
            head_dim=query.shape[3],
            physical_rank=32,
        )
        if error is not None:
            raise ValueError(error)
        expected_output_arena = (
            full_attention_backward_arena_shape(
                tokens=tokens,
                hidden_features=hidden_features,
                kv_features=kv_features,
            )
            if resident_kv_projection
            else output_attention_arena_shape(
                tokens=tokens,
                hidden_features=hidden_features,
                query_heads=query.shape[1],
                kv_heads=key.shape[1],
                head_dim=query.shape[3],
            )
        )
        if output_attention_arena is None:
            raise ValueError("resident output+attention requires an arena")
        if (
            tuple(output_attention_arena.shape) != expected_output_arena
            or output_attention_arena.dtype != hidden_states.dtype
        ):
            raise ValueError(
                "invalid output+attention arena: expected "
                f"{hidden_states.dtype} {expected_output_arena}, got "
                f"{output_attention_arena.dtype} "
                f"{tuple(output_attention_arena.shape)}"
            )
    elif output_attention_arena is None:
        output_attention_arena = gradient_arena
    if resident_kv_projection:
        error = full_attention_qkv_projection_support_error(
            tokens=tokens,
            hidden_features=hidden_features,
            kv_features=kv_features,
        )
        if error is not None:
            raise ValueError(error)
        expected_arena = qkv_resident_backward_arena_shape(
            tokens=tokens,
            hidden_features=hidden_features,
            kv_features=kv_features,
        )
        if kv_projection_arena is None:
            raise ValueError("resident K/V projection requires an output arena")
        if (
            tuple(kv_projection_arena.shape) != expected_arena
            or kv_projection_arena.dtype != hidden_states.dtype
        ):
            raise ValueError(
                "invalid K/V projection arena: expected "
                f"{hidden_states.dtype} {expected_arena}, got "
                f"{kv_projection_arena.dtype} {tuple(kv_projection_arena.shape)}"
            )
    elif kv_projection_arena is None:
        # Custom autograd inputs must remain tensors. The core arena is never
        # read by the disabled projection phase and is a safe placeholder.
        kv_projection_arena = gradient_arena
    return NativeFullAttentionBackwardFunction.apply(
        exact_output.detach(),
        hidden_states,
        probability.detach(),
        query.detach(),
        key.detach(),
        value.detach(),
        cosine.detach(),
        sine.detach(),
        head_output.detach(),
        o_hidden.detach(),
        q_hidden.detach(),
        k_hidden.detach(),
        v_hidden.detach(),
        q_parts[0], q_parts[1], q_parts[2],
        k_parts[0], k_parts[1], k_parts[2],
        v_parts[0], v_parts[1], v_parts[2],
        o_parts[0], o_parts[1], o_parts[2],
        float(q_parts[3]),
        float(k_parts[3]),
        float(v_parts[3]),
        float(o_parts[3]),
        float(attention_scale),
        gradient_arena,
        bool(resident_core),
        kv_projection_arena,
        bool(resident_kv_projection),
        output_attention_arena,
        bool(resident_output_attention),
    )


attach_full_attention_backward.experimental_backend = (
    "full_attention_multiphase_exact_opt_in"
)
