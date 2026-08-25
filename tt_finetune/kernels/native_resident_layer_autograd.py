"""Autograd boundary for the exact resident Transformer-layer owners.

Forward keeps the accepted full-layer checkpoint arena private.  Backward
feeds that arena directly to the accepted twelve-program layer owner and
returns gradients only for the logical LoRA parameters and the residual
stream.  Frozen base weights and RMSNorm parameters remain ordinary inputs so
the owner has one canonical view of model state without registering duplicate
trainable buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.nn.functional as F

from .tt_lang_lora import pad_lora_weights
from .tt_lang_output_attention_resident_backward import (
    output_attention_projection_arena_shape,
)
from .tt_lang_resident_layer_autograd_owner import (
    build_tt_lang_resident_layer_autograd_owner,
)
from .tt_lang_resident_layer_forward_trainable_owner import (
    build_tt_lang_resident_layer_forward_trainable_owner,
)
from .tt_lang_resident_layer_forward_compact_owner import (
    build_tt_lang_resident_layer_forward_compact_owner,
)
from .native_full_attention_forward_autograd import (
    _pack_qwen_grouped_qkv_lora,
    _pack_smollm_grouped_qvk_lora,
)


# Opt-in Python-side references used only by correctness diagnostics.  They
# do not add an output to StableHLO or alter any numerical program.  Forward
# overwrites the slot on every layer, leaving layer N-1; backward records only
# its first invocation, which is the same top layer.
_LAYER_AUTOGRAD_DIAGNOSTIC: dict[str, object] = {}


def reset_layer_autograd_diagnostic() -> None:
    _LAYER_AUTOGRAD_DIAGNOSTIC.clear()


@dataclass(frozen=True)
class ResidentLayerAutogradConfig:
    source_bundle: str
    sequence_length: int
    hidden_features: int
    intermediate_features: int
    kv_features: int
    query_heads: int
    kv_heads: int
    head_dim: int
    physical_rank: int
    lora_scale: float
    attention_scale: float
    input_rmsnorm_epsilon: float
    post_attention_rmsnorm_epsilon: float
    has_bias: bool
    compact_forward: bool = False
    o_lora_k_tiles_per_phase: int | None = None
    canonical_exact_attention: bool = False


class NativeResidentTransformerLayerFunction(torch.autograd.Function):
    """Connect the full-layer forward arena to the exact backward owner."""

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        q_weight, q_weight_t, q_bias, q_a, q_b,
        k_weight, k_weight_t, k_bias, k_a, k_b,
        v_weight, v_weight_t, v_bias, v_a, v_b,
        o_weight, o_weight_t, o_a, o_b,
        cosine, sine, mask,
        input_gamma, post_gamma, packed_static,
        gate_weight, gate_a, gate_b,
        up_weight, up_a, up_b,
        down_weight, down_a, down_b,
        forward_arena, backward_arena,
        gate_da_seed, gate_db_seed, up_da_seed, up_db_seed,
        down_da_seed, down_db_seed,
        fp32_seed, fp32_seed_two, bf16_seed,
        config,
    ):
        hidden_shape = tuple(hidden_states.shape)
        flat_hidden = hidden_states.reshape(
            config.sequence_length, config.hidden_features
        )
        physical = tuple(
            pad_lora_weights(a, b)
            for a, b in (
                (q_a, q_b), (k_a, k_b), (v_a, v_b), (o_a, o_b),
                (gate_a, gate_b), (up_a, up_b), (down_a, down_b),
            )
        )
        (
            (q_a32, q_b32), (k_a32, k_b32), (v_a32, v_b32),
            (o_a32, o_b32), (gate_a32, gate_b32),
            (up_a32, up_b32), (down_a32, down_b32),
        ) = physical
        canonical_smol_qvk = bool(
            config.canonical_exact_attention
            and not config.has_bias
            and int(q_a.shape[0]) == 8
        )
        canonical_qwen_qkv = bool(
            config.canonical_exact_attention
            and config.has_bias
            and int(q_a.shape[0]) == 16
        )
        if canonical_smol_qvk:
            grouped_a, q_b32, k_b32, v_b32 = (
                _pack_smollm_grouped_qvk_lora(
                    q_a, q_b, k_a, k_b, v_a, v_b
                )
            )
            q_a32 = k_a32 = v_a32 = grouped_a
        elif canonical_qwen_qkv:
            grouped_a, q_b32, k_b32, v_b32 = (
                _pack_qwen_grouped_qkv_lora(
                    q_a, q_b, k_a, k_b, v_a, v_b
                )
            )
            grouped_weight = torch.cat(
                (q_weight, k_weight, v_weight), dim=0
            )
            grouped_bias = torch.cat(
                (
                    q_bias[0, : q_weight.shape[0]],
                    k_bias[0, : k_weight.shape[0]],
                    v_bias[0, : v_weight.shape[0]],
                ),
                dim=0,
            )
            if os.environ.get(
                "TT_FINETUNE_QWEN_LAYER_EXTERNAL_PREFIX", "0"
            ) == "1":
                # Match Qwen2RMSNorm's operation and cast boundaries exactly.
                # torch.nn.functional.rms_norm lowers to a different reduction
                # schedule on TT and is not a bit-exact diagnostic reference.
                normalized_fp32 = flat_hidden.float()
                variance = normalized_fp32.pow(2).mean(
                    -1, keepdim=True
                )
                normalized_fp32 = normalized_fp32 * torch.rsqrt(
                    variance + config.input_rmsnorm_epsilon
                )
                normalized = input_gamma[0] * normalized_fp32.to(
                    flat_hidden.dtype
                )
                grouped_h = normalized @ grouped_a.transpose(0, 1)
                grouped_base = F.linear(
                    normalized, grouped_weight, grouped_bias
                )
                q_a32 = k_a32 = v_a32 = grouped_h
                q_weight_t = k_weight_t = v_weight_t = grouped_base
                q_bias = k_bias = v_bias = grouped_bias.reshape(1, -1)
            else:
                # The compact attention block consumes these static grouped
            # operands only after its exact private input RMSNorm. It then
            # creates grouped H/base as compiler-owned L1 tensors inside the
            # same artifact-v4 call.
                q_a32 = k_a32 = v_a32 = grouped_a
                # Keep the grouped frozen weight in the same physical
                # orientation consumed by torch.nn.functional.linear.  The
                # resident descriptor carries transpose_b=True, which lets
                # TTNN select the identical reduction/program schedule rather
                # than treating a pre-transposed L1 operand as a plain
                # matmul.
                q_weight_t = k_weight_t = v_weight_t = (
                    grouped_weight.contiguous()
                )
                # Preserve the broadcast bias as one logical row.  Expanding
                # it to every token changes TTNN Linear's fused-bias program
                # selection and therefore its BF16 reduction/packing
                # schedule, even though the mathematical values repeat.
                grouped_bias = grouped_bias.reshape(1, -1).contiguous()
                q_bias = k_bias = v_bias = grouped_bias
        forward_builder = (
            build_tt_lang_resident_layer_forward_compact_owner
            if config.compact_forward
            else build_tt_lang_resident_layer_forward_trainable_owner
        )
        operation = forward_builder(
            sequence_length=config.sequence_length,
            hidden_features=config.hidden_features,
            intermediate_features=config.intermediate_features,
            kv_features=config.kv_features,
            query_heads=config.query_heads,
            kv_heads=config.kv_heads,
            head_dim=config.head_dim,
            physical_rank=config.physical_rank,
            lora_scale=config.lora_scale,
            attention_scale=config.attention_scale,
            has_bias=config.has_bias,
            input_rmsnorm_epsilon=config.input_rmsnorm_epsilon,
            post_attention_rmsnorm_epsilon=(
                config.post_attention_rmsnorm_epsilon
            ),
            o_lora_k_tiles_per_phase=(
                config.o_lora_k_tiles_per_phase
            ),
            **(
                {
                    "canonical_native_o_path": True,
                    "exact_attention_tail": True,
                    "canonical_bf16_attention": False,
                    "canonical_grouped_qkv_rank": canonical_smol_qvk,
                    "logical_rank": int(o_a.shape[0]),
                }
                if config.compact_forward
                and config.canonical_exact_attention
                else {}
            ),
        )
        forward_arguments = [
            flat_hidden,
            q_weight_t, q_bias, q_a32, q_b32,
            k_weight_t, k_bias, k_a32, k_b32,
            v_weight_t, v_bias, v_a32, v_b32,
            o_weight_t, o_a32, o_b32,
            cosine.reshape(config.sequence_length, config.head_dim),
            sine.reshape(config.sequence_length, config.head_dim),
            mask.reshape(config.sequence_length, config.sequence_length),
            input_gamma, packed_static,
            gate_a32, gate_b32, up_a32, up_b32, down_a32, down_b32,
        ]
        if config.compact_forward:
            # The compact artifact has three explicit public scaffolds that
            # become private L1 tensors after composition: FP32 RMS storage,
            # a BF16 RMS/result template, and the hidden-sized handoff target.
            # They do not cross the autograd boundary as semantic outputs.
            forward_arguments.extend((fp32_seed, bf16_seed, bf16_seed))
        forward_arguments.append(forward_arena)
        arena = operation(*forward_arguments)
        if os.environ.get("TT_FINETUNE_CAPTURE_LAYER_AUTOGRAD") == "1":
            _LAYER_AUTOGRAD_DIAGNOSTIC["top_forward_arena"] = arena
            _LAYER_AUTOGRAD_DIAGNOSTIC["top_hidden"] = flat_hidden
        ctx.config = config
        ctx.hidden_shape = hidden_shape
        ctx.logical_rank = int(q_a.shape[0])
        ctx.save_for_backward(
            arena, flat_hidden, input_gamma, post_gamma,
            q_weight, q_a, q_b, k_weight, k_a, k_b,
            v_weight, v_a, v_b, o_weight, o_a, o_b,
            cosine, sine,
            gate_weight, gate_a, gate_b,
            up_weight, up_a, up_b,
            down_weight, down_a, down_b,
            backward_arena,
            gate_da_seed, gate_db_seed, up_da_seed, up_db_seed,
            down_da_seed, down_db_seed,
            fp32_seed, fp32_seed_two, bf16_seed,
        )
        return arena[
            : config.sequence_length, : config.hidden_features
        ].reshape(hidden_shape)

    @staticmethod
    def backward(ctx, output_gradient):
        config = ctx.config
        (
            arena, hidden, input_gamma, post_gamma,
            q_weight, q_a, q_b, k_weight, k_a, k_b,
            v_weight, v_a, v_b, o_weight, o_a, o_b,
            cosine, sine,
            gate_weight, gate_a, gate_b,
            up_weight, up_a, up_b,
            down_weight, down_a, down_b,
            backward_arena,
            gate_da_seed, gate_db_seed, up_da_seed, up_db_seed,
            down_da_seed, down_db_seed,
            fp32_seed, fp32_seed_two, bf16_seed,
        ) = ctx.saved_tensors
        (
            (q_a32, q_b32), (k_a32, k_b32), (v_a32, v_b32),
            (o_a32, o_b32), (gate_a32, gate_b32),
            (up_a32, up_b32), (down_a32, down_b32),
        ) = tuple(
            pad_lora_weights(a, b)
            for a, b in (
                (q_a, q_b), (k_a, k_b), (v_a, v_b), (o_a, o_b),
                (gate_a, gate_b), (up_a, up_b), (down_a, down_b),
            )
        )
        operation = build_tt_lang_resident_layer_autograd_owner(
            source_bundle=config.source_bundle,
            sequence_length=config.sequence_length,
            hidden_features=config.hidden_features,
            intermediate_features=config.intermediate_features,
            kv_features=config.kv_features,
            query_heads=config.query_heads,
            kv_heads=config.kv_heads,
            head_dim=config.head_dim,
            physical_rank=config.physical_rank,
            lora_scale=config.lora_scale,
            attention_scale=config.attention_scale,
            rmsnorm_epsilon=config.input_rmsnorm_epsilon,
        )
        operation_arguments = (
            output_gradient.reshape(
                config.sequence_length, config.hidden_features
            ),
            hidden, input_gamma, post_gamma,
            fp32_seed, fp32_seed_two, bf16_seed,
            o_weight, o_b32, o_a32,
            cosine.reshape(config.sequence_length, config.head_dim),
            sine.reshape(config.sequence_length, config.head_dim),
            q_weight, q_b32, q_a32,
            k_weight, k_b32, k_a32,
            v_weight, v_b32, v_a32,
            down_weight, down_b32, down_a32,
            gate_weight, gate_b32, gate_a32,
            up_weight, up_b32, up_a32,
            arena,
            backward_arena,
            gate_da_seed, gate_db_seed, up_da_seed, up_db_seed,
            down_da_seed, down_db_seed,
        )
        result = operation(*operation_arguments)
        if (
            os.environ.get("TT_FINETUNE_CAPTURE_LAYER_AUTOGRAD") == "1"
            and "top_backward_result" not in _LAYER_AUTOGRAD_DIAGNOSTIC
        ):
            _LAYER_AUTOGRAD_DIAGNOSTIC["top_output_gradient"] = output_gradient
            _LAYER_AUTOGRAD_DIAGNOSTIC["top_backward_result"] = result
            _LAYER_AUTOGRAD_DIAGNOSTIC["top_backward_operation"] = operation
            _LAYER_AUTOGRAD_DIAGNOSTIC["top_backward_arguments"] = (
                operation_arguments
            )
            _LAYER_AUTOGRAD_DIAGNOSTIC["top_backward_parameters"] = (
                q_weight, q_a32, q_b32,
                k_weight, k_a32, k_b32,
                v_weight, v_a32, v_b32,
                o_weight, o_a32, o_b32,
            )
        attention, gate_da, gate_db, up_da, up_db, down_da, down_db = result
        rank = ctx.logical_rank
        hidden_features = config.hidden_features
        intermediate_features = config.intermediate_features
        kv_features = config.kv_features
        projection_rows, _ = output_attention_projection_arena_shape(
            tokens=config.sequence_length,
            hidden_features=hidden_features,
        )
        q_da = attention[
            projection_rows : projection_rows + 32, :hidden_features
        ][:rank]
        k_da = attention[
            projection_rows + 32 : projection_rows + 64, :hidden_features
        ][:rank]
        v_da = attention[
            projection_rows + 64 : projection_rows + 96, :hidden_features
        ][:rank]
        q_db_start = projection_rows + 96
        q_db = attention[
            q_db_start : q_db_start + hidden_features, :32
        ][:, :rank]
        k_db_start = q_db_start + hidden_features
        k_db = attention[
            k_db_start : k_db_start + kv_features, :32
        ][:, :rank]
        v_db = attention[
            k_db_start + kv_features : k_db_start + 2 * kv_features, :32
        ][:, :rank]
        o_da = attention[
            config.sequence_length : config.sequence_length + 32,
            :hidden_features,
        ][:rank]
        o_db = attention[
            config.sequence_length + 32 :
            config.sequence_length + 32 + hidden_features,
            :32,
        ][:, :rank]

        gradients = [None] * 47
        gradients[0] = attention[
            : config.sequence_length, :hidden_features
        ].reshape(ctx.hidden_shape)
        gradients[4], gradients[5] = q_da, q_db
        gradients[9], gradients[10] = k_da, k_db
        gradients[14], gradients[15] = v_da, v_db
        gradients[18], gradients[19] = o_da, o_db
        gradients[27] = gate_da[:rank, :hidden_features]
        gradients[28] = gate_db[:intermediate_features, :rank]
        gradients[30] = up_da[:rank, :hidden_features]
        gradients[31] = up_db[:intermediate_features, :rank]
        gradients[33] = down_da[:rank, :intermediate_features]
        gradients[34] = down_db[:hidden_features, :rank]
        return tuple(gradients)


def resident_transformer_layer_autograd(*arguments):
    """Functional spelling kept small for the integration installer."""

    return NativeResidentTransformerLayerFunction.apply(*arguments)
