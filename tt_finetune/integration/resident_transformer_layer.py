"""Opt-in full Transformer-layer forward owner for Blackhole qualification.

This installer replaces the complete Llama/Qwen decoder-layer forward with the
accepted exact TT-Metal owner.  It is intentionally forward-only until the
arena-backed whole-layer backward ABI is connected; callers must opt in
explicitly and the ordinary module remains the safe fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType

import torch
from torch import nn

from tt_finetune.kernels.tt_lang_attention_forward_qkv_resident import (
    build_tt_lang_attention_forward_qkv_resident,
)
from tt_finetune.kernels.native_resident_layer_autograd import (
    ResidentLayerAutogradConfig,
    resident_transformer_layer_autograd,
)
from tt_finetune.kernels.tt_lang_lora import pad_lora_weights, padded_rank
from tt_finetune.kernels.tt_lang_output_attention_resident_backward import (
    full_attention_backward_arena_shape,
)
from tt_finetune.kernels.tt_lang_resident_layer_forward_trainable_owner import (
    build_tt_lang_resident_layer_forward_trainable_owner,
)
from tt_finetune.kernels.tt_lang_resident_layer_forward_compact_owner import (
    build_tt_lang_resident_layer_forward_compact_owner,
)


def _active_adapter(projection: nn.Module) -> str:
    active = getattr(projection, "active_adapter", "default")
    if isinstance(active, str):
        return active
    active = tuple(active)
    if len(active) != 1:
        raise ValueError("resident layer requires exactly one active adapter")
    return active[0]


def _parts(
    projection: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    active = _active_adapter(projection)
    try:
        weight = projection.base_layer.weight
        a = projection.lora_A[active].weight
        b = projection.lora_B[active].weight
        scale = float(projection.scaling[active])
        dropout = projection.lora_dropout[active]
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError("resident layer requires an ordinary PEFT LoRA linear") from error
    if float(getattr(dropout, "p", 0.0)) != 0.0:
        raise ValueError("resident layer requires zero LoRA dropout")
    if bool(getattr(projection, "use_dora", {}).get(active, False)):
        raise ValueError("resident layer does not support DoRA")
    return weight, a, b, scale


def _packed_tile_pages(*tensors: torch.Tensor) -> torch.Tensor:
    pages = []
    for tensor in tensors:
        if tensor.ndim != 2 or tensor.shape[0] % 32 or tensor.shape[1] % 32:
            raise ValueError(f"packed layer tensor is not tile aligned: {tensor.shape}")
        row_tiles, column_tiles = tensor.shape[0] // 32, tensor.shape[1] // 32
        pages.append(
            tensor.reshape(row_tiles, 32, column_tiles, 32)
            .permute(0, 2, 1, 3)
            .reshape(row_tiles * column_tiles, 32, 32)
        )
    return torch.cat(pages, dim=0).reshape(-1, 32).contiguous()


def _tiled_bias(projection: nn.Module, *, dtype: torch.dtype) -> torch.Tensor:
    bias = getattr(projection.base_layer, "bias", None)
    if bias is None:
        bias = torch.zeros(
            projection.out_features,
            dtype=dtype,
            device=projection.base_layer.weight.device,
        )
    return bias.detach().reshape(1, -1).expand(32, -1).contiguous()


def _register_buffer(module: nn.Module, name: str, value: torch.Tensor) -> None:
    module.register_buffer(name, value.detach().contiguous(), persistent=False)


def _resident_layer_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_values=None,
    use_cache: bool | None = False,
    cache_position: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs,
) -> torch.Tensor:
    if not bool(getattr(module, "_tt_resident_layer_forward_enabled", False)):
        return module._tt_resident_layer_original_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
    if past_key_values is not None or bool(use_cache):
        raise RuntimeError("resident Transformer layer does not support KV cache")
    if position_embeddings is None:
        raise RuntimeError("resident Transformer layer requires precomputed RoPE")
    if attention_mask is None:
        # Transformers may elide an all-ones 2-D mask and communicate causal
        # semantics only through the selected attention implementation.  The
        # native owner consumes the values directly, so retain one immutable
        # causal tile matrix per layer instead of rebuilding it in the graph.
        attention_mask = module._tt_layer_causal_mask
    if hidden_states.shape[0] != 1 or hidden_states.shape[1] != 128:
        raise RuntimeError("resident Transformer layer currently requires [1,128,H]")

    cosine, sine = position_embeddings
    flat_hidden = hidden_states.reshape(128, module._tt_layer_hidden_features)
    if bool(getattr(module, "_tt_resident_layer_autograd_enabled", False)):
        q = _parts(module.self_attn.q_proj)
        k = _parts(module.self_attn.k_proj)
        v = _parts(module.self_attn.v_proj)
        o = _parts(module.self_attn.o_proj)
        gate = _parts(module.mlp.gate_proj)
        up = _parts(module.mlp.up_proj)
        down = _parts(module.mlp.down_proj)
        return resident_transformer_layer_autograd(
            hidden_states,
            q[0], module._tt_layer_q_weight_t, module._tt_layer_q_bias,
            q[1], q[2],
            k[0], module._tt_layer_k_weight_t, module._tt_layer_k_bias,
            k[1], k[2],
            v[0], module._tt_layer_v_weight_t, module._tt_layer_v_bias,
            v[1], v[2],
            o[0], module._tt_layer_o_weight_t, o[1], o[2],
            cosine, sine, attention_mask,
            module._tt_layer_input_gamma, module._tt_layer_post_gamma,
            module._tt_layer_packed_post_parameters,
            gate[0], gate[1], gate[2],
            up[0], up[1], up[2],
            down[0], down[1], down[2],
            module._tt_resident_layer_forward_arena,
            module._tt_resident_layer_backward_arena,
            module._tt_layer_gate_da_seed, module._tt_layer_gate_db_seed,
            module._tt_layer_up_da_seed, module._tt_layer_up_db_seed,
            module._tt_layer_down_da_seed, module._tt_layer_down_db_seed,
            module._tt_layer_fp32_seed, module._tt_layer_fp32_seed_two,
            module._tt_layer_bf16_seed,
            module._tt_resident_layer_autograd_config,
        )
    arena = module._tt_resident_layer_forward_operation(
        flat_hidden,
        module._tt_layer_q_weight_t,
        module._tt_layer_q_bias,
        module._tt_layer_q_a,
        module._tt_layer_q_b,
        module._tt_layer_k_weight_t,
        module._tt_layer_k_bias,
        module._tt_layer_k_a,
        module._tt_layer_k_b,
        module._tt_layer_v_weight_t,
        module._tt_layer_v_bias,
        module._tt_layer_v_a,
        module._tt_layer_v_b,
        module._tt_layer_o_weight_t,
        module._tt_layer_o_a,
        module._tt_layer_o_b,
        cosine.reshape(128, module._tt_layer_head_dim),
        sine.reshape(128, module._tt_layer_head_dim),
        attention_mask.reshape(128, 128),
        module._tt_layer_input_gamma,
        module._tt_layer_packed_post_parameters,
        module._tt_resident_layer_forward_arena,
    )
    # Integrated mode mirrors the exact final residual into page zero.  The
    # displaced attention diagnostic output is not a backward checkpoint.
    return arena[:128, : module._tt_layer_hidden_features].reshape_as(
        hidden_states
    )


@dataclass(frozen=True)
class ResidentTransformerLayerInstallReport:
    installed_paths: tuple[str, ...]
    skipped: dict[str, str]

    @property
    def installed_count(self) -> int:
        return len(self.installed_paths)


def install_resident_transformer_layer_forward(
    model: nn.Module, *, sequence_length: int = 128,
    source_bundle: str | None = None,
    compact_forward: bool = False,
    o_lora_k_tiles_per_phase: int | None = None,
    canonical_exact_attention: bool = False,
) -> ResidentTransformerLayerInstallReport:
    """Install the exact full-layer owner behind an opt-in flag.

    Supplying ``source_bundle`` connects the live LoRA parameters and the
    accepted arena-backed backward owner.  Omitting it preserves the accepted
    forward-only qualification path and its safe behavior.
    """

    installed: list[str] = []
    skipped: dict[str, str] = {}
    if sequence_length != 128:
        return ResidentTransformerLayerInstallReport(
            (), {"*": "resident layer currently requires sequence length 128"}
        )

    for path, layer in tuple(model.named_modules()):
        if not path or not all(
            hasattr(layer, name)
            for name in (
                "self_attn", "mlp", "input_layernorm",
                "post_attention_layernorm",
            )
        ):
            continue
        attention, mlp = layer.self_attn, layer.mlp
        if not all(
            hasattr(attention, name)
            for name in ("q_proj", "k_proj", "v_proj", "o_proj", "head_dim")
        ) or not all(
            hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj")
        ):
            continue
        try:
            q = _parts(attention.q_proj)
            k = _parts(attention.k_proj)
            v = _parts(attention.v_proj)
            o = _parts(attention.o_proj)
            gate = _parts(mlp.gate_proj)
            up = _parts(mlp.up_proj)
            down = _parts(mlp.down_proj)
            scales = {parts[3] for parts in (q, k, v, o, gate, up, down)}
            if len(scales) != 1:
                raise ValueError("resident layer requires one shared LoRA scale")
            hidden = int(attention.q_proj.in_features)
            intermediate = int(mlp.gate_proj.out_features)
            kv_features = int(attention.k_proj.out_features)
            head_dim = int(attention.head_dim)
            query_heads = int(attention.q_proj.out_features) // head_dim
            kv_heads = kv_features // head_dim
            rank = int(q[1].shape[0])
            if any(int(parts[1].shape[0]) != rank for parts in (k, v, o, gate, up, down)):
                raise ValueError("resident layer requires one logical rank")
            if not 1 <= rank <= 128:
                raise ValueError(
                    "resident layer rank must be within the qualified "
                    "rank-scaling envelope"
                )
            physical_rank = padded_rank(rank)
            qkv_biases = tuple(
                getattr(projection.base_layer, "bias", None) is not None
                for projection in (attention.q_proj, attention.k_proj, attention.v_proj)
            )
            if len(set(qkv_biases)) != 1:
                raise ValueError("Q/K/V bias presence must match")
            if getattr(attention.o_proj.base_layer, "bias", None) is not None:
                raise ValueError("resident O projection requires no bias")
            if float(getattr(attention, "attention_dropout", 0.0)) != 0.0:
                raise ValueError("resident layer requires zero attention dropout")

            padded = []
            for parts in (q, k, v, o, gate, up, down):
                padded.append(pad_lora_weights(parts[1].detach(), parts[2].detach()))
            (q_a, q_b), (k_a, k_b), (v_a, v_b), (o_a, o_b), (
                gate_a, gate_b
            ), (up_a, up_b), (down_a, down_b) = padded
            dtype = q[0].dtype
            input_gamma = layer.input_layernorm.weight.detach().reshape(
                1, hidden
            ).expand(32, hidden).contiguous()
            post_gamma = layer.post_attention_layernorm.weight.detach().reshape(
                1, hidden
            ).expand(32, hidden).contiguous()
            packed_post = _packed_tile_pages(
                post_gamma,
                torch.zeros((128, 128), dtype=dtype, device=post_gamma.device),
                torch.cat((gate[0], up[0]), dim=0).T.contiguous(),
                gate_a, gate_b, up_a, up_b,
                down[0].T.contiguous(), down_a, down_b,
            )
            operation_arguments = dict(
                sequence_length=128,
                hidden_features=hidden,
                kv_features=kv_features,
                query_heads=query_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                physical_rank=physical_rank,
                lora_scale=float(next(iter(scales))),
                attention_scale=float(attention.scaling),
                has_bias=bool(qkv_biases[0]),
                input_rmsnorm_epsilon=float(layer.input_layernorm.variance_epsilon),
                post_attention_rmsnorm_epsilon=float(
                    layer.post_attention_layernorm.variance_epsilon
                ),
            )
            if source_bundle is not None:
                forward_builder = (
                    build_tt_lang_resident_layer_forward_compact_owner
                    if compact_forward
                    else build_tt_lang_resident_layer_forward_trainable_owner
                )
                operation = forward_builder(
                    intermediate_features=intermediate,
                    o_lora_k_tiles_per_phase=o_lora_k_tiles_per_phase,
                    **(
                        {
                            "canonical_native_o_path": True,
                            "exact_attention_tail": True,
                            "canonical_bf16_attention": False,
                            "canonical_grouped_qkv_rank": (
                                not qkv_biases[0] and rank == 8
                            ),
                            "canonical_separate_qkv_rank": False,
                            "logical_rank": rank,
                        }
                        if compact_forward and canonical_exact_attention
                        else {}
                    ),
                    **operation_arguments,
                )
            else:
                operation = build_tt_lang_attention_forward_qkv_resident(
                    resident_gate_up_intermediate_features=intermediate,
                    resident_full_mlp_forward=True,
                    publish_layer_output_at_page_zero=True,
                    **operation_arguments,
                )

            for short, projection, a, b in (
                ("q", attention.q_proj, q_a, q_b),
                ("k", attention.k_proj, k_a, k_b),
                ("v", attention.v_proj, v_a, v_b),
                ("o", attention.o_proj, o_a, o_b),
            ):
                _register_buffer(
                    layer, f"_tt_layer_{short}_weight_t",
                    projection.base_layer.weight.detach().T.contiguous(),
                )
                if source_bundle is None:
                    _register_buffer(layer, f"_tt_layer_{short}_a", a)
                    _register_buffer(layer, f"_tt_layer_{short}_b", b)
            for short, projection in (
                ("q", attention.q_proj),
                ("k", attention.k_proj),
                ("v", attention.v_proj),
            ):
                _register_buffer(
                    layer, f"_tt_layer_{short}_bias",
                    _tiled_bias(projection, dtype=dtype),
                )
            _register_buffer(layer, "_tt_layer_input_gamma", input_gamma)
            _register_buffer(layer, "_tt_layer_post_gamma", post_gamma)
            _register_buffer(
                layer,
                "_tt_layer_causal_mask",
                torch.triu(
                    torch.full(
                        (128, 128), float("-inf"), dtype=dtype,
                        device=input_gamma.device,
                    ),
                    diagonal=1,
                ),
            )
            _register_buffer(
                layer, "_tt_layer_packed_post_parameters", packed_post
            )
            _register_buffer(
                layer,
                "_tt_resident_layer_forward_arena",
                torch.zeros(operation.output_shape, dtype=dtype),
            )
            if source_bundle is not None:
                attention_arena_shape = full_attention_backward_arena_shape(
                    tokens=128,
                    hidden_features=hidden,
                    kv_features=kv_features,
                )

                def output_seed(
                    name: str, shape: tuple[int, int], nonce: int
                ) -> None:
                    value = torch.zeros(shape, dtype=dtype)
                    value[0, 0] = float(nonce) / 128.0
                    _register_buffer(layer, name, value)

                output_seed(
                    "_tt_resident_layer_backward_arena",
                    attention_arena_shape,
                    1,
                )
                output_seed(
                    "_tt_layer_gate_da_seed", (physical_rank, hidden), 2
                )
                output_seed(
                    "_tt_layer_gate_db_seed", (intermediate, physical_rank), 3
                )
                output_seed(
                    "_tt_layer_up_da_seed", (physical_rank, hidden), 4
                )
                output_seed(
                    "_tt_layer_up_db_seed", (intermediate, physical_rank), 5
                )
                output_seed(
                    "_tt_layer_down_da_seed", (physical_rank, intermediate), 6
                )
                output_seed(
                    "_tt_layer_down_db_seed", (hidden, physical_rank), 7
                )
                _register_buffer(
                    layer, "_tt_layer_fp32_seed",
                    torch.zeros((128, hidden), dtype=torch.float32),
                )
                _register_buffer(
                    layer, "_tt_layer_fp32_seed_two",
                    torch.zeros((128, hidden), dtype=torch.float32),
                )
                _register_buffer(
                    layer, "_tt_layer_bf16_seed",
                    torch.zeros((128, hidden), dtype=dtype),
                )
                layer._tt_resident_layer_autograd_config = (
                    ResidentLayerAutogradConfig(
                        source_bundle=str(source_bundle),
                        sequence_length=128,
                        hidden_features=hidden,
                        intermediate_features=intermediate,
                        kv_features=kv_features,
                        query_heads=query_heads,
                        kv_heads=kv_heads,
                        head_dim=head_dim,
                        physical_rank=physical_rank,
                        lora_scale=float(next(iter(scales))),
                        attention_scale=float(attention.scaling),
                        input_rmsnorm_epsilon=float(
                            layer.input_layernorm.variance_epsilon
                        ),
                        post_attention_rmsnorm_epsilon=float(
                            layer.post_attention_layernorm.variance_epsilon
                        ),
                        has_bias=bool(qkv_biases[0]),
                        compact_forward=compact_forward,
                        o_lora_k_tiles_per_phase=(
                            o_lora_k_tiles_per_phase
                        ),
                        canonical_exact_attention=(
                            canonical_exact_attention
                        ),
                    )
                )
                layer._tt_resident_layer_autograd_enabled = True
            else:
                layer._tt_resident_layer_autograd_enabled = False
            layer._tt_layer_hidden_features = hidden
            layer._tt_layer_head_dim = head_dim
            layer._tt_resident_layer_forward_operation = operation
            layer._tt_resident_layer_forward_enabled = True
            layer._tt_resident_layer_original_forward = layer.forward
            layer.forward = MethodType(_resident_layer_forward, layer)
            installed.append(path)
        except (ValueError, AttributeError) as error:
            skipped[path] = str(error)

    return ResidentTransformerLayerInstallReport(tuple(installed), skipped)
