"""Opt-in Transformers attention interface with one resident backward owner."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from types import MethodType

import torch
import torch.nn.functional as F
from torch import nn
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from tt_finetune.kernels.native_attention_autograd import (
    attach_resident_attention_backward,
)
from tt_finetune.kernels.native_full_attention_autograd import (
    attach_full_attention_backward,
    full_attention_qkv_projection_support_error,
)
from tt_finetune.kernels.native_full_attention_forward_autograd import (
    resident_full_attention_forward,
)
from tt_finetune.kernels.full_attention_forward_planner import (
    plan_full_attention_forward,
)
from tt_finetune.kernels.tt_lang_qkv_resident_backward import (
    qkv_resident_backward_arena_shape,
)
from tt_finetune.kernels.tt_lang_output_attention_resident_backward import (
    full_attention_backward_arena_shape,
    output_attention_arena_shape,
    output_attention_support_error,
)


ATTENTION_INTERFACE_NAME = "tt_resident_attention_backward"
_ATTENTION_ARENA_NONCES = itertools.count(512)


def _attention_arena_nonce(code: int) -> float:
    """Return a unique exactly representable negative BF16 graph identity."""

    exponent, mantissa = divmod(code, 128)
    return -math.ldexp(1.0 + mantissa / 128.0, exponent)


def _repeat_kv(hidden_states: torch.Tensor, groups: int) -> torch.Tensor:
    batch, kv_heads, sequence, head_dim = hidden_states.shape
    if groups == 1:
        return hidden_states
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, kv_heads, groups, sequence, head_dim)
        .reshape(batch, kv_heads * groups, sequence, head_dim)
    )


def _exact_eager_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **_kwargs,
):
    groups = int(module.num_key_value_groups)
    expanded_key = _repeat_kv(key, groups)
    expanded_value = _repeat_kv(value, groups)
    probability = torch.matmul(query, expanded_key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        probability = probability + attention_mask[
            :, :, :, : expanded_key.shape[-2]
        ]
    probability_fp32 = F.softmax(
        probability, dim=-1, dtype=torch.float32
    )
    probability = probability_fp32.to(query.dtype)
    probability = F.dropout(
        probability, p=dropout, training=module.training
    )
    head_output = torch.matmul(probability, expanded_value)

    supported = (
        bool(getattr(module, "_tt_resident_attention_enabled", False))
        and query.dtype in {torch.float32, torch.bfloat16}
        and query.shape[0] == 1
        and query.shape[-2] == 128
        and query.shape[-1] % 32 == 0
        and query.shape[1] % key.shape[1] == 0
        and (query.shape[1] // key.shape[1]) % 2 == 1
        and query.shape[1] * 4 + key.shape[1] * 4 <= 110
        and float(dropout) == 0.0
    )
    if supported and bool(
        getattr(module, "_tt_resident_attention_block_enabled", False)
    ):
        # The outer module wrapper owns o_proj and the projection tails. Keep
        # the exact forward tensors private until o_proj has produced the
        # final module output; do not attach the smaller core-only owner.
        module._tt_pending_full_attention = (
            probability_fp32,
            query,
            key,
            value,
            head_output.transpose(1, 2).contiguous(),
        )
    elif supported:
        arena = module._tt_resident_attention_gradient_arena
        head_output = attach_resident_attention_backward(
            head_output,
            probability_fp32,
            query,
            key,
            value,
            scale=float(scaling),
            gradient_arena=arena,
            resident_backward=bool(
                getattr(module, "_tt_resident_attention_native", True)
            ),
        )
    return head_output.transpose(1, 2).contiguous(), probability


def _attention_adapter_parts(
    projection: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return frozen weight and one ordinary PEFT adapter, allowing bias."""

    active = getattr(projection, "active_adapter", "default")
    if not isinstance(active, str):
        active = tuple(active)
        if len(active) != 1:
            raise ValueError("full attention owner requires one adapter")
        active = active[0]
    try:
        weight = projection.base_layer.weight
        lora_a = projection.lora_A[active].weight
        lora_b = projection.lora_B[active].weight
        scale = float(projection.scaling[active])
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError("attention projection is not a PEFT LoRA linear") from error
    dropout = projection.lora_dropout[active]
    if float(getattr(dropout, "p", 0.0)) != 0.0:
        raise ValueError("full attention owner requires zero LoRA dropout")
    use_dora = getattr(projection, "use_dora", {})
    if bool(use_dora.get(active, False)):
        raise ValueError("full attention owner does not support DoRA")
    return weight, lora_a, lora_b, scale


def _attention_active_adapter(projection: nn.Module) -> str:
    active = getattr(projection, "active_adapter", "default")
    if isinstance(active, str):
        return active
    active = tuple(active)
    if len(active) != 1:
        raise ValueError("full attention owner requires one adapter")
    return active[0]


def _capture_attention_lora_hidden(
    owner: nn.Module, name: str
):
    """Create a hook that keeps an already-produced rank-space activation."""

    def capture(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        owner._tt_pending_lora_hidden[name] = output

    return capture


def _full_attention_module_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """Wrap the unchanged Transformers forward with full backward ownership."""

    if bool(getattr(module, "_tt_resident_attention_enabled", False)) and bool(
        getattr(module, "_tt_resident_attention_forward_enabled", False)
    ):
        if past_key_values is not None:
            raise RuntimeError("resident attention forward does not support KV cache")
        if attention_mask is None:
            raise RuntimeError("resident attention forward requires a causal mask")
        if bool(kwargs.get("output_attentions", False)):
            raise RuntimeError(
                "resident attention forward does not materialize attention weights"
            )
        cosine, sine = position_embeddings
        query_heads = int(module.num_key_value_groups) * int(
            getattr(module, "num_key_value_heads", 0)
            or module.k_proj.out_features // module.head_dim
        )
        kv_heads = query_heads // int(module.num_key_value_groups)
        output = resident_full_attention_forward(
            hidden_states,
            q_parts=_attention_adapter_parts(module.q_proj),
            k_parts=_attention_adapter_parts(module.k_proj),
            v_parts=_attention_adapter_parts(module.v_proj),
            o_parts=_attention_adapter_parts(module.o_proj),
            q_weight_t=module._tt_forward_q_weight_t,
            k_weight_t=module._tt_forward_k_weight_t,
            v_weight_t=module._tt_forward_v_weight_t,
            o_weight_t=module._tt_forward_o_weight_t,
            q_bias_tiled=module._tt_forward_q_bias_tiled,
            k_bias_tiled=module._tt_forward_k_bias_tiled,
            v_bias_tiled=module._tt_forward_v_bias_tiled,
            cosine=cosine,
            sine=sine,
            mask=attention_mask,
            forward_arena=module._tt_resident_attention_forward_arena,
            backward_arena=module._tt_resident_output_attention_arena,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=int(module.head_dim),
            attention_scale=float(module.scaling),
            has_bias=bool(module._tt_forward_has_bias),
            o_lora_k_tiles_per_phase=(
                module._tt_forward_o_lora_k_tiles_per_phase
            ),
            dedicated_o_rank_producer=bool(
                module._tt_forward_dedicated_o_rank_producer
            ),
            canonical_native_o_path=bool(
                module._tt_forward_canonical_native_o_path
            ),
            exact_attention_tail=bool(
                module._tt_forward_exact_attention_tail
            ),
            canonical_bf16_attention=bool(
                module._tt_forward_canonical_bf16_attention
            ),
        )
        return output, None

    module._tt_pending_full_attention = None
    module._tt_pending_lora_hidden = {}
    result = module._tt_full_attention_original_forward(
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=past_key_values,
        cache_position=cache_position,
        **kwargs,
    )
    pending = module._tt_pending_full_attention
    lora_hidden = module._tt_pending_lora_hidden
    module._tt_pending_full_attention = None
    module._tt_pending_lora_hidden = {}
    if pending is None:
        return result
    if past_key_values is not None:
        # Cache mutation changes K/V ownership and is not part of the training
        # contract. The interface should have failed closed before this point.
        raise RuntimeError("full resident attention does not support KV cache")
    output, attention_weights = result
    probability, query, key, value, head_output = pending
    resident_output_attention = bool(
        getattr(module, "_tt_resident_output_attention_enabled", False)
    )
    if resident_output_attention and "o" not in lora_hidden:
        # Never substitute a recomputed rank activation into an exact fused
        # backward. Fall back to the separately qualified path for this
        # invocation if another forward owner bypassed the capture hook.
        resident_output_attention = False
    o_hidden = lora_hidden.get("o", head_output)
    if all(name in lora_hidden for name in ("q", "k", "v")):
        q_hidden = lora_hidden["q"]
        k_hidden = lora_hidden["k"]
        v_hidden = lora_hidden["v"]
    elif bool(
        getattr(module, "_tt_resident_attention_projection_enabled", False)
    ):
        # A different experimental forward owner may bypass the lora_A
        # modules and therefore their capture hooks. Fail closed to the same
        # exact recomputation used by stage 1 instead of crashing or using a
        # stale activation. The benchmark avoids this combination so accepted
        # performance measurements never include duplicate work.
        _, q_a, _, _ = _attention_adapter_parts(module.q_proj)
        _, k_a, _, _ = _attention_adapter_parts(module.k_proj)
        _, v_a, _, _ = _attention_adapter_parts(module.v_proj)
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        q_hidden = flat_hidden @ q_a.transpose(0, 1)
        k_hidden = flat_hidden @ k_a.transpose(0, 1)
        v_hidden = flat_hidden @ v_a.transpose(0, 1)
    else:
        # Disabled stage-2 code never consumes these tensor placeholders.
        q_hidden = hidden_states
        k_hidden = hidden_states
        v_hidden = hidden_states
    cosine, sine = position_embeddings
    output = attach_full_attention_backward(
        output,
        hidden_states,
        probability,
        query,
        key,
        value,
        cosine,
        sine,
        head_output.reshape(*hidden_states.shape[:-1], -1),
        o_hidden,
        q_hidden,
        k_hidden,
        v_hidden,
        q_parts=_attention_adapter_parts(module.q_proj),
        k_parts=_attention_adapter_parts(module.k_proj),
        v_parts=_attention_adapter_parts(module.v_proj),
        o_parts=_attention_adapter_parts(module.o_proj),
        attention_scale=float(module.scaling),
        gradient_arena=module._tt_resident_attention_gradient_arena,
        resident_core=bool(module._tt_resident_attention_native),
        kv_projection_arena=getattr(
            module, "_tt_resident_attention_kv_projection_arena", None
        ),
        resident_kv_projection=bool(
            getattr(module, "_tt_resident_attention_projection_enabled", False)
        ),
        output_attention_arena=getattr(
            module, "_tt_resident_output_attention_arena", None
        ),
        resident_output_attention=resident_output_attention,
    )
    return output, attention_weights


@dataclass(frozen=True)
class ResidentAttentionInstallReport:
    installed_paths: tuple[str, ...]
    skipped: dict[str, str]
    fallbacks: dict[str, str]
    previous_implementation: str

    @property
    def installed_count(self) -> int:
        return len(self.installed_paths)


def install_resident_attention_backward(
    model: nn.Module,
    *,
    sequence_length: int = 128,
    native_backward: bool = True,
    full_block_backward: bool = False,
    native_projection_backward: bool = False,
    native_output_attention_backward: bool = False,
    native_full_forward: bool = False,
    o_lora_k_tiles_per_phase: int | None = None,
    dedicated_o_rank_producer: bool = False,
    canonical_native_o_path: bool = False,
    exact_attention_tail: bool = False,
    canonical_bf16_attention: bool = False,
) -> ResidentAttentionInstallReport:
    """Install the exact candidate behind a Transformers attention key."""

    ALL_ATTENTION_FUNCTIONS.register(
        ATTENTION_INTERFACE_NAME, _exact_eager_forward
    )
    # Mask construction is selected independently from attention arithmetic.
    # Without this registration Transformers treats the custom key like a
    # backend that owns masking and passes ``None`` here, silently removing the
    # causal mask.  The resident owner intentionally preserves eager forward
    # arithmetic, so it must also preserve the eager mask contract.
    ALL_MASK_ATTENTION_FUNCTIONS.register(
        ATTENTION_INTERFACE_NAME,
        ALL_MASK_ATTENTION_FUNCTIONS["eager"],
    )
    if native_full_forward:
        native_output_attention_backward = True
        native_projection_backward = True
        native_backward = True
    if native_output_attention_backward:
        full_block_backward = True
    installed: list[str] = []
    skipped: dict[str, str] = {}
    fallbacks: dict[str, str] = {}
    for path, module in tuple(model.named_modules()):
        if not path or not all(
            hasattr(module, name)
            for name in (
                "q_proj", "k_proj", "v_proj", "o_proj",
                "num_key_value_groups", "head_dim",
            )
        ):
            continue
        query_heads = int(module.num_key_value_groups) * int(
            getattr(module, "num_key_value_heads", 0)
            or module.k_proj.out_features // module.head_dim
        )
        kv_heads = query_heads // int(module.num_key_value_groups)
        group = query_heads // kv_heads
        workers = query_heads * (sequence_length // 32)
        reducers = kv_heads * (sequence_length // 32)
        if (
            sequence_length != 128
            or int(module.head_dim) % 32
            or group % 2 != 1
            or workers + reducers > 110
        ):
            skipped[path] = (
                f"unsupported resident geometry qh={query_heads}, "
                f"kvh={kv_heads}, s={sequence_length}, d={module.head_dim}"
            )
            continue
        parameter = next(module.parameters())
        arena_rows = (query_heads + 2 * kv_heads) * sequence_length
        arena_nonce_code = next(_ATTENTION_ARENA_NONCES)
        module.register_buffer(
            "_tt_resident_attention_gradient_arena",
            torch.full(
                (arena_rows, int(module.head_dim)),
                _attention_arena_nonce(arena_nonce_code),
                dtype=parameter.dtype,
                device=parameter.device,
            ),
            persistent=False,
        )
        module._tt_resident_attention_arena_nonce_code = arena_nonce_code
        module._tt_resident_attention_enabled = True
        module._tt_resident_attention_native = bool(native_backward)
        module._tt_resident_attention_block_enabled = bool(
            full_block_backward
        )
        module._tt_resident_attention_projection_enabled = False
        module._tt_resident_output_attention_enabled = False
        module._tt_resident_attention_forward_enabled = False
        if full_block_backward:
            try:
                for projection_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    _attention_adapter_parts(getattr(module, projection_name))
            except ValueError as error:
                skipped[path] = str(error)
                module._tt_resident_attention_enabled = False
                module._tt_resident_attention_block_enabled = False
                continue
            if native_output_attention_backward:
                hidden_features = int(module.q_proj.in_features)
                qkv_projection_supported = (
                    native_projection_backward
                    and full_attention_qkv_projection_support_error(
                        tokens=sequence_length,
                        hidden_features=hidden_features,
                        kv_features=int(module.k_proj.out_features),
                    )
                    is None
                )
                error = (
                    "resident output+attention requires native backward"
                    if not native_backward
                    else output_attention_support_error(
                        sequence_length=sequence_length,
                        hidden_features=hidden_features,
                        query_heads=query_heads,
                        kv_heads=kv_heads,
                        head_dim=int(module.head_dim),
                        physical_rank=32,
                    )
                )
                if error is not None:
                    fallbacks[path] = error
                else:
                    output_arena_nonce_code = next(_ATTENTION_ARENA_NONCES)
                    module.register_buffer(
                        "_tt_resident_output_attention_arena",
                        torch.full(
                            (
                                full_attention_backward_arena_shape(
                                    tokens=sequence_length,
                                    hidden_features=hidden_features,
                                    kv_features=int(module.k_proj.out_features),
                                )
                                if qkv_projection_supported
                                else output_attention_arena_shape(
                                    tokens=sequence_length,
                                    hidden_features=hidden_features,
                                    query_heads=query_heads,
                                    kv_heads=kv_heads,
                                    head_dim=int(module.head_dim),
                                )
                            ),
                            _attention_arena_nonce(output_arena_nonce_code),
                            dtype=parameter.dtype,
                            device=parameter.device,
                        ),
                        persistent=False,
                    )
                    module._tt_resident_output_attention_arena_nonce_code = (
                        output_arena_nonce_code
                    )
                    module._tt_resident_output_attention_enabled = True
            if native_projection_backward:
                hidden_features = int(module.q_proj.in_features)
                kv_features = int(module.k_proj.out_features)
                error = full_attention_qkv_projection_support_error(
                    tokens=sequence_length,
                    hidden_features=hidden_features,
                    kv_features=kv_features,
                )
                if error is not None:
                    skipped[path] = error
                    module._tt_resident_attention_enabled = False
                    module._tt_resident_attention_block_enabled = False
                    continue
                kv_arena_nonce_code = next(_ATTENTION_ARENA_NONCES)
                module.register_buffer(
                    "_tt_resident_attention_kv_projection_arena",
                    torch.full(
                        qkv_resident_backward_arena_shape(
                            tokens=sequence_length,
                            hidden_features=hidden_features,
                            kv_features=kv_features,
                        ),
                        _attention_arena_nonce(kv_arena_nonce_code),
                        dtype=parameter.dtype,
                        device=parameter.device,
                    ),
                    persistent=False,
                )
                module._tt_resident_attention_kv_arena_nonce_code = (
                    kv_arena_nonce_code
                )
                module._tt_resident_attention_projection_enabled = True
            if native_full_forward:
                hidden_features = int(module.q_proj.in_features)
                kv_features = int(module.k_proj.out_features)
                q_bias = getattr(module.q_proj.base_layer, "bias", None)
                k_bias = getattr(module.k_proj.base_layer, "bias", None)
                v_bias = getattr(module.v_proj.base_layer, "bias", None)
                bias_presence = tuple(
                    bias is not None for bias in (q_bias, k_bias, v_bias)
                )
                if len(set(bias_presence)) != 1:
                    fallbacks[path] = "Q/K/V bias presence must match"
                elif getattr(module.o_proj.base_layer, "bias", None) is not None:
                    fallbacks[path] = "resident O projection requires no bias"
                elif float(getattr(module, "attention_dropout", 0.0)) != 0.0:
                    fallbacks[path] = "resident attention forward requires zero dropout"
                elif not module._tt_resident_output_attention_enabled:
                    fallbacks[path] = "full forward requires accepted v90 backward"
                elif not module._tt_resident_attention_projection_enabled:
                    fallbacks[path] = "full forward requires resident Q/K/V backward"
                else:
                    forward_plan = plan_full_attention_forward(
                        tokens=sequence_length,
                        hidden_features=hidden_features,
                        kv_features=kv_features,
                        query_heads=query_heads,
                        kv_heads=kv_heads,
                        head_dim=int(module.head_dim),
                        physical_rank=32,
                        qkv_bias=bool(bias_presence[0]),
                    )
                    for short_name, projection in (
                        ("q", module.q_proj), ("k", module.k_proj),
                        ("v", module.v_proj), ("o", module.o_proj),
                    ):
                        weight = projection.base_layer.weight.detach()
                        module.register_buffer(
                            f"_tt_forward_{short_name}_weight_t",
                            weight.transpose(0, 1).contiguous(),
                            persistent=False,
                        )
                    for short_name, projection in (
                        ("q", module.q_proj), ("k", module.k_proj),
                        ("v", module.v_proj),
                    ):
                        bias = getattr(projection.base_layer, "bias", None)
                        if bias is None:
                            bias = torch.zeros(
                                projection.out_features,
                                dtype=parameter.dtype,
                                device=parameter.device,
                            )
                        module.register_buffer(
                            f"_tt_forward_{short_name}_bias_tiled",
                            bias.detach().reshape(1, -1).expand(32, -1).contiguous(),
                            persistent=False,
                        )
                    forward_nonce = next(_ATTENTION_ARENA_NONCES)
                    module.register_buffer(
                        "_tt_resident_attention_forward_arena",
                        torch.full(
                            forward_plan.arena.physical_shape,
                            _attention_arena_nonce(forward_nonce),
                            dtype=parameter.dtype,
                            device=parameter.device,
                        ),
                        persistent=False,
                    )
                    module._tt_forward_has_bias = bool(bias_presence[0])
                    module._tt_forward_o_lora_k_tiles_per_phase = (
                        o_lora_k_tiles_per_phase
                    )
                    module._tt_forward_dedicated_o_rank_producer = bool(
                        dedicated_o_rank_producer
                    )
                    module._tt_forward_canonical_native_o_path = bool(
                        canonical_native_o_path
                    )
                    module._tt_forward_exact_attention_tail = bool(
                        exact_attention_tail
                    )
                    module._tt_forward_canonical_bf16_attention = bool(
                        canonical_bf16_attention
                    )
                    module._tt_resident_attention_forward_enabled = True
            module._tt_full_attention_original_forward = module.forward
            module._tt_pending_lora_hidden = {}
            for projection_name, short_name in (
                ("q_proj", "q"), ("k_proj", "k"),
                ("v_proj", "v"), ("o_proj", "o")
            ):
                projection = getattr(module, projection_name)
                active = _attention_active_adapter(projection)
                projection.lora_A[active].register_forward_hook(
                    _capture_attention_lora_hidden(module, short_name)
                )
            module.forward = MethodType(_full_attention_module_forward, module)
        installed.append(path)

    previous = str(getattr(model.config, "_attn_implementation", "eager"))
    if installed:
        model.config._attn_implementation = ATTENTION_INTERFACE_NAME
    return ResidentAttentionInstallReport(
        installed_paths=tuple(installed),
        skipped=skipped,
        fallbacks=fallbacks,
        previous_implementation=previous,
    )
