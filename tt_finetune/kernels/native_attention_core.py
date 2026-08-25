"""Shape-safe experimental entry point for resident GQA backward."""

from __future__ import annotations

import torch

from .attention_block_planner import plan_attention_block
from .tt_lang_attention_core_backward import (
    build_tt_lang_attention_core_backward,
)
from .tt_lang_attention_resident_backward import (
    build_tt_lang_attention_resident_backward,
)


def make_attention_gradient_arena(
    query: torch.Tensor, key: torch.Tensor
) -> torch.Tensor:
    """Allocate the packed [dQ; dK; dV] output with query placement."""

    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must have [B,H,S,D] shapes")
    rows = query.shape[0] * (
        query.shape[1] + 2 * key.shape[1]
    ) * query.shape[2]
    return torch.zeros(
        (rows, query.shape[3]), dtype=query.dtype, device=query.device
    )


def native_attention_core_backward(
    probability: torch.Tensor,
    output_gradient: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
    arena: torch.Tensor | None = None,
    math_fidelity: str = "HiFi4",
    matmul_full_fp32: bool = True,
    resident_owner: bool = False,
    expanded_key: torch.Tensor | None = None,
    expanded_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return experimental native dQ/dK/dV views.

    This function intentionally requires explicit selection by its caller. It
    is not used by the exact trainer path until silicon parity is established.
    """

    tensors = (output_gradient, query, key, value)
    if query.dtype not in {torch.float32, torch.bfloat16}:
        raise ValueError("native attention core requires float32 or bfloat16")
    if any(tensor.dtype != query.dtype for tensor in tensors):
        raise ValueError("native attention vectors must share one dtype")
    expected_probability_dtype = (
        torch.float32
        if resident_owner and query.dtype == torch.bfloat16
        else query.dtype
    )
    if probability.dtype != expected_probability_dtype:
        raise ValueError(
            "resident BF16 attention requires the saved FP32 softmax "
            f"probability, got {probability.dtype}"
        )
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("Q/K/V must be [B,H,S,D], with equal K/V shapes")
    if output_gradient.shape != query.shape:
        raise ValueError("output_gradient must match query")
    batch, query_heads, sequence, head_dim = query.shape
    if probability.shape != (batch, query_heads, sequence, sequence):
        raise ValueError("probability must be [B,QH,S,S]")
    if key.shape[0] != batch or key.shape[2:] != (sequence, head_dim):
        raise ValueError("key geometry must match query batch/sequence/dim")
    plan_attention_block(
        batch_size=batch,
        sequence_length=sequence,
        query_heads=query_heads,
        kv_heads=key.shape[1],
        head_dim=head_dim,
    )
    if batch != 1:
        raise ValueError("first silicon attention core supports batch size one")
    if arena is None:
        arena = make_attention_gradient_arena(query, key)
    expected_shape = (
        (query_heads + 2 * key.shape[1]) * sequence,
        head_dim,
    )
    if tuple(arena.shape) != expected_shape or arena.dtype != query.dtype:
        raise ValueError(
            f"arena must be {query.dtype} {expected_shape}, got {tuple(arena.shape)}"
        )
    if resident_owner:
        operation = build_tt_lang_attention_resident_backward(
            sequence_length=sequence,
            query_heads=query_heads,
            kv_heads=key.shape[1],
            head_dim=head_dim,
            scale=scale,
            dtype_name=(
                "bfloat16" if query.dtype == torch.bfloat16 else "float32"
            ),
        )
        operation_key = key
        operation_value = value
    else:
        operation = build_tt_lang_attention_core_backward(
            sequence_length=sequence,
            query_heads=query_heads,
            kv_heads=key.shape[1],
            head_dim=head_dim,
            scale=scale,
            math_fidelity=math_fidelity,
            matmul_full_fp32=matmul_full_fp32,
        )
        operation_key = key
        operation_value = value
    packed = operation(
        probability.reshape(-1, sequence),
        output_gradient.reshape(-1, head_dim),
        query.reshape(-1, head_dim),
        operation_key.reshape(-1, head_dim),
        operation_value.reshape(-1, head_dim),
        arena,
    )
    q_rows = query_heads * sequence
    kv_rows = key.shape[1] * sequence
    dq = packed[:q_rows].reshape_as(query)
    dk = packed[q_rows : q_rows + kv_rows].reshape_as(key)
    dv = packed[q_rows + kv_rows :].reshape_as(value)
    return dq, dk, dv
