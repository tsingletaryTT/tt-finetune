"""Compact attention-only owner with a canonical TTNN O projection.

The established resident attention kernel owns Q/K/V, RoPE and causal
attention.  This outer artifact keeps that program unchanged, then replaces
only its O projection and saved O-rank checkpoint with the same TTNN program
families selected by the uninstrumented training graph.  The head, rank
activation and all arithmetic intermediates are compiler-private L1 tensors;
the existing checkpoint arena remains the sole public result.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import os
from typing import Callable

import torch

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_attention_forward_qkv_resident import (
    build_tt_lang_attention_forward_qkv_resident,
)
from .tt_lang_grouped_qkv_rank_path import (
    build_tt_lang_grouped_qkv_rank_path,
)
from .tt_lang_native_lora_o_path import build_tt_lang_native_lora_o_path
from .tt_lang_native_qk_prefix import build_tt_lang_native_qk_prefix
from .tt_lang_native_attention_fp32_tail import (
    build_tt_lang_native_attention_fp32_tail_compact_value_private,
)
from .tt_lang_native_attention_bf16_core import (
    build_tt_lang_native_attention_bf16_core_compact_private,
)
from .tt_lang_owner_aligned_copy import build_tt_lang_owner_aligned_copy
from .tt_lang_program_composition import (
    compile_serialized_child,
    compile_serialized_sequence_children,
    make_serialized_head_to_token_copy_bitwise,
    make_serialized_page_copy_bitwise,
    remap_serialized_child,
    retarget_tensor_writes_to_packed_pages,
    serialized_child_tensor_ids,
)
from .tt_lang_raw_fp32_checkpoint import build_raw_fp32_checkpoint_pack
from .tt_lang_fp32_pow2_scale import build_fp32_pow2_scale


ttl = None
tt_lang_operation = None


def _remap_all(program: dict, mapping: dict[int, int]) -> dict:
    used = serialized_child_tensor_ids(program)
    return remap_serialized_child(
        program, {tensor: mapping[tensor] for tensor in used}
    )


def _compile_placeholder_like(
    tensor, *, shape: tuple[int, ...], dtype=None,
):
    try:
        return type(tensor)(
            shape=shape,
            dtype=tensor.dtype if dtype is None else dtype,
            layout=tensor.layout,
            memory_config=tensor.memory_config(),
            device=tensor.device(),
        )
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(
            "resident attention owner requires shape-only compile operands"
        ) from exc


def canonical_bf16_attention_owner_internal_specs(
    *,
    sequence_length: int,
    hidden_features: int,
    kv_heads: int,
    query_heads: int,
    head_dim: int,
    physical_rank: int,
    output_shape: tuple[int, int],
    grouped_rank_width: int | None = None,
    arena_memory: str = "l1",
    bf16_l1_tensor_indices: tuple[int, ...] = tuple(range(9)),
) -> list[dict]:
    """Return the exact BF16 owner's compiler-private tensor ABI.

    Keeping this ABI explicit lets larger attention-block and Transformer
    owners flatten the accepted forward owner without re-exposing any of its
    Q/K/V, score, probability or O-projection intermediates to PJRT.
    """

    score_shape = [1, query_heads, sequence_length, sequence_length]
    expanded_shape = [1, query_heads, sequence_length, head_dim]
    specs: list[dict] = []
    def bf16_memory(index: int) -> str:
        return "l1" if index in bf16_l1_tensor_indices else "clone"
    if grouped_rank_width is not None:
        specs.append({
            "source_operand": 0,
            "memory": "l1",
            "shape": [sequence_length, grouped_rank_width],
        })
    specs.extend([
        {
            "source_operand": 21,
            "memory": "clone" if arena_memory == "dram" else "l1",
            "shape": list(output_shape),
            "layout": "tile",
        },
        {
            "source_operand": 0, "memory": "l1",
            "shape": expanded_shape, "data_format": "BFloat16",
            "layout": "tile",
        },
        {
            "source_operand": 0, "memory": "l1",
            "shape": [1, kv_heads, sequence_length, head_dim],
            "data_format": "BFloat16", "layout": "tile",
        },
        {
            "source_operand": 0, "memory": bf16_memory(2),
            "shape": score_shape, "data_format": "BFloat16",
            "layout": "tile",
        },
        {
            "source_operand": 0, "memory": "l1",
            "shape": [1, kv_heads, sequence_length, head_dim],
            "data_format": "BFloat16", "layout": "tile",
        },
        {
            "source_operand": 0, "memory": bf16_memory(0),
            "shape": expanded_shape, "data_format": "BFloat16",
            "layout": "tile",
        },
        {
            "source_operand": 0, "memory": "l1",
            "shape": [sequence_length, hidden_features],
            "data_format": "BFloat16", "layout": "tile",
        },
        {
            "source_operand": 0, "memory": bf16_memory(1),
            "shape": expanded_shape, "data_format": "BFloat16",
            "layout": "tile",
        },
        {
            "source_operand": 0, "memory": "l1",
            "shape": expanded_shape, "data_format": "BFloat16",
            "layout": "tile",
        },
        {
            "source_operand": 0, "memory": bf16_memory(3),
            "shape": [1, 1, 1, 1], "data_format": "BFloat16",
            "layout": "tile",
        },
        *(
            {
                "source_operand": 0, "memory": bf16_memory(4 + index),
                "shape": score_shape, "data_format": data_format,
                "layout": "tile",
            }
            for index, data_format in enumerate((
                "BFloat16", "BFloat16", "Float32", "Float32", "BFloat16"
            ))
        ),
        {
            "source_operand": 0, "memory": "l1",
            "shape": [sequence_length, hidden_features],
            "data_format": "BFloat16", "layout": "tile",
        },
        {
            "source_operand": 0, "memory": "l1",
            "shape": [sequence_length, physical_rank],
            "data_format": "BFloat16", "layout": "tile",
        },
        {"source_operand": 0, "memory": "l1", "layout": "tile"},
        {"source_operand": 0, "memory": "l1", "layout": "tile"},
        {
            "source_operand": 0, "memory": "l1", "shape": [1, 1],
            "layout": "tile",
        },
        {"source_operand": 0, "memory": "l1", "layout": "tile"},
    ])
    return specs


def canonical_fp32_attention_owner_internal_specs(
    *,
    sequence_length: int,
    hidden_features: int,
    kv_heads: int,
    query_heads: int,
    head_dim: int,
    physical_rank: int,
    output_shape: tuple[int, int],
    grouped_rank_width: int | None = None,
    arena_memory: str = "l1",
    raw_memory: str = "l1",
) -> list[dict]:
    """Return the post-lowering-qualified FP32 attention private ABI."""

    score = [1, query_heads, sequence_length, sequence_length]
    expanded = [1, query_heads, sequence_length, head_dim]
    compact = [1, kv_heads, sequence_length, head_dim]
    specs: list[dict] = []
    if grouped_rank_width is not None:
        specs.append({
            "source_operand": 0,
            "memory": "l1",
            "shape": [sequence_length, grouped_rank_width],
            "data_format": "BFloat16",
            "layout": "tile",
        })
    specs.extend([
        {
            "source_operand": 21,
            "memory": "clone" if arena_memory == "dram" else "l1",
            "shape": list(output_shape),
            "layout": "tile",
        },
        {"source_operand": 0, "memory": "l1", "shape": expanded,
         "data_format": "BFloat16", "layout": "tile"},
        {"source_operand": 0, "memory": "l1", "shape": compact,
         "data_format": "BFloat16", "layout": "tile"},
        {
            "source_operand": 0,
            "memory": "clone" if raw_memory == "dram" else "l1",
            "shape": score,
            "data_format": "Float32",
            "layout": "tile",
        },
        {"source_operand": 0, "memory": "l1", "shape": compact,
         "data_format": "BFloat16", "layout": "tile"},
        {"source_operand": 0, "memory": "l1", "shape": expanded,
         "data_format": "BFloat16", "layout": "tile"},
        {"source_operand": 0, "memory": "l1",
         "shape": [sequence_length, hidden_features],
         "data_format": "BFloat16", "layout": "tile"},
        {"source_operand": 0, "memory": "l1", "shape": expanded,
         "data_format": "BFloat16", "layout": "tile"},
        {"source_operand": 0, "memory": "l1", "shape": [1, 1, 1, 1],
         "data_format": "Float32", "layout": "tile"},
        {"source_operand": 0, "memory": "l1", "shape": score,
         "data_format": "Float32", "layout": "tile"},
        {"source_operand": 0, "memory": "l1",
         "shape": [1, 1, sequence_length, sequence_length],
         "data_format": "Float32", "layout": "tile"},
        {"source_operand": 0, "memory": "l1", "shape": score,
         "data_format": "Float32", "layout": "tile"},
        {"source_operand": 0, "memory": "l1", "shape": score,
         "data_format": "Float32", "layout": "tile"},
        {"source_operand": 0, "memory": "l1", "shape": expanded,
         "data_format": "Float32", "layout": "tile"},
        {"source_operand": 0, "memory": "l1", "shape": expanded,
         "data_format": "Float32", "layout": "tile"},
        {"source_operand": 0, "memory": "l1",
         "shape": [sequence_length, hidden_features],
         "data_format": "BFloat16", "layout": "tile"},
        {"source_operand": 0, "memory": "l1",
         "shape": [sequence_length, physical_rank],
         "data_format": "BFloat16", "layout": "tile"},
        {"source_operand": 0, "memory": "l1", "layout": "tile"},
        {"source_operand": 0, "memory": "l1", "layout": "tile"},
        {"source_operand": 0, "memory": "l1", "shape": [1, 1],
         "layout": "tile"},
        {"source_operand": 0, "memory": "l1", "layout": "tile"},
        {"source_operand": 0, "memory": "l1", "shape": expanded,
         "data_format": "Float32", "layout": "tile"},
        {"source_operand": 0, "memory": "l1", "shape": compact,
         "data_format": "Float32", "layout": "tile"},
        {"source_operand": 0, "memory": "l1", "shape": expanded,
         "data_format": "Float32", "layout": "tile"},
    ])
    return specs


@lru_cache(maxsize=8)
def build_tt_lang_resident_attention_forward_owner(
    *,
    sequence_length: int,
    hidden_features: int,
    kv_features: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    physical_rank: int,
    logical_rank: int,
    lora_scale: float,
    attention_scale: float,
    has_bias: bool = False,
    bias_after_bf16_base: bool = False,
    grouped_qkv_lora: bool = False,
    active_lora_roles: tuple[str, ...] = ("q", "k", "v", "o"),
    canonical_grouped_qkv_rank: bool = False,
    canonical_separate_qkv_rank: bool = False,
    exact_attention_tail: bool = False,
    fused_exact_attention_owner: bool = False,
    canonical_bf16_attention: bool = False,
    base_k_tiles_per_phase: int = 1,
    o_lora_k_tiles_per_phase: int | None = None,
    canonical_rank_k_tiles_per_phase: int = 2,
) -> Callable:
    """Build one public attention call with a private canonical O tail."""

    if not 1 <= logical_rank <= physical_rank:
        raise ValueError("logical rank must fit the physical rank tile")
    if physical_rank % 32 or physical_rank > 128:
        raise ValueError("canonical O owner requires tiled rank up to 128")
    if canonical_grouped_qkv_rank and (has_bias or logical_rank != 8):
        raise ValueError(
            "canonical Smol grouped-QKV rank currently requires biasless rank 8"
        )
    if canonical_separate_qkv_rank and canonical_grouped_qkv_rank:
        raise ValueError(
            "grouped and separate canonical Q/K/V rank prefixes are exclusive"
        )
    if canonical_separate_qkv_rank and (has_bias or not exact_attention_tail):
        raise ValueError(
            "separate canonical Q/K/V rank prefixes require the biasless "
            "exact attention tail"
        )
    if fused_exact_attention_owner and not exact_attention_tail:
        raise ValueError(
            "fused exact attention ownership requires the exact attention contract"
        )
    if fused_exact_attention_owner and canonical_bf16_attention:
        raise ValueError(
            "the fused exact attention owner currently implements the captured "
            "BF16-boundary/FP32-softmax contract"
        )
    outer_exact_tail = bool(
        exact_attention_tail and not fused_exact_attention_owner
    )
    if "TT_FINETUNE_O_RANK_K_PHASE" in os.environ:
        canonical_rank_k_tiles_per_phase = int(
            os.environ["TT_FINETUNE_O_RANK_K_PHASE"]
        )
    exact_tail_arena_memory = os.getenv(
        "TT_FINETUNE_EXACT_TAIL_ARENA_MEMORY", "l1"
    ).strip().lower()
    if exact_tail_arena_memory not in {"l1", "dram"}:
        raise ValueError(
            "TT_FINETUNE_EXACT_TAIL_ARENA_MEMORY must be 'l1' or 'dram'"
        )
    exact_tail_raw_memory = os.getenv(
        "TT_FINETUNE_EXACT_TAIL_RAW_MEMORY", "l1"
    ).strip().lower()
    if exact_tail_raw_memory not in {"l1", "dram"}:
        raise ValueError(
            "TT_FINETUNE_EXACT_TAIL_RAW_MEMORY must be 'l1' or 'dram'"
        )
    exact_tail_stage = os.getenv(
        "TT_FINETUNE_EXACT_TAIL_STAGE", "full"
    ).strip().lower()
    if exact_tail_stage not in {
        "full", "prefix", "raw", "scale", "tail", "native", "probpack", "headpack",
        "rankpack",
    }:
        raise ValueError(
            "TT_FINETUNE_EXACT_TAIL_STAGE must be one of "
            "'full', 'prefix', 'raw', 'scale', 'tail', 'native', 'probpack', 'headpack' "
            "or 'rankpack'"
        )

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError("resident attention owner requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    attention = build_tt_lang_attention_forward_qkv_resident(
        sequence_length=sequence_length,
        hidden_features=hidden_features,
        kv_features=kv_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        physical_rank=physical_rank,
        lora_scale=lora_scale,
        attention_scale=attention_scale,
        base_k_tiles_per_phase=base_k_tiles_per_phase,
        lora_k_tiles_per_phase=(
            2 if fused_exact_attention_owner else None
        ),
        o_lora_k_tiles_per_phase=o_lora_k_tiles_per_phase,
        has_bias=has_bias,
        bias_after_bf16_base=bias_after_bf16_base,
        grouped_qkv_lora=grouped_qkv_lora,
        active_lora_roles=active_lora_roles,
        smol_grouped_qkv_lora=canonical_grouped_qkv_rank,
        separate_qkv_lora=canonical_separate_qkv_rank,
        precomputed_separate_qkv_delta=canonical_separate_qkv_rank,
        rebase_projection_cbs=True,
        # The accepted resident child still owns the Q/K/V projections and
        # RoPE, but its fused attention reduction selects a different
        # reduction tree once the operands become private L1 tensors.  The
        # exact outer owner consumes the Q/K checkpoints directly and runs
        # the explicitly pinned canonical QK schedule below instead.
        publish_raw_score_checkpoint=(
            os.environ.get(
                "TT_FINETUNE_PUBLISH_RAW_ATTENTION_SCORE", "0"
            ) == "1"
        ),
        # The canonical outer owner recomputes the exact attention core and O
        # projection from the resident Q/K/V checkpoints.  Running the legacy
        # attention tail and O inside the prefix child was pure duplicate
        # work.  Compile the child as Q/K/V -> RoPE only and retain the same
        # private arena handoff.
        projection_rope_prefix_only=(
            outer_exact_tail
            and os.environ.get(
                "TT_FINETUNE_PROJECTION_ROPE_PREFIX_ONLY", "1"
            ) != "0"
        ),
        canonical_moreh_softmax=(
            (outer_exact_tail or fused_exact_attention_owner)
            and os.environ.get(
                "TT_FINETUNE_CANONICAL_MOREH_SOFTMAX", "0"
            ) == "1"
        ),
        # The accepted resident model path promotes RoPE Q/K before QK and
        # keeps the raw score in FP32.  The fused owner must retain that same
        # public numerical contract; the BF16-boundary variant is a distinct
        # isolated attention contract and cannot be substituted here.
        fp32_canonical_attention=fused_exact_attention_owner,
    )
    grouped_rank = (
        build_tt_lang_grouped_qkv_rank_path(
            tokens=sequence_length,
            hidden_features=hidden_features,
            logical_rank=logical_rank,
        )
        if canonical_grouped_qkv_rank else None
    )
    target_aware_o = (
        os.environ.get("TT_FINETUNE_TARGET_AWARE_O", "0") == "1"
    )
    o_lora_active = "o" in active_lora_roles or not target_aware_o
    native_o = build_tt_lang_native_lora_o_path(
        tokens=sequence_length,
        hidden_features=hidden_features,
        logical_rank=logical_rank,
        scale=lora_scale,
        frozen_weight_transposed=True,
        # TTNN's all-DRAM canonical selector uses two K tiles for the
        # 128x576 @ 8x576.T SmolLM2 operation.  Keeping this explicit avoids
        # the L1 destination selecting a different one-tile reduction tree.
        rank_k_tiles_per_phase=canonical_rank_k_tiles_per_phase,
        lora_active=o_lora_active,
    )
    direct_copy = build_tt_lang_owner_aligned_copy(
        rows=sequence_length,
        columns=hidden_features,
    )
    rank_copy = build_tt_lang_owner_aligned_copy(
        rows=sequence_length,
        columns=physical_rank,
    )
    head_page = attention.forward_plan.arena.slice("head_output").page_offset
    rank_page = attention.forward_plan.arena.slice(
        "o_rank_hidden"
    ).page_offset
    value_page = attention.forward_plan.arena.slice("value").page_offset
    probability_page = attention.forward_plan.arena.slice(
        "probability_fp32"
    ).page_offset
    query_page = attention.forward_plan.arena.slice("query").page_offset
    key_page = attention.forward_plan.arena.slice("key").page_offset

    exact_tail = (
        build_tt_lang_native_attention_fp32_tail_compact_value_private(
            sequence_length=sequence_length,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            scale=attention_scale,
        )
        if outer_exact_tail and not canonical_bf16_attention else None
    )
    bf16_l1_tensor_indices = (
        (0, 1, 3)
        if canonical_bf16_attention and logical_rank >= 32
        else tuple(range(9))
    )
    bf16_core = (
        build_tt_lang_native_attention_bf16_core_compact_private(
            sequence_length=sequence_length,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            scale=attention_scale,
            private_memory=(
                "l1"
                if len(bf16_l1_tensor_indices) == 9 else "hybrid"
            ),
            l1_tensor_indices=bf16_l1_tensor_indices,
        )
        if outer_exact_tail and canonical_bf16_attention else None
    )
    native_qk = (
        build_tt_lang_native_qk_prefix(
            sequence_length=sequence_length,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )
        if outer_exact_tail and not canonical_bf16_attention else None
    )
    probability_pack = (
        build_raw_fp32_checkpoint_pack(
            rows=query_heads * sequence_length,
            columns=sequence_length,
            arena_rows=attention.output_shape[0],
            arena_columns=attention.output_shape[1],
            page_offset=probability_page,
        )
        if outer_exact_tail else None
    )
    debug_stable_fp32_head = bool(
        outer_exact_tail
        and not canonical_bf16_attention
        and os.environ.get(
            "TT_FINETUNE_DEBUG_PUBLISH_STABLE_FP32_HEAD", "0"
        ) == "1"
    )
    fp32_head_pack = (
        build_raw_fp32_checkpoint_pack(
            rows=query_heads * sequence_length,
            columns=head_dim,
            arena_rows=attention.output_shape[0],
            arena_columns=attention.output_shape[1],
            # Diagnostic-only: replace the first half of the probability
            # checkpoint after its normal publication. The run terminates at
            # the layer checkpoint, so no training consumer observes this.
            page_offset=probability_page,
        )
        if debug_stable_fp32_head else None
    )
    exact_scale = (
        build_fp32_pow2_scale(
            rows=query_heads * sequence_length,
            columns=sequence_length,
            shift=3,
            canonical_ttnn_multiply=True,
        )
        if (
            outer_exact_tail
            and not canonical_bf16_attention
            and attention_scale == 0.125
        ) else None
    )
    if outer_exact_tail and not canonical_bf16_attention and exact_scale is None:
        raise ValueError(
            "the exact resident attention tail currently requires scale 0.125"
        )
    compact_value_copy = (
        build_tt_lang_owner_aligned_copy(
            rows=sequence_length * kv_heads,
            columns=head_dim,
        )
        if outer_exact_tail else None
    )
    query_copy = (
        build_tt_lang_owner_aligned_copy(
            rows=sequence_length * query_heads,
            columns=head_dim,
        )
        if outer_exact_tail else None
    )
    key_copy = (
        build_tt_lang_owner_aligned_copy(
            rows=sequence_length * kv_heads,
            columns=head_dim,
        )
        if outer_exact_tail else None
    )
    arena_copy = (
        build_tt_lang_owner_aligned_copy(
            rows=attention.output_shape[0],
            columns=attention.output_shape[1],
        )
        if outer_exact_tail else None
    )

    @ttl.operation(grid="full")
    def _probe(
        t00, t01, t02, t03, t04, t05, t06, t07,
        t08, t09, t10, t11, t12, t13, t14, t15,
        t16, t17, t18, t19, t20, t21,
    ):
        source = ttl.make_dataflow_buffer_like(t00, shape=(1, 1), block_count=1)
        output = ttl.make_dataflow_buffer_like(t21, shape=(1, 1), block_count=1)

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                transaction = ttl.copy(t00[0:1, 0:1], tile)
                transaction.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with output.reserve() as target:
                    target.store(tile)

        @ttl.datamovement()
        def write():
            with output.wait() as tile:
                transaction = ttl.copy(tile, t21[0:1, 0:1])
                transaction.wait()

    # The artifact-v4 owner serializes several independently compiled child
    # operations.  Its own TT-XLA disk-cache key must therefore change when
    # any child implementation changes, even when all public shapes and
    # arguments remain identical.  Previously this ended in a fixed ``v6``;
    # after the resident QK reduction was changed to TTNN's canonical simple
    # BMM schedule, the full model could still load the old parent artifact.
    # Hash the actual decorated child identities so dependency changes are
    # propagated through the composition boundary automatically.
    dependency_operations = (
        attention,
        grouped_rank,
        native_o,
        direct_copy,
        rank_copy,
        exact_tail,
        bf16_core,
        native_qk,
        probability_pack,
        fp32_head_pack,
        exact_scale,
        query_copy,
        key_copy,
        compact_value_copy,
        arena_copy,
    )
    dependency_identity = []
    for dependency in dependency_operations:
        if dependency is None:
            continue
        entry = getattr(dependency, "_tt_lang_operation_entry", None)
        if entry is None:
            raise RuntimeError(
                "resident attention owner dependency has no TT-Lang identity"
            )
        dependency_identity.append(
            f"{entry.operation_id}:{entry.version_tag}"
        )
    dependency_digest = hashlib.sha256(
        "|".join(dependency_identity).encode()
    ).hexdigest()[:16]
    rank_prefix_programs = (
        int(canonical_grouped_qkv_rank)
        + 6 * int(canonical_separate_qkv_rank)
    )
    fused_exact_internal_specs = (
        ([{
            "source_operand": 0,
            "memory": "l1",
            "shape": [sequence_length, 3 * logical_rank],
        }] if canonical_grouped_qkv_rank else [])
        + ([{
            "source_operand": 0,
            "memory": "l1",
            "shape": [sequence_length, physical_rank],
            "data_format": "BFloat16",
            "layout": "tile",
        } for _ in range(3)] if canonical_separate_qkv_rank else [])
        + ([{
            "source_operand": 0,
            "memory": "l1",
            "shape": [sequence_length, columns],
            "data_format": "BFloat16",
            "layout": "tile",
        } for columns in (hidden_features, kv_features, kv_features)]
           if canonical_separate_qkv_rank else [])
    )

    variant = (
        f"{sequence_length}x{hidden_features}.kv{kv_features}."
        f"q{query_heads}.k{kv_heads}.d{head_dim}.pr{physical_rank}."
        f"lr{logical_rank}.bias{int(has_bias)}."
        f"grouped{int(grouped_qkv_lora)}."
        f"targets{'-'.join(active_lora_roles) or 'none'}."
        f"canonicalqvk{int(canonical_grouped_qkv_rank)}."
        f"separateqvk{int(canonical_separate_qkv_rank)}."
        f"exacttail{int(exact_attention_tail)}."
        f"fusedexact{int(fused_exact_attention_owner)}."
        f"bf16core{int(canonical_bf16_attention)}."
        f"{('arenadram.' if exact_attention_tail and exact_tail_arena_memory == 'dram' else '')}"
        f"{('rawdram.' if exact_attention_tail and exact_tail_raw_memory == 'dram' else '')}"
        f"{((exact_tail_stage + 'only.') if exact_attention_tail and exact_tail_stage != 'full' else '')}"
        f"debugfp32head{int(debug_stable_fp32_head)}."
        f"rankk{canonical_rank_k_tiles_per_phase}."
        f"deps{dependency_digest}.v10"
    )

    @tt_lang_operation(
        operation_id=f"tt_finetune.resident_attention_forward_owner.{variant}",
        arg_roles=("in",) * 21 + ("out",),
        version_tag=f"resident-attention-forward-owner-{variant}",
    )
    def operation(
        t00, t01, t02, t03, t04, t05, t06, t07,
        t08, t09, t10, t11, t12, t13, t14, t15,
        t16, t17, t18, t19, t20, t21,
    ):
        tensors = (
            t00, t01, t02, t03, t04, t05, t06, t07,
            t08, t09, t10, t11, t12, t13, t14, t15,
            t16, t17, t18, t19, t20, t21,
        )
        grouped_rank_seed = _compile_placeholder_like(
            t00, shape=(sequence_length, 3 * logical_rank)
        )
        separate_rank_seeds = (
            tuple(
                _compile_placeholder_like(
                    t00, shape=(sequence_length, physical_rank)
                )
                for _ in range(3)
            )
            if canonical_separate_qkv_rank else ()
        )
        separate_delta_seeds = (
            (
                _compile_placeholder_like(
                    t00, shape=(sequence_length, hidden_features)
                ),
                _compile_placeholder_like(
                    t00, shape=(sequence_length, kv_features)
                ),
                _compile_placeholder_like(
                    t00, shape=(sequence_length, kv_features)
                ),
            )
            if canonical_separate_qkv_rank else ()
        )
        private_base = 22 + rank_prefix_programs
        private_arena_seed = (
            _compile_placeholder_like(t21, shape=attention.output_shape)
            if outer_exact_tail else None
        )
        attention_tensors = list(tensors)
        if canonical_grouped_qkv_rank:
            attention_tensors[3] = grouped_rank_seed
            attention_tensors[7] = grouped_rank_seed
            attention_tensors[11] = grouped_rank_seed
        elif canonical_separate_qkv_rank:
            attention_tensors[3] = separate_rank_seeds[0]
            attention_tensors[7] = separate_rank_seeds[1]
            attention_tensors[11] = separate_rank_seeds[2]
            # The resident reader interprets each remapped B operand as a
            # token-major, already reduced LoRA delta when the separate exact
            # path is selected.
            attention_tensors[4] = separate_delta_seeds[0]
            attention_tensors[8] = separate_delta_seeds[1]
            attention_tensors[12] = separate_delta_seeds[2]
        if outer_exact_tail:
            attention_tensors[21] = private_arena_seed
        attention_child = compile_serialized_child(
            attention, tuple(attention_tensors)
        )
        grouped_rank_child = None
        if grouped_rank is not None:
            grouped_rank_children = compile_serialized_sequence_children(
                grouped_rank, (t00, t03, grouped_rank_seed)
            )
            if len(grouped_rank_children) != 1:
                raise RuntimeError("canonical grouped Q/V/K child count changed")
            grouped_rank_child = grouped_rank_children[0]
        children = []
        if grouped_rank_child is not None:
            children.append(
                _remap_all(grouped_rank_child, {0: 0, 1: 3, 2: 22})
            )
        if canonical_separate_qkv_rank:
            # The untouched biasless graph computes three independent
            # all-DRAM rank projections.  The former ungrouped resident path
            # recomputed them inside the projection kernel with a different
            # spill schedule at logical rank 32.  Keep the three standard A
            # tensors, reproduce the captured two-K TTNN schedule explicitly,
            # and retain each physical rank tile in private L1.
            rank_compute = {
                "math_fidelity": "HiFi4",
                "math_approx_mode": True,
                "fp32_dest_acc_en": True,
                "packer_l1_acc": False,
                "dst_full_sync_en": False,
            }
            rank_program = {
                "family": "2d", "grid": [11, 10],
                "in0_block_w": 2,
                "out_subblock_h": 1, "out_subblock_w": 1,
                "out_block_h": 1, "out_block_w": 1,
                "per_core_m": 1, "per_core_n": 1,
                "transpose_mcast": False, "fuse_batch": False,
            }
            rank_children = [
                {
                    "operation": "ttnn.matmul",
                    "operand_indices": [0, operand, 22 + index],
                    "transpose_a": False,
                    "transpose_b": True,
                    "compute_config": rank_compute,
                }
                for index, operand in enumerate((3, 7, 11))
            ]
            # resident runtime exactness froze this two-K schedule for one physical
            # rank tile.  TTNN deliberately chooses a different reduction
            # schedule when N spans multiple tiles; forcing the rank-32
            # program independently on every tile changes BF16 spill points.
            # Preserve the accepted R32 contract and let TTNN's shape-aware
            # selector reproduce the generic reference for R64--R128.
            if physical_rank == 32:
                for rank_child in rank_children:
                    rank_child["program_config"] = rank_program
            children.extend(rank_children)
            delta_compute = {
                "math_fidelity": "HiFi4",
                "math_approx_mode": False,
                "fp32_dest_acc_en": True,
                "packer_l1_acc": False,
                "dst_full_sync_en": False,
            }
            children.extend(
                {
                    "operation": "ttnn.matmul",
                    "operand_indices": [
                        22 + index,
                        operand,
                        25 + index,
                    ],
                    "transpose_a": False,
                    "transpose_b": True,
                    "compute_config": delta_compute,
                }
                for index, operand in enumerate((4, 8, 12))
            )
        attention_mapping = {
            index: (
                22 if canonical_grouped_qkv_rank and index in {3, 7, 11}
                else (
                    22 + {3: 0, 7: 1, 11: 2}[index]
                    if canonical_separate_qkv_rank
                    and index in {3, 7, 11}
                    else 25 + {4: 0, 8: 1, 12: 2}[index]
                    if canonical_separate_qkv_rank
                    and index in {4, 8, 12}
                    else private_base
                    if outer_exact_tail and index == 21
                    else index
                )
            )
            for index in serialized_child_tensor_ids(attention_child)
        }
        children.append(_remap_all(attention_child, attention_mapping))

        if outer_exact_tail:
            assert probability_pack is not None
            assert query_copy is not None
            assert key_copy is not None
            assert compact_value_copy is not None
            assert arena_copy is not None
            if canonical_bf16_attention:
                assert bf16_core is not None
            else:
                assert exact_tail is not None
                assert native_qk is not None
                assert exact_scale is not None

            arena_id = private_base
            query_id = private_base + 1
            key_id = private_base + 2
            raw_score_id = private_base + 3
            compact_value_id = private_base + 4
            head4_id = private_base + 5
            head2_id = private_base + 6
            tail_private_base = private_base + 7
            # Both exact tails own eight private tensors.  The canonical BF16
            # core uses local IDs 5..13, but its raw QK tensor (local 7) is
            # deliberately mapped to ``raw_score_id`` above the tail range.
            tail_private_count = 8
            o_output_id = tail_private_base + tail_private_count
            o_rank_id = o_output_id + 1
            o_internal_base = o_rank_id + 1
            qk_internal_base = o_internal_base + 4

            query_seed = _compile_placeholder_like(
                t00,
                shape=(1, query_heads, sequence_length, head_dim),
            )
            key_seed = _compile_placeholder_like(
                t00,
                shape=(1, kv_heads, sequence_length, head_dim),
            )
            raw_score_seed = _compile_placeholder_like(
                t00,
                shape=(1, query_heads, sequence_length, sequence_length),
                dtype=torch.float32,
            )
            flat_query_source = _compile_placeholder_like(
                t00, shape=(sequence_length * query_heads, head_dim)
            )
            flat_query_destination = _compile_placeholder_like(
                t00, shape=(sequence_length * query_heads, head_dim)
            )
            flat_key_source = _compile_placeholder_like(
                t00, shape=(sequence_length * kv_heads, head_dim)
            )
            flat_key_destination = _compile_placeholder_like(
                t00, shape=(sequence_length * kv_heads, head_dim)
            )
            compact_value_seed = _compile_placeholder_like(
                t00,
                shape=(1, kv_heads, sequence_length, head_dim),
            )
            flat_value_source = _compile_placeholder_like(
                t00, shape=(sequence_length * kv_heads, head_dim)
            )
            flat_value_destination = _compile_placeholder_like(
                t00, shape=(sequence_length * kv_heads, head_dim)
            )
            head4_seed = _compile_placeholder_like(
                t00,
                shape=(1, query_heads, sequence_length, head_dim),
            )
            flat_head_source = _compile_placeholder_like(
                t00, shape=(sequence_length, hidden_features)
            )
            flat_head_destination = _compile_placeholder_like(
                t00, shape=(sequence_length, hidden_features)
            )
            head2_seed = _compile_placeholder_like(
                t00, shape=(sequence_length, hidden_features)
            )
            probability_flat_seed = _compile_placeholder_like(
                t00,
                shape=(query_heads * sequence_length, sequence_length),
                dtype=torch.float32,
            )
            fp32_head_flat_seed = _compile_placeholder_like(
                t00,
                shape=(query_heads * sequence_length, head_dim),
                dtype=torch.float32,
            )
            scaled_score_seed = _compile_placeholder_like(
                t00,
                shape=(1, query_heads, sequence_length, sequence_length),
                dtype=torch.float32,
            )
            o_output_seed = _compile_placeholder_like(
                t00, shape=(sequence_length, hidden_features)
            )
            o_rank_seed = _compile_placeholder_like(
                t00, shape=(sequence_length, physical_rank)
            )

            query_child = compile_serialized_child(
                query_copy,
                (flat_query_source, flat_query_destination),
            )
            query_child = make_serialized_page_copy_bitwise(
                query_child,
                total_pages=(
                    query_heads * (sequence_length // 32) * (head_dim // 32)
                ),
                column_tiles=head_dim // 32,
                source_page_offset=query_page,
            )
            key_child = compile_serialized_child(
                key_copy,
                (flat_key_source, flat_key_destination),
            )
            key_child = make_serialized_page_copy_bitwise(
                key_child,
                total_pages=(
                    kv_heads * (sequence_length // 32) * (head_dim // 32)
                ),
                column_tiles=head_dim // 32,
                source_page_offset=key_page,
            )
            qk_children = []
            qk_map = {}
            if not canonical_bf16_attention:
                qk_children = compile_serialized_sequence_children(
                    native_qk,
                    (query_seed, key_seed, raw_score_seed),
                )
                if len(qk_children) != native_qk.physical_programs:
                    raise RuntimeError("canonical private QK child count changed")
                qk_map = {
                    0: query_id,
                    1: key_id,
                    2: raw_score_id,
                    3: qk_internal_base,
                    4: qk_internal_base + 1,
                    5: qk_internal_base + 2,
                }
            value_child = compile_serialized_child(
                compact_value_copy,
                (flat_value_source, flat_value_destination),
            )
            # The resident QKV writer already checkpoints V in
            # [kv_head, sequence_tile, head_dimension_tile] order.  A prior
            # composition treated it as token-major and permuted the pages a
            # second time, feeding the exact FP32 tail the wrong KV rows.
            value_child = make_serialized_page_copy_bitwise(
                value_child,
                total_pages=(
                    kv_heads * (sequence_length // 32) * (head_dim // 32)
                ),
                column_tiles=head_dim // 32,
                source_page_offset=value_page,
            )
            tail_children = []
            scale_child = None
            tail_map = {}
            bf16_children = []
            bf16_map = {}
            if canonical_bf16_attention:
                bf16_children = compile_serialized_sequence_children(
                    bf16_core,
                    (
                        query_seed,
                        key_seed,
                        t18,
                        compact_value_seed,
                        head4_seed,
                    ),
                )
                if len(bf16_children) != bf16_core.physical_programs:
                    raise RuntimeError("canonical BF16 attention child count changed")
                bf16_map = {
                    0: query_id,
                    1: key_id,
                    2: 18,
                    3: compact_value_id,
                    4: head4_id,
                    5: tail_private_base,
                    6: tail_private_base + 1,
                    7: raw_score_id,
                    8: tail_private_base + 2,
                    9: tail_private_base + 3,
                    10: tail_private_base + 4,
                    11: tail_private_base + 5,
                    12: tail_private_base + 6,
                    13: tail_private_base + 7,
                }
            else:
                tail_children = compile_serialized_sequence_children(
                    exact_tail,
                    (raw_score_seed, t18, compact_value_seed, head4_seed),
                )
                if len(tail_children) != exact_tail.physical_programs:
                    raise RuntimeError(
                        "exact private attention-tail child count changed"
                    )
                scale_child = compile_serialized_child(
                    exact_scale, (raw_score_seed, scaled_score_seed)
                )
                tail_map = {
                    0: raw_score_id,
                    1: 18,
                    2: compact_value_id,
                    3: head4_id,
                }
                tail_map.update({
                    local: tail_private_base + local - 4
                    for local in range(4, 12)
                })

            head_to_token_child = compile_serialized_child(
                direct_copy,
                (flat_head_source, flat_head_destination),
            )
            head_to_token_child = make_serialized_head_to_token_copy_bitwise(
                head_to_token_child,
                sequence_tiles=sequence_length // 32,
                query_heads=query_heads,
                head_dimension_tiles=head_dim // 32,
                source_page_offset=0,
            )

            native_children = compile_serialized_sequence_children(
                native_o,
                (head2_seed, t13, t14, t15, o_output_seed, o_rank_seed),
            )
            if len(native_children) != native_o.physical_programs:
                raise RuntimeError("canonical O child count changed")
            native_map = {
                0: head2_id,
                1: 13,
                2: 14,
                3: 15,
                4: o_output_id,
                5: o_rank_id,
                6: o_internal_base,
                7: o_internal_base + 1,
                8: o_internal_base + 2,
                9: o_internal_base + 3,
            }

            probability_child = compile_serialized_child(
                probability_pack,
                (probability_flat_seed, private_arena_seed),
            )
            fp32_head_child = (
                compile_serialized_child(
                    fp32_head_pack,
                    (fp32_head_flat_seed, private_arena_seed),
                )
                if fp32_head_pack is not None else None
            )
            head_checkpoint_child = compile_serialized_child(
                direct_copy,
                (flat_head_source, flat_head_destination),
            )
            head_checkpoint_child = make_serialized_page_copy_bitwise(
                head_checkpoint_child,
                total_pages=(
                    query_heads * (sequence_length // 32) * (head_dim // 32)
                ),
                column_tiles=head_dim // 32,
                source_page_offset=0,
            )
            head_checkpoint_child = retarget_tensor_writes_to_packed_pages(
                head_checkpoint_child,
                packed_tensor=1,
                tensor_page_offsets={
                    1: (
                        query_page
                        if os.environ.get(
                            "TT_FINETUNE_DEBUG_HEAD_AT_QUERY", "0"
                        ) == "1"
                        else head_page
                    )
                },
            )
            rank_checkpoint_child = compile_serialized_child(
                rank_copy, (o_rank_seed, private_arena_seed)
            )
            rank_checkpoint_child = make_serialized_page_copy_bitwise(
                rank_checkpoint_child,
                total_pages=(sequence_length // 32) * (physical_rank // 32),
                column_tiles=physical_rank // 32,
                source_page_offset=0,
            )
            rank_checkpoint_child = retarget_tensor_writes_to_packed_pages(
                rank_checkpoint_child,
                packed_tensor=1,
                tensor_page_offsets={1: rank_page},
            )
            output_checkpoint_child = compile_serialized_child(
                direct_copy, (o_output_seed, private_arena_seed)
            )
            output_checkpoint_child = make_serialized_page_copy_bitwise(
                output_checkpoint_child,
                total_pages=(sequence_length // 32) * (hidden_features // 32),
                column_tiles=hidden_features // 32,
                source_page_offset=0,
            )
            final_arena_child = compile_serialized_child(
                arena_copy, (private_arena_seed, t21)
            )
            final_arena_child = make_serialized_page_copy_bitwise(
                final_arena_child,
                total_pages=attention.forward_plan.arena.used_pages,
                column_tiles=hidden_features // 32,
                source_page_offset=0,
            )

            probability_id = (
                tail_private_base + 6
                if canonical_bf16_attention
                else tail_private_base + 5
            )
            attention_core_children = (
                [_remap_all(child, bf16_map) for child in bf16_children]
                if canonical_bf16_attention
                else [
                    *(_remap_all(child, qk_map) for child in qk_children),
                    # The generic TTNN scalar-broadcast multiply selects a
                    # different numerical path for a compiler-owned L1 input.
                    # Keep GQA expansion, replace full+multiply with one exact
                    # IEEE power-of-two program, then retain the FP32 tail.
                    _remap_all(tail_children[0], tail_map),
                    _remap_all(
                        scale_child,
                        {0: raw_score_id, 1: tail_private_base + 2},
                    ),
                    *(
                        _remap_all(child, tail_map)
                        for child in tail_children[3:]
                    ),
                ]
            )
            children.extend([
                _remap_all(query_child, {0: arena_id, 1: query_id}),
                _remap_all(key_child, {0: arena_id, 1: key_id}),
                _remap_all(value_child, {0: arena_id, 1: compact_value_id}),
                *attention_core_children,
                _remap_all(
                    head_to_token_child, {0: head4_id, 1: head2_id}
                ),
                *(_remap_all(child, native_map) for child in native_children),
                _remap_all(
                    probability_child,
                    {
                        0: (
                            raw_score_id
                            if os.environ.get(
                                "TT_FINETUNE_DEBUG_PUBLISH_STABLE_RAW_SCORE",
                                "0",
                            ) == "1"
                            else probability_id
                        ),
                        1: arena_id,
                    },
                ),
                *(
                    [_remap_all(
                        fp32_head_child,
                        {0: tail_private_base + 7, 1: arena_id},
                    )]
                    if fp32_head_child is not None else []
                ),
                _remap_all(
                    head_checkpoint_child, {0: head4_id, 1: arena_id}
                ),
                *(
                    [_remap_all(
                        rank_checkpoint_child, {0: o_rank_id, 1: arena_id}
                    )]
                    if o_lora_active else []
                ),
                _remap_all(
                    output_checkpoint_child, {0: o_output_id, 1: arena_id}
                ),
                _remap_all(final_arena_child, {0: arena_id, 1: 21}),
            ])
            if exact_tail_stage != "full":
                # Retain the complete exact prefix (resident QKV/RoPE,
                # Q/K page relays and the four-program canonical QK path)
                # before truncating later qualification boundaries.
                prefix_programs = 7 + rank_prefix_programs
                retained_programs = {
                    "prefix": prefix_programs,
                    "raw": prefix_programs,
                    "scale": prefix_programs + 2,
                    "tail": prefix_programs + 10,
                    "native": prefix_programs + 10 + native_o.physical_programs,
                    "probpack": (
                        prefix_programs + 11 + native_o.physical_programs
                    ),
                    "headpack": (
                        prefix_programs + 12 + native_o.physical_programs
                    ),
                    "rankpack": (
                        prefix_programs + 13 + native_o.physical_programs
                    ),
                }[exact_tail_stage]
                if exact_tail_stage in {"raw", "scale"}:
                    if canonical_bf16_attention:
                        raise ValueError(
                            "scale boundary diagnostic requires the FP32 tail"
                        )
                    children = children[:retained_programs] + [
                        _remap_all(
                            probability_child,
                            {
                                0: (
                                    raw_score_id
                                    if exact_tail_stage == "raw"
                                    else tail_private_base + 2
                                ),
                                1: arena_id,
                            },
                        ),
                        _remap_all(
                            final_arena_child, {0: arena_id, 1: 21}
                        ),
                    ]
                else:
                    children = children[:retained_programs] + [
                        _remap_all(final_arena_child, {0: arena_id, 1: 21})
                    ]

            score_shape = [
                1, query_heads, sequence_length, sequence_length
            ]
            expanded_shape = [
                1, query_heads, sequence_length, head_dim
            ]
            if canonical_bf16_attention:
                # Local IDs 5,6 and 8..13 from the exact BF16 core.  Local 7
                # (the raw QK result) is mapped to raw_score_id and described
                # by the fixed owner prefix below.
                tail_internal_tensors = [
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": expanded_shape,
                        "data_format": "BFloat16",
                        "layout": "tile",
                    },
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": expanded_shape,
                        "data_format": "BFloat16",
                        "layout": "tile",
                    },
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": [1, 1, 1, 1],
                        "data_format": "BFloat16",
                        "layout": "tile",
                    },
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": score_shape,
                        "data_format": "BFloat16",
                        "layout": "tile",
                    },
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": score_shape,
                        "data_format": "BFloat16",
                        "layout": "tile",
                    },
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": score_shape,
                        "data_format": "Float32",
                        "layout": "tile",
                    },
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": score_shape,
                        "data_format": "Float32",
                        "layout": "tile",
                    },
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": score_shape,
                        "data_format": "BFloat16",
                        "layout": "tile",
                    },
                ]
            else:
                # Preserve the established compact FP32-tail layout.
                tail_internal_tensors = [
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": expanded_shape,
                        "data_format": "BFloat16",
                        "layout": "tile",
                    },
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": [1, 1, 1, 1],
                        "data_format": "Float32",
                        "layout": "tile",
                    },
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": score_shape,
                        "data_format": "Float32",
                        "layout": "tile",
                    },
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": [1, 1, sequence_length, sequence_length],
                        "data_format": "Float32",
                        "layout": "tile",
                    },
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": score_shape,
                        "data_format": "Float32",
                        "layout": "tile",
                    },
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": score_shape,
                        "data_format": "Float32",
                        "layout": "tile",
                    },
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": expanded_shape,
                        "data_format": "Float32",
                        "layout": "tile",
                    },
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": expanded_shape,
                        "data_format": "Float32",
                        "layout": "tile",
                    },
                ]

            internal_tensors = (
                ([{
                    "source_operand": 0,
                    "memory": "l1",
                    "shape": [sequence_length, 3 * logical_rank],
                }] if canonical_grouped_qkv_rank else [])
                + ([{
                    "source_operand": 0,
                    "memory": "l1",
                    "shape": [sequence_length, physical_rank],
                    "data_format": "BFloat16",
                    "layout": "tile",
                } for _ in range(3)] if canonical_separate_qkv_rank else [])
                + ([{
                    "source_operand": 0,
                    "memory": "l1",
                    "shape": [sequence_length, columns],
                    "data_format": "BFloat16",
                    "layout": "tile",
                } for columns in (hidden_features, kv_features, kv_features)]
                   if canonical_separate_qkv_rank else [])
                + [
                    {
                        "source_operand": 21,
                        "memory": (
                            "clone"
                            if exact_tail_arena_memory == "dram"
                            else "l1"
                        ),
                        "shape": list(attention.output_shape),
                        "layout": "tile",
                    },
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": [
                            1, query_heads, sequence_length, head_dim
                        ],
                        "data_format": "BFloat16",
                        "layout": "tile",
                    },
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": [1, kv_heads, sequence_length, head_dim],
                        "data_format": "BFloat16",
                        "layout": "tile",
                    },
                    {
                        "source_operand": 0,
                        "memory": (
                            "l1"
                            if canonical_bf16_attention
                            else "clone"
                            if exact_tail_raw_memory == "dram"
                            else "l1"
                        ),
                        "shape": [
                            1,
                            query_heads,
                            sequence_length,
                            sequence_length,
                        ],
                        "data_format": (
                            "BFloat16"
                            if canonical_bf16_attention
                            else "Float32"
                        ),
                        "layout": "tile",
                    },
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": [1, kv_heads, sequence_length, head_dim],
                        "layout": "tile",
                    },
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": [1, query_heads, sequence_length, head_dim],
                        "layout": "tile",
                    },
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": [sequence_length, hidden_features],
                        "layout": "tile",
                    },
                    *tail_internal_tensors,
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": [sequence_length, hidden_features],
                        "layout": "tile",
                    },
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": [sequence_length, physical_rank],
                        "layout": "tile",
                    },
                    {"source_operand": 0, "memory": "l1", "layout": "tile"},
                    {"source_operand": 0, "memory": "l1", "layout": "tile"},
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": [1, 1],
                        "layout": "tile",
                    },
                    {"source_operand": 0, "memory": "l1", "layout": "tile"},
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": [
                            1, query_heads, sequence_length, head_dim
                        ],
                        "data_format": "Float32",
                        "layout": "tile",
                    },
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": [1, kv_heads, sequence_length, head_dim],
                        "data_format": "Float32",
                        "layout": "tile",
                    },
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": [
                            1, query_heads, sequence_length, head_dim
                        ],
                        "data_format": "Float32",
                        "layout": "tile",
                    },
                ]
            )
            if canonical_bf16_attention:
                internal_tensors = canonical_bf16_attention_owner_internal_specs(
                    sequence_length=sequence_length,
                    hidden_features=hidden_features,
                    kv_heads=kv_heads,
                    query_heads=query_heads,
                    head_dim=head_dim,
                    physical_rank=physical_rank,
                    output_shape=attention.output_shape,
                    grouped_rank_width=(
                        3 * logical_rank
                        if canonical_grouped_qkv_rank else None
                    ),
                    arena_memory=exact_tail_arena_memory,
                    bf16_l1_tensor_indices=bf16_l1_tensor_indices,
                )
        elif fused_exact_attention_owner:
            # The exact Q/K/V -> RoPE -> QK -> softmax -> P@V -> O path is
            # already one physical TT-Metal child.  Only optional rank-prefix
            # tensors belong to the composition layer; every attention
            # intermediate remains in that child's CB/L1 allocation and the
            # child writes the final checkpoint arena directly to tensor 21.
            internal_tensors = fused_exact_internal_specs
        else:
            rank_seed = _compile_placeholder_like(
                t00, shape=(sequence_length, physical_rank)
            )
            native_children = compile_serialized_sequence_children(
                native_o, (t00, t13, t14, t15, t21, rank_seed)
            )
            if len(native_children) != native_o.physical_programs:
                raise RuntimeError("canonical O child count changed")
            head_copy = compile_serialized_child(direct_copy, (t21, t00))
            head_copy = make_serialized_head_to_token_copy_bitwise(
                head_copy,
                sequence_tiles=sequence_length // 32,
                query_heads=query_heads,
                head_dimension_tiles=head_dim // 32,
                source_page_offset=head_page,
            )
            checkpoint_copy = compile_serialized_child(
                rank_copy, (rank_seed, t21)
            )
            checkpoint_copy = make_serialized_page_copy_bitwise(
                checkpoint_copy,
                total_pages=(sequence_length // 32) * (physical_rank // 32),
                column_tiles=physical_rank // 32,
                source_page_offset=0,
            )
            checkpoint_copy = retarget_tensor_writes_to_packed_pages(
                checkpoint_copy,
                packed_tensor=1,
                tensor_page_offsets={1: rank_page},
            )
            output_copy = compile_serialized_child(direct_copy, (t00, t21))
            native_map = {
                0: private_base,
                1: 13,
                2: 14,
                3: 15,
                4: private_base + 1,
                5: private_base + 2,
                6: private_base + 3,
                7: private_base + 4,
                8: private_base + 5,
                9: private_base + 6,
            }
            children.extend([
                _remap_all(head_copy, {0: 21, 1: private_base}),
                *(_remap_all(child, native_map) for child in native_children),
                _remap_all(
                    checkpoint_copy, {0: private_base + 2, 1: 21}
                ),
                _remap_all(output_copy, {0: private_base + 1, 1: 21}),
            ])
            internal_tensors = (
                ([{
                    "source_operand": 0,
                    "memory": "l1",
                    "shape": [sequence_length, 3 * logical_rank],
                }] if canonical_grouped_qkv_rank else [])
                + ([{
                    "source_operand": 0,
                    "memory": "l1",
                    "shape": [sequence_length, physical_rank],
                    "data_format": "BFloat16",
                    "layout": "tile",
                } for _ in range(3)] if canonical_separate_qkv_rank else [])
                + ([{
                    "source_operand": 0,
                    "memory": "l1",
                    "shape": [sequence_length, columns],
                    "data_format": "BFloat16",
                    "layout": "tile",
                } for columns in (hidden_features, kv_features, kv_features)]
                   if canonical_separate_qkv_rank else [])
                + [
                    {"source_operand": 0, "memory": "l1"},
                    {"source_operand": 0, "memory": "l1"},
                    {
                        "source_operand": 0,
                        "memory": "l1",
                        "shape": [sequence_length, physical_rank],
                    },
                    {"source_operand": 0, "memory": "l1"},
                    {"source_operand": 0, "memory": "l1"},
                    {"source_operand": 0, "memory": "l1", "shape": [1, 1]},
                    {"source_operand": 0, "memory": "l1"},
                ]
            )

        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(*tensors)
        closure = _probe.__closure__ or ()
        for name, cell in zip(_probe.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                compiled.program_sequence = children
                compiled.internal_tensors = internal_tensors
            break
        return result

    operation.artifact_format = 4
    # The established resident attention child is one physical TT-Metal
    # program.  Exact mode additionally relays Q/K from the private arena and
    # owns the four-program canonical QK schedule before the private tail.
    full_physical_programs = (
        1 + rank_prefix_programs
        if fused_exact_attention_owner else
        (
            (
                1 + rank_prefix_programs
                + 3 + bf16_core.physical_programs + 1
                + native_o.physical_programs + 5 - int(not o_lora_active)
            )
            if canonical_bf16_attention else
            (
                1 + rank_prefix_programs
                + 2 + native_qk.physical_programs
                + 1 + exact_tail.physical_programs + 1 - 1
                + native_o.physical_programs + 5 - int(not o_lora_active)
            )
        )
        if exact_attention_tail else
        1 + rank_prefix_programs + 3 + native_o.physical_programs
    ) + int(debug_stable_fp32_head)
    if outer_exact_tail and exact_tail_stage != "full":
        prefix_programs = 7 + rank_prefix_programs
        retained_programs = {
            "prefix": prefix_programs,
            "raw": prefix_programs,
            "scale": prefix_programs + 2,
            "tail": prefix_programs + 10,
            "native": prefix_programs + 10 + native_o.physical_programs,
            "probpack": prefix_programs + 11 + native_o.physical_programs,
            "headpack": prefix_programs + 12 + native_o.physical_programs,
            "rankpack": prefix_programs + 13 + native_o.physical_programs,
        }[exact_tail_stage]
        # Stage diagnostics always append the final private-arena publication
        # after truncating the child sequence.
        operation.physical_programs = retained_programs + (
            2 if exact_tail_stage in {"raw", "scale"} else 1
        )
    else:
        operation.physical_programs = full_physical_programs
    operation.public_custom_calls = 1
    operation.compiler_owned_l1_tensors = (
        rank_prefix_programs
        if fused_exact_attention_owner else
        (21 if canonical_bf16_attention else 24)
        + rank_prefix_programs
        if exact_attention_tail else
        7 + int(canonical_grouped_qkv_rank)
    )
    operation.output_shape = attention.output_shape
    operation.forward_plan = attention.forward_plan
    operation.intermediate_dram_bytes = 0
    operation.checkpoint_dram_bytes = attention.checkpoint_dram_bytes
    operation.projection_cb_bytes = attention.projection_cb_bytes
    operation.rebase_projection_cbs = attention.rebase_projection_cbs
    operation.reverse_handoff_noc_bytes = attention.reverse_handoff_noc_bytes
    operation.safe_default = False
    operation.experimental_backend = (
        "attention_forward_full_exact_fused_owner_v1"
        if fused_exact_attention_owner else
        "attention_forward_full_canonical_qk_private_tail_v2"
        if exact_attention_tail else
        "attention_forward_canonical_qvk_o_v2"
        if canonical_grouped_qkv_rank else
        "attention_forward_canonical_o_v1"
    )
    operation.canonical_grouped_qkv_rank = bool(canonical_grouped_qkv_rank)
    operation.canonical_separate_qkv_rank = bool(canonical_separate_qkv_rank)
    operation.exact_attention_tail = bool(exact_attention_tail)
    operation.fused_exact_attention_owner = bool(fused_exact_attention_owner)
    operation.canonical_bf16_attention = bool(canonical_bf16_attention)
    operation.exact_tail_arena_memory = exact_tail_arena_memory
    operation.exact_tail_stage = exact_tail_stage
    operation.canonical_rank_k_tiles_per_phase = canonical_rank_k_tiles_per_phase
    exact_internal_specs = (
        canonical_bf16_attention_owner_internal_specs(
            sequence_length=sequence_length,
            hidden_features=hidden_features,
            kv_heads=kv_heads,
            query_heads=query_heads,
            head_dim=head_dim,
            physical_rank=physical_rank,
            output_shape=attention.output_shape,
            grouped_rank_width=(
                3 * logical_rank if canonical_grouped_qkv_rank else None
            ),
            arena_memory=exact_tail_arena_memory,
            bf16_l1_tensor_indices=bf16_l1_tensor_indices,
        )
        if outer_exact_tail and canonical_bf16_attention else
        canonical_fp32_attention_owner_internal_specs(
            sequence_length=sequence_length,
            hidden_features=hidden_features,
            kv_heads=kv_heads,
            query_heads=query_heads,
            head_dim=head_dim,
            physical_rank=physical_rank,
            output_shape=attention.output_shape,
            grouped_rank_width=(
                3 * logical_rank if canonical_grouped_qkv_rank else None
            ),
            arena_memory=exact_tail_arena_memory,
            raw_memory=exact_tail_raw_memory,
        )
        if outer_exact_tail else (
            fused_exact_internal_specs
            if fused_exact_attention_owner else None
        )
    )
    if (
        outer_exact_tail
        and exact_internal_specs is not None
        and canonical_separate_qkv_rank
    ):
        exact_internal_specs = [
            {
                "source_operand": 0,
                "memory": "l1",
                "shape": [sequence_length, physical_rank],
                "data_format": "BFloat16",
                "layout": "tile",
            }
            for _ in range(3)
        ] + [
            {
                "source_operand": 0,
                "memory": "l1",
                "shape": [sequence_length, columns],
                "data_format": "BFloat16",
                "layout": "tile",
            }
            for columns in (hidden_features, kv_features, kv_features)
        ] + exact_internal_specs
    operation.internal_tensor_specs = exact_internal_specs
    operation._ttl_program = _probe
    return operation


__all__ = [
    "build_tt_lang_resident_attention_forward_owner",
    "canonical_bf16_attention_owner_internal_specs",
    "canonical_fp32_attention_owner_internal_specs",
]
