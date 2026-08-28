"""Compact-program resident attention-block forward owner.

One StableHLO/PJRT custom call executes five manageable TT-Metal programs:

  exact input RMSNorm
  -> resident Q/K/V + RoPE + causal attention + O
  -> attention-output arena extraction
  -> exact residual add
  -> exact post-attention RMSNorm

The normalized input/residual and extracted attention output use a two-buffer
compiler-owned L1 ping-pong pair. The checkpoint arena and final hidden state
are the only framework-visible results. This deliberately avoids the
pathological host compile time of one giant generated C++ translation unit
while retaining the same device-resident ownership boundary.
"""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_attention_forward_qkv_resident import (
    build_tt_lang_attention_forward_qkv_resident,
)
from .tt_lang_resident_attention_forward_owner import (
    build_tt_lang_resident_attention_forward_owner,
)
from .tt_lang_exact_residual_add import build_tt_lang_exact_residual_add
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
from .tt_lang_native_lora_o_path import build_tt_lang_native_lora_o_path
from .tt_lang_rmsnorm_forward import build_tt_lang_rmsnorm_forward


ttl = None
tt_lang_operation = None


def _remap_all(program: dict, mapping: dict[int, int]) -> dict:
    used = serialized_child_tensor_ids(program)
    return remap_serialized_child(
        program, {tensor: mapping[tensor] for tensor in used}
    )


def _compile_placeholder_like(tensor, *, shape: tuple[int, int]):
    """Create a shape-only sibling for nested child serialization.

    The PJRT bridge resolves an owner with ``_StubTtnnTensor`` operands.  Such
    compile-only values intentionally do not implement tensor slicing, but a
    nested native operation still needs accurate output metadata for its
    logical rank tensor.  Reconstruct the same lightweight tensor class with
    a different shape; no runtime tensor or device allocation is created.
    """

    try:
        return type(tensor)(
            shape=shape,
            dtype=tensor.dtype,
            layout=tensor.layout,
            memory_config=tensor.memory_config(),
            device=tensor.device(),
        )
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(
            "resident attention owner requires shape-only PJRT compile operands"
        ) from exc


@lru_cache(maxsize=8)
def build_tt_lang_resident_attention_block_forward_owner(
    *,
    sequence_length: int,
    hidden_features: int,
    kv_features: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    physical_rank: int,
    lora_scale: float,
    attention_scale: float,
    rmsnorm_epsilon: float,
    has_bias: bool = False,
    active_lora_roles: tuple[str, ...] = ("q", "k", "v", "o"),
    bias_after_bf16_base: bool = False,
    grouped_qkv_lora: bool = False,
    base_k_tiles_per_phase: int = 1,
    o_lora_k_tiles_per_phase: int | None = None,
    canonical_native_o_path: bool = False,
    logical_rank: int | None = None,
    exact_attention_tail: bool = False,
    fused_exact_attention_owner: bool = False,
    canonical_bf16_attention: bool = False,
    canonical_grouped_qkv_rank: bool = False,
    canonical_separate_qkv_rank: bool = False,
    debug_stop_after: str | None = None,
) -> Callable:
    """Build an exact compact-program attention-sublayer owner."""

    if debug_stop_after not in (
        None, "rmsnorm", "double_rmsnorm", "attention", "residual",
        "post_rms_public",
    ):
        raise ValueError(
            "debug_stop_after must be rmsnorm, double_rmsnorm, attention, "
            "residual, post_rms_public or None"
        )
    if canonical_native_o_path and (
        logical_rank is None
        or not 1 <= int(logical_rank) <= physical_rank
        or physical_rank % 32
        or physical_rank > 128
    ):
        raise ValueError(
            "canonical native O path requires logical rank inside a tiled "
            "physical rank up to 128"
        )
    if canonical_native_o_path and debug_stop_after in (
        "rmsnorm", "double_rmsnorm"
    ):
        raise ValueError(
            "canonical native O path requires an attention-containing prefix"
        )
    if canonical_bf16_attention and not exact_attention_tail:
        raise ValueError("canonical BF16 attention requires the exact tail")
    if exact_attention_tail and not canonical_native_o_path:
        raise ValueError("exact attention tail requires canonical native O")
    if canonical_grouped_qkv_rank and canonical_separate_qkv_rank:
        raise ValueError(
            "grouped and separate canonical Q/K/V rank prefixes are exclusive"
        )
    attention_roles = {"q", "k", "v", "o"}
    active_lora_roles = tuple(
        role for role in dict.fromkeys(active_lora_roles)
        if role in attention_roles
    )

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError(
                "resident attention-block forward owner requires TT-XLA"
            ) from exc
        ttl, tt_lang_operation = ttl_module, decorator

    rms = build_tt_lang_rmsnorm_forward(
        sequence_length=sequence_length,
        hidden_size=hidden_features,
        epsilon=rmsnorm_epsilon,
        # The real Transformers graph lowers this expression to TTNN's fused
        # RMSNorm numerical schedule.  The legacy native stage exposes an
        # intermediate BF16 boundary and is not an exact oracle for learned
        # gamma.  Keep the compact resident block on the qualified physical
        # TTNN schedule (one final BF16 pack) instead.
        stage="ttnn_fused_final",
    )
    attention_arguments = dict(
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
        o_lora_k_tiles_per_phase=o_lora_k_tiles_per_phase,
        has_bias=has_bias,
        bias_after_bf16_base=bias_after_bf16_base,
        grouped_qkv_lora=grouped_qkv_lora,
        active_lora_roles=active_lora_roles,
    )
    if exact_attention_tail:
        attention = build_tt_lang_resident_attention_forward_owner(
            **attention_arguments,
            logical_rank=int(logical_rank),
            canonical_grouped_qkv_rank=canonical_grouped_qkv_rank,
            canonical_separate_qkv_rank=canonical_separate_qkv_rank,
            exact_attention_tail=True,
            fused_exact_attention_owner=fused_exact_attention_owner,
            canonical_bf16_attention=canonical_bf16_attention,
        )
    else:
        attention = build_tt_lang_attention_forward_qkv_resident(
            **attention_arguments,
            # This owner is composable across transformer layers without a
            # PJRT boundary, so every projection compute CB starts a fresh
            # epoch just like its reader and writer views.
            rebase_projection_cbs=True,
        )
    qwen_grouped_prefix = bool(
        exact_attention_tail and grouped_qkv_lora and has_bias
        and os.environ.get("TT_FINETUNE_QWEN_LAYER_EXTERNAL_PREFIX", "0") != "1"
    )
    grouped_qkv_features = hidden_features + 2 * kv_features
    hidden_tiles = hidden_features // 32
    output_page = attention.forward_plan.arena.slice("output").page_offset
    if output_page % hidden_tiles:
        raise RuntimeError("attention output is not row-aligned in its arena")
    extract = build_tt_lang_owner_aligned_copy(
        rows=sequence_length,
        columns=hidden_features,
        source_row_offset=(output_page // hidden_tiles) * 32,
    )
    head_page = attention.forward_plan.arena.slice(
        "head_output"
    ).page_offset
    o_rank_page = attention.forward_plan.arena.slice(
        "o_rank_hidden"
    ).page_offset
    o_rank_checkpoint = build_tt_lang_owner_aligned_copy(
        rows=sequence_length,
        # Copy every physical rank tile, including tail padding, so backward
        # observes the exact native matmul output without a framework edge.
        columns=physical_rank,
    )
    add = build_tt_lang_exact_residual_add(
        rows=sequence_length,
        columns=hidden_features,
        owner_aligned=True,
    )
    direct_copy = build_tt_lang_owner_aligned_copy(
        rows=sequence_length,
        columns=hidden_features,
    )

    @ttl.operation(grid="full")
    def _probe(
        t00, t01, t02, t03, t04, t05, t06, t07,
        t08, t09, t10, t11, t12, t13, t14, t15,
        t16, t17, t18, t19, t20, t21, t22, t23, t24,
    ):
        source = ttl.make_dataflow_buffer_like(
            t00, shape=(1, 1), block_count=1
        )
        output = ttl.make_dataflow_buffer_like(
            t24, shape=(1, 1), block_count=1
        )

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                tx = ttl.copy(t00[0:1, 0:1], tile)
                tx.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with output.reserve() as target:
                    target.store(tile)

        @ttl.datamovement()
        def write():
            with output.wait() as tile:
                tx = ttl.copy(tile, t24[0:1, 0:1])
                tx.wait()

    variant = (
        f"{sequence_length}x{hidden_features}.kv{kv_features}."
        f"q{query_heads}.k{kv_heads}.d{head_dim}.r{physical_rank}."
        f"bias{int(has_bias)}.grouped{int(grouped_qkv_lora)}."
        f"targets{'-'.join(active_lora_roles) or 'none'}."
        f"stop{debug_stop_after or 'full'}."
        f"oloraphase{o_lora_k_tiles_per_phase}."
        f"nativeo{int(canonical_native_o_path)}."
        f"rank{logical_rank}.exact{int(exact_attention_tail)}."
        f"fusedexact{int(fused_exact_attention_owner)}."
        f"bf16{int(canonical_bf16_attention)}."
        f"groupedrank{int(canonical_grouped_qkv_rank)}."
        f"separaterank{int(canonical_separate_qkv_rank)}."
        f"qwenprefix{int(qwen_grouped_prefix)}.ttnnrms1.v22"
    )

    @tt_lang_operation(
        operation_id=(
            f"tt_finetune.resident_attention_block_forward_owner.{variant}"
        ),
        arg_roles=("in",) * 23 + ("out", "out"),
        version_tag=f"resident-attention-block-forward-owner-{variant}",
    )
    def operation(
        t00, t01, t02, t03, t04, t05, t06, t07,
        t08, t09, t10, t11, t12, t13, t14, t15,
        t16, t17, t18, t19, t20, t21, t22, t23, t24,
    ):
        tensors = (
            t00, t01, t02, t03, t04, t05, t06, t07,
            t08, t09, t10, t11, t12, t13, t14, t15,
            t16, t17, t18, t19, t20, t21, t22, t23, t24,
        )

        # Compile each accepted child with public placeholders. Tensor IDs
        # 25..26 form a compiler-owned hidden-sized L1 ping-pong pair. The
        # normalized input in 25 is dead after attention, so residual add
        # deliberately reuses it instead of allocating a third hidden tensor.
        # t21/t22 are distinct FP32/BF16 scaffolds required to preserve every
        # child-local tensor ID during artifact serialization.
        input_rms = compile_serialized_child(
            rms, (t00, t19, t22, t21, t24)
        )
        grouped_base_placeholders = tuple(
            _compile_placeholder_like(
                t00, shape=(sequence_length, grouped_qkv_features)
            )
            for _ in range(3)
        )
        grouped_h_placeholders = tuple(
            _compile_placeholder_like(
                t00, shape=(sequence_length, 3 * int(logical_rank))
            )
            for _ in range(3)
        )
        attention_operands = (
            t24,
            grouped_base_placeholders[0] if qwen_grouped_prefix else t01,
            t02,
            grouped_h_placeholders[0] if qwen_grouped_prefix else t03,
            t04,
            grouped_base_placeholders[1] if qwen_grouped_prefix else t05,
            t06,
            grouped_h_placeholders[1] if qwen_grouped_prefix else t07,
            t08,
            grouped_base_placeholders[2] if qwen_grouped_prefix else t09,
            t10,
            grouped_h_placeholders[2] if qwen_grouped_prefix else t11,
            t12,
            t13, t14, t15,
            t16, t17, t18, t19, t20, t23,
        )
        attention_children = (
            compile_serialized_sequence_children(
                attention, attention_operands
            )
            if exact_attention_tail else
            [compile_serialized_child(attention, attention_operands)]
        )
        expected_attention_programs = (
            attention.physical_programs if exact_attention_tail else 1
        )
        if len(attention_children) != expected_attention_programs:
            raise RuntimeError("resident attention child count changed")
        extract_child = compile_serialized_child(extract, (t23, t24))
        direct_copy_child = compile_serialized_child(
            direct_copy, (t23, t24)
        )
        add_child = compile_serialized_child(add, (t24, t00, t22))
        post_rms = compile_serialized_child(
            rms, (t00, t20, t22, t21, t24)
        )

        input_rms_child = _remap_all(
            input_rms, {0: 0, 1: 19, 2: 22, 3: 21, 4: 25}
        )
        attention_map = {
            0: 25,
            **{index: index for index in range(1, 21)},
            21: 23,
        }
        if exact_attention_tail:
            attention_map.update({
                22 + index: 27 + index
                for index in range(attention.compiler_owned_l1_tensors)
            })
        qwen_prefix_children = []
        if qwen_grouped_prefix:
            qwen_private_base = 27 + attention.compiler_owned_l1_tensors
            grouped_h_id = qwen_private_base
            grouped_base_id = qwen_private_base + 1
            compute = {"math_fidelity": "HiFi4", "fp32_dest_acc_en": True}
            qwen_prefix_children = [
                {
                    "operation": "ttnn.matmul",
                    "operand_indices": [25, 3, grouped_h_id],
                    "transpose_a": False,
                    "transpose_b": True,
                    "bias_rank1": True,
                    "compute_config": compute,
                    "program_config": {
                        "family": "2d", "grid": [11, 10],
                        "in0_block_w": 1,
                        "out_subblock_h": 1, "out_subblock_w": 1,
                        "out_block_h": 1, "out_block_w": 1,
                        "per_core_m": 1, "per_core_n": 1,
                        "transpose_mcast": False, "fuse_batch": False,
                    },
                },
                {
                    "operation": "ttnn.linear",
                    "operand_indices": [25, 1, 2, grouped_base_id],
                    "transpose_a": False,
                    "transpose_b": True,
                    "compute_config": compute,
                },
            ]
            attention_map.update({
                1: grouped_base_id, 5: grouped_base_id, 9: grouped_base_id,
                3: grouped_h_id, 7: grouped_h_id, 11: grouped_h_id,
            })
        attention_pipeline = [
            _remap_all(child, attention_map)
            for child in attention_children
        ]
        if canonical_native_o_path and not exact_attention_tail:
            native_o = build_tt_lang_native_lora_o_path(
                tokens=sequence_length,
                hidden_features=hidden_features,
                logical_rank=int(logical_rank),
                scale=lora_scale,
                frozen_weight_transposed=True,
                rank_k_tiles_per_phase=(
                    (1 if hidden_features == 576 else 2)
                    if o_lora_k_tiles_per_phase is None
                    else int(o_lora_k_tiles_per_phase)
                ),
            )
            rank_hidden_seed = _compile_placeholder_like(
                # The integrated attention ABI pads every LoRA A/B pair to
                # one physical rank tile.  Unlike the isolated native-O
                # qualifier (which receives unpadded rank-8/16 weights), this
                # matmul therefore has a logical width of 32 and backward's
                # arena checkpoint stores that complete physical tile.
                t00, shape=(sequence_length, physical_rank)
            )
            native_o_children = compile_serialized_sequence_children(
                native_o,
                (t00, t13, t14, t15, t24, rank_hidden_seed),
            )
            if len(native_o_children) != native_o.physical_programs:
                raise RuntimeError("canonical native O child count changed")
            native_o_map = {
                0: 27,
                1: 13,
                2: 14,
                3: 15,
                4: 28,
                5: 29,
                6: 30,
                7: 31,
                8: 32,
                9: 33,
            }
            head_extract_child = compile_serialized_child(
                direct_copy, (t23, t24)
            )
            head_extract_child = make_serialized_head_to_token_copy_bitwise(
                head_extract_child,
                sequence_tiles=sequence_length // 32,
                query_heads=query_heads,
                head_dimension_tiles=head_dim // 32,
                source_page_offset=head_page,
            )
            output_copy_child = compile_serialized_child(
                direct_copy, (t24, t23)
            )
            o_rank_checkpoint_child = compile_serialized_child(
                # Relay the complete padded rank tile; this is the physical
                # checkpoint ABI already consumed by resident backward.
                o_rank_checkpoint, (rank_hidden_seed, t23)
            )
            o_rank_checkpoint_child = make_serialized_page_copy_bitwise(
                o_rank_checkpoint_child,
                total_pages=(sequence_length // 32) * (physical_rank // 32),
                column_tiles=physical_rank // 32,
                source_page_offset=0,
            )
            o_rank_checkpoint_child = retarget_tensor_writes_to_packed_pages(
                o_rank_checkpoint_child,
                packed_tensor=1,
                tensor_page_offsets={1: o_rank_page},
            )
            attention_pipeline.extend(
                [
                    _remap_all(head_extract_child, {0: 23, 1: 27}),
                    *(
                        _remap_all(child, native_o_map)
                        for child in native_o_children
                    ),
                    _remap_all(
                        o_rank_checkpoint_child, {0: 29, 1: 23}
                    ),
                    _remap_all(output_copy_child, {0: 28, 1: 23}),
                ]
            )
        if debug_stop_after == "rmsnorm":
            # Publish the private RMSNorm handoff solely to identify whether
            # the first compact program is safe across repeated invocations.
            children = [
                input_rms_child,
                _remap_all(direct_copy_child, {0: 25, 1: 24}),
            ]
            internal_tensor_count = 1
        elif debug_stop_after == "double_rmsnorm":
            children = [
                input_rms_child,
                _remap_all(
                    post_rms, {0: 25, 1: 20, 2: 22, 3: 21, 4: 24}
                ),
            ]
            internal_tensor_count = 1
        elif debug_stop_after == "attention":
            # The accepted attention arena is public; extract its O result
            # directly to the public hidden output for a balanced prefix.
            children = [
                input_rms_child,
                *qwen_prefix_children,
                *attention_pipeline,
                _remap_all(extract_child, {0: 23, 1: 24}),
            ]
            internal_tensor_count = 1
        elif debug_stop_after == "residual":
            children = [
                input_rms_child,
                *qwen_prefix_children,
                *attention_pipeline,
                _remap_all(extract_child, {0: 23, 1: 26}),
                _remap_all(add_child, {0: 26, 1: 0, 2: 24}),
            ]
            internal_tensor_count = 2
        elif debug_stop_after == "post_rms_public":
            # Diagnostic only: route the residual through the public hidden
            # buffer before post RMSNorm, and publish normalized tiles into
            # the first hidden-sized arena region. This distinguishes a
            # third compiler-owned-L1 lifetime defect from cross-program
            # compute-engine state left by attention.
            children = [
                input_rms_child,
                *qwen_prefix_children,
                *attention_pipeline,
                _remap_all(extract_child, {0: 23, 1: 26}),
                _remap_all(add_child, {0: 26, 1: 0, 2: 24}),
                _remap_all(
                    post_rms, {0: 24, 1: 20, 2: 22, 3: 21, 4: 23}
                ),
            ]
            internal_tensor_count = 2
        else:
            children = [
                input_rms_child,
                *qwen_prefix_children,
                *attention_pipeline,
                _remap_all(extract_child, {0: 23, 1: 26}),
                _remap_all(add_child, {0: 26, 1: 0, 2: 25}),
                _remap_all(
                    post_rms, {0: 25, 1: 20, 2: 22, 3: 21, 4: 24}
                ),
            ]
            internal_tensor_count = 2

        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(*tensors)
        closure = _probe.__closure__ or ()
        for name, cell in zip(_probe.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                compiled.program_sequence = children
                base_internal_count = (
                    2
                    if canonical_native_o_path or exact_attention_tail
                    else internal_tensor_count
                )
                compiled.internal_tensors = [
                    {
                        "source_operand": 0,
                        "memory": (
                            "clone" if qwen_grouped_prefix and index == 0
                            else "l1"
                        ),
                        "shape": [sequence_length, hidden_features],
                        "data_format": "BFloat16",
                        "layout": "tile",
                    }
                    for index in range(base_internal_count)
                ]
                if exact_attention_tail:
                    for spec in attention.internal_tensor_specs:
                        mapped = dict(spec)
                        if mapped.get("source_operand") == 21:
                            mapped["source_operand"] = 23
                        compiled.internal_tensors.append(mapped)
                    if qwen_grouped_prefix:
                        compiled.internal_tensors.extend(
                            {
                                "source_operand": 0,
                                "memory": memory,
                                "shape": shape,
                                "data_format": "BFloat16",
                                "layout": "tile",
                            }
                            for shape, memory in (
                                (
                                    [sequence_length, 3 * int(logical_rank)],
                                    "l1",
                                ),
                                (
                                    [sequence_length, grouped_qkv_features],
                                    "clone",
                                ),
                            )
                        )
                elif canonical_native_o_path:
                    compiled.internal_tensors.extend(
                        [
                            {"source_operand": 0, "memory": "l1"},
                            {"source_operand": 0, "memory": "l1"},
                            {
                                "source_operand": 0,
                                "memory": "l1",
                                "shape": [
                                    sequence_length, physical_rank
                                ],
                            },
                            {"source_operand": 0, "memory": "l1"},
                            {"source_operand": 0, "memory": "l1"},
                            {
                                "source_operand": 0,
                                "memory": "l1",
                                "shape": [1, 1],
                            },
                            {"source_operand": 0, "memory": "l1"},
                        ]
                    )
            break
        return result

    operation.artifact_format = 4
    operation.physical_programs = {
        "rmsnorm": 2,
        "double_rmsnorm": 2,
        "attention": 3,
        "residual": 4,
        "post_rms_public": 5,
        None: 5,
    }[debug_stop_after] + (
        attention.physical_programs - 1
        if exact_attention_tail else 0
    ) + (
        3 + build_tt_lang_native_lora_o_path(
            tokens=sequence_length,
            hidden_features=hidden_features,
            logical_rank=int(logical_rank),
            scale=lora_scale,
            frozen_weight_transposed=True,
            rank_k_tiles_per_phase=(
                (1 if hidden_features == 576 else 2)
                if o_lora_k_tiles_per_phase is None
                else int(o_lora_k_tiles_per_phase)
            ),
        ).physical_programs
        if canonical_native_o_path and not exact_attention_tail else 0
    ) + (
        2 if qwen_grouped_prefix and debug_stop_after not in (
            "rmsnorm", "double_rmsnorm"
        ) else 0
    )
    operation.compiler_owned_l1_tensors = {
        "rmsnorm": 1,
        "double_rmsnorm": 1,
        "attention": 1,
        "residual": 2,
        "post_rms_public": 2,
        None: 2,
    }[debug_stop_after] + (
        attention.compiler_owned_l1_tensors
        if exact_attention_tail else
        7 if canonical_native_o_path else 0
    ) + (
        2 if qwen_grouped_prefix and debug_stop_after not in (
            "rmsnorm", "double_rmsnorm"
        ) else 0
    )
    operation.output_shape = attention.output_shape
    operation.forward_plan = attention.forward_plan
    operation.final_hidden_shape = (sequence_length, hidden_features)
    operation.attention_output_page_offset = output_page
    operation.checkpoint_dram_bytes = attention.checkpoint_dram_bytes
    operation.projection_cb_bytes = attention.projection_cb_bytes
    operation.projection_cb_rebase = attention.rebase_projection_cbs
    operation.reverse_handoff_noc_bytes = attention.reverse_handoff_noc_bytes
    operation.intermediate_dram_bytes = 0
    operation.safe_default = False
    operation.experimental_backend = (
        "attention_block_forward_compact_native_o_v1"
        if canonical_native_o_path
        else "attention_block_forward_compact_v1"
    )
    operation.canonical_native_o_path = bool(canonical_native_o_path)
    operation.exact_attention_tail = bool(exact_attention_tail)
    operation.canonical_bf16_attention = bool(canonical_bf16_attention)
    operation.canonical_grouped_qkv_rank = bool(canonical_grouped_qkv_rank)
    operation.canonical_separate_qkv_rank = bool(
        canonical_separate_qkv_rank
    )
    operation.internal_tensor_specs = (
        [
            {
                "source_operand": 0,
                "memory": "clone" if qwen_grouped_prefix else "l1",
                "shape": [sequence_length, hidden_features],
                "data_format": "BFloat16",
                "layout": "tile",
            },
            {
                "source_operand": 0,
                "memory": "l1",
                "shape": [sequence_length, hidden_features],
                "data_format": "BFloat16",
                "layout": "tile",
            },
        ]
        + [
            {
                **dict(spec),
                "source_operand": (
                    23 if spec.get("source_operand") == 21
                    else spec.get("source_operand", 0)
                ),
            }
            for spec in attention.internal_tensor_specs
        ]
        + (
            [
                {
                    "source_operand": 0,
                    "memory": memory,
                    "shape": shape,
                    "data_format": "BFloat16",
                    "layout": "tile",
                }
                for shape, memory in (
                    ([sequence_length, 3 * int(logical_rank)], "l1"),
                    ([sequence_length, grouped_qkv_features], "clone"),
                )
            ] if qwen_grouped_prefix else []
        )
        if exact_attention_tail else None
    )
    # TTNN destination tensors used by the Qwen grouped-QKV prefix carry
    # runtime ownership/lifecycle state.  A standalone layer may reuse them,
    # but a flattened multi-layer artifact must give each layer a distinct
    # destination.  Publish their indices relative to internal_tensor_specs
    # so the outer schedule can rebind only these two tensors without
    # duplicating the rest of the resident attention scratch plan.
    operation.qwen_grouped_prefix_internal_indices = (
        tuple(range(len(operation.internal_tensor_specs) - 2,
                    len(operation.internal_tensor_specs)))
        if qwen_grouped_prefix and operation.internal_tensor_specs is not None
        else ()
    )
    operation.logical_rank = logical_rank
    operation.debug_stop_after = debug_stop_after
    operation._ttl_program = _probe
    return operation
