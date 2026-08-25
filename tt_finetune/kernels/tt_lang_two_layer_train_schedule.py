"""One device schedule for two resident forwards followed by two backwards.

The schedule is intentionally the first training *interval*, not another
micro-kernel.  Two full-layer checkpoint arenas are compiler-owned DRAM
tensors because their lifetime spans the complete forward/reverse gap.  The
layer-0 output copied for layer-1 backward, reverse dHidden chain, and all
short-lived backward scratch tensors are compiler-owned L1 tensors.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_arena_page_copy import build_tt_lang_arena_page_copy
from .tt_lang_program_composition import (
    compile_serialized_child,
    compile_serialized_sequence_children,
    remap_serialized_child,
    serialized_child_tensor_ids,
)
from .tt_lang_resident_layer_autograd_split_owner import (
    build_tt_lang_resident_layer_autograd_split_owner,
)
from .tt_lang_resident_layer_forward_trainable_owner import (
    build_tt_lang_resident_layer_forward_trainable_owner,
)
from .tt_lang_owner_aligned_copy import build_tt_lang_owner_aligned_copy


ttl = None
tt_lang_operation = None


INPUTS = 75
OUTPUTS = 15
EXTERNAL_TENSORS = INPUTS + OUTPUTS
CHECKPOINT0 = EXTERNAL_TENSORS
CHECKPOINT1 = CHECKPOINT0 + 1
LAYER1_HIDDEN = CHECKPOINT1 + 1
REVERSE_CHAIN = LAYER1_HIDDEN + 1
LAYER1_SCRATCH = REVERSE_CHAIN + 1
LAYER0_SCRATCH = LAYER1_SCRATCH + 7
INTERNAL_TENSORS = 18
LAYER0_BLOCK = 7
LAYER1_BLOCK = 41
LAYER0_OUTPUT = 75
LAYER1_OUTPUT = 82
FINAL_DHIDDEN = 89


def _remap_used(program: dict, mapping: dict[int, int]) -> dict:
    used = serialized_child_tensor_ids(program)
    return remap_serialized_child(
        program, {tensor: mapping[tensor] for tensor in used}
    )


def _forward_map(block: int, checkpoint: int, hidden: int) -> dict[int, int]:
    mapping = {0: hidden, 27: checkpoint}
    mapping.update({local: block + local - 1 for local in range(1, 27)})
    return mapping


def _backward_map(
    *,
    block: int,
    checkpoint: int,
    dy: int,
    hidden: int,
    output: int,
    dhidden: int,
    scratch: int,
) -> dict[int, int]:
    mapping = {
        0: dy,
        1: hidden,
        2: block + 18,
        3: block + 26,
        4: 2,
        5: 3,
        6: 4,
        7: block + 27,
        8: block + 14,
        9: block + 13,
        10: block + 15,
        11: block + 16,
        12: block + 28,
        13: block + 3,
        14: block + 2,
        15: block + 29,
        16: block + 7,
        17: block + 6,
        18: block + 30,
        19: block + 11,
        20: block + 10,
        21: block + 31,
        22: block + 25,
        23: block + 24,
        24: block + 32,
        25: block + 21,
        26: block + 20,
        27: block + 33,
        28: block + 23,
        29: block + 22,
        30: checkpoint,
        31: output,
        32: output + 1,
        33: output + 2,
        34: output + 3,
        35: output + 4,
        36: output + 5,
        37: output + 6,
        38: dhidden,
    }
    mapping.update({local: scratch + local - 39 for local in range(39, 46)})
    return mapping


@lru_cache(maxsize=8)
def build_tt_lang_two_layer_train_schedule(
    *,
    source_bundle: str,
    sequence_length: int,
    hidden_features: int,
    intermediate_features: int,
    kv_features: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    physical_rank: int,
    lora_scale: float,
    attention_scale: float,
    rmsnorm_epsilon: float,
    has_bias: bool = False,
    debug_stop_after_layer1_backward: bool = False,
    debug_backward_scratch_memory: str = "l1",
    debug_hidden_memory: str = "l1",
    debug_publish_layer1_dhidden_in_arena: bool = False,
    debug_stop_after_hidden_copy: bool = False,
    debug_stop_after_layer1_mlp: bool = False,
    debug_stop_after_layer1_mlp_internal_copy: bool = False,
    debug_align_layer1_mlp_edge: bool = False,
    debug_align_layer1_rstd_edge: bool = False,
    debug_direct_layer1_stage_output: bool = False,
    debug_tap_layer1_rstd: bool = False,
    debug_tap_layer1_mlp: bool = False,
    debug_layer1_attention_stage: str | None = None,
    debug_skip_layer1_mlp: bool = False,
    debug_extract_layer0_checkpoint: bool = False,
    debug_external_checkpoint_storage: bool = False,
    debug_public_checkpoint_storage: bool = False,
) -> Callable:
    """Build one exact two-forward/two-backward program-sequence owner."""
    if debug_backward_scratch_memory not in ("l1", "clone"):
        raise ValueError("backward scratch memory must be 'l1' or 'clone'")
    if debug_hidden_memory not in ("l1", "clone"):
        raise ValueError("hidden memory must be 'l1' or 'clone'")

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError("two-layer train schedule requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    forward0 = build_tt_lang_resident_layer_forward_trainable_owner(
        sequence_length=sequence_length,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        kv_features=kv_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        physical_rank=physical_rank,
        lora_scale=lora_scale,
        attention_scale=attention_scale,
        has_bias=has_bias,
        input_rmsnorm_epsilon=rmsnorm_epsilon,
        post_attention_rmsnorm_epsilon=rmsnorm_epsilon,
        hidden_page_base=0,
        publish_layer_output_at_page_zero=False,
    )
    forward1 = build_tt_lang_resident_layer_forward_trainable_owner(
        sequence_length=sequence_length,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        kv_features=kv_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        physical_rank=physical_rank,
        lora_scale=lora_scale,
        attention_scale=attention_scale,
        has_bias=has_bias,
        input_rmsnorm_epsilon=rmsnorm_epsilon,
        post_attention_rmsnorm_epsilon=rmsnorm_epsilon,
        hidden_page_base=forward0.layer_output_page_offset,
        publish_layer_output_at_page_zero=False,
    )
    hidden_copy = build_tt_lang_arena_page_copy(
        rows=sequence_length,
        columns=hidden_features,
        source_page_offset=forward0.layer_output_page_offset,
    )
    mlp_edge_copy = build_tt_lang_owner_aligned_copy(
        rows=sequence_length, columns=hidden_features
    )
    backward = build_tt_lang_resident_layer_autograd_split_owner(
        source_bundle=source_bundle,
        sequence_length=sequence_length,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        kv_features=kv_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        physical_rank=physical_rank,
        lora_scale=lora_scale,
        attention_scale=attention_scale,
        rmsnorm_epsilon=rmsnorm_epsilon,
        debug_attention_stop_after=debug_layer1_attention_stage,
    )

    @ttl.operation(grid="full")
    def _probe(
        t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
        t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
        t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
        t30, t31, t32, t33, t34, t35, t36, t37, t38, t39,
        t40, t41, t42, t43, t44, t45, t46, t47, t48, t49,
        t50, t51, t52, t53, t54, t55, t56, t57, t58, t59,
        t60, t61, t62, t63, t64, t65, t66, t67, t68, t69,
        t70, t71, t72, t73, t74, t75, t76, t77, t78, t79,
        t80, t81, t82, t83, t84, t85, t86, t87, t88, t89,
    ):
        probe_source = t02 if debug_layer1_attention_stage == "post_rstd" else t00
        source = ttl.make_dataflow_buffer_like(
            probe_source, shape=(1, 1), block_count=1
        )
        output = ttl.make_dataflow_buffer_like(t89, shape=(1, 1), block_count=1)

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                tx = ttl.copy(probe_source[0:1, 0:1], tile)
                tx.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with output.reserve() as target:
                    target.store(tile)

        @ttl.datamovement()
        def write():
            with output.wait() as tile:
                tx = ttl.copy(tile, t89[0:1, 0:1])
                tx.wait()

    variant = (
        f"{Path(source_bundle).name}.{sequence_length}x{hidden_features}."
        f"i{intermediate_features}.q{query_heads}.k{kv_heads}."
        f"d{head_dim}.r{physical_rank}.bias{int(has_bias)}."
        f"stopb1{int(debug_stop_after_layer1_backward)}."
        f"dhinb1arena{int(debug_publish_layer1_dhidden_in_arena)}."
        f"stopcopy{int(debug_stop_after_hidden_copy)}."
        f"stopb1mlp{int(debug_stop_after_layer1_mlp)}."
        f"stopb1mlpcopy{int(debug_stop_after_layer1_mlp_internal_copy)}."
        f"alignb1mlp{int(debug_align_layer1_mlp_edge)}."
        f"alignb1rstd{int(debug_align_layer1_rstd_edge)}."
        f"directb1stage{int(debug_direct_layer1_stage_output)}."
        f"tapb1rstd{int(debug_tap_layer1_rstd)}."
        f"tapb1mlp{int(debug_tap_layer1_mlp)}."
        f"b1stage{debug_layer1_attention_stage or 'full'}."
        f"skipb1mlp{int(debug_skip_layer1_mlp)}."
        f"extractcp0{int(debug_extract_layer0_checkpoint)}."
        f"externalcp{int(debug_external_checkpoint_storage)}."
        f"publiccp{int(debug_public_checkpoint_storage)}."
        f"hidden{debug_hidden_memory}."
        f"scratch{debug_backward_scratch_memory}.v4"
    )

    @tt_lang_operation(
        operation_id=f"tt_finetune.two_layer_train_schedule.{variant}",
        arg_roles=("in",) * INPUTS + ("out",) * OUTPUTS,
        version_tag=f"two-layer-train-schedule-{variant}",
    )
    def operation(
        t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
        t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
        t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
        t30, t31, t32, t33, t34, t35, t36, t37, t38, t39,
        t40, t41, t42, t43, t44, t45, t46, t47, t48, t49,
        t50, t51, t52, t53, t54, t55, t56, t57, t58, t59,
        t60, t61, t62, t63, t64, t65, t66, t67, t68, t69,
        t70, t71, t72, t73, t74, t75, t76, t77, t78, t79,
        t80, t81, t82, t83, t84, t85, t86, t87, t88, t89,
    ):
        tensors = (
            t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
            t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
            t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
            t30, t31, t32, t33, t34, t35, t36, t37, t38, t39,
            t40, t41, t42, t43, t44, t45, t46, t47, t48, t49,
            t50, t51, t52, t53, t54, t55, t56, t57, t58, t59,
            t60, t61, t62, t63, t64, t65, t66, t67, t68, t69,
            t70, t71, t72, t73, t74, t75, t76, t77, t78, t79,
            t80, t81, t82, t83, t84, t85, t86, t87, t88, t89,
        )

        checkpoint0_id = (
            LAYER0_OUTPUT if debug_public_checkpoint_storage else (
                5 if debug_external_checkpoint_storage else CHECKPOINT0
            )
        )
        checkpoint1_id = (
            LAYER1_OUTPUT if debug_public_checkpoint_storage else (
                6 if debug_external_checkpoint_storage else CHECKPOINT1
            )
        )

        f0 = compile_serialized_child(
            forward0,
            (tensors[0], *tensors[LAYER0_BLOCK:LAYER0_BLOCK + 26], tensors[5]),
        )
        f1 = compile_serialized_child(
            forward1,
            (tensors[5], *tensors[LAYER1_BLOCK:LAYER1_BLOCK + 26], tensors[6]),
        )
        copy_child = compile_serialized_child(
            hidden_copy, (tensors[5], tensors[0])
        )

        def backward_arguments(block: int, checkpoint_operand: int, output: int):
            return (
                tensors[1], tensors[0], tensors[block + 18],
                tensors[block + 26], tensors[2], tensors[3], tensors[4],
                tensors[block + 27], tensors[block + 14], tensors[block + 13],
                tensors[block + 15], tensors[block + 16],
                tensors[block + 28], tensors[block + 3], tensors[block + 2],
                tensors[block + 29], tensors[block + 7], tensors[block + 6],
                tensors[block + 30], tensors[block + 11], tensors[block + 10],
                tensors[block + 31], tensors[block + 25], tensors[block + 24],
                tensors[block + 32], tensors[block + 21], tensors[block + 20],
                tensors[block + 33], tensors[block + 23], tensors[block + 22],
                tensors[checkpoint_operand],
                *tensors[output:output + 7], tensors[FINAL_DHIDDEN],
            )

        b1 = compile_serialized_sequence_children(
            backward,
            backward_arguments(LAYER1_BLOCK, 6, LAYER1_OUTPUT),
        )
        need_b0 = not (
            debug_stop_after_hidden_copy
            or debug_stop_after_layer1_mlp
            or debug_stop_after_layer1_mlp_internal_copy
            or debug_stop_after_layer1_backward
            or debug_layer1_attention_stage is not None
        )
        b0 = (
            compile_serialized_sequence_children(
                backward,
                backward_arguments(LAYER0_BLOCK, 5, LAYER0_OUTPUT),
            )
            if need_b0 else []
        )
        if (
            len(b1) != backward.physical_programs
            or (need_b0 and len(b0) != backward.physical_programs)
        ):
            raise RuntimeError("split layer backward physical count changed")

        b1_map = _backward_map(
            block=LAYER1_BLOCK,
            checkpoint=(
                checkpoint0_id
                if debug_extract_layer0_checkpoint else checkpoint1_id
            ),
            dy=1,
            hidden=LAYER1_HIDDEN,
            output=LAYER1_OUTPUT,
            dhidden=(
                LAYER1_OUTPUT
                if (
                    debug_publish_layer1_dhidden_in_arena
                    or debug_layer1_attention_stage == "attention"
                )
                else (
                    FINAL_DHIDDEN
                    if (
                        debug_stop_after_layer1_backward
                        or debug_layer1_attention_stage is not None
                    ) else REVERSE_CHAIN
                )
            ),
            scratch=LAYER1_SCRATCH,
        )
        if debug_skip_layer1_mlp:
            # Attention-owner diagnostic ABI: bypass the MLP producer and
            # feed the public layer dY directly into the post-RMS backward
            # consumer. This keeps the diagnostic mathematically identical
            # to the standalone attention-block owner.
            b1_map[39] = 1
        b0_map = _backward_map(
            block=LAYER0_BLOCK,
            checkpoint=checkpoint0_id,
            dy=REVERSE_CHAIN,
            hidden=0,
            output=LAYER0_OUTPUT,
            dhidden=FINAL_DHIDDEN,
            scratch=LAYER0_SCRATCH,
        )
        forward_prefix = [
            _remap_used(f0, _forward_map(LAYER0_BLOCK, checkpoint0_id, 0)),
            _remap_used(
                f1, _forward_map(LAYER1_BLOCK, checkpoint1_id, checkpoint0_id)
            ),
            _remap_used(
                copy_child,
                {
                    0: checkpoint0_id,
                    1: (
                        FINAL_DHIDDEN
                        if debug_stop_after_hidden_copy else LAYER1_HIDDEN
                    ),
                },
            ),
        ]
        aligned_b1_map = dict(b1_map)
        aligned_b1_map[39] = REVERSE_CHAIN
        mlp_edge_copy_child = (
            compile_serialized_child(
                mlp_edge_copy, (tensors[0], tensors[FINAL_DHIDDEN])
            )
            if (
                debug_align_layer1_mlp_edge
                or debug_stop_after_layer1_mlp_internal_copy
            ) else None
        )
        rstd_edge_copy_child = (
            compile_serialized_child(mlp_edge_copy, (tensors[2], tensors[2]))
            if debug_align_layer1_rstd_edge else None
        )
        rstd_tap_child = (
            compile_serialized_child(
                mlp_edge_copy, (tensors[2], tensors[LAYER0_OUTPUT])
            )
            if debug_tap_layer1_rstd else None
        )
        mlp_tap_child = (
            compile_serialized_child(
                mlp_edge_copy, (tensors[0], tensors[LAYER0_OUTPUT])
            )
            if debug_tap_layer1_mlp else None
        )
        rstd_aligned_b1_map = dict(b1_map)
        rstd_aligned_b1_map[42] = LAYER0_SCRATCH + 3
        ordinary_b1_children = [
            *(_remap_used(child, b1_map) for child in b1)
        ]
        if debug_align_layer1_rstd_edge:
            ordinary_b1_children = [
                *(_remap_used(child, b1_map) for child in b1[:4]),
                _remap_used(
                    rstd_edge_copy_child,
                    {0: LAYER1_SCRATCH + 3, 1: LAYER0_SCRATCH + 3},
                ),
                *(
                    _remap_used(child, rstd_aligned_b1_map)
                    for child in b1[4:]
                ),
            ]
        prefix = [
            *forward_prefix,
            *(
                [
                    _remap_used(b1[0], b1_map),
                    _remap_used(
                        mlp_edge_copy_child,
                        {0: LAYER1_SCRATCH, 1: REVERSE_CHAIN},
                    ),
                    *(
                        _remap_used(child, aligned_b1_map)
                        for child in b1[1:]
                    ),
                ]
                if debug_align_layer1_mlp_edge else [
                    *ordinary_b1_children
                ]
            ),
        ]
        b1_mlp_map = dict(b1_map)
        b1_mlp_map[39] = FINAL_DHIDDEN
        layer1_mlp_prefix = [
            *forward_prefix,
            _remap_used(b1[0], b1_mlp_map),
        ]
        layer1_mlp_internal_copy_prefix = (
            [
                *forward_prefix,
                _remap_used(b1[0], b1_map),
                _remap_used(
                    mlp_edge_copy_child,
                    {0: LAYER1_SCRATCH, 1: FINAL_DHIDDEN},
                ),
            ]
            if debug_stop_after_layer1_mlp_internal_copy else None
        )
        children = (
            forward_prefix if debug_stop_after_hidden_copy else (
                layer1_mlp_internal_copy_prefix
                if debug_stop_after_layer1_mlp_internal_copy else (
                layer1_mlp_prefix if debug_stop_after_layer1_mlp else (
                    prefix if debug_stop_after_layer1_backward else [
                        *prefix,
                        *(_remap_used(child, b0_map) for child in b0),
                    ]
                ))
            )
        )
        if debug_layer1_attention_stage is not None:
            stage_prefix = (
                forward_prefix[:1]
                if debug_extract_layer0_checkpoint else forward_prefix
            )
            if (
                debug_direct_layer1_stage_output
                and debug_layer1_attention_stage == "post_backward"
            ):
                direct_map = dict(b1_map)
                direct_map[40] = FINAL_DHIDDEN
                children = [
                    *stage_prefix,
                    *(
                        _remap_used(child, b1_map)
                        for child in b1[:5]
                    ),
                    _remap_used(b1[5], direct_map),
                ]
            else:
                stage_children = [
                    *(_remap_used(child, b1_map) for child in b1)
                ]
                if debug_tap_layer1_rstd:
                    stage_children = [
                        *stage_children[:4],
                        _remap_used(
                            rstd_tap_child,
                            {
                                0: LAYER1_SCRATCH + 3,
                                1: LAYER0_OUTPUT,
                            },
                        ),
                        *stage_children[4:],
                    ]
                if debug_tap_layer1_mlp:
                    stage_children = [
                        stage_children[0],
                        _remap_used(
                            mlp_tap_child,
                            {0: LAYER1_SCRATCH, 1: LAYER0_OUTPUT},
                        ),
                        *stage_children[1:],
                    ]
                children = [
                    *stage_prefix,
                    *((
                        _remap_used(child, b1_map) for child in b1[1:]
                    ) if debug_skip_layer1_mlp else (
                        child for child in stage_children
                    )),
                ]

        dump_path = os.environ.get("TT_FINETUNE_TWO_LAYER_SEQUENCE_DUMP")
        if dump_path:
            Path(dump_path).write_text(
                json.dumps({"programs": children}, indent=2) + "\n",
                encoding="utf-8",
            )

        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(*tensors)
        closure = _probe.__closure__ or ()
        for name, cell in zip(_probe.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                compiled.program_sequence = children
                compiled.internal_tensors = [
                    {"source_operand": 5, "memory": "clone"},
                    {"source_operand": 6, "memory": "clone"},
                    {"source_operand": 0, "memory": debug_hidden_memory},
                    {"source_operand": 1, "memory": "l1"},
                    *(
                        {
                            "source_operand": source,
                            "memory": debug_backward_scratch_memory,
                        }
                        for source in (1, 0, 0, 2, 0, 1, 1)
                    ),
                    *(
                        {
                            "source_operand": source,
                            "memory": debug_backward_scratch_memory,
                        }
                        for source in (1, 0, 0, 2, 0, 1, 1)
                    ),
                ]
            break
        return result

    operation.artifact_format = 4
    operation.layer_count = 2
    operation.physical_programs = (
        3 if debug_stop_after_hidden_copy else (
            5 if debug_stop_after_layer1_mlp_internal_copy else (
            4 if debug_stop_after_layer1_mlp else (
                (1 if debug_extract_layer0_checkpoint else 3)
                + backward.physical_programs - int(debug_skip_layer1_mlp)
                - int(
                    debug_direct_layer1_stage_output
                    and debug_layer1_attention_stage == "post_backward"
                )
                + int(debug_tap_layer1_rstd)
                + int(debug_tap_layer1_mlp)
                if debug_layer1_attention_stage is not None else (
                    3 + backward.physical_programs
                    + int(debug_align_layer1_mlp_edge)
                    + int(debug_align_layer1_rstd_edge)
                    if debug_stop_after_layer1_backward else (
                        3 + 2 * backward.physical_programs
                        + int(debug_align_layer1_mlp_edge)
                        + int(debug_align_layer1_rstd_edge)
                    )
                )
            ))
        )
    )
    operation.debug_stop_after_layer1_backward = (
        debug_stop_after_layer1_backward
    )
    operation.debug_publish_layer1_dhidden_in_arena = (
        debug_publish_layer1_dhidden_in_arena
    )
    operation.compiler_owned_internal_tensors = INTERNAL_TENSORS
    operation.compiler_owned_checkpoint_tensors = 2
    operation.compiler_owned_l1_tensors = 16
    operation.framework_visible_interlayer_tensors = 0
    operation.framework_visible_checkpoints = 0
    operation.intermediate_dram_bytes = 0
    operation.checkpoint_dram_bytes = 2 * forward0.checkpoint_dram_bytes
    operation.safe_default = False
    operation.source_bundle = source_bundle
    operation._ttl_program = _probe
    return operation
