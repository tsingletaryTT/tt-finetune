"""Exact compact-program attention-sublayer backward owner.

The owner consumes the accepted forward checkpoint arena directly and keeps
all normalization/residual gradient handoffs in four compiler-owned L1
tensors.  Cheap residual and RMS statistics are recomputed on device instead
of adding more backward-lifetime checkpoint traffic.

One StableHLO custom call executes these physical epochs::

    extract O -> recompute post residual -> recompute post rstd
    -> post RMS backward -> recompute normalized input
    -> full resident attention backward -> recompute input rstd
    -> input RMS backward -> final residual-gradient add

Only the packed final dHidden + LoRA-gradient arena is public.
"""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_exact_residual_add import build_tt_lang_exact_residual_add
from .tt_lang_output_attention_resident_backward import (
    build_tt_lang_output_attention_resident_backward,
)
from .tt_lang_owner_aligned_copy import build_tt_lang_owner_aligned_copy
from .tt_lang_program_composition import (
    compile_serialized_child,
    remap_serialized_child,
    serialized_child_tensor_ids,
)
from .resident_rms_attention_fusion import (
    fuse_post_rms_backward_into_attention_program,
)
from .tt_lang_resident_layer_backward_owner import _divert_attention_hidden
from .tt_lang_rmsnorm_backward import build_tt_lang_rmsnorm_backward
from .tt_lang_rmsnorm_forward import build_tt_lang_rmsnorm_forward


ttl = None
tt_lang_operation = None


def _remap_all(program: dict, mapping: dict[int, int]) -> dict:
    used = serialized_child_tensor_ids(program)
    return remap_serialized_child(
        program, {tensor: mapping[tensor] for tensor in used}
    )


@lru_cache(maxsize=8)
def build_tt_lang_resident_attention_block_backward_owner(
    *,
    sequence_length: int,
    hidden_features: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    physical_rank: int,
    lora_scale: float,
    attention_scale: float,
    rmsnorm_epsilon: float,
    full_layer_residual: bool = False,
    fused_post_rms_attention: bool = False,
    debug_stop_after: str | None = None,
    debug_attention_dy_memory: str = "l1",
) -> Callable:
    """Build the exact attention sublayer backward ownership boundary."""

    if debug_stop_after not in (
        None, "post_extract", "post_residual", "post_rstd", "post_backward", "input_norm",
        "zero_input", "direct_attention", "attention_dram",
        "attention_input", "attention_dy_internal",
        "attention_dy_post_direct", "attention_dy_copy",
        "attention_hidden_internal", "attention_fused_rms", "attention",
        "input_backward",
    ):
        raise ValueError("invalid attention-block backward diagnostic stage")
    if debug_attention_dy_memory not in {"l1", "clone"}:
        raise ValueError("attention dY memory must be l1 or clone")

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError(
                "resident attention-block backward owner requires TT-XLA"
            ) from exc
        ttl, tt_lang_operation = ttl_module, decorator

    attention = build_tt_lang_output_attention_resident_backward(
        sequence_length=sequence_length,
        hidden_features=hidden_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        physical_rank=physical_rank,
        lora_scale=lora_scale,
        attention_scale=attention_scale,
        resident_qkv_projection=True,
        forward_checkpoint_arena=True,
    )
    rms_final = build_tt_lang_rmsnorm_forward(
        sequence_length=sequence_length,
        hidden_size=hidden_features,
        epsilon=rmsnorm_epsilon,
    )
    rms_rstd = build_tt_lang_rmsnorm_forward(
        sequence_length=sequence_length,
        hidden_size=hidden_features,
        epsilon=rmsnorm_epsilon,
        stage="rstd_expanded_fp32",
    )
    rms_backward = build_tt_lang_rmsnorm_backward(
        sequence_length=sequence_length,
        hidden_size=hidden_features,
        compact_l1=True,
    )
    copy_op = build_tt_lang_owner_aligned_copy(
        rows=sequence_length, columns=hidden_features,
    )
    add = build_tt_lang_exact_residual_add(
        rows=sequence_length,
        columns=hidden_features,
        owner_aligned=True,
    )

    @ttl.operation(grid="full")
    def _probe(
        t00, t01, t02, t03, t04, t05, t06,
        t07, t08, t09, t10, t11,
        t12, t13, t14, t15, t16, t17,
        t18, t19, t20, t21, t22,
    ):
        probe_source = t04 if debug_stop_after == "post_rstd" else t00
        source = ttl.make_dataflow_buffer_like(
            probe_source, shape=(1, 1), block_count=1
        )
        output = ttl.make_dataflow_buffer_like(
            t22, shape=(1, 1), block_count=1
        )

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
                tx = ttl.copy(tile, t22[0:1, 0:1])
                tx.wait()

    diagnostic_waypoint_tag = (
        ".wp" if os.environ.get("TT_FINETUNE_FUSED_RMS_WAYPOINTS") == "1"
        else ""
    )
    variant = (
        f"{sequence_length}x{hidden_features}.q{query_heads}.k{kv_heads}."
        f"d{head_dim}.r{physical_rank}."
        f"stop{debug_stop_after or 'full'}."
        f"layerres{int(full_layer_residual)}."
        f"fusedrmsattn{int(fused_post_rms_attention)}."
        f"v13{diagnostic_waypoint_tag}"
    )

    @tt_lang_operation(
        operation_id=(
            f"tt_finetune.resident_attention_block_backward_owner.{variant}"
        ),
        arg_roles=("in",) * 22 + ("out",),
        version_tag=f"resident-attention-block-backward-owner-{variant}",
    )
    def operation(
        t00, t01, t02, t03, t04, t05, t06,
        t07, t08, t09, t10, t11,
        t12, t13, t14, t15, t16, t17,
        t18, t19, t20, t21, t22,
    ):
        tensors = (
            t00, t01, t02, t03, t04, t05, t06,
            t07, t08, t09, t10, t11,
            t12, t13, t14, t15, t16, t17,
            t18, t19, t20, t21, t22,
        )

        # Public ABI:
        #   0 input hidden, 1 output dY, 2/3 input+post RMS gamma expanded
        #   over token rows (forward consumes its first tile row),
        #   4/5 independent FP32 scaffolds, 6 BF16 scaffold,
        #   7..20 attention parameters/rotary tensors,
        #   21 forward checkpoint arena, 22 packed backward arena.
        # Private IDs 23..27 form the BF16 handoffs plus one reusable FP32
        # rstd buffer. ID27 is the canonical aligned view required when an
        # RMSNorm writer feeds the attention TensorAccessor directly.
        # ID28 is a sacrificial runtime result slot: current generic_op
        # re-registers the final IO tensor after every physical program.
        post_rstd_diagnostic = debug_stop_after == "post_rstd"
        post_backward_output_placeholder = (
            t02 if post_rstd_diagnostic else t22
        )
        input_backward_output_placeholder = (
            t03 if post_rstd_diagnostic else t22
        )
        extract_child = compile_serialized_child(copy_op, (t21, t06))
        add_child = compile_serialized_child(add, (t06, t00, t01))
        post_rstd_child = compile_serialized_child(
            rms_rstd, (t00, t03, t04, t04, t05)
        )
        post_backward_child = compile_serialized_child(
            rms_backward,
            (t00, t01, t03, t05, t06, post_backward_output_placeholder),
        )
        input_norm_child = compile_serialized_child(
            rms_final, (t00, t02, t06, t04, t01)
        )
        attention_child = None if post_rstd_diagnostic else compile_serialized_child(
            attention,
            (
                t01, t07, t08, t09, t10, t11, t00,
                t12, t13, t14, t15, t16, t17,
                t18, t19, t20, t21, t22,
            ),
        )
        input_rstd_child = compile_serialized_child(
            rms_rstd, (t00, t02, t04, t04, t05)
        )
        input_backward_child = compile_serialized_child(
            rms_backward,
            (t00, t01, t02, t05, t06, input_backward_output_placeholder),
        )

        # Lifetimes:
        #   23: extracted O -> post-RMS dResidual
        #   24: post residual -> attention dNormalizedInput
        #   25: post rstd -> input rstd (FP32)
        #   26: normalized input -> input-RMS dX
        #   27: canonical aligned post-RMS dResidual for attention
        prefix_children = [
            _remap_all(extract_child, {0: 21, 1: 23}),
            _remap_all(add_child, {0: 23, 1: 0, 2: 24}),
            _remap_all(
                post_rstd_child, {0: 24, 1: 3, 2: 4, 3: 4, 4: 25}
            ),
            _remap_all(
                input_norm_child, {0: 0, 1: 2, 2: 6, 3: 4, 4: 26}
            ),
            _remap_all(
                post_backward_child,
                {0: 24, 1: 1, 2: 3, 3: 25, 4: 6, 5: 23},
            ),
        ]
        full_fused_owner = (
            fused_post_rms_attention and debug_stop_after is None
        )
        attention_handoff_tensor = 26 if full_fused_owner else 27
        attention_hidden_tensor = 23 if full_fused_owner else 26
        attention_map = {
            0: attention_handoff_tensor,
            1: 7, 2: 8, 3: 9, 4: 10, 5: 11,
            6: attention_hidden_tensor,
            7: 12, 8: 13, 9: 14,
            10: 15, 11: 16, 12: 17,
            13: 18, 14: 19, 15: 20,
            16: 21, 17: 22,
        }
        raw_attention = attention_child
        attention_child = (
            None if raw_attention is None
            else _remap_all(raw_attention, attention_map)
        )
        fused_post_rms_attention_program = (
            fuse_post_rms_backward_into_attention_program(
                prefix_children[4],
                (
                    _divert_attention_hidden(
                        attention_child,
                        arena_tensor=22,
                        hidden_tensor=24,
                    )
                    if fused_post_rms_attention and debug_stop_after is None
                    else attention_child
                ),
                hidden_tiles=hidden_features // 32,
                kv_tiles=(kv_heads * head_dim) // 32,
                handoff_tensor_index=attention_handoff_tensor,
            )
            if (
                debug_stop_after == "attention_fused_rms"
                or (fused_post_rms_attention and debug_stop_after is None)
            )
            else None
        )
        if debug_stop_after in {"direct_attention", "attention_dram"}:
            direct_map = dict(attention_map)
            if debug_stop_after == "direct_attention":
                direct_map[0] = 1
                direct_map[6] = 0
                children = [_remap_all(raw_attention, direct_map)]
                internal_tensor_count = 0
            else:
                direct_map[0] = 21
                direct_map[6] = 26
                children = prefix_children + [
                    _remap_all(extract_child, {0: 23, 1: 24}),
                    _remap_all(add_child, {0: 24, 1: 6, 2: 21}),
                    _remap_all(raw_attention, direct_map),
                ]
                internal_tensor_count = 4
        elif debug_stop_after == "zero_input":
            children = [_remap_all(extract_child, {0: 6, 1: 22})]
            internal_tensor_count = 0
        elif debug_stop_after == "post_extract":
            children = [_remap_all(extract_child, {0: 21, 1: 22})]
            internal_tensor_count = 0
        elif debug_stop_after == "post_residual":
            children = prefix_children[:2] + [
                _remap_all(extract_child, {0: 24, 1: 22})
            ]
            internal_tensor_count = 2
        elif debug_stop_after == "post_rstd":
            children = prefix_children[:2] + [
                _remap_all(
                    post_rstd_child,
                    {0: 24, 1: 3, 2: 4, 3: 4, 4: 22},
                )
            ]
            internal_tensor_count = 2
        elif debug_stop_after == "post_backward":
            children = prefix_children + [
                _remap_all(extract_child, {0: 23, 1: 22})
            ]
            internal_tensor_count = 4
        elif debug_stop_after == "input_norm":
            children = prefix_children[:4] + [
                _remap_all(extract_child, {0: 26, 1: 22})
            ]
            internal_tensor_count = 4
        elif debug_stop_after == "attention":
            children = prefix_children + [
                _remap_all(extract_child, {0: 23, 1: 24}),
                _remap_all(add_child, {0: 24, 1: 6, 2: 27}),
                attention_child,
            ]
            internal_tensor_count = 5
        elif debug_stop_after == "attention_input":
            children = prefix_children + [
                _remap_all(extract_child, {0: 23, 1: 24}),
                _remap_all(add_child, {0: 24, 1: 6, 2: 27}),
                _remap_all(extract_child, {0: 27, 1: 22}),
            ]
            internal_tensor_count = 5
        elif debug_stop_after == "attention_dy_internal":
            dy_internal_map = dict(attention_map)
            dy_internal_map[6] = 0
            children = prefix_children + [
                _remap_all(extract_child, {0: 23, 1: 24}),
                _remap_all(add_child, {0: 24, 1: 6, 2: 27}),
                _remap_all(raw_attention, dy_internal_map),
            ]
            internal_tensor_count = 5
        elif debug_stop_after == "attention_dy_post_direct":
            dy_direct_map = dict(attention_map)
            dy_direct_map[0] = 23
            dy_direct_map[6] = 0
            children = prefix_children + [
                _remap_all(raw_attention, dy_direct_map),
            ]
            internal_tensor_count = 4
        elif debug_stop_after == "attention_dy_copy":
            dy_copy_map = dict(attention_map)
            dy_copy_map[0] = 24
            dy_copy_map[6] = 0
            children = prefix_children + [
                _remap_all(extract_child, {0: 23, 1: 24}),
                _remap_all(raw_attention, dy_copy_map),
            ]
            internal_tensor_count = 4
        elif debug_stop_after == "attention_hidden_internal":
            hidden_internal_map = dict(attention_map)
            hidden_internal_map[0] = 1
            children = prefix_children + [
                _remap_all(raw_attention, hidden_internal_map),
            ]
            internal_tensor_count = 5
        elif debug_stop_after == "attention_fused_rms":
            children = prefix_children[:4] + [fused_post_rms_attention_program]
            internal_tensor_count = 4
        elif debug_stop_after == "input_backward":
            children = prefix_children + [
                _remap_all(extract_child, {0: 23, 1: 24}),
                _remap_all(add_child, {0: 24, 1: 6, 2: 27}),
                _divert_attention_hidden(
                    attention_child,
                    arena_tensor=22,
                    hidden_tensor=24,
                ),
                _remap_all(
                    input_rstd_child,
                    {0: 0, 1: 2, 2: 4, 3: 4, 4: 25},
                ),
                _remap_all(
                    input_backward_child,
                    {0: 0, 1: 24, 2: 2, 3: 25, 4: 6, 5: 26},
                ),
                _remap_all(extract_child, {0: 26, 1: 22}),
            ]
            internal_tensor_count = 4
        else:
            attention_epoch = (
                [fused_post_rms_attention_program]
                if fused_post_rms_attention
                else [
                    prefix_children[4],
                    _remap_all(extract_child, {0: 23, 1: 24}),
                    _remap_all(add_child, {0: 24, 1: 6, 2: 27}),
                    _divert_attention_hidden(
                        attention_child,
                        arena_tensor=22,
                        hidden_tensor=24,
                    ),
                ]
            )
            prefix_epoch = (
                prefix_children[:3]
                + [
                    _remap_all(
                        input_norm_child,
                        {0: 0, 1: 2, 2: 6, 3: 4, 4: 23},
                    )
                ]
                if fused_post_rms_attention
                else prefix_children[:4]
            )
            children = prefix_epoch + attention_epoch + [
                _remap_all(
                    input_rstd_child,
                    {0: 0, 1: 2, 2: 4, 3: 4, 4: 25},
                ),
                _remap_all(
                    input_backward_child,
                    {
                        0: 0, 1: 24, 2: 2, 3: 25, 4: 6,
                        5: 23 if fused_post_rms_attention else 26,
                    },
                ),
                _remap_all(
                    add_child,
                    {
                        0: (
                            6
                            if full_layer_residual
                            else (
                                attention_handoff_tensor
                                if fused_post_rms_attention else 23
                            )
                        ),
                        1: 23 if fused_post_rms_attention else 26,
                        2: 22,
                    },
                ),
            ]
            internal_tensor_count = 3 if fused_post_rms_attention else 4

        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(*tensors)
        closure = _probe.__closure__ or ()
        for name, cell in zip(_probe.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                compiled.program_sequence = children
                internal_specs = [
                    {
                        "source_operand": 0,
                        "memory": debug_attention_dy_memory,
                    },
                    {
                        "source_operand": 0,
                        "memory": debug_attention_dy_memory,
                    },
                    {
                        "source_operand": 4,
                        "memory": debug_attention_dy_memory,
                    },
                    {
                        "source_operand": 0,
                        "memory": debug_attention_dy_memory,
                    },
                    {
                        "source_operand": 1,
                        "memory": debug_attention_dy_memory,
                    },
                    {"source_operand": 0, "memory": "l1"},
                ]
                # Keep the final private slot for generic_op result
                # registration.  In the fused RMS->attention path ID27 also
                # becomes the real distributed L1 dY handoff: the RMS writer
                # scatters one tile at a time into it and attention consumes it
                # inside the same physical TT-Metal program.
                needs_sacrificial_result = internal_tensor_count > 0
                compiled.internal_tensors = (
                    internal_specs[:internal_tensor_count]
                    + ([internal_specs[-1]] if needs_sacrificial_result else [])
                )
                dump_root = os.environ.get("TT_FINETUNE_OWNER_DUMP_DIR")
                if dump_root:
                    destination = Path(dump_root)
                    destination.mkdir(parents=True, exist_ok=True)
                    (destination / f"attention_block_backward_{variant}.json").write_text(
                        json.dumps({
                            "programs": children,
                            "internal_tensors": compiled.internal_tensors,
                        }, indent=2) + "\n",
                        encoding="utf-8",
                    )
            break
        return result

    operation.artifact_format = 4
    operation.physical_programs = {
        "post_extract": 1,
        "post_residual": 3,
        "post_rstd": 3,
        "zero_input": 1,
        "direct_attention": 1,
        "attention_dram": 8,
        "post_backward": 6,
        "input_norm": 5,
        "attention_input": 8,
        "attention_dy_internal": 8,
        "attention_dy_post_direct": 6,
        "attention_dy_copy": 7,
        "attention_hidden_internal": 6,
        "attention_fused_rms": 5,
        "attention": 8,
        "input_backward": 11,
        None: 8 if fused_post_rms_attention else 11,
    }[debug_stop_after]
    operation.compiler_owned_l1_tensors = {
        "post_extract": 0,
        "post_residual": 2,
        "post_rstd": 2,
        "zero_input": 0,
        "direct_attention": 0,
        "attention_dram": 4,
        "post_backward": 4,
        "input_norm": 4,
        "attention_input": 5,
        "attention_dy_internal": 5,
        "attention_dy_post_direct": 4,
        "attention_dy_copy": 4,
        "attention_hidden_internal": 5,
        "attention_fused_rms": 4,
        "attention": 5,
        "input_backward": 5,
        None: 5,
    }[debug_stop_after] + (
        1 if debug_stop_after not in {
            "zero_input", "direct_attention", "post_extract",
        } else 0
    )
    operation.output_shape = attention.arena_shape
    operation.intermediate_dram_bytes = 0
    operation.checkpoint_recompute = (
        "post_residual+post_rstd+input_normalized+input_rstd"
    )
    operation.safe_default = False
    operation.experimental_backend = (
        "attention_block_backward_fused_rms_attention_v1"
        if fused_post_rms_attention
        else "attention_block_backward_compact_v1"
    )
    operation.fused_post_rms_attention = bool(fused_post_rms_attention)
    operation.full_layer_residual = bool(full_layer_residual)
    operation.debug_stop_after = debug_stop_after
    operation._ttl_program = _probe
    return operation
