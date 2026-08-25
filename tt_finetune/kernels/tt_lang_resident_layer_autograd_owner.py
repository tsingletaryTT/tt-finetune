"""Exact arena-backed full Transformer-layer backward owner.

One public custom call executes the accepted full-MLP backward program first
and then the accepted attention-sublayer backward sequence.  The MLP reader is
retargeted to the unified full-layer forward arena, its dX remains in a
compiler-owned L1 tensor, and that tensor becomes post-RMS dY for attention.
Only the final dHidden/attention-gradient arena and six MLP LoRA gradients are
public.
"""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Callable

from .full_attention_forward_planner import plan_full_attention_forward
from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_program_composition import (
    compile_serialized_child,
    compile_serialized_sequence_children,
    remap_serialized_child,
    retarget_full_mlp_checkpoint_arena,
    serialized_child_tensor_ids,
)
from .tt_lang_owner_aligned_copy import build_tt_lang_owner_aligned_copy
from .tt_lang_resident_attention_block_backward_owner import (
    build_tt_lang_resident_attention_block_backward_owner,
)
from .tt_lang_resident_layer_backward_pair import (
    build_tt_lang_resident_layer_backward_pair,
)


ttl = None
tt_lang_operation = None


def _remap_used(program: dict, mapping: dict[int, int]) -> dict:
    used = serialized_child_tensor_ids(program)
    return remap_serialized_child(
        program, {tensor: mapping[tensor] for tensor in used}
    )


@lru_cache(maxsize=8)
def build_tt_lang_resident_layer_autograd_owner(
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
    debug_attention_stop_after: str | None = None,
    debug_attention_dy_memory: str = "l1",
    pre_rms_copy: bool | None = None,
) -> Callable:
    """Build one exact backward ownership boundary for a complete layer."""
    if sequence_length != 128:
        raise ValueError("resident layer autograd currently requires 128 tokens")
    if intermediate_features <= 0 or intermediate_features % 32:
        raise ValueError("intermediate width must be positive and tile aligned")

    forward_plan = plan_full_attention_forward(
        tokens=sequence_length,
        hidden_features=hidden_features,
        kv_features=kv_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        physical_rank=physical_rank,
        include_attention_block=True,
    )
    sequence_tiles = sequence_length // 32
    hidden_tiles = hidden_features // 32
    intermediate_tiles = intermediate_features // 32
    gate_page_offset = forward_plan.arena.used_pages
    up_page_offset = gate_page_offset + sequence_tiles * intermediate_tiles
    gate_h_page_offset = up_page_offset + sequence_tiles * intermediate_tiles
    up_h_page_offset = gate_h_page_offset + sequence_tiles
    normalized_input_page_offset = forward_plan.arena.slice(
        "post_attention_normalized"
    ).page_offset
    hidden_pages = sequence_tiles * hidden_tiles
    # Blackhole exposes 110 worker-local L1 banks.  An interleaved allocation
    # which is only slightly larger than one page per bank (Qwen has 112
    # hidden pages) does not reserve a complete second page in every bank.
    # Pages 110/111 then wrap into the neighboring allocation.  Give private
    # resident handoffs a whole-bank-round physical backing while kernels keep
    # using the unchanged first ``hidden_pages`` linear page IDs.
    rounded_l1_pages = (
        ((hidden_pages + 109) // 110) * 110
        if hidden_pages > 110 else hidden_pages
    )
    padded_internal_shape = (
        [32, rounded_l1_pages * 32]
        if rounded_l1_pages != hidden_pages else None
    )

    def _internal(source_operand: int, memory: str = "l1") -> dict:
        spec = {"source_operand": source_operand, "memory": memory}
        if memory == "l1" and padded_internal_shape is not None:
            spec.update({"shape": padded_internal_shape, "layout": "tile"})
        return spec

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError(
                "resident layer autograd owner requires TT-XLA"
            ) from exc
        ttl, tt_lang_operation = ttl_module, decorator

    mlp_source = build_tt_lang_resident_layer_backward_pair(
        source_bundle=source_bundle,
        program_sequence=True,
        program_components=("mlp",),
    )
    mlp_dx_diagnostic = debug_attention_stop_after == "mlp_dx"
    mlp_dx_prefix_programs = {
        "mlp_dx_after_extract": 1,
        "mlp_dx_after_residual": 2,
        "mlp_dx_after_rstd": 3,
        "mlp_dx_after_prefix": 4,
    }.get(debug_attention_stop_after)
    mlp_dx_after_prefix_diagnostic = mlp_dx_prefix_programs is not None
    rstd_private_diagnostic = debug_attention_stop_after == "rstd_private"
    private_after_prefix_tensor = {
        "post_residual_private_after_prefix": 40,
        "rstd_private_after_prefix": 41,
    }.get(debug_attention_stop_after)
    post_backward_direct_diagnostic = (
        debug_attention_stop_after == "post_backward_direct_output"
    )
    pre_rms_copy = (
        os.environ.get("TT_FINETUNE_LAYER_PRE_RMS_COPY") == "1"
        if pre_rms_copy is None else bool(pre_rms_copy)
    )
    attention_stage = (
        "input_norm" if (
            mlp_dx_after_prefix_diagnostic or rstd_private_diagnostic
            or private_after_prefix_tensor is not None
        ) else "post_backward" if post_backward_direct_diagnostic
        else debug_attention_stop_after
    )
    attention = None if mlp_dx_diagnostic else build_tt_lang_resident_attention_block_backward_owner(
        sequence_length=sequence_length,
        hidden_features=hidden_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        physical_rank=physical_rank,
        lora_scale=lora_scale,
        attention_scale=attention_scale,
        rmsnorm_epsilon=rmsnorm_epsilon,
        full_layer_residual=True,
        debug_stop_after=attention_stage,
        debug_attention_dy_memory=debug_attention_dy_memory,
    )
    copy_mlp_dx = build_tt_lang_owner_aligned_copy(
        rows=sequence_length, columns=hidden_features
    )

    @ttl.operation(grid="full")
    def _probe(
        t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
        t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
        t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
        t30, t31, t32, t33, t34, t35, t36, t37,
    ):
        probe_source = (
            t04 if debug_attention_stop_after in {
                "post_rstd", "rstd_private", "rstd_private_after_prefix"
            }
            else t00
        )
        source = ttl.make_dataflow_buffer_like(
            probe_source, shape=(1, 1), block_count=1
        )
        output = ttl.make_dataflow_buffer_like(t31, shape=(1, 1), block_count=1)

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                transaction = ttl.copy(probe_source[0:1, 0:1], tile)
                transaction.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with output.reserve() as target:
                    target.store(tile)

        @ttl.datamovement()
        def write():
            with output.wait() as tile:
                transaction = ttl.copy(tile, t31[0:1, 0:1])
                transaction.wait()

    variant = (
        f"{Path(source_bundle).name}.{sequence_length}x{hidden_features}."
        f"i{intermediate_features}.q{query_heads}.k{kv_heads}."
        f"d{head_dim}.r{physical_rank}."
        f"attn{debug_attention_stop_after or 'full'}.v2"
        f".dymem{debug_attention_dy_memory}"
        f".preramscopy{int(pre_rms_copy)}"
    )

    @tt_lang_operation(
        operation_id=f"tt_finetune.resident_layer_autograd_owner.{variant}",
        arg_roles=("in",) * 31 + ("out",) * 7,
        version_tag=f"resident-layer-autograd-owner-{variant}",
    )
    def operation(
        t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
        t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
        t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
        t30, t31, t32, t33, t34, t35, t36, t37,
    ):
        tensors = (
            t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
            t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
            t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
            t30, t31, t32, t33, t34, t35, t36, t37,
        )

        # Compile the already accepted MLP child through its source-bundle
        # gate. Placeholder values need only carry the correct dtype/shape;
        # compact-program tensor IDs are remapped below.
        mlp_arguments = (
            t00, t21, t22, t30, t23, t30, t30, t30,
            t24, t25, t30, t26, t27, t28, t30, t29, t30,
            # Unselected attention component placeholders preserve the source
            # pair's fixed 42-tensor public ABI during compilation.
            t00, t07, t08, t09, t10, t11, t01,
            t12, t13, t14, t15, t16, t17, t18, t19, t20, t30,
            # Pair DPS slots: private MLP dX, six public MLP gradients,
            # unselected attention arena placeholder.
            t32, t32, t33, t34, t35, t36, t37, t32,
        )
        mlp_children = compile_serialized_sequence_children(
            mlp_source, mlp_arguments
        )
        if len(mlp_children) != 1:
            raise RuntimeError("accepted source bundle no longer has one MLP child")
        mlp_child = retarget_full_mlp_checkpoint_arena(
            mlp_children[0],
            sequence_tiles=sequence_tiles,
            hidden_tiles=hidden_tiles,
            intermediate_tiles=intermediate_tiles,
            gate_page_offset=gate_page_offset,
            up_page_offset=up_page_offset,
            gate_h_page_offset=gate_h_page_offset,
            up_h_page_offset=up_h_page_offset,
            normalized_input_page_offset=normalized_input_page_offset,
        )
        # Source-bundle local ABI: 0..16 inputs, 34..40 outputs.
        mlp_map = {
            0: 0, 1: 21, 2: 22, 3: 30, 4: 23,
            5: 30, 6: 30, 7: 30,
            8: 24, 9: 25, 10: 30, 11: 26,
            12: 27, 13: 28, 14: 30, 15: 29, 16: 30,
            34: 38, 35: 32, 36: 33, 37: 34,
            38: 35, 39: 36, 40: 37,
        }
        mlp_child = _remap_used(mlp_child, mlp_map)

        # The attention-block owner is compiled with ordinary placeholders,
        # then flattened so this owner remains one public custom call rather
        # than a nested wrapper. Its local output-dY becomes private MLP dX.
        attention_arguments = (
            t01, t00, t02, t03, t04, t05, t06,
            t07, t08, t09, t10, t11, t12, t13,
            t14, t15, t16, t17, t18, t19, t20, t30, t31,
        )
        if debug_attention_stop_after == "post_rstd":
            attention_arguments = (*attention_arguments[:-1], t04)
        elif rstd_private_diagnostic or debug_attention_stop_after == "rstd_private_after_prefix":
            attention_arguments = (*attention_arguments[:-1], t32)
        attention_children = (
            [] if mlp_dx_diagnostic else
            compile_serialized_sequence_children(attention, attention_arguments)
        )
        if not mlp_dx_diagnostic and len(attention_children) != attention.physical_programs:
            raise RuntimeError(
                "attention-block physical child count changed"
            )
        attention_map = {
            0: 1, 1: 38, 2: 2, 3: 3, 4: 4, 5: 5, 6: 0,
            7: 7, 8: 8, 9: 9, 10: 10, 11: 11,
            12: 12, 13: 13, 14: 14, 15: 15, 16: 16,
            17: 17, 18: 18, 19: 19, 20: 20,
            21: 30, 22: 31,
            23: 39, 24: 40, 25: 41, 26: 42, 27: 43, 28: 44,
        }
        if debug_attention_stop_after == "post_rstd":
            attention_map[22] = 31
        remapped_attention = []
        for child in attention_children:
            remapped_attention.append(_remap_used(child, attention_map))

        # Diagnostic/compatibility epoch: qualify the long-lived MLP dX L1
        # handoff immediately before post-RMS backward.  This deliberately
        # remains inside the same public custom call and never touches DRAM.
        # It distinguishes stale producer visibility from arithmetic state
        # carried across physical TT-Metal programs.
        if pre_rms_copy and len(remapped_attention) >= 5:
            pre_rms_copy_child = compile_serialized_child(
                copy_mlp_dx, (t00, t31)
            )
            pre_rms_copy_child = _remap_used(
                pre_rms_copy_child, {0: 38, 1: 45}
            )
            post_backward = remapped_attention[4]
            post_backward_map = {
                tensor: (45 if tensor == 38 else tensor)
                for tensor in serialized_child_tensor_ids(post_backward)
            }
            remapped_attention = [
                *remapped_attention[:4], pre_rms_copy_child,
                _remap_used(post_backward, post_backward_map),
                *remapped_attention[5:],
            ]

        if mlp_dx_diagnostic or mlp_dx_after_prefix_diagnostic:
            copy_child = compile_serialized_child(copy_mlp_dx, (t00, t31))
            copy_child = _remap_used(copy_child, {0: 38, 1: 31})
            remapped_attention = (
                [copy_child]
                if mlp_dx_diagnostic
                else [
                    *remapped_attention[:mlp_dx_prefix_programs], copy_child
                ]
            )
        elif rstd_private_diagnostic:
            copy_child = compile_serialized_child(copy_mlp_dx, (t04, t31))
            copy_child = _remap_used(copy_child, {0: 41, 1: 31})
            remapped_attention = [*remapped_attention[:3], copy_child]
        elif private_after_prefix_tensor is not None:
            copy_source = (
                t04 if private_after_prefix_tensor == 41 else t00
            )
            copy_child = compile_serialized_child(
                copy_mlp_dx, (copy_source, t31)
            )
            copy_child = _remap_used(
                copy_child, {0: private_after_prefix_tensor, 1: 31}
            )
            remapped_attention = [*remapped_attention[:4], copy_child]
        elif post_backward_direct_diagnostic:
            direct = remapped_attention[4]
            direct_map = {
                tensor: tensor for tensor in serialized_child_tensor_ids(direct)
            }
            direct_map[39] = 31
            direct = _remap_used(direct, direct_map)
            remapped_attention = [*remapped_attention[:4], direct]

        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(*tensors)
        closure = _probe.__closure__ or ()
        for name, cell in zip(_probe.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                compiled.program_sequence = [mlp_child, *remapped_attention]
                internal_tensors = [
                    _internal(0, debug_attention_dy_memory),
                    _internal(1),
                    _internal(1),
                    _internal(4),
                    _internal(1),
                    _internal(0),
                    _internal(0),
                ]
                if pre_rms_copy:
                    internal_tensors.append(_internal(0))
                forced_internal_memory = os.environ.get(
                    "TT_FINETUNE_LAYER_INTERNAL_MEMORY"
                )
                if forced_internal_memory in {"l1", "clone"}:
                    internal_tensors = [
                        {**spec, "memory": forced_internal_memory}
                        for spec in internal_tensors
                    ]
                compiled.internal_tensors = (
                    internal_tensors[:1]
                    if mlp_dx_diagnostic else internal_tensors
                )
                dump_root = os.environ.get("TT_FINETUNE_LAYER_OWNER_DUMP_DIR")
                if dump_root:
                    destination = Path(dump_root)
                    destination.mkdir(parents=True, exist_ok=True)
                    (destination / f"layer_autograd_{variant}.json").write_text(
                        json.dumps(
                            {
                                "programs": compiled.program_sequence,
                                "internal_tensors": compiled.internal_tensors,
                            },
                            indent=2,
                        ) + "\n",
                        encoding="utf-8",
                    )
            break
        return result

    operation.artifact_format = 4
    operation.physical_programs = (
        2 if mlp_dx_diagnostic else
        2 + mlp_dx_prefix_programs if mlp_dx_after_prefix_diagnostic else
        5 if rstd_private_diagnostic else
        6 if private_after_prefix_tensor is not None else
        6 if post_backward_direct_diagnostic else
        1 + attention.physical_programs
    ) + (1 if pre_rms_copy and not mlp_dx_diagnostic else 0)
    operation.compiler_owned_l1_tensors = (
        1 if mlp_dx_diagnostic else 7
    ) + (1 if pre_rms_copy and not mlp_dx_diagnostic else 0)
    operation.internal_tensor_specs = (
        [_internal(0, debug_attention_dy_memory)]
        if mlp_dx_diagnostic else [
            _internal(0, debug_attention_dy_memory),
            _internal(1), _internal(1), _internal(4),
            _internal(1), _internal(0), _internal(0),
            *([_internal(0)] if pre_rms_copy else []),
        ]
    )
    operation.intermediate_dram_bytes = 0
    operation.forward_arena_input = 30
    operation.safe_default = False
    operation.experimental_backend = "full_layer_arena_autograd_v1"
    operation.debug_attention_stop_after = debug_attention_stop_after
    operation._ttl_program = _probe
    return operation
