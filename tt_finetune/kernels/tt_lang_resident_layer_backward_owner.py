"""Experimental complete Transformer-layer backward owner.

One public TT-Lang custom call owns compact physical programs for MLP
backward, both RMSNorm/residual paths and attention backward. Hidden-gradient
handoffs are compiler-owned L1 tensors and never appear in StableHLO/PJRT.
"""

from __future__ import annotations

import copy
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_exact_residual_add import build_tt_lang_exact_residual_add
from .tt_lang_owner_aligned_copy import build_tt_lang_owner_aligned_copy
from .tt_lang_program_composition import (
    compile_serialized_child,
    compile_serialized_sequence_children,
    remap_serialized_child,
    serialized_child_tensor_ids,
)
from .tt_lang_resident_layer_backward_pair import (
    build_tt_lang_resident_layer_backward_pair,
)
from .tt_lang_rmsnorm_backward import build_tt_lang_rmsnorm_backward


ttl = None
tt_lang_operation = None


def _remap_pair_child(program: dict, mapping: dict[int, int]) -> dict:
    used = serialized_child_tensor_ids(program)
    return remap_serialized_child(
        program, {tensor: mapping[tensor] for tensor in used}
    )


def _divert_attention_hidden(
    program: dict, *, arena_tensor: int, hidden_tensor: int
) -> dict:
    """Keep final attention dHidden private while retaining adapter arena writes."""
    result = copy.deepcopy(program)
    marker = 'DeviceZoneScopedN("resident_down_final_dx_write")'
    candidates = [
        kernel for kernel in result["kernels"]
        if marker in kernel["cpp_source"]
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"attention final-dHidden writer count changed: {len(candidates)}"
        )
    kernel = candidates[0]
    if arena_tensor not in kernel["tensor_indices"]:
        raise RuntimeError("attention writer no longer references its output arena")
    common_position = len(kernel["tensor_indices"])
    kernel["tensor_indices"].append(hidden_tensor)
    source = kernel["cpp_source"]
    marker_position = source.index(marker)
    declaration_start = source.rfind(
        "    const auto arena = TensorAccessor(", 0, marker_position
    )
    if declaration_start < 0:
        raise RuntimeError("attention dHidden arena declaration was not found")
    declaration_end = source.find(";\n", declaration_start)
    if declaration_end < 0:
        raise RuntimeError("attention dHidden arena declaration is unterminated")
    declaration = source[declaration_start:declaration_end + 2]
    accessor = re.search(
        rf"get_tensor_accessor_args_cta_offset<\s*{arena_tensor}\s*,\s*(\d+)\s*>\(\)",
        declaration,
    )
    if accessor is None:
        raise RuntimeError("attention arena accessor descriptor changed")
    total_cbs = int(accessor.group(1))
    hidden_declaration = f'''    const uint32_t resident_hidden_addr =
        get_common_arg_val<uint32_t>({common_position});
    const auto resident_hidden = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                {hidden_tensor}, {total_cbs}>(), {common_position}>(),
        resident_hidden_addr, q_kTileBytes);
'''
    source = (
        source[:declaration_end + 2]
        + hidden_declaration
        + source[declaration_end + 2:]
    )
    marker_position = source.index(marker)
    before, after = source[:marker_position], source[marker_position:]
    if ", arena," not in after:
        raise RuntimeError("attention final dHidden write target changed")
    after = after.replace(", arena,", ", resident_hidden,", 1)
    kernel["cpp_source"] = before + after
    return result


@lru_cache(maxsize=8)
def build_tt_lang_resident_layer_backward_owner(
    *, source_bundle: str, rows: int, columns: int
) -> Callable:
    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("resident layer backward owner requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    pair = build_tt_lang_resident_layer_backward_pair(
        source_bundle=source_bundle, program_sequence=True
    )
    rms = build_tt_lang_rmsnorm_backward(
        sequence_length=rows, hidden_size=columns, compact_l1=True
    )
    copy_op = build_tt_lang_owner_aligned_copy(rows=rows, columns=columns)
    add = build_tt_lang_exact_residual_add(
        rows=rows, columns=columns, owner_aligned=True
    )

    @ttl.operation(grid="full")
    def _probe(
        t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
        t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
        t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
        t30, t31, t32, t33, t34, t35, t36, t37, t38, t39,
        t40, t41, t42, t43, t44, t45, t46, t47, t48,
    ):
        # Keep this dummy program deliberately scalar.  TTL compiles the probe
        # before the format-v4 serializer replaces it with the physical child
        # sequence, and its capture pass cannot lower a tuple of tensor values.
        source = ttl.make_dataflow_buffer_like(t00, shape=(1, 1), block_count=1)
        output = ttl.make_dataflow_buffer_like(t41, shape=(1, 1), block_count=1)

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
                tx = ttl.copy(tile, t41[0:1, 0:1])
                tx.wait()

    @tt_lang_operation(
        operation_id=(
            f"tt_finetune.resident_layer_backward_owner."
            f"{Path(source_bundle).name}.{rows}x{columns}.v1"
        ),
        arg_roles=("in",) * 41 + ("out",) * 8,
        version_tag=(
            f"resident-layer-backward-owner-{Path(source_bundle).name}-"
            f"{rows}x{columns}-v1"
        ),
    )
    def operation(
        t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
        t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
        t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
        t30, t31, t32, t33, t34, t35, t36, t37, t38, t39,
        t40, t41, t42, t43, t44, t45, t46, t47, t48,
    ):
        tensors = (
            t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
            t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
            t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
            t30, t31, t32, t33, t34, t35, t36, t37, t38, t39,
            t40, t41, t42, t43, t44, t45, t46, t47, t48,
        )
        pair_arguments = tuple(tensors[:34]) + tuple(tensors[41:49])
        pair_children = compile_serialized_sequence_children(pair, pair_arguments)
        if len(pair_children) != 2:
            raise RuntimeError("accepted layer pair no longer has MLP+attention children")
        mlp_child, attention_child = pair_children

        # Compile the exact numerical boundary children with distinct ABI
        # placeholders; their tensor IDs are remapped below.
        post_rms = compile_serialized_child(
            rms, (tensors[34], tensors[41], tensors[35], tensors[36], tensors[37], tensors[42])
        )
        input_rms = compile_serialized_child(
            rms, (tensors[38], tensors[41], tensors[39], tensors[40], tensors[37], tensors[42])
        )
        copy_child = compile_serialized_child(copy_op, (tensors[41], tensors[42]))
        add_child = compile_serialized_child(add, (tensors[41], tensors[0], tensors[42]))

        # Pair local IDs 0..41 -> whole-owner external/private table.
        pair_map = {index: index for index in range(34)}
        pair_map.update({34: 49})
        pair_map.update({index: index + 7 for index in range(35, 41)})
        pair_map[41] = 48
        mlp_child = _remap_pair_child(mlp_child, pair_map)

        # Attention consumes private attention dY and writes only adapter
        # gradients to the public arena; final dHidden is a private tensor.
        pair_map[17] = 52
        attention_child = _remap_pair_child(attention_child, pair_map)
        attention_child = _divert_attention_hidden(
            attention_child, arena_tensor=48, hidden_tensor=53
        )

        children = [
            mlp_child,
            remap_serialized_child(
                post_rms, {0: 34, 1: 49, 2: 35, 3: 36, 4: 37, 5: 50}
            ),
            remap_serialized_child(copy_child, {0: 50, 1: 51}),
            remap_serialized_child(add_child, {0: 51, 1: 0, 2: 52}),
            attention_child,
            remap_serialized_child(copy_child, {0: 53, 1: 54}),
            remap_serialized_child(
                input_rms, {0: 38, 1: 54, 2: 39, 3: 40, 4: 37, 5: 55}
            ),
            remap_serialized_child(copy_child, {0: 55, 1: 56}),
            remap_serialized_child(add_child, {0: 56, 1: 0, 2: 41}),
        ]

        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(*tensors)
        closure = _probe.__closure__ or ()
        for name, cell in zip(_probe.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                compiled.program_sequence = children
                compiled.internal_tensors = [
                    {"source_operand": 0, "memory": "l1"}
                    for _ in range(8)
                ]
            break
        return result

    operation.artifact_format = 4
    operation.physical_programs = 9
    operation.compiler_owned_l1_tensors = 8
    operation.safe_default = False
    operation.source_bundle = source_bundle
    operation._ttl_program = _probe
    return operation
