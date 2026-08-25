"""One PJRT owner for two exact resident Transformer-layer backwards.

Backward executes layer 1 before layer 0.  Layer 1's final hidden gradient is
an artifact-v4 compiler-owned L1 tensor and is consumed directly as layer 0's
incoming gradient.  The two already-qualified nine-program layer owners are
flattened into one ordered sequence; they are never nested and their compact
TT-Metal sources are not physically concatenated.

External ABI
------------
``t00`` is the top gradient. ``t01..t40`` and ``t41..t80`` are the remaining
inputs for layer 0 and layer 1 respectively. ``t81..t88`` are layer 0's eight
ordinary outputs and ``t89..t95`` are layer 1's seven adapter-gradient outputs
(its final dHidden is private).  Consequently the only inter-layer edge is an
L1 tensor invisible to StableHLO/PJRT.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_program_composition import (
    compile_serialized_sequence_children,
    remap_serialized_child,
    serialized_child_tensor_ids,
)
from .tt_lang_resident_layer_backward_owner import (
    build_tt_lang_resident_layer_backward_owner,
)


ttl = None
tt_lang_operation = None


EXTERNAL_INPUTS = 81
EXTERNAL_OUTPUTS = 15
EXTERNAL_TENSORS = EXTERNAL_INPUTS + EXTERNAL_OUTPUTS
CHAIN_TENSOR = EXTERNAL_TENSORS
LAYER1_SCRATCH_BASE = CHAIN_TENSOR + 1
LAYER0_SCRATCH_BASE = LAYER1_SCRATCH_BASE + 8
INTERNAL_TENSORS = 17


def two_layer_backward_tensor_maps() -> tuple[dict[int, int], dict[int, int]]:
    """Return local-layer to outer tensor maps in execution order.

    A complete layer owner has 49 external tensors (41 inputs, 8 outputs) and
    eight local artifact-v4 scratch tensors numbered 49..56.
    """

    later: dict[int, int] = {0: 0}
    later.update({local: 40 + local for local in range(1, 41)})
    later[41] = CHAIN_TENSOR
    later.update({local: 47 + local for local in range(42, 49)})
    later.update(
        {local: LAYER1_SCRATCH_BASE + local - 49 for local in range(49, 57)}
    )

    earlier: dict[int, int] = {0: CHAIN_TENSOR}
    earlier.update({local: local for local in range(1, 41)})
    earlier.update({local: 40 + local for local in range(41, 49)})
    earlier.update(
        {local: LAYER0_SCRATCH_BASE + local - 49 for local in range(49, 57)}
    )
    return later, earlier


def _remap_used(program: dict, mapping: Mapping[int, int]) -> dict:
    used = serialized_child_tensor_ids(program)
    return remap_serialized_child(
        program, {tensor: int(mapping[tensor]) for tensor in used}
    )


@lru_cache(maxsize=8)
def build_tt_lang_two_layer_backward_schedule(
    *, source_bundle: str, rows: int, columns: int
) -> Callable:
    """Build the first exact two-layer resident backward device schedule."""

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "two-layer backward schedule requires TT-XLA"
            ) from exc
        ttl, tt_lang_operation = ttl_module, decorator

    layer = build_tt_lang_resident_layer_backward_owner(
        source_bundle=source_bundle, rows=rows, columns=columns
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
        t90, t91, t92, t93, t94, t95,
    ):
        source = ttl.make_dataflow_buffer_like(
            t00, shape=(1, 1), block_count=1
        )
        output = ttl.make_dataflow_buffer_like(
            t81, shape=(1, 1), block_count=1
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
                tx = ttl.copy(tile, t81[0:1, 0:1])
                tx.wait()

    bundle_name = Path(source_bundle).name

    @tt_lang_operation(
        operation_id=(
            f"tt_finetune.two_layer_backward_schedule."
            f"{bundle_name}.{rows}x{columns}.v1"
        ),
        arg_roles=("in",) * EXTERNAL_INPUTS + ("out",) * EXTERNAL_OUTPUTS,
        version_tag=(
            f"two-layer-backward-schedule-{bundle_name}-"
            f"{rows}x{columns}-v1"
        ),
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
        t90, t91, t92, t93, t94, t95,
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
            t90, t91, t92, t93, t94, t95,
        )

        # Compile each accepted owner with shape-compatible external
        # placeholders, then remap its complete 0..56 tensor table. The later
        # layer's local output 41 is deliberately mapped to CHAIN_TENSOR.
        later_arguments = (
            tensors[0], *tensors[41:81], tensors[81], *tensors[89:96]
        )
        # The compile-only placeholder for local input 0 must be a different
        # Python tensor object from local output 41.  Reusing t81 for both
        # lets TT-Lang canonicalize the two tensor IDs and corrupts the
        # accepted residual-add child ABI.  Runtime remapping still connects
        # the real input to CHAIN_TENSOR below.
        earlier_arguments = (
            tensors[0], *tensors[1:41], *tensors[81:89]
        )
        later_children = compile_serialized_sequence_children(
            layer, later_arguments
        )
        earlier_children = compile_serialized_sequence_children(
            layer, earlier_arguments
        )
        if len(later_children) != 9 or len(earlier_children) != 9:
            raise RuntimeError(
                "accepted layer owner no longer contains nine physical programs"
            )
        later_map, earlier_map = two_layer_backward_tensor_maps()
        children = [
            *(_remap_used(child, later_map) for child in later_children),
            *(_remap_used(child, earlier_map) for child in earlier_children),
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
                    for _ in range(INTERNAL_TENSORS)
                ]
            break
        return result

    operation.artifact_format = 4
    operation.layer_count = 2
    operation.physical_programs = 18
    operation.compiler_owned_l1_tensors = INTERNAL_TENSORS
    operation.framework_visible_interlayer_tensors = 0
    operation.intermediate_dram_bytes = 0
    operation.safe_default = False
    operation.source_bundle = source_bundle
    operation._ttl_program = _probe
    return operation
