"""One public owner for exact clipping over a packed gradient arena."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

from .optimizer_gradient_arena_planner import (
    plan_multi_layer_optimizer_gradient_arena,
)
from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_clip_coefficient import build_tt_lang_clip_coefficient
from .tt_lang_gradient_l2_norm import build_tt_lang_gradient_l2_norm
from .tt_lang_gradient_scale import build_tt_lang_gradient_scale
from .tt_lang_grouped_gradient_l2_norm import (
    build_tt_lang_grouped_gradient_l2_norm,
)
from .tt_lang_program_composition import (
    compile_serialized_child,
    remap_serialized_child,
    serialized_child_tensor_ids,
)


ttl = None
tt_lang_operation = None


def _remap_used(program: dict, mapping: dict[int, int]) -> dict:
    used = serialized_child_tensor_ids(program)
    return remap_serialized_child(
        program, {tensor: mapping[tensor] for tensor in used}
    )


@lru_cache(maxsize=8)
def build_tt_lang_packed_gradient_clip_owner(
    *, layer_count: int, sequence_length: int, hidden_features: int,
    intermediate_features: int, kv_features: int, physical_rank: int = 32,
    diagnostic_stage: str = "full",
    grouped_norms: bool = False,
) -> Callable:
    if diagnostic_stage not in {"full", "norms", "coefficient"}:
        raise ValueError(
            "diagnostic_stage must be full, norms, or coefficient"
        )
    plan = plan_multi_layer_optimizer_gradient_arena(
        layer_count=layer_count,
        tokens=sequence_length,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        kv_features=kv_features,
        physical_rank=physical_rank,
    )
    norm_pages = len(plan.slices)
    grouped_norm = (
        build_tt_lang_grouped_gradient_l2_norm(
            layer_count=layer_count,
            sequence_length=sequence_length,
            hidden_features=hidden_features,
            intermediate_features=intermediate_features,
            kv_features=kv_features,
            physical_rank=physical_rank,
        )
        if grouped_norms else None
    )
    norm_operations = [
        build_tt_lang_gradient_l2_norm(
            page_offset=region.page_offset,
            height_tiles=(region.logical_shape[0] + 31) // 32,
            width_tiles=(region.logical_shape[1] + 31) // 32,
            output_page_offset=index,
            norm_arena_pages=norm_pages,
        )
        for index, region in enumerate(plan.slices)
    ] if not grouped_norms else []
    total_operation = build_tt_lang_gradient_l2_norm(
        page_offset=0,
        height_tiles=norm_pages,
        width_tiles=1,
        output_page_offset=0,
        # The global reduction has one scalar output and no state to carry.
        # Keeping a multi-page arena here both wastes traffic and makes its
        # input template alias the packed norm input during child compilation.
        norm_arena_pages=1,
    )
    coefficient_operation = build_tt_lang_clip_coefficient()
    scale_operation = build_tt_lang_gradient_scale(
        gradient_page_offset=plan.scratch_pages,
        gradient_pages=layer_count * plan.layer_gradient_pages,
    )

    global ttl, tt_lang_operation
    if ttl is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("packed clipping owner requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid=(1, 1))
    def _probe(
        gradient, scaler, norm_template, norm_output_template,
        constants, coefficient_template, clipped,
    ):
        source = ttl.make_dataflow_buffer_like(
            gradient, shape=(1, 1), block_count=1
        )
        output = ttl.make_dataflow_buffer_like(
            clipped, shape=(1, 1), block_count=1
        )

        @ttl.datamovement()
        def read():
            with source.reserve() as value:
                tx = ttl.copy(gradient[0:1, 0:1], value)
                tx.wait()

        @ttl.compute()
        def compute():
            with source.wait() as value:
                with output.reserve() as result:
                    result.store(value)

        @ttl.datamovement()
        def write():
            with output.wait() as value:
                tx = ttl.copy(value, clipped[0:1, 0:1])
                tx.wait()

    variant = (
        f"l{layer_count}.{sequence_length}x{hidden_features}."
        f"i{intermediate_features}.kv{kv_features}.{diagnostic_stage}."
        f"grouped{int(grouped_norms)}.periodicnorm1.v8"
    )

    @tt_lang_operation(
        operation_id=f"tt_finetune.packed_gradient_clip_owner.{variant}",
        arg_roles=("in",) * 6 + ("out",),
        version_tag=f"packed-gradient-clip-owner-{variant}",
    )
    def operation(
        gradient, scaler, norm_template, norm_output_template,
        constants, coefficient_template, clipped,
    ):
        children = []
        if grouped_norms:
            child_target = 6 if diagnostic_stage == "norms" else 7
            child = compile_serialized_child(
                grouped_norm, (gradient, scaler, norm_output_template)
            )
            children.append(_remap_used(child, {
                0: 0, 1: 1, 2: child_target,
            }))
            current = child_target
        else:
            current, target = 2, 7
            for index, norm_operation in enumerate(norm_operations):
                child_target = (
                    6
                    if diagnostic_stage == "norms" and index + 1 == norm_pages
                    else target
                )
                child = compile_serialized_child(
                    # These must be distinct symbolic tensors while compiling
                    # the child. TT-Lang aliases identical arguments; passing
                    # ``norm_template`` twice would read the fresh output.
                    norm_operation,
                    (gradient, scaler, norm_template, norm_output_template),
                )
                children.append(_remap_used(child, {
                    0: 0, 1: 1, 2: current, 3: child_target,
                }))
                current = child_target
                target = 8 if target == 7 else 7

        if diagnostic_stage == "norms":
            total = coefficient = scale = None
        else:
            # Internal IDs: 6/7 ping-pong norm arenas, 8 total norm arena,
            # 9 full coefficient tile.
            total = compile_serialized_child(
                total_operation,
                # ``constants`` is a distinct, unused carry template because
                # norm_arena_pages=1 replaces its sole page.  The one-tile
                # coefficient template is the correctly sized scalar output.
                (norm_template, scaler, constants, coefficient_template),
            )
            children.append(_remap_used(total, {0: current, 1: 1, 2: 4, 3: 9}))
            coefficient_target = 6 if diagnostic_stage == "coefficient" else 10
            coefficient = compile_serialized_child(
                coefficient_operation,
                (norm_template, constants, coefficient_template),
            )
            children.append(_remap_used(
                coefficient, {0: 9, 1: 4, 2: coefficient_target}
            ))
            if diagnostic_stage == "coefficient":
                scale = None
            else:
                scale = compile_serialized_child(
                    scale_operation, (gradient, coefficient_template, clipped)
                )
                children.append(_remap_used(scale, {0: 0, 1: 10, 2: 6}))

        # Keep the compact descriptors available for host-side ABI/lifetime
        # diagnostics.  They contain no tensor data and do not participate in
        # execution; exposing them avoids having to depend on TT-Lang's
        # short-lived compile cache after the custom-call artifact is emitted.
        operation._debug_program_sequence = tuple(children)

        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(
            gradient, scaler, norm_template, norm_output_template,
            constants, coefficient_template, clipped,
        )
        closure = _probe.__closure__ or ()
        for name, cell in zip(_probe.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                compiled.program_sequence = children
                compiled.internal_tensors = [
                    {"source_operand": 2, "memory": "clone"},
                    {"source_operand": 2, "memory": "clone"},
                    *(
                        [] if diagnostic_stage == "norms" else [
                            {"source_operand": 5, "memory": "clone"},
                            *(
                                [{"source_operand": 5, "memory": "clone"}]
                                if diagnostic_stage == "full" else []
                            ),
                        ]
                    ),
                ]
            break
        return result

    operation.artifact_format = 4
    operation.physical_programs = (1 if grouped_norms else norm_pages) + {
        "norms": 0,
        "coefficient": 2,
        "full": 3,
    }[diagnostic_stage]
    operation.compiler_owned_internal_tensors = (
        2 if diagnostic_stage == "norms" else 4
    )
    operation.diagnostic_stage = diagnostic_stage
    operation.logical_gradient_tensors = norm_pages
    operation.grouped_norms = grouped_norms
    operation.gradient_page_offset = plan.scratch_pages
    operation.gradient_pages = layer_count * plan.layer_gradient_pages
    operation.safe_default = False
    operation.experimental_backend = "packed_exact_gradient_clip_owner_v1"
    operation._ttl_program = _probe
    return operation
