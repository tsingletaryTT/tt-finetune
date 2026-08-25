"""Two-layer gradient arena owner reading adapters from packed AdamW state."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

from .optimizer_gradient_arena_planner import (
    plan_multi_layer_optimizer_gradient_arena,
)
from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_program_composition import (
    compile_serialized_sequence_children,
    remap_serialized_child,
    retarget_tensor_reads_to_packed_pages,
    serialized_child_tensor_ids,
)
from .tt_lang_two_layer_adamw_schedule import _adapter_page_offsets
from .tt_lang_two_layer_optimizer_arena_schedule import (
    build_tt_lang_two_layer_optimizer_arena_schedule,
)


ttl = None
tt_lang_operation = None


def _remap_used(program: dict, mapping: dict[int, int]) -> dict:
    used = serialized_child_tensor_ids(program)
    return remap_serialized_child(
        program, {tensor: mapping[tensor] for tensor in used}
    )


@lru_cache(maxsize=8)
def build_tt_lang_two_layer_packed_optimizer_arena_schedule(
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
) -> Callable:
    """Return ``(gradient_arena, dHidden)`` from one packed state input."""

    plan = plan_multi_layer_optimizer_gradient_arena(
        layer_count=2,
        tokens=sequence_length,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        kv_features=kv_features,
    )
    base = build_tt_lang_two_layer_optimizer_arena_schedule(
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
        has_bias=has_bias,
    )

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError(
                "packed two-layer arena schedule requires TT-XLA"
            ) from exc
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid="full")
    def _probe(
        t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
        t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
        t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
        t30, t31, t32, t33, t34, t35, t36, t37, t38, t39,
        t40, t41, t42, t43, t44, t45, t46, t47, t48, t49,
        t50, t51, t52, t53, t54, t55, t56, t57, t58, t59,
        t60, t61, t62, t63, t64, t65, t66, t67, t68, t69,
        t70, t71, t72, t73, t74, t75, t76, t77,
    ):
        source = ttl.make_dataflow_buffer_like(t00, shape=(1, 1), block_count=1)
        target = ttl.make_dataflow_buffer_like(t76, shape=(1, 1), block_count=1)

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                tx = ttl.copy(t00[0:1, 0:1], tile)
                tx.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with target.reserve() as output:
                    output.store(tile)

        @ttl.datamovement()
        def write():
            with target.wait() as output:
                tx = ttl.copy(output, t76[0:1, 0:1])
                tx.wait()

    variant = (
        f"{Path(source_bundle).name}.{sequence_length}x{hidden_features}."
        f"i{intermediate_features}.kv{kv_features}.r{physical_rank}."
        f"bias{int(has_bias)}.v1"
    )

    @tt_lang_operation(
        operation_id=f"tt_finetune.two_layer_packed_arena.{variant}",
        arg_roles=("in",) * 76 + ("out", "out"),
        version_tag=f"two-layer-packed-arena-{variant}",
    )
    def operation(
        t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
        t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
        t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
        t30, t31, t32, t33, t34, t35, t36, t37, t38, t39,
        t40, t41, t42, t43, t44, t45, t46, t47, t48, t49,
        t50, t51, t52, t53, t54, t55, t56, t57, t58, t59,
        t60, t61, t62, t63, t64, t65, t66, t67, t68, t69,
        t70, t71, t72, t73, t74, t75, t76, t77,
    ):
        tensors = (
            t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
            t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
            t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
            t30, t31, t32, t33, t34, t35, t36, t37, t38, t39,
            t40, t41, t42, t43, t44, t45, t46, t47, t48, t49,
            t50, t51, t52, t53, t54, t55, t56, t57, t58, t59,
            t60, t61, t62, t63, t64, t65, t66, t67, t68, t69,
            t70, t71, t72, t73, t74, t75, t76, t77,
        )
        children = compile_serialized_sequence_children(
            base, (*tensors[:75], tensors[76], tensors[77])
        )
        offsets = _adapter_page_offsets(plan)
        seen = {tensor: 0 for tensor in offsets}
        transformed = []
        for child in children:
            used = serialized_child_tensor_ids(child)
            for tensor in seen:
                if tensor in used:
                    seen[tensor] += 1
            transformed.append(retarget_tensor_reads_to_packed_pages(
                child,
                packed_tensor=96,
                tensor_page_offsets=offsets,
            ))
        missing = [tensor for tensor, count in seen.items() if count == 0]
        if missing:
            raise RuntimeError(f"packed adapter readers missing: {missing}")

        mapping = {index: index for index in range(75)}
        mapping[75] = 76
        mapping[76] = 77
        mapping.update({index: index + 1 for index in range(77, 95)})
        mapping[96] = 75
        transformed = [_remap_used(child, mapping) for child in transformed]

        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(*tensors)
        closure = _probe.__closure__ or ()
        for name, cell in zip(_probe.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                compiled.program_sequence = transformed
                compiled.internal_tensors = [
                    {"source_operand": 5, "memory": "clone"},
                    {"source_operand": 6, "memory": "clone"},
                    {"source_operand": 0, "memory": "l1"},
                    {"source_operand": 1, "memory": "l1"},
                    *(
                        {"source_operand": source, "memory": "l1"}
                        for source in (1, 0, 0, 2, 0, 1, 1)
                    ),
                    *(
                        {"source_operand": source, "memory": "l1"}
                        for source in (1, 0, 0, 2, 0, 1, 1)
                    ),
                ]
            break
        return result

    operation.artifact_format = 4
    operation.layer_count = 2
    operation.physical_programs = base.physical_programs
    operation.compiler_owned_internal_tensors = 18
    operation.compiler_owned_checkpoint_tensors = 2
    operation.compiler_owned_l1_tensors = 16
    operation.framework_visible_gradient_tensors = 1
    operation.packed_parameter_reads = True
    operation.gradient_arena_shape = plan.physical_shape
    operation.gradient_page_offset = plan.scratch_pages
    operation.gradient_pages = 2 * plan.layer_gradient_pages
    operation.safe_default = False
    operation.experimental_backend = "two_layer_packed_optimizer_arena_v1"
    operation._ttl_program = _probe
    return operation
