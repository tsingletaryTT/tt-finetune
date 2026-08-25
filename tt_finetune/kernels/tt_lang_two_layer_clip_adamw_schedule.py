"""Two resident transformer layers, exact clipping, then exact AdamW.

The accepted two-layer backward/AdamW owner is intentionally left unchanged.
This experimental resident owner reuses its physical children, inserts the
accepted packed clipping sequence at the private gradient-arena boundary, and
publishes the same two results: updated optimizer state and final dHidden.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

from .optimizer_gradient_arena_planner import (
    plan_multi_layer_optimizer_gradient_arena,
)
from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_packed_gradient_clip_owner import (
    build_tt_lang_packed_gradient_clip_owner,
)
from .tt_lang_program_composition import (
    compile_serialized_sequence_children,
    remap_serialized_child,
    serialized_child_tensor_ids,
)
from .tt_lang_two_layer_adamw_schedule import (
    build_tt_lang_two_layer_adamw_schedule,
)


ttl = None
tt_lang_operation = None


def _remap_used(program: dict, mapping: dict[int, int]) -> dict:
    used = serialized_child_tensor_ids(program)
    return remap_serialized_child(
        program, {tensor: mapping[tensor] for tensor in used}
    )


@lru_cache(maxsize=8)
def build_tt_lang_two_layer_clip_adamw_schedule(
    *,
    source_bundle: str,
    descriptor_bundle: str,
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
    packed_parameter_reads: bool = True,
) -> Callable:
    """Return one public forward/backward/clip/AdamW custom call.

    External tensors 0..81 retain the accepted AdamW-owner ABI.  The six
    clipping allocation/constant operands are 82..87, followed by updated
    state and final dHidden outputs 88..89.  Every gradient and clipping
    intermediate is compiler-owned.
    """

    plan = plan_multi_layer_optimizer_gradient_arena(
        layer_count=2,
        tokens=sequence_length,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        kv_features=kv_features,
    )
    base = build_tt_lang_two_layer_adamw_schedule(
        source_bundle=source_bundle,
        descriptor_bundle=descriptor_bundle,
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
        packed_parameter_reads=packed_parameter_reads,
    )
    clip = build_tt_lang_packed_gradient_clip_owner(
        layer_count=2,
        sequence_length=sequence_length,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        kv_features=kv_features,
    )

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError(
                "two-layer clip/AdamW schedule requires TT-XLA"
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
        t70, t71, t72, t73, t74, t75, t76, t77, t78, t79,
        t80, t81, t82, t83, t84, t85, t86, t87, t88, t89,
    ):
        source = ttl.make_dataflow_buffer_like(t00, shape=(1, 1), block_count=1)
        target = ttl.make_dataflow_buffer_like(t88, shape=(1, 1), block_count=1)

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                transaction = ttl.copy(t00[0:1, 0:1], tile)
                transaction.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with target.reserve() as output:
                    output.store(tile)

        @ttl.datamovement()
        def write():
            with target.wait() as output:
                transaction = ttl.copy(output, t88[0:1, 0:1])
                transaction.wait()

    variant = (
        f"{Path(source_bundle).name}.{sequence_length}x{hidden_features}."
        f"i{intermediate_features}.kv{kv_features}.r{physical_rank}."
        f"bias{int(has_bias)}.packed{int(packed_parameter_reads)}.v1"
    )

    @tt_lang_operation(
        operation_id=f"tt_finetune.two_layer_clip_adamw_schedule.{variant}",
        arg_roles=("in",) * 88 + ("out", "out"),
        version_tag=f"two-layer-clip-adamw-schedule-{variant}",
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
        base_children = compile_serialized_sequence_children(
            base, (*tensors[:82], tensors[88], tensors[89])
        )
        if len(base_children) != base.physical_programs:
            raise RuntimeError("two-layer AdamW physical program count changed")
        optimizer_start = len(base_children) - 4
        if optimizer_start <= 0:
            raise RuntimeError("two-layer AdamW child boundary is invalid")

        clip_children = compile_serialized_sequence_children(
            clip,
            (
                tensors[76], tensors[82], tensors[83], tensors[84],
                tensors[85], tensors[86], tensors[87],
            ),
        )
        if len(clip_children) != clip.physical_programs:
            raise RuntimeError("packed clipping physical program count changed")

        # Six new external operands shift the accepted owner's private table:
        # old 84..105 become 90..111.  Backward writes the raw arena to 90;
        # clipping writes private arena 112; AdamW reads 112 instead of 90.
        base_mapping = {index: index for index in range(82)}
        base_mapping.update({82: 88, 83: 89})
        base_mapping.update({index: index + 6 for index in range(84, 106)})
        backward_children = [
            _remap_used(child, base_mapping)
            for child in base_children[:optimizer_start]
        ]
        optimizer_mapping = dict(base_mapping)
        optimizer_mapping[84] = 112
        optimizer_children = [
            _remap_used(child, optimizer_mapping)
            for child in base_children[optimizer_start:]
        ]

        clip_mapping = {
            0: 90, 1: 82, 2: 83, 3: 84, 4: 85, 5: 86, 6: 112,
            7: 113, 8: 114, 9: 115, 10: 116,
        }
        remapped_clip = [
            _remap_used(child, clip_mapping) for child in clip_children
        ]
        children = [
            *backward_children, *remapped_clip, *optimizer_children,
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
                    # 90..111: exact accepted owner internals, shifted by six.
                    {"source_operand": 76, "memory": "clone"},
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
                    {"source_operand": 80, "memory": "clone"},
                    {"source_operand": 81, "memory": "clone"},
                    {"source_operand": 81, "memory": "clone"},
                    # 112: clipped gradient; 113..116: exact clip handoffs.
                    {"source_operand": 87, "memory": "clone"},
                    {"source_operand": 83, "memory": "clone"},
                    {"source_operand": 83, "memory": "clone"},
                    {"source_operand": 86, "memory": "clone"},
                    {"source_operand": 86, "memory": "clone"},
                ]
            break
        return result

    operation.artifact_format = 4
    operation.layer_count = 2
    operation.physical_programs = base.physical_programs + clip.physical_programs
    operation.compiler_owned_internal_tensors = 27
    operation.framework_visible_gradient_tensors = 0
    operation.framework_visible_clip_tensors = 0
    operation.gradient_arena_shape = plan.physical_shape
    operation.gradient_page_offset = plan.scratch_pages
    operation.gradient_pages = 2 * plan.layer_gradient_pages
    operation.optimizer_rows = base.optimizer_rows
    operation.packed_parameter_reads = packed_parameter_reads
    operation.safe_default = False
    operation.experimental_backend = "two_layer_clip_adamw_schedule_v1"
    operation._ttl_program = _probe
    return operation
