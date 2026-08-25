"""Two-layer device trace with one optimizer-aligned gradient arena output."""

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
    retarget_full_attention_gradient_outputs,
    retarget_full_mlp_gradient_outputs,
    serialized_child_tensor_ids,
)
from .tt_lang_two_layer_train_schedule import (
    build_tt_lang_two_layer_train_schedule,
)


ttl = None
tt_lang_operation = None


def _remap_used(program: dict, mapping: dict[int, int]) -> dict:
    used = serialized_child_tensor_ids(program)
    return remap_serialized_child(
        program, {tensor: mapping[tensor] for tensor in used}
    )


def _is_full_attention_writer(program: dict) -> bool:
    return any(
        all(
            marker in kernel["cpp_source"]
            for marker in (
                "FORCE_INLINE void v_writer_phase()",
                "FORCE_INLINE void k_writer_phase()",
                "FORCE_INLINE void q_writer_phase()",
                "Final output-projection adapter gradients",
            )
        )
        for kernel in program["kernels"]
    )


@lru_cache(maxsize=8)
def build_tt_lang_two_layer_optimizer_arena_schedule(
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
    """Run two forwards/backwards and publish one arena plus final dHidden."""

    if physical_rank != 32:
        raise ValueError("optimizer gradient arena currently requires rank 32")
    plan = plan_multi_layer_optimizer_gradient_arena(
        layer_count=2,
        tokens=sequence_length,
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
                "two-layer optimizer-arena schedule requires TT-XLA"
            ) from exc
        ttl, tt_lang_operation = ttl_module, decorator

    base = build_tt_lang_two_layer_train_schedule(
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

    @ttl.operation(grid="full")
    def _probe(
        t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
        t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
        t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
        t30, t31, t32, t33, t34, t35, t36, t37, t38, t39,
        t40, t41, t42, t43, t44, t45, t46, t47, t48, t49,
        t50, t51, t52, t53, t54, t55, t56, t57, t58, t59,
        t60, t61, t62, t63, t64, t65, t66, t67, t68, t69,
        t70, t71, t72, t73, t74, t75, t76,
    ):
        source = ttl.make_dataflow_buffer_like(t00, shape=(1, 1), block_count=1)
        output = ttl.make_dataflow_buffer_like(t76, shape=(1, 1), block_count=1)

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
                tx = ttl.copy(tile, t76[0:1, 0:1])
                tx.wait()

    variant = (
        f"{Path(source_bundle).name}.{sequence_length}x{hidden_features}."
        f"i{intermediate_features}.kv{kv_features}.r{physical_rank}."
        f"bias{int(has_bias)}.v1"
    )

    @tt_lang_operation(
        operation_id=f"tt_finetune.two_layer_optimizer_arena.{variant}",
        arg_roles=("in",) * 75 + ("out", "out"),
        version_tag=f"two-layer-optimizer-gradient-arena-{variant}",
    )
    def operation(
        t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
        t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
        t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
        t30, t31, t32, t33, t34, t35, t36, t37, t38, t39,
        t40, t41, t42, t43, t44, t45, t46, t47, t48, t49,
        t50, t51, t52, t53, t54, t55, t56, t57, t58, t59,
        t60, t61, t62, t63, t64, t65, t66, t67, t68, t69,
        t70, t71, t72, t73, t74, t75, t76,
    ):
        tensors = (
            t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
            t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
            t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
            t30, t31, t32, t33, t34, t35, t36, t37, t38, t39,
            t40, t41, t42, t43, t44, t45, t46, t47, t48, t49,
            t50, t51, t52, t53, t54, t55, t56, t57, t58, t59,
            t60, t61, t62, t63, t64, t65, t66, t67, t68, t69,
            t70, t71, t72, t73, t74, t75, t76,
        )
        children = compile_serialized_sequence_children(
            base, (*tensors[:75], *([t75] * 14), t76)
        )
        if len(children) != base.physical_programs:
            raise RuntimeError("two-layer physical program count changed")
        backward_programs = (base.physical_programs - 3) // 2
        if 3 + 2 * backward_programs != len(children):
            raise RuntimeError("two-layer forward/backward partition changed")

        # Reverse execution is layer 1 then layer 0. Both reuse arena pages
        # [0:scratch_pages); only their final-gradient tails are disjoint.
        for layer, start in ((1, 3), (0, 3 + backward_programs)):
            offset = lambda name, layer=layer: plan.slice(layer, name).page_offset
            children[start] = retarget_full_mlp_gradient_outputs(
                children[start],
                gate_a_page_offset=offset("gate_a"),
                gate_b_page_offset=offset("gate_b"),
                up_a_page_offset=offset("up_a"),
                up_b_page_offset=offset("up_b"),
                down_a_page_offset=offset("down_a"),
                down_b_page_offset=offset("down_b"),
            )
            attention_children = [
                index for index in range(start, start + backward_programs)
                if _is_full_attention_writer(children[index])
            ]
            if len(attention_children) != 1:
                raise RuntimeError(
                    f"layer {layer} attention writer count changed: "
                    f"{len(attention_children)}"
                )
            index = attention_children[0]
            children[index] = retarget_full_attention_gradient_outputs(
                children[index],
                o_a_page_offset=offset("o_a"),
                o_b_page_offset=offset("o_b"),
                q_a_page_offset=offset("q_a"),
                k_a_page_offset=offset("k_a"),
                v_a_page_offset=offset("v_a"),
                q_b_page_offset=offset("q_b"),
                k_b_page_offset=offset("k_b"),
                v_b_page_offset=offset("v_b"),
            )

        # Base external IDs: 0..74 inputs, 75..88 gradients, 89 dHidden.
        # Base private IDs 90..107 become this owner's 77..94.
        mapping = {index: index for index in range(75)}
        mapping.update({index: 75 for index in range(75, 89)})
        mapping[89] = 76
        mapping.update({index: index - 13 for index in range(90, 108)})
        remapped = [_remap_used(child, mapping) for child in children]

        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(*tensors)
        closure = _probe.__closure__ or ()
        for name, cell in zip(_probe.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                compiled.program_sequence = remapped
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
    operation.checkpoint_dram_bytes = base.checkpoint_dram_bytes
    operation.gradient_arena_shape = plan.physical_shape
    operation.gradient_pages = 2 * plan.layer_gradient_pages
    operation.gradient_page_offset = plan.scratch_pages
    operation.gradient_plan = plan
    operation.framework_visible_gradient_tensors = 1
    operation.intermediate_dram_bytes = 0
    operation.safe_default = False
    operation.experimental_backend = "two_layer_optimizer_gradient_arena_v1"
    operation._ttl_program = _probe
    return operation
