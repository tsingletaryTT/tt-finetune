"""One exact N-layer forward/backward/clip/AdamW ownership boundary."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

from .optimizer_gradient_arena_planner import (
    plan_multi_layer_optimizer_gradient_arena,
)
from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_adamw_descriptor_owner import (
    build_tt_lang_adamw_descriptor_owner,
)
from .tt_lang_multi_layer_gradient_arena_schedule import (
    build_tt_lang_multi_layer_gradient_arena_schedule,
)
from .tt_lang_packed_gradient_clip_owner import (
    build_tt_lang_packed_gradient_clip_owner,
)
from .tt_lang_program_composition import (
    bind_serialized_child,
    compile_serialized_sequence_children,
    serialized_child_tensor_ids,
)


ttl = None
tt_lang_operation = None


def _remap_used(program: dict, mapping: dict[int, int]) -> dict:
    used = serialized_child_tensor_ids(program)
    return bind_serialized_child(
        program, {tensor: mapping[tensor] for tensor in used}
    )


@lru_cache(maxsize=8)
def build_tt_lang_multi_layer_clip_adamw_schedule(
    *,
    layer_count: int,
    block_page_offsets: tuple[tuple[int, ...], ...],
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
    grouped_norms: bool = True,
) -> Callable:
    """Flatten the accepted four phase owners into one public custom call.

    Public operands 0..50 are the accepted packed multi-layer inputs. 51/52
    are shape-only parity-arena templates. 53..58 are clipping constants and
    allocation templates. 59..63 are AdamW scalar/scratch templates. The only
    public results are updated packed optimizer state (64) and final dHidden
    (65). Raw gradients, clipped gradients, checkpoints, and all phase scratch
    remain compiler-owned.
    """

    if layer_count < 2:
        raise ValueError("multi-layer step requires at least two layers")
    plan = plan_multi_layer_optimizer_gradient_arena(
        layer_count=layer_count,
        tokens=sequence_length,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        kv_features=kv_features,
    )
    common = dict(
        layer_count=layer_count,
        block_page_offsets=block_page_offsets,
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
    forward = build_tt_lang_multi_layer_gradient_arena_schedule(
        **common, execution_phase="forward"
    )
    backward = build_tt_lang_multi_layer_gradient_arena_schedule(
        **common,
        reuse_backward_scratch=True,
        execution_phase="backward",
    )
    clip = build_tt_lang_packed_gradient_clip_owner(
        layer_count=layer_count,
        sequence_length=sequence_length,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        kv_features=kv_features,
        grouped_norms=grouped_norms,
    )
    adamw = build_tt_lang_adamw_descriptor_owner(
        descriptor_bundle=descriptor_bundle
    )
    optimizer_rows = layer_count * plan.layer_gradient_pages * 32
    if (
        adamw.rows != optimizer_rows
        or adamw.gradient_page_offset != plan.scratch_pages
        or adamw.gradient_pages != layer_count * plan.layer_gradient_pages
    ):
        raise ValueError("AdamW descriptor does not match gradient arena")

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError("multi-layer step owner requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid="full")
    def _probe(
        t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
        t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
        t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
        t30, t31, t32, t33, t34, t35, t36, t37, t38, t39,
        t40, t41, t42, t43, t44, t45, t46, t47, t48, t49,
        t50, t51, t52, t53, t54, t55, t56, t57, t58, t59,
        t60, t61, t62, t63, t64, t65,
    ):
        source = ttl.make_dataflow_buffer_like(
            t00, shape=(1, 1), block_count=1
        )
        target = ttl.make_dataflow_buffer_like(
            t64, shape=(1, 1), block_count=1
        )

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
                transaction = ttl.copy(output, t64[0:1, 0:1])
                transaction.wait()

    variant = (
        f"{Path(source_bundle).name}.l{layer_count}."
        f"{sequence_length}x{hidden_features}.i{intermediate_features}."
        f"kv{kv_features}.r{physical_rank}.bias{int(has_bias)}."
        f"grouped{int(grouped_norms)}.scratchreuse1.compactabi1."
        "periodicnorm1.v5"
    )

    @tt_lang_operation(
        operation_id=f"tt_finetune.multi_layer_clip_adamw.{variant}",
        arg_roles=("in",) * 64 + ("out", "out"),
        version_tag=f"multi-layer-clip-adamw-{variant}",
    )
    def operation(
        t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
        t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
        t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
        t30, t31, t32, t33, t34, t35, t36, t37, t38, t39,
        t40, t41, t42, t43, t44, t45, t46, t47, t48, t49,
        t50, t51, t52, t53, t54, t55, t56, t57, t58, t59,
        t60, t61, t62, t63, t64, t65,
    ):
        tensors = (
            t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
            t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
            t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
            t30, t31, t32, t33, t34, t35, t36, t37, t38, t39,
            t40, t41, t42, t43, t44, t45, t46, t47, t48, t49,
            t50, t51, t52, t53, t54, t55, t56, t57, t58, t59,
            t60, t61, t62, t63, t64, t65,
        )
        forward_children = compile_serialized_sequence_children(
            forward, tensors[:53]
        )
        backward_children = compile_serialized_sequence_children(
            backward, (*tensors[:51], tensors[58], tensors[65])
        )
        clip_children = compile_serialized_sequence_children(
            clip,
            (
                tensors[14], tensors[53], tensors[54], tensors[55],
                tensors[56], tensors[57], tensors[58],
            ),
        )
        adamw_children = compile_serialized_sequence_children(
            adamw,
            (
                tensors[16], tensors[58], tensors[59], tensors[60],
                tensors[61], tensors[62], tensors[63], tensors[64],
            ),
        )
        expected = (
            forward.physical_programs,
            backward.physical_programs,
            clip.physical_programs,
            adamw.physical_programs,
        )
        actual = tuple(map(len, (
            forward_children, backward_children, clip_children, adamw_children
        )))
        if actual != expected:
            raise RuntimeError(
                f"multi-layer child counts changed: {actual} != {expected}"
            )

        # Private tensor table:
        # 66/67 parity checkpoints, 68 raw and 69 clipped gradients,
        # 70..73 clipping scratch, 74..76 AdamW scratch, and 77 onward the
        # accepted backward owner's 3+7*layer_count private tensors.
        forward_map = {index: index for index in range(51)}
        forward_map.update({51: 66, 52: 67})
        remapped_forward = [
            _remap_used(child, forward_map) for child in forward_children
        ]

        backward_map = {index: index for index in range(53)}
        backward_map.update({5: 66, 6: 67, 51: 68, 52: 65})
        backward_private = backward.compiler_owned_internal_tensors
        backward_map.update({
            53 + index: 77 + index for index in range(backward_private)
        })
        remapped_backward = [
            _remap_used(child, backward_map) for child in backward_children
        ]

        clip_map = {
            0: 68, 1: 53, 2: 54, 3: 55, 4: 56, 5: 57, 6: 69,
            7: 70, 8: 71, 9: 72, 10: 73,
        }
        remapped_clip = [
            _remap_used(child, clip_map) for child in clip_children
        ]
        adamw_map = {
            0: 16, 1: 69, 2: 59, 3: 60, 4: 61, 5: 62, 6: 63,
            7: 64, 8: 74, 9: 75, 10: 76,
        }
        remapped_adamw = [
            _remap_used(child, adamw_map) for child in adamw_children
        ]
        children = [
            *remapped_forward,
            *remapped_backward,
            *remapped_clip,
            *remapped_adamw,
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
                    {"source_operand": 51, "memory": "clone"},
                    {"source_operand": 52, "memory": "clone"},
                    {"source_operand": 14, "memory": "clone"},
                    {"source_operand": 58, "memory": "clone"},
                    {"source_operand": 54, "memory": "clone"},
                    {"source_operand": 54, "memory": "clone"},
                    {"source_operand": 57, "memory": "clone"},
                    {"source_operand": 57, "memory": "clone"},
                    {"source_operand": 62, "memory": "clone"},
                    {"source_operand": 63, "memory": "clone"},
                    {"source_operand": 63, "memory": "clone"},
                    {"source_operand": 0, "memory": "l1"},
                    {"source_operand": 1, "memory": "l1"},
                    {"source_operand": 1, "memory": "l1"},
                    *(
                        {
                            "source_operand": source,
                            "memory": "l1",
                        }
                        for _layer in range(backward.backward_scratch_groups)
                        for _slot, source in enumerate(
                            (1, 0, 0, 2, 0, 1, 1)
                        )
                    ),
                ]
            break
        return result

    operation.artifact_format = 4
    operation.layer_count = layer_count
    operation.physical_programs = sum((
        forward.physical_programs,
        backward.physical_programs,
        clip.physical_programs,
        adamw.physical_programs,
    ))
    operation.public_custom_calls = 1
    operation.compiler_owned_internal_tensors = 11 + (
        backward.compiler_owned_internal_tensors
    )
    operation.backward_scratch_groups = backward.backward_scratch_groups
    operation.l1_spill_tensors = 0
    operation.l1_spill_reason = None
    operation.framework_visible_checkpoint_tensors = 0
    operation.framework_visible_gradient_tensors = 0
    operation.framework_visible_clip_tensors = 0
    operation.optimizer_rows = optimizer_rows
    operation.gradient_arena_shape = plan.physical_shape
    operation.gradient_page_offset = plan.scratch_pages
    operation.gradient_pages = layer_count * plan.layer_gradient_pages
    operation.safe_default = False
    operation.experimental_backend = "multi_layer_resident_step_v1"
    operation._ttl_program = _probe
    return operation
