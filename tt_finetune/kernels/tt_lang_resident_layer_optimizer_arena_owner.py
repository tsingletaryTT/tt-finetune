"""Exact layer backward writing all 14 LoRA gradients to one dense arena.

This is the backward-to-optimizer ownership boundary. It composes the already
accepted full-layer owner, preserves its physical programs, and changes only
final writer tensor IDs/page addresses. The attention dHidden workspace stays
at the arena prefix until input RMSNorm backward consumes it; final dHidden is
published separately for the preceding layer.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

from .optimizer_gradient_arena_planner import (
    plan_layer_optimizer_gradient_arena,
)
from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_program_composition import (
    compile_serialized_sequence_children,
    remap_serialized_child,
    retarget_full_attention_gradient_outputs,
    retarget_full_mlp_gradient_outputs,
    serialized_child_tensor_ids,
)
from .tt_lang_resident_layer_autograd_owner import (
    build_tt_lang_resident_layer_autograd_owner,
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
def build_tt_lang_resident_layer_optimizer_arena_owner(
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
) -> Callable:
    """Return one layer backward with ``(dense_gradient_arena, dHidden)``."""

    plan = plan_layer_optimizer_gradient_arena(
        tokens=sequence_length,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        kv_features=kv_features,
    )
    if physical_rank != 32:
        raise ValueError("optimizer gradient arena currently requires rank 32")

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError(
                "optimizer-arena layer owner requires TT-XLA"
            ) from exc
        ttl, tt_lang_operation = ttl_module, decorator

    base = build_tt_lang_resident_layer_autograd_owner(
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
    )

    @ttl.operation(grid="full")
    def _probe(
        t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
        t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
        t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
        t30, t31, t32,
    ):
        source = ttl.make_dataflow_buffer_like(t00, shape=(1, 1), block_count=1)
        output = ttl.make_dataflow_buffer_like(t32, shape=(1, 1), block_count=1)

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
                tx = ttl.copy(tile, t32[0:1, 0:1])
                tx.wait()

    variant = (
        f"{Path(source_bundle).name}.{sequence_length}x{hidden_features}."
        f"i{intermediate_features}.kv{kv_features}.r{physical_rank}.v1"
    )

    @tt_lang_operation(
        operation_id=f"tt_finetune.layer_optimizer_arena.{variant}",
        arg_roles=("in",) * 31 + ("out", "out"),
        version_tag=f"layer-optimizer-gradient-arena-{variant}",
    )
    def operation(
        t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
        t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
        t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
        t30, t31, t32,
    ):
        tensors = (
            t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
            t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
            t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
            t30, t31, t32,
        )
        # Base ABI has one attention arena and six MLP outputs. Passing the
        # same placeholder is compilation-only; all physical children are
        # flattened and explicitly remapped below.
        children = compile_serialized_sequence_children(
            base, (*tensors[:31], *([t31] * 7))
        )
        if len(children) != base.physical_programs:
            raise RuntimeError("full-layer physical child count changed")

        offset = lambda name: plan.slice(name).page_offset
        children[0] = retarget_full_mlp_gradient_outputs(
            children[0],
            gate_a_page_offset=offset("gate_a"),
            gate_b_page_offset=offset("gate_b"),
            up_a_page_offset=offset("up_a"),
            up_b_page_offset=offset("up_b"),
            down_a_page_offset=offset("down_a"),
            down_b_page_offset=offset("down_b"),
        )
        attention_indices = [
            index for index, child in enumerate(children)
            if _is_full_attention_writer(child)
        ]
        if len(attention_indices) != 1:
            raise RuntimeError(
                "full-layer attention writer child count changed: "
                f"{len(attention_indices)}"
            )
        index = attention_indices[0]
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

        # Base IDs 31..37 are its seven public outputs; collapse them to the
        # compact arena. Base IDs 38..44 are seven compiler-owned L1 tensors.
        ordinary_map = {index: index for index in range(31)}
        ordinary_map.update({index: 31 for index in range(31, 38)})
        ordinary_map.update({index: index - 5 for index in range(38, 45)})
        remapped = [
            _remap_used(child, ordinary_map) for child in children[:-1]
        ]
        final_map = dict(ordinary_map)
        final_map[31] = 32
        remapped.append(_remap_used(children[-1], final_map))

        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(*tensors)
        closure = _probe.__closure__ or ()
        for name, cell in zip(_probe.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                compiled.program_sequence = remapped
                compiled.internal_tensors = [
                    {"source_operand": source, "memory": "l1"}
                    for source in (0, 1, 1, 4, 1, 0, 0)
                ]
            break
        return result

    operation.artifact_format = 4
    operation.physical_programs = base.physical_programs
    operation.compiler_owned_l1_tensors = 7
    operation.gradient_arena_shape = plan.physical_shape
    operation.gradient_page_offset = plan.gradient_page_offset
    operation.gradient_pages = plan.gradient_pages
    operation.gradient_plan = plan
    operation.intermediate_dram_bytes = 0
    operation.safe_default = False
    operation.experimental_backend = "layer_optimizer_gradient_arena_v1"
    operation._ttl_program = _probe
    return operation
