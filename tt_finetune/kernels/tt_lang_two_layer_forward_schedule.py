"""One PJRT owner for two exact resident Transformer-layer forwards.

The first layer writes its documented checkpoint arena.  The second layer
consumes the first arena's page-zero ``layer_output`` region directly, so no
StableHLO slice, host dispatch or second hidden-state tensor exists between
the layers.  Both checkpoint arenas intentionally remain DRAM-backed because
they must survive the model-wide forward/backward gap.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_attention_forward_qkv_resident import (
    build_tt_lang_attention_forward_qkv_resident,
)
from .tt_lang_program_composition import (
    compile_serialized_child,
    remap_serialized_child,
    serialized_child_tensor_ids,
)


ttl = None
tt_lang_operation = None


def _remap_all(program: dict, mapping: dict[int, int]) -> dict:
    used = serialized_child_tensor_ids(program)
    return remap_serialized_child(
        program, {tensor: mapping[tensor] for tensor in used}
    )


@lru_cache(maxsize=8)
def build_tt_lang_two_layer_forward_schedule(
    *,
    sequence_length: int,
    hidden_features: int,
    kv_features: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    physical_rank: int,
    lora_scale: float,
    attention_scale: float,
    intermediate_features: int,
    rmsnorm_epsilon: float,
    has_bias: bool = False,
    base_k_tiles_per_phase: int = 1,
    gate_up_base_k_tiles_per_phase: int = 1,
    down_base_k_tiles_per_phase: int = 1,
) -> Callable:
    """Build the first exact multi-layer device-scheduled forward owner."""

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("two-layer forward schedule requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    layer = build_tt_lang_attention_forward_qkv_resident(
        sequence_length=sequence_length,
        hidden_features=hidden_features,
        kv_features=kv_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        physical_rank=physical_rank,
        lora_scale=lora_scale,
        attention_scale=attention_scale,
        base_k_tiles_per_phase=base_k_tiles_per_phase,
        has_bias=has_bias,
        input_rmsnorm_epsilon=rmsnorm_epsilon,
        post_attention_rmsnorm_epsilon=rmsnorm_epsilon,
        resident_gate_up_intermediate_features=intermediate_features,
        resident_gate_up_base_k_tiles_per_phase=(
            gate_up_base_k_tiles_per_phase
        ),
        resident_full_mlp_forward=True,
        resident_down_base_k_tiles_per_phase=down_base_k_tiles_per_phase,
        hidden_page_base=0,
    )
    chained_layer = build_tt_lang_attention_forward_qkv_resident(
        sequence_length=sequence_length,
        hidden_features=hidden_features,
        kv_features=kv_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        physical_rank=physical_rank,
        lora_scale=lora_scale,
        attention_scale=attention_scale,
        base_k_tiles_per_phase=base_k_tiles_per_phase,
        has_bias=has_bias,
        input_rmsnorm_epsilon=rmsnorm_epsilon,
        post_attention_rmsnorm_epsilon=rmsnorm_epsilon,
        resident_gate_up_intermediate_features=intermediate_features,
        resident_gate_up_base_k_tiles_per_phase=(
            gate_up_base_k_tiles_per_phase
        ),
        resident_full_mlp_forward=True,
        resident_down_base_k_tiles_per_phase=down_base_k_tiles_per_phase,
        hidden_page_base=layer.layer_output_page_offset,
    )

    @ttl.operation(grid="full")
    def _probe(
        t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
        t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
        t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
        t30, t31, t32, t33, t34, t35, t36, t37, t38, t39,
        t40, t41, t42,
    ):
        source = ttl.make_dataflow_buffer_like(
            t00, shape=(1, 1), block_count=1
        )
        output = ttl.make_dataflow_buffer_like(
            t41, shape=(1, 1), block_count=1
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
                tx = ttl.copy(tile, t41[0:1, 0:1])
                tx.wait()

    variant = (
        f"{sequence_length}x{hidden_features}x{intermediate_features}."
        f"q{query_heads}.kv{kv_heads}.d{head_dim}.r{physical_rank}."
        f"bias{int(has_bias)}"
    )

    @tt_lang_operation(
        operation_id=f"tt_finetune.two_layer_forward_schedule.{variant}.v1",
        arg_roles=("in",) * 41 + ("out",) * 2,
        version_tag=f"two-layer-forward-schedule-{variant}-v1",
    )
    def operation(
        t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
        t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
        t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
        t30, t31, t32, t33, t34, t35, t36, t37, t38, t39,
        t40, t41, t42,
    ):
        tensors = (
            t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
            t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
            t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
            t30, t31, t32, t33, t34, t35, t36, t37, t38, t39,
            t40, t41, t42,
        )
        first_arguments = tuple(tensors[:21]) + (tensors[41],)
        second_arguments = (
            tensors[41], *tuple(tensors[21:41]), tensors[42]
        )
        first = compile_serialized_child(layer, first_arguments)
        second = compile_serialized_child(chained_layer, second_arguments)

        first_map = {index: index for index in range(21)}
        first_map[21] = 41
        second_map = {0: 41}
        second_map.update({index: 20 + index for index in range(1, 21)})
        second_map[21] = 42
        children = [
            _remap_all(first, first_map),
            _remap_all(second, second_map),
        ]

        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(*tensors)
        closure = _probe.__closure__ or ()
        for name, cell in zip(_probe.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                compiled.program_sequence = children
            break
        return result

    operation.artifact_format = 4
    operation.physical_programs = 2
    operation.layer_count = 2
    operation.output_shape = layer.output_shape
    operation.layer_output_page_offset = layer.layer_output_page_offset
    operation.chain_input_page_offset = layer.layer_output_page_offset
    operation.intermediate_dram_bytes = 0
    operation.checkpoint_dram_bytes = 2 * layer.checkpoint_dram_bytes
    operation.safe_default = False
    operation._ttl_program = _probe
    return operation
