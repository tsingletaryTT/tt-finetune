"""Full resident layer forward with canonical PEFT MLP LoRA operands.

The accepted forward program historically read gate/up/down A/B from a
detached immutable pack. This direct single-program owner preserves that exact
physical program but redirects those six reads to explicit tensor operands. Optimizer
updates therefore become visible without rebuilding frozen base-weight packs.
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
    retarget_full_mlp_forward_trainable_parameters,
    serialized_child_tensor_ids,
)


ttl = None
tt_lang_operation = None


@lru_cache(maxsize=8)
def build_tt_lang_resident_layer_forward_trainable_owner(
    *,
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
    has_bias: bool,
    input_rmsnorm_epsilon: float,
    post_attention_rmsnorm_epsilon: float,
    hidden_page_base: int = 0,
    publish_layer_output_at_page_zero: bool = True,
    o_lora_k_tiles_per_phase: int | None = None,
) -> Callable:
    """Build the exact trainable-state forward owner for one full layer."""
    base = build_tt_lang_attention_forward_qkv_resident(
        sequence_length=sequence_length,
        hidden_features=hidden_features,
        kv_features=kv_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        physical_rank=physical_rank,
        lora_scale=lora_scale,
        attention_scale=attention_scale,
        o_lora_k_tiles_per_phase=o_lora_k_tiles_per_phase,
        has_bias=has_bias,
        input_rmsnorm_epsilon=input_rmsnorm_epsilon,
        post_attention_rmsnorm_epsilon=post_attention_rmsnorm_epsilon,
        resident_gate_up_intermediate_features=intermediate_features,
        resident_full_mlp_forward=True,
        hidden_page_base=hidden_page_base,
        publish_layer_output_at_page_zero=publish_layer_output_at_page_zero,
    )
    offsets = base.mlp_parameter_pack_offsets

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError(
                "trainable resident layer forward requires TT-XLA"
            ) from exc
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid="full")
    def _probe(
        t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
        t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
        t20, t21, t22, t23, t24, t25, t26, t27,
    ):
        source = ttl.make_dataflow_buffer_like(t00, shape=(1, 1), block_count=1)
        output = ttl.make_dataflow_buffer_like(t27, shape=(1, 1), block_count=1)

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                transaction = ttl.copy(t00[0:1, 0:1], tile)
                transaction.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with output.reserve() as target:
                    target.store(tile)

        @ttl.datamovement()
        def write():
            with output.wait() as tile:
                transaction = ttl.copy(tile, t27[0:1, 0:1])
                transaction.wait()

    variant = (
        f"{sequence_length}x{hidden_features}.i{intermediate_features}."
        f"q{query_heads}.k{kv_heads}.d{head_dim}.r{physical_rank}."
        f"bias{int(has_bias)}.hidden{hidden_page_base}."
        f"page0{int(publish_layer_output_at_page_zero)}.v3"
    )

    @tt_lang_operation(
        operation_id=(
            f"tt_finetune.resident_layer_forward_trainable_owner.{variant}"
        ),
        arg_roles=("in",) * 27 + ("out",),
        version_tag=f"resident-layer-forward-trainable-owner-{variant}",
    )
    def operation(
        t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
        t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
        t20, t21, t22, t23, t24, t25, t26, t27,
    ):
        tensors = (
            t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
            t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
            t20, t21, t22, t23, t24, t25, t26, t27,
        )
        child = compile_serialized_child(base, (*tensors[:21], t27))
        used = serialized_child_tensor_ids(child)
        mapping = {local: local for local in used if local <= 20}
        if 21 in used:
            mapping[21] = 27
        child = remap_serialized_child(child, mapping)
        child = retarget_full_mlp_forward_trainable_parameters(
            child,
            gate_a_tensor=21,
            gate_b_tensor=22,
            up_a_tensor=23,
            up_b_tensor=24,
            down_a_tensor=25,
            down_b_tensor=26,
            gate_a_pack_base=offsets["gate_a"],
            gate_b_pack_base=offsets["gate_b"],
            up_a_pack_base=offsets["up_a"],
            up_b_pack_base=offsets["up_b"],
            down_a_pack_base=offsets["down_a"],
            down_b_pack_base=offsets["down_b"],
        )

        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(*tensors)
        closure = _probe.__closure__ or ()
        for name, cell in zip(_probe.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                # This owner has exactly one accepted physical child. Emit it
                # directly as a normal format-1/2/3 artifact. Wrapping one
                # program in artifact-v4 adds a second runtime lifecycle and,
                # on the current Blackhole stack, can leave only the probe
                # program visible or wedge the completion queue.
                compiled.serialized_program_override = child
            break
        return result

    operation.artifact_format = 3
    operation.physical_programs = 1
    operation.output_shape = base.output_shape
    operation.forward_plan = base.forward_plan
    operation.layer_output_page_offset = base.layer_output_page_offset
    operation.hidden_page_base = hidden_page_base
    operation.publishes_layer_output_at_page_zero = bool(
        publish_layer_output_at_page_zero
    )
    operation.layer_used_pages = base.layer_used_pages
    operation.layer_allocated_pages = base.layer_allocated_pages
    operation.mlp_gate_page_offset = base.mlp_gate_page_offset
    operation.mlp_up_page_offset = base.mlp_up_page_offset
    operation.mlp_gate_h_page_offset = base.mlp_gate_h_page_offset
    operation.mlp_up_h_page_offset = base.mlp_up_h_page_offset
    operation.resident_gate_up_intermediate_features = (
        base.resident_gate_up_intermediate_features
    )
    # Preserve the accepted owner's profiling contract.  Qualification and
    # trainer reports must not special-case the trainable retarget wrapper.
    operation.intermediate_dram_bytes = base.intermediate_dram_bytes
    operation.checkpoint_dram_bytes = base.checkpoint_dram_bytes
    operation.projection_cb_bytes = base.projection_cb_bytes
    operation.reverse_handoff_noc_bytes = base.reverse_handoff_noc_bytes
    operation.trainable_mlp_parameters = True
    operation.safe_default = False
    operation.experimental_backend = "full_layer_forward_trainable_v1"
    operation._ttl_program = _probe
    return operation
