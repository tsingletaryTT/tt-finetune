"""Complete layer backward with dHidden separated from adapter gradients.

The accepted arena-backed autograd owner writes attention adapter gradients
and final dHidden into one public arena.  Multi-layer schedules need dHidden as
a private L1 edge while still publishing the adapter-gradient arena.  This
wrapper flattens the accepted twelve physical children and retargets only the
last exact residual-add destination to a distinct DPS result.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_program_composition import (
    compile_serialized_sequence_children,
    remap_serialized_child,
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


@lru_cache(maxsize=8)
def build_tt_lang_resident_layer_autograd_split_owner(
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
    debug_stop_after_mlp: bool = False,
    debug_attention_stop_after: str | None = None,
    pre_rms_copy: bool | None = None,
) -> Callable:
    """Return one exact layer backward with seven gradients plus dHidden."""

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError("split layer autograd owner requires TT-XLA") from exc
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
        debug_attention_stop_after=debug_attention_stop_after,
        pre_rms_copy=pre_rms_copy,
    )
    base_internal_specs = tuple(
        dict(spec) for spec in getattr(base, "internal_tensor_specs", ())
    )

    @ttl.operation(grid="full")
    def _probe(
        t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
        t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
        t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
        t30, t31, t32, t33, t34, t35, t36, t37, t38,
    ):
        probe_source = (
            t04 if debug_attention_stop_after == "post_rstd" else t00
        )
        source = ttl.make_dataflow_buffer_like(
            probe_source, shape=(1, 1), block_count=1
        )
        output = ttl.make_dataflow_buffer_like(t38, shape=(1, 1), block_count=1)

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                tx = ttl.copy(probe_source[0:1, 0:1], tile)
                tx.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with output.reserve() as target:
                    target.store(tile)

        @ttl.datamovement()
        def write():
            with output.wait() as tile:
                tx = ttl.copy(tile, t38[0:1, 0:1])
                tx.wait()

    variant = (
        f"{Path(source_bundle).name}.{sequence_length}x{hidden_features}."
        f"i{intermediate_features}.q{query_heads}.k{kv_heads}."
        f"d{head_dim}.r{physical_rank}."
        f"stopmlp{int(debug_stop_after_mlp)}."
        f"attn{debug_attention_stop_after or 'full'}."
        f"prermscopy{int(bool(pre_rms_copy))}.v3"
    )

    @tt_lang_operation(
        operation_id=f"tt_finetune.resident_layer_autograd_split.{variant}",
        arg_roles=("in",) * 31 + ("out",) * 8,
        version_tag=f"resident-layer-autograd-split-{variant}",
    )
    def operation(
        t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
        t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
        t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
        t30, t31, t32, t33, t34, t35, t36, t37, t38,
    ):
        tensors = (
            t00, t01, t02, t03, t04, t05, t06, t07, t08, t09,
            t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
            t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
            t30, t31, t32, t33, t34, t35, t36, t37, t38,
        )
        children = compile_serialized_sequence_children(
            base, tuple(tensors[:38])
        )
        if len(children) != base.physical_programs:
            raise RuntimeError(
                "arena-backed layer owner physical child count changed: "
                f"compiled={len(children)}, declared={base.physical_programs}"
            )

        ordinary_map = {index: index for index in range(38)}
        ordinary_map.update({
            index: index + 1
            for index in range(38, 38 + len(base_internal_specs))
        })
        if debug_stop_after_mlp:
            mlp_map = dict(ordinary_map)
            # Base internal tensor 38 is the retargeted MLP dX. Publish it in
            # the split owner's shape-compatible eighth output and stop before
            # post-RMS/attention children consume it.
            mlp_map[38] = 38
            remapped = [_remap_used(children[0], mlp_map)]
        else:
            remapped = [
                _remap_used(child, ordinary_map) for child in children[:-1]
            ]
            final_map = dict(ordinary_map)
            if debug_attention_stop_after == "post_rstd":
                final_map[4] = 38
            else:
                final_map[31] = 38
            remapped.append(_remap_used(children[-1], final_map))

        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(*tensors)
        closure = _probe.__closure__ or ()
        for name, cell in zip(_probe.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                compiled.program_sequence = remapped
                compiled.internal_tensors = (
                    [] if debug_stop_after_mlp
                    else [dict(spec) for spec in base_internal_specs]
                )
            break
        return result

    operation.artifact_format = 4
    operation.physical_programs = (
        1 if debug_stop_after_mlp else base.physical_programs
    )
    operation.compiler_owned_l1_tensors = (
        0 if debug_stop_after_mlp else len(base_internal_specs)
    )
    operation.internal_tensor_specs = (
        () if debug_stop_after_mlp else base_internal_specs
    )
    operation.debug_stop_after_mlp = debug_stop_after_mlp
    operation.debug_attention_stop_after = debug_attention_stop_after
    operation.intermediate_dram_bytes = 0
    operation.safe_default = False
    operation.source_bundle = source_bundle
    operation._ttl_program = _probe
    return operation
