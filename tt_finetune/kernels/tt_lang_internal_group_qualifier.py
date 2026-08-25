"""Diagnostic one-call group projection through a compiler-owned tensor."""

from __future__ import annotations

from functools import lru_cache

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID
from .tt_lang_owner_aligned_copy import build_tt_lang_owner_aligned_copy
from .tt_lang_program_composition import (
    compile_serialized_child,
    compiled_ttl_program,
    remap_serialized_child,
    retarget_tensor_writes_to_packed_pages,
)
from .tt_lang_resident_grouped_gate_up_forward import (
    build_tt_lang_resident_grouped_gate_up_forward,
)
from .tt_lang_semaphore_scrub import build_tt_lang_semaphore_scrub

ttl = None
tt_lang_operation = None


@lru_cache(maxsize=8)
def build_tt_lang_internal_group_qualifier(
    *, scale: float, tokens: int, hidden_features: int,
    intermediate_features: int, base_k_tiles_per_phase: int,
):
    global ttl, tt_lang_operation
    if ttl is None:
        import ttl as ttl_module
        from tt_torch.tt_lang import tt_lang_operation as decorator
        ttl, tt_lang_operation = ttl_module, decorator
    group = build_tt_lang_resident_grouped_gate_up_forward(
        scale=scale, tokens=tokens, hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        base_k_tiles_per_phase=base_k_tiles_per_phase, worker_count=110,
    )
    copy = build_tt_lang_owner_aligned_copy(
        rows=tokens, columns=2 * intermediate_features + 64
    )
    scrub = build_tt_lang_semaphore_scrub()

    @ttl.operation(grid=BLACKHOLE_P150A_GRID)
    def _probe(x, w, a0, b0, a1, b1, output):
        source = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=1)
        target = ttl.make_dataflow_buffer_like(output, shape=(1, 1), block_count=1)
        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                tx = ttl.copy(x[0:1, 0:1], tile); tx.wait()
        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with target.reserve() as result: result.store(tile)
        @ttl.datamovement()
        def write():
            with target.wait() as tile:
                tx = ttl.copy(tile, output[0:1, 0:1]); tx.wait()

    @tt_lang_operation(
        operation_id=(f"tt_finetune.internal_group_qualifier.{tokens}."
                      f"{hidden_features}.{intermediate_features}.v1"),
        arg_roles=("in",) * 6 + ("out",),
        version_tag="internal-group-qualifier-v1",
    )
    def operation(x, w, a0, b0, a1, b1, output):
        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(x, w, a0, b0, a1, b1, output)
        compiled = compiled_ttl_program(_probe)
        group_child = compile_serialized_child(
            group, (x, w, a0, b0, a1, b1, output)
        )
        copy_child = compile_serialized_child(copy, (output, output))
        scrub_child = compile_serialized_child(scrub, (x, x))
        compiled.program_sequence = [
            scrub_child,
            remap_serialized_child(group_child, {6: 7}),
            remap_serialized_child(copy_child, {0: 7, 1: 6}),
        ]
        compiled.internal_tensors = [{
            "source_operand": 6, "memory": "l1",
            "shape": [tokens, 2 * intermediate_features + 64],
            "data_format": "BFloat16", "layout": "tile",
        }]
        return result
    operation.artifact_format = 4
    operation._ttl_program = _probe
    return operation


@lru_cache(maxsize=8)
def build_tt_lang_internal_group_split_qualifier(
    *, scale: float, tokens: int, hidden_features: int,
    intermediate_features: int, base_k_tiles_per_phase: int,
):
    from .tt_lang_resident_layer_forward_compact_owner import (
        _split_grouped_mlp_writes,
    )
    global ttl, tt_lang_operation
    if ttl is None:
        import ttl as ttl_module
        from tt_torch.tt_lang import tt_lang_operation as decorator
        ttl, tt_lang_operation = ttl_module, decorator
    group = build_tt_lang_resident_grouped_gate_up_forward(
        scale=scale, tokens=tokens, hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        base_k_tiles_per_phase=base_k_tiles_per_phase, worker_count=110,
    )
    copy = build_tt_lang_owner_aligned_copy(
        rows=tokens, columns=intermediate_features
    )
    scrub = build_tt_lang_semaphore_scrub()
    @ttl.operation(grid=BLACKHOLE_P150A_GRID)
    def _probe(x,w,a0,b0,a1,b1,output):
        a=ttl.make_dataflow_buffer_like(x,shape=(1,1),block_count=1)
        b=ttl.make_dataflow_buffer_like(output,shape=(1,1),block_count=1)
        @ttl.datamovement()
        def read():
            with a.reserve() as t:
                z=ttl.copy(x[0:1,0:1],t);z.wait()
        @ttl.compute()
        def compute():
            with a.wait() as t:
                with b.reserve() as o:o.store(t)
        @ttl.datamovement()
        def write():
            with b.wait() as t:
                z=ttl.copy(t,output[0:1,0:1]);z.wait()
    @tt_lang_operation(
        operation_id=f"tt_finetune.internal_group_split_qualifier.{tokens}.{hidden_features}.{intermediate_features}.v1",
        arg_roles=("in",)*6+("out",),version_tag="internal-group-split-qualifier-v1")
    def operation(x,w,a0,b0,a1,b1,output):
        _clear_ttl_compile_wrapper_cache(_probe);result=_probe(x,w,a0,b0,a1,b1,output)
        c=compiled_ttl_program(_probe)
        g=compile_serialized_child(group,(x,w,a0,b0,a1,b1,output))
        g=_split_grouped_mlp_writes(g,gate_page_offset=0,up_page_offset=4*(intermediate_features//32),gate_h_page_offset=0,up_h_page_offset=0)
        g=remap_serialized_child(g,{0:0,1:1,2:2,3:3,4:4,5:5,6:7,7:8,8:6})
        cg=compile_serialized_child(copy,(output,output));cg=retarget_tensor_writes_to_packed_pages(cg,packed_tensor=1,tensor_page_offsets={1:0});cg=remap_serialized_child(cg,{0:7,1:6})
        cu=compile_serialized_child(copy,(output,output));cu=retarget_tensor_writes_to_packed_pages(cu,packed_tensor=1,tensor_page_offsets={1:4*(intermediate_features//32)});cu=remap_serialized_child(cu,{0:8,1:6})
        c.program_sequence=[compile_serialized_child(scrub,(x,x)),g,cg,cu]
        c.internal_tensors=[
            {"source_operand":6,"memory":"l1","shape":[tokens,intermediate_features],"data_format":"BFloat16","layout":"tile"},
            {"source_operand":6,"memory":"l1","shape":[tokens,intermediate_features],"data_format":"BFloat16","layout":"tile"},
        ]
        return result
    operation.artifact_format=4;operation._ttl_program=_probe
    return operation
