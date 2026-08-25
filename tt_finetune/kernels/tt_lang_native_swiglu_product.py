"""TTNN-native SiLU(gate)*up qualifier."""
from __future__ import annotations
from functools import lru_cache
from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID
from .tt_lang_program_composition import compiled_ttl_program

ttl = None
tt_lang_operation = None

@lru_cache(maxsize=8)
def build_tt_lang_native_swiglu_product(*, tokens: int, features: int):
    global ttl, tt_lang_operation
    if ttl is None:
        import ttl as ttl_module
        from tt_torch.tt_lang import tt_lang_operation as decorator
        ttl, tt_lang_operation = ttl_module, decorator
    @ttl.operation(grid=BLACKHOLE_P150A_GRID)
    def _probe(gate, up, output):
        a=ttl.make_dataflow_buffer_like(gate,shape=(1,1),block_count=1)
        b=ttl.make_dataflow_buffer_like(output,shape=(1,1),block_count=1)
        @ttl.datamovement()
        def read():
            with a.reserve() as t:
                x=ttl.copy(gate[0:1,0:1],t);x.wait()
        @ttl.compute()
        def compute():
            with a.wait() as t:
                with b.reserve() as o:o.store(t)
        @ttl.datamovement()
        def write():
            with b.wait() as t:
                x=ttl.copy(t,output[0:1,0:1]);x.wait()
    @tt_lang_operation(
        operation_id=f"tt_finetune.native_swiglu_product.{tokens}.{features}.v1",
        arg_roles=("in","in","out"), version_tag="native-swiglu-product-v1")
    def operation(gate,up,output):
        _clear_ttl_compile_wrapper_cache(_probe);result=_probe(gate,up,output)
        c=compiled_ttl_program(_probe)
        c.program_sequence=[
            {"operation":"ttnn.silu","operand_indices":[0,3]},
            {"operation":"ttnn.multiply","operand_indices":[3,1,2]},
        ]
        c.internal_tensors=[{"source_operand":0,"memory":"l1","shape":[tokens,features],"data_format":"BFloat16","layout":"tile"}]
        return result
    operation.artifact_format=4;operation._ttl_program=_probe
    return operation
