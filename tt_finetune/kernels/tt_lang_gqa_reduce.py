"""Exact one-pack BF16 GQA member reduction probe."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID


ttl = None
tt_lang_operation = None


@lru_cache(maxsize=8)
def build_tt_lang_gqa_reduce(
    *, sequence_length: int, query_heads: int, kv_heads: int, head_dim: int,
) -> Callable:
    if sequence_length != 128 or sequence_length % 32 or head_dim % 32:
        raise ValueError("GQA reduction probe requires S=128 and tiled head dim")
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    seq_tiles = sequence_length // 32
    dim_tiles = head_dim // 32
    group = query_heads // kv_heads
    if kv_heads * seq_tiles > 110:
        raise ValueError("GQA reduction probe exceeds the worker grid")

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        import ttl as ttl_module
        from tt_torch.tt_lang import tt_lang_operation as decorator
        ttl = ttl_module
        tt_lang_operation = decorator

    @ttl.operation(grid=BLACKHOLE_P150A_GRID, fp32_dest_acc_en=True)
    def _program(input_tensor, arena):
        def dfb(tensor, blocks):
            return ttl.make_dataflow_buffer_like(
                tensor, shape=(1, 1), block_count=max(1, blocks)
            )
        input_cb = dfb(input_tensor, dim_tiles * group)
        zero = dfb(arena, 1)
        output = dfb(arena, dim_tiles)

        @ttl.datamovement()
        def read():
            with input_cb.reserve() as block:
                tx = ttl.copy(input_tensor[0:1, 0:1], block); tx.wait()

        @ttl.compute()
        def compute():
            with input_cb.wait() as item:
                with output.reserve() as result:
                    result.store(item)
            x, y = ttl.node(dims=2)
            if y * 11 + x < 0:
                with zero.wait() as _: pass

        @ttl.datamovement()
        def write():
            with output.wait() as item:
                tx = ttl.copy(item, arena[0:1, 0:1]); tx.wait()

    variant = hashlib.sha256(
        f"gqa-reduce-v1:{sequence_length}:{query_heads}:{kv_heads}:{head_dim}".encode()
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.gqa_reduce.{variant}",
        arg_roles=("in", "out"),
        version_tag=f"gqa-reduce-v1-{variant}",
    )
    def operation(input_tensor, arena):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(input_tensor, arena)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 3:
                    raise RuntimeError(f"GQA reduce expected 3 CBs, got {len(compiled.cb_configs)}")
                reader_indices = [
                    compiled.kernel_tensor_indices[i]
                    for i, (path, thread) in enumerate(compiled.kernel_paths)
                    if thread != "compute" and "read" in Path(path).name
                ]
                writer_indices = [
                    compiled.kernel_tensor_indices[i]
                    for i, (path, thread) in enumerate(compiled.kernel_paths)
                    if thread != "compute" and "write" in Path(path).name
                ]
                if reader_indices != [[0]] or writer_indices != [[1]]:
                    raise RuntimeError(
                        f"GQA reduce ABI changed: reader={reader_indices}, writer={writer_indices}"
                    )
                replacements = {
                    "@SEQ_TILES@": str(seq_tiles), "@DIM_TILES@": str(dim_tiles),
                    "@KV_HEADS@": str(kv_heads), "@GROUP@": str(group),
                }
                cpp = Path(__file__).with_name("cpp")
                sources = {
                    "read": (cpp / "gqa_reduce_reader.cpp").read_text(),
                    "compute": (cpp / "gqa_reduce_compute.cpp").read_text(),
                    "write": (cpp / "gqa_reduce_writer.cpp").read_text(),
                }
                for kind, source in tuple(sources.items()):
                    for placeholder, value in replacements.items():
                        source = source.replace(placeholder, value)
                    if re.search(r"@[A-Z0-9_]+@", source):
                        raise RuntimeError(f"unexpanded GQA reduce {kind} placeholder")
                    sources[kind] = source
                for path, thread in compiled.kernel_paths:
                    kind = "compute" if thread == "compute" else (
                        "read" if "read" in Path(path).name else "write"
                    )
                    Path(path).write_text(sources[kind], encoding="utf-8")
                for config in compiled.kernel_configs:
                    if hasattr(config, "math_fidelity"):
                        enum_type = type(config.math_fidelity)
                        config.math_fidelity = getattr(enum_type, "HiFi4")
                    if hasattr(config, "math_approx_mode"):
                        config.math_approx_mode = True
                    if hasattr(config, "fp32_dest_acc_en"):
                        config.fp32_dest_acc_en = True
            break
        return result

    operation.group = group
    return operation
