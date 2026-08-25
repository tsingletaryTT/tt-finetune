"""TTNN-compatible multicore FP32 attention dK-head diagnostic."""

from __future__ import annotations
import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .attention_block_planner import plan_attention_block
from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID

ttl = None
tt_lang_operation = None


@lru_cache(maxsize=16)
def build_tt_lang_attention_dk_reference(
    *, sequence_length: int, query_heads: int, kv_heads: int, head_dim: int
) -> Callable:
    plan = plan_attention_block(
        batch_size=1, sequence_length=sequence_length,
        query_heads=query_heads, kv_heads=kv_heads, head_dim=head_dim,
    )
    if plan.query_chunks != 1:
        raise ValueError("reference dK currently requires sequence 128")
    if plan.reference_dk_workers > 110:
        raise ValueError("reference dK schedule exceeds 110 workers")
    seq_tiles, dim_tiles = sequence_length // 32, head_dim // 32

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("native attention requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(
        grid=BLACKHOLE_P150A_GRID,
        fp32_dest_acc_en=True,
        options="--ttl-matmul-full-fp32",
    )
    def _program(query, scaled_ds, output):
        def dfb(tensor, blocks):
            return ttl.make_dataflow_buffer_like(tensor, shape=(1, 1), block_count=blocks)
        q_column = dfb(query, seq_tiles)
        ds = dfb(scaled_ds, seq_tiles * seq_tiles)
        qt = dfb(output, seq_tiles)
        dk_raw = dfb(output, seq_tiles)
        dk = dfb(output, seq_tiles)
        partial = dfb(output, seq_tiles)

        @ttl.datamovement()
        def read():
            with q_column.reserve() as block:
                tx = ttl.copy(query[0:1, 0:1], block)
                tx.wait()
            with q_column.reserve() as block:
                tx = ttl.copy(scaled_ds[0:1, 0:1], block)
                tx.wait()

        @ttl.compute()
        def compute():
            for _ in range(2):
                with q_column.wait() as item:
                    with dk.reserve() as result:
                        result.store(item)
            node_col, node_row = ttl.node(dims=2)
            if node_row * 11 + node_col < 0:
                with ds.wait() as _unused: pass
                with qt.wait() as _unused: pass
                with dk_raw.wait() as _unused: pass
                with partial.wait() as _unused: pass

        @ttl.datamovement()
        def write():
            for _ in range(2):
                with dk.wait() as item:
                    tx = ttl.copy(item, output[0:1, 0:1])
                    tx.wait()

    variant = hashlib.sha256(
        f"attention-dk-reference-v1:{sequence_length}:{query_heads}:{kv_heads}:{head_dim}".encode()
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.attention_dk_reference.{variant}",
        arg_roles=("in", "in", "out"),
        version_tag=f"attention-dk-reference-{variant}",
    )
    def operation(query, scaled_ds, output):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(query, scaled_ds, output)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 6:
                    raise RuntimeError(f"reference dK expected six CBs, got {len(compiled.cb_configs)}")
                readers = [
                    compiled.kernel_tensor_indices[i]
                    for i, (path, thread) in enumerate(compiled.kernel_paths)
                    if thread != "compute" and "read" in Path(path).name
                ]
                writers = [
                    compiled.kernel_tensor_indices[i]
                    for i, (path, thread) in enumerate(compiled.kernel_paths)
                    if thread != "compute" and "write" in Path(path).name
                ]
                if len(readers) != 1 or sorted(readers[0]) != [0, 1]:
                    raise RuntimeError(f"reference dK reader ABI changed: {readers!r}")
                if writers != [[2]]:
                    raise RuntimeError(f"reference dK writer ABI changed: {writers!r}")
                positions = {tensor: pos for pos, tensor in enumerate(readers[0])}
                repl = {
                    "@SEQ_TILES@": str(seq_tiles),
                    "@DIM_TILES@": str(dim_tiles),
                    "@QUERY_HEADS@": str(query_heads),
                }
                cpp = Path(__file__).with_name("cpp")
                sources = {
                    "read": (cpp / "attention_dk_reference_reader.cpp").read_text(),
                    "compute": (cpp / "attention_dk_reference_compute.cpp").read_text(),
                    "write": (cpp / "attention_dk_reference_writer.cpp").read_text(),
                }
                sources["read"] = re.sub(
                    r"get_common_arg_val<uint32_t>\((\d+)\)",
                    lambda m: f"get_common_arg_val<uint32_t>({positions[int(m.group(1))]})",
                    sources["read"],
                )
                sources["read"] = re.sub(
                    r"get_tensor_accessor_args_cta_offset<(\d+), 6>\(\),\s*(\d+)",
                    lambda m: f"get_tensor_accessor_args_cta_offset<{m.group(1)}, 6>(), {positions[int(m.group(1))]}",
                    sources["read"],
                )
                for kind, source in tuple(sources.items()):
                    for key, value in repl.items():
                        source = source.replace(key, value)
                    if re.search(r"@[A-Z0-9_]+@", source):
                        raise RuntimeError(f"unexpanded reference dK {kind} placeholder")
                    sources[kind] = source
                for path, thread in compiled.kernel_paths:
                    kind = "compute" if thread == "compute" else (
                        "read" if "read" in Path(path).name else "write"
                    )
                    Path(path).write_text(sources[kind])
                for config in compiled.kernel_configs:
                    if hasattr(config, "math_fidelity"):
                        enum_type = type(config.math_fidelity)
                        config.math_fidelity = getattr(enum_type, "HiFi4")
                    if hasattr(config, "math_approx_mode"):
                        config.math_approx_mode = False
                    if hasattr(config, "fp32_dest_acc_en"):
                        config.fp32_dest_acc_en = True
            break
        return result

    operation.plan = plan
    return operation
