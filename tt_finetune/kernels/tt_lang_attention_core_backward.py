"""One-call FP32 GQA attention-core backward prototype for Blackhole."""

from __future__ import annotations

import hashlib
import re
import struct
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .attention_block_planner import plan_attention_block
from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID


ttl = None
tt_lang_operation = None


@lru_cache(maxsize=16)
def build_tt_lang_attention_core_backward(
    *,
    sequence_length: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    scale: float,
    math_fidelity: str = "HiFi4",
    matmul_full_fp32: bool = True,
    long_k_phase_tiles: int = 1,
    fp32_dest_acc_en: bool = True,
) -> Callable:
    """Build the resident dQ/dK/dV program for batch-one causal GQA."""

    plan = plan_attention_block(
        batch_size=1,
        sequence_length=sequence_length,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )
    if plan.query_chunks != 1:
        raise ValueError("the first silicon attention schedule requires sequence 128")
    if plan.kv_group_size % 2 != 1:
        raise ValueError("the first exact GQA accumulator requires an odd group")
    if math_fidelity not in {"HiFi2", "HiFi3", "HiFi4"}:
        raise ValueError("math_fidelity must be HiFi2, HiFi3, or HiFi4")
    if long_k_phase_tiles not in {1, 2, 4}:
        raise ValueError("long_k_phase_tiles must be 1, 2, or 4")

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import (
                tt_lang_operation as tt_lang_operation_decorator,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("native attention requires TT-XLA") from exc
        ttl = ttl_module
        tt_lang_operation = tt_lang_operation_decorator

    seq_tiles = sequence_length // 32
    dim_tiles = head_dim // 32
    if seq_tiles % long_k_phase_tiles:
        raise ValueError(
            "sequence tiles must be divisible by long_k_phase_tiles"
        )
    score_tiles = seq_tiles * seq_tiles
    vector_tiles = seq_tiles * dim_tiles
    group = query_heads // kv_heads
    arena_tile_rows = (query_heads + 2 * kv_heads) * seq_tiles
    arena_tile_cols = dim_tiles
    scale_bits = struct.unpack("<I", struct.pack("<f", float(scale)))[0]

    @ttl.operation(
        grid=BLACKHOLE_P150A_GRID,
        fp32_dest_acc_en=fp32_dest_acc_en,
        options=(
            "--ttl-matmul-full-fp32"
            if matmul_full_fp32
            else "--no-ttl-matmul-full-fp32"
        ),
    )
    def _program(probability, output_gradient, query, key, value, arena):
        def dfb(tensor, blocks):
            return ttl.make_dataflow_buffer_like(
                tensor, shape=(1, 1), block_count=blocks
            )

        p = dfb(probability, score_tiles)
        dout = dfb(output_gradient, vector_tiles)
        q = dfb(query, vector_tiles)
        k = dfb(key, vector_tiles)
        v = dfb(value, vector_tiles)
        scaler = dfb(arena, 1)
        pt = dfb(arena, score_tiles)
        vt = dfb(arena, vector_tiles)
        dv = dfb(arena, vector_tiles)
        dp = dfb(arena, score_tiles)
        product = dfb(arena, score_tiles)
        rowsum = dfb(arena, seq_tiles)
        ds = dfb(arena, score_tiles)
        dq = dfb(arena, vector_tiles)
        dst = dfb(arena, score_tiles)
        dk = dfb(arena, vector_tiles)
        dk_a = dfb(arena, vector_tiles)
        dk_b = dfb(arena, vector_tiles)
        dv_a = dfb(arena, vector_tiles)
        dv_b = dfb(arena, vector_tiles)
        dk_final = dfb(arena, vector_tiles)
        dv_final = dfb(arena, vector_tiles)
        vector_scratch = dfb(arena, vector_tiles)
        @ttl.datamovement()
        def read():
            # ABI probes; generated source is replaced after lowering.
            with p.reserve() as block:
                tx = ttl.copy(probability[0:1, 0:1], block)
                tx.wait()
            with p.reserve() as block:
                tx = ttl.copy(output_gradient[0:1, 0:1], block)
                tx.wait()
            with p.reserve() as block:
                tx = ttl.copy(query[0:1, 0:1], block)
                tx.wait()
            with p.reserve() as block:
                tx = ttl.copy(key[0:1, 0:1], block)
                tx.wait()
            with p.reserve() as block:
                tx = ttl.copy(value[0:1, 0:1], block)
                tx.wait()

        @ttl.compute()
        def compute():
            for _ in range(5):
                with p.wait() as item:
                    with dq.reserve() as output:
                        output.store(item)
            node_col, node_row = ttl.node(dims=2)
            if node_row * 11 + node_col < 0:
                with dout.wait() as _unused: pass
                with q.wait() as _unused: pass
                with k.wait() as _unused: pass
                with v.wait() as _unused: pass
                with scaler.wait() as _unused: pass
                with pt.wait() as _unused: pass
                with vt.wait() as _unused: pass
                with dv.wait() as _unused: pass
                with dp.wait() as _unused: pass
                with product.wait() as _unused: pass
                with rowsum.wait() as _unused: pass
                with ds.wait() as _unused: pass
                with dst.wait() as _unused: pass
                with dk.wait() as _unused: pass
                with dk_a.wait() as _unused: pass
                with dk_b.wait() as _unused: pass
                with dv_a.wait() as _unused: pass
                with dv_b.wait() as _unused: pass
                with dk_final.wait() as _unused: pass
                with dv_final.wait() as _unused: pass
                with vector_scratch.wait() as _unused: pass

        @ttl.datamovement()
        def write():
            for _ in range(5):
                with dq.wait() as item:
                    tx = ttl.copy(item, arena[0:1, 0:1])
                    tx.wait()

    variant = hashlib.sha256(
        (
            f"attention-core-bwd-v6-fidelity:{sequence_length}:{query_heads}:"
            f"{kv_heads}:{head_dim}:{scale:.17g}:{math_fidelity}:"
            f"fullfp32={int(matmul_full_fp32)}:phase={long_k_phase_tiles}:"
            f"fp32dest={int(fp32_dest_acc_en)}"
        ).encode("ascii")
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.attention_core_backward.{variant}",
        arg_roles=("in", "in", "in", "in", "in", "out"),
        version_tag=f"attention-core-backward-{variant}",
    )
    def operation(probability, output_gradient, query, key, value, arena):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(
            probability, output_gradient, query, key, value, arena
        )
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 23:
                    raise RuntimeError(
                        "attention backward expected 23 CBs, got "
                        f"{len(compiled.cb_configs)}"
                    )
                reader_indices = [
                    compiled.kernel_tensor_indices[index]
                    for index, (path, thread_type) in enumerate(
                        compiled.kernel_paths
                    )
                    if thread_type != "compute" and "read" in Path(path).name
                ]
                writer_indices = [
                    compiled.kernel_tensor_indices[index]
                    for index, (path, thread_type) in enumerate(
                        compiled.kernel_paths
                    )
                    if thread_type != "compute" and "write" in Path(path).name
                ]
                if (
                    len(reader_indices) != 1
                    or sorted(reader_indices[0]) != [0, 1, 2, 3, 4]
                ):
                    raise RuntimeError(
                        f"attention reader ABI changed: {reader_indices!r}"
                    )
                if writer_indices != [[5]]:
                    raise RuntimeError(
                        f"attention writer ABI changed: {writer_indices!r}"
                    )
                replacements = {
                    "@SEQ_TILES@": str(seq_tiles),
                    "@DIM_TILES@": str(dim_tiles),
                    "@QUERY_HEADS@": str(query_heads),
                    "@KV_HEADS@": str(kv_heads),
                    "@SCALE_BITS@": str(scale_bits),
                    "@LONG_K_PHASE_TILES@": str(long_k_phase_tiles),
                }
                cpp_dir = Path(__file__).with_name("cpp")
                sources = {
                    "read": (cpp_dir / "attention_core_backward_reader.cpp").read_text(encoding="utf-8"),
                    "compute": (cpp_dir / "attention_core_backward_compute.cpp").read_text(encoding="utf-8"),
                    "write": (cpp_dir / "attention_core_backward_writer.cpp").read_text(encoding="utf-8"),
                }
                reader_common_position = {
                    tensor_index: common_position
                    for common_position, tensor_index in enumerate(
                        reader_indices[0]
                    )
                }
                sources["read"] = re.sub(
                    r"get_common_arg_val<uint32_t>\((\d+)\)",
                    lambda match: (
                        "get_common_arg_val<uint32_t>("
                        f"{reader_common_position[int(match.group(1))]})"
                    ),
                    sources["read"],
                )
                sources["read"] = re.sub(
                    r"get_tensor_accessor_args_cta_offset<(\d+), 23>\(\),\s*(\d+)",
                    lambda match: (
                        "get_tensor_accessor_args_cta_offset<"
                        f"{match.group(1)}, 23>(), "
                        f"{reader_common_position[int(match.group(1))]}"
                    ),
                    sources["read"],
                )
                for kind, source in tuple(sources.items()):
                    for placeholder, value_text in replacements.items():
                        source = source.replace(placeholder, value_text)
                    if re.search(r"@[A-Z0-9_]+@", source):
                        raise RuntimeError(
                            f"unexpanded attention {kind} placeholder"
                        )
                    sources[kind] = source
                for path, thread_type in compiled.kernel_paths:
                    kind = (
                        "compute"
                        if thread_type == "compute"
                        else ("read" if "read" in Path(path).name else "write")
                    )
                    Path(path).write_text(sources[kind], encoding="utf-8")
                for config in compiled.kernel_configs:
                    if hasattr(config, "math_fidelity"):
                        enum_type = type(config.math_fidelity)
                        config.math_fidelity = getattr(enum_type, math_fidelity)
                    if hasattr(config, "math_approx_mode"):
                        config.math_approx_mode = False
                    if hasattr(config, "fp32_dest_acc_en"):
                        config.fp32_dest_acc_en = bool(fp32_dest_acc_en)
            break
        return result

    operation.output_tile_shape = (arena_tile_rows, arena_tile_cols)
    operation.plan = plan
    operation.group_size = group
    operation.math_fidelity = math_fidelity
    operation.matmul_full_fp32 = bool(matmul_full_fp32)
    operation.long_k_phase_tiles = long_k_phase_tiles
    operation.fp32_dest_acc_en = bool(fp32_dest_acc_en)
    return operation
