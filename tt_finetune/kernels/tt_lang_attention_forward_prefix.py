"""Exact-candidate resident RoPE + causal QK forward prefix.

This is the missing compute prefix in front of the accepted resident
softmax/value program.  Its public operation is a qualification boundary; the
same C++ bodies are intended to be embedded in the full one-call owner after
their numerical schedule is accepted on both model geometries.
"""

from __future__ import annotations

import hashlib
import inspect
import re
import struct
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID


ttl = None
tt_lang_operation = None


def attention_forward_prefix_arena_shape(
    *, sequence_length: int, query_heads: int, kv_heads: int, head_dim: int,
) -> tuple[int, int]:
    if sequence_length != 128 or head_dim != 64:
        raise ValueError("first forward prefix requires S=128 and D=64")
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    seq_tiles = sequence_length // 32
    dim_tiles = head_dim // 32
    pages = (
        query_heads * seq_tiles * dim_tiles
        + kv_heads * seq_tiles * dim_tiles
        + query_heads * seq_tiles * seq_tiles
    )
    return pages * 32, 32


@lru_cache(maxsize=8)
def build_tt_lang_attention_forward_prefix(
    *, sequence_length: int, query_heads: int, kv_heads: int,
    head_dim: int, scale: float,
) -> Callable:
    arena_shape = attention_forward_prefix_arena_shape(
        sequence_length=sequence_length,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )
    seq_tiles = sequence_length // 32
    dim_tiles = head_dim // 32
    hidden_tiles = query_heads * dim_tiles
    kv_tiles = kv_heads * dim_tiles
    scale_bits = struct.unpack("<I", struct.pack("<f", float(scale)))[0]

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("native attention prefix requires TT-XLA") from exc
        ttl = ttl_module
        tt_lang_operation = decorator

    options: dict[str, object] = {
        "grid": BLACKHOLE_P150A_GRID,
        "fp32_dest_acc_en": True,
        "options": "--ttl-matmul-full-fp32",
    }
    if "math_approx_mode" in inspect.signature(ttl.operation).parameters:
        options["math_approx_mode"] = False

    @ttl.operation(**options)
    def _program(query, key, cosine, sine, mask, arena):
        def dfb(tensor, blocks):
            return ttl.make_dataflow_buffer_like(
                tensor, shape=(1, 1), block_count=max(1, blocks)
            )

        q = dfb(query, dim_tiles)
        q_cos = dfb(cosine, dim_tiles)
        q_sin = dfb(sine, dim_tiles)
        k = dfb(key, seq_tiles * dim_tiles)
        k_cos = dfb(cosine, seq_tiles * dim_tiles)
        k_sin = dfb(sine, seq_tiles * dim_tiles)
        q_cos_product = dfb(arena, dim_tiles)
        q_sin_product = dfb(arena, dim_tiles)
        q_rotated = dfb(arena, dim_tiles)
        q_saved = dfb(arena, dim_tiles)
        k_cos_product = dfb(arena, seq_tiles * dim_tiles)
        k_sin_product = dfb(arena, seq_tiles * dim_tiles)
        k_rotated = dfb(arena, seq_tiles * dim_tiles)
        k_saved = dfb(arena, seq_tiles * dim_tiles)
        key_transposed = dfb(arena, seq_tiles * dim_tiles)
        score = dfb(arena, seq_tiles)
        scaled_score = dfb(arena, seq_tiles)
        mask_tiles = dfb(mask, seq_tiles)
        masked_score = dfb(arena, seq_tiles)
        qk_partial = dfb(arena, 1)

        @ttl.datamovement()
        def read():
            with q.reserve() as block:
                tx = ttl.copy(query[0:1, 0:1], block); tx.wait()
            with k.reserve() as block:
                tx = ttl.copy(key[0:1, 0:1], block); tx.wait()
            with q_cos.reserve() as block:
                tx = ttl.copy(cosine[0:1, 0:1], block); tx.wait()
            with q_sin.reserve() as block:
                tx = ttl.copy(sine[0:1, 0:1], block); tx.wait()
            with k_cos.reserve() as block:
                tx = ttl.copy(cosine[0:1, 0:1], block); tx.wait()
            with k_sin.reserve() as block:
                tx = ttl.copy(sine[0:1, 0:1], block); tx.wait()
            with mask_tiles.reserve() as block:
                tx = ttl.copy(mask[0:1, 0:1], block); tx.wait()

        @ttl.compute()
        def compute():
            with q.wait() as item:
                with masked_score.reserve() as result:
                    result.store(item)
            node_x, node_y = ttl.node(dims=2)
            if node_x + node_y < 0:
                with q_cos.wait() as _: pass
                with q_sin.wait() as _: pass
                with k.wait() as _: pass
                with k_cos.wait() as _: pass
                with k_sin.wait() as _: pass
                with q_cos_product.wait() as _: pass
                with q_sin_product.wait() as _: pass
                with q_rotated.wait() as _: pass
                with q_saved.wait() as _: pass
                with k_cos_product.wait() as _: pass
                with k_sin_product.wait() as _: pass
                with k_rotated.wait() as _: pass
                with k_saved.wait() as _: pass
                with key_transposed.wait() as _: pass
                with score.wait() as _: pass
                with scaled_score.wait() as _: pass
                with mask_tiles.wait() as _: pass
                with qk_partial.wait() as _: pass

        @ttl.datamovement()
        def write():
            with masked_score.wait() as item:
                tx = ttl.copy(item, arena[0:1, 0:1]); tx.wait()

    variant = hashlib.sha256(
        (
            f"attention-forward-prefix-v9:{sequence_length}:{query_heads}:"
            f"{kv_heads}:{head_dim}:{scale:.17g}"
        ).encode()
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.attention_forward_prefix.{variant}",
        arg_roles=("in", "in", "in", "in", "in", "out"),
        version_tag=f"attention-forward-prefix-v9-{variant}",
    )
    def operation(query, key, cosine, sine, mask, arena):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(query, key, cosine, sine, mask, arena)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 20:
                    raise RuntimeError(
                        "attention forward prefix expected 20 CBs, got "
                        f"{len(compiled.cb_configs)}"
                    )
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
                if len(reader_indices) != 1 or sorted(reader_indices[0]) != [0, 1, 2, 3, 4]:
                    raise RuntimeError(f"attention prefix reader ABI changed: {reader_indices!r}")
                if writer_indices != [[5]]:
                    raise RuntimeError(f"attention prefix writer ABI changed: {writer_indices!r}")
                positions = {tensor: pos for pos, tensor in enumerate(reader_indices[0])}
                replacements = {
                    "@SEQ_TILES@": str(seq_tiles),
                    "@DIM_TILES@": str(dim_tiles),
                    "@HIDDEN_TILES@": str(hidden_tiles),
                    "@KV_TILES@": str(kv_tiles),
                    "@QUERY_HEADS@": str(query_heads),
                    "@KV_HEADS@": str(kv_heads),
                    "@SCALE_BITS@": str(scale_bits),
                    "@FP32_CANONICAL_ATTENTION@": "0",
                    "@PUBLISH_RAW_SCORE_CHECKPOINT@": "0",
                }
                cpp = Path(__file__).with_name("cpp")
                sources = {
                    "read": (cpp / "attention_forward_prefix_reader.cpp").read_text(),
                    "compute": (cpp / "attention_forward_prefix_compute.cpp").read_text(),
                    "write": (cpp / "attention_forward_prefix_writer.cpp").read_text(),
                }
                sources["read"] = re.sub(
                    r"get_common_arg_val<uint32_t>\((\d+)\)",
                    lambda match: f"get_common_arg_val<uint32_t>({positions[int(match.group(1))]})",
                    sources["read"],
                )
                for kind, source in tuple(sources.items()):
                    for placeholder, replacement in replacements.items():
                        source = source.replace(placeholder, replacement)
                    if re.search(r"@[A-Z0-9_]+@", source):
                        raise RuntimeError(f"unexpanded attention prefix {kind} placeholder")
                    sources[kind] = source
                for path, thread in compiled.kernel_paths:
                    kind = "compute" if thread == "compute" else (
                        "read" if "read" in Path(path).name else "write"
                    )
                    Path(path).write_text(sources[kind], encoding="utf-8")

                from ttl.dataflow_buffer import CompilerAllocatedDFBConfig
                capacities = (
                    dim_tiles, dim_tiles, dim_tiles,
                    seq_tiles * dim_tiles, seq_tiles * dim_tiles,
                    seq_tiles * dim_tiles, dim_tiles, dim_tiles, dim_tiles,
                    dim_tiles, seq_tiles * dim_tiles, seq_tiles * dim_tiles,
                    seq_tiles * dim_tiles, seq_tiles * dim_tiles,
                    seq_tiles * dim_tiles, seq_tiles, seq_tiles, seq_tiles,
                    seq_tiles, 1,
                )
                for cb_index, blocks in enumerate(capacities):
                    config = CompilerAllocatedDFBConfig(
                        dfb_index=cb_index, num_tiles=1,
                        data_format="bfloat16", block_count=blocks,
                    )
                    config._cb_index = cb_index
                    compiled.cb_configs[cb_index] = config
                partial = CompilerAllocatedDFBConfig(
                    dfb_index=19, num_tiles=1,
                    data_format="float32", block_count=1,
                )
                partial._cb_index = 19
                compiled.cb_configs[19] = partial
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

    operation.output_shape = arena_shape
    operation.query_pages = query_heads * seq_tiles * dim_tiles
    operation.key_pages = kv_heads * seq_tiles * dim_tiles
    operation.score_pages = query_heads * seq_tiles * seq_tiles
    return operation
