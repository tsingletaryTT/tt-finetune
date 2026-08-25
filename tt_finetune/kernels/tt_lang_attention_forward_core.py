"""Resident exact-candidate softmax + value forward core for Blackhole."""

from __future__ import annotations

import hashlib
import inspect
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID


ttl = None
tt_lang_operation = None


def attention_forward_core_arena_shape(
    *, sequence_length: int, query_heads: int, head_dim: int
) -> tuple[int, int]:
    values = (sequence_length, head_dim)
    if any(value <= 0 or value % 32 for value in values) or query_heads <= 0:
        raise ValueError("attention forward core dimensions must be positive /32")
    seq_tiles = sequence_length // 32
    dim_tiles = head_dim // 32
    # One BF16 head tile row plus two physical BF16 pages for every raw FP32
    # probability tile.  No diagnostic probability copy is part of the
    # production ABI.
    pages = query_heads * seq_tiles * (dim_tiles + 2 * seq_tiles)
    rows = -(-pages // dim_tiles) * 32
    return rows, head_dim


@lru_cache(maxsize=8)
def build_tt_lang_attention_forward_core(
    *, sequence_length: int, query_heads: int, kv_heads: int, head_dim: int,
    value_k_phase_tiles: int = 1,
) -> Callable:
    if sequence_length != 128 or head_dim <= 0 or head_dim % 32:
        raise ValueError("first attention forward core requires S=128 and D /32")
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if query_heads * (sequence_length // 32) > 60:
        raise ValueError("query-row jobs exceed the 60-core attention island")
    seq_tiles = sequence_length // 32
    dim_tiles = head_dim // 32
    if value_k_phase_tiles != 1:
        raise ValueError(
            "exact TTNN P@V contract requires one K tile per spill phase"
        )
    arena_shape = attention_forward_core_arena_shape(
        sequence_length=sequence_length,
        query_heads=query_heads,
        head_dim=head_dim,
    )

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("native attention forward requires TT-XLA") from exc
        ttl = ttl_module
        tt_lang_operation = decorator

    options: dict[str, object] = {
        "grid": BLACKHOLE_P150A_GRID,
        "fp32_dest_acc_en": True,
        "options": "--ttl-matmul-full-fp32",
    }
    if "math_approx_mode" in inspect.signature(ttl.operation).parameters:
        options["math_approx_mode"] = True

    @ttl.operation(**options)
    def _program(scores, value, arena):
        def dfb(tensor, blocks):
            return ttl.make_dataflow_buffer_like(
                tensor, shape=(1, 1), block_count=max(1, blocks)
            )

        score = dfb(scores, seq_tiles)
        score_fp32 = dfb(arena, seq_tiles)
        max_scaler = dfb(arena, 1)
        maximum = dfb(arena, 1)
        exps = dfb(arena, seq_tiles)
        sum_scaler = dfb(arena, 1)
        reciprocal = dfb(arena, 1)
        probability = dfb(arena, seq_tiles)
        probability_saved = dfb(arena, seq_tiles)
        probability_bf16 = dfb(arena, seq_tiles)
        value_full = dfb(value, seq_tiles * dim_tiles)
        partial = dfb(arena, dim_tiles)
        head = dfb(arena, dim_tiles)

        @ttl.datamovement()
        def read():
            with score.reserve() as block:
                tx = ttl.copy(scores[0:1, 0:1], block); tx.wait()
            with value_full.reserve() as block:
                tx = ttl.copy(value[0:1, 0:1], block); tx.wait()

        @ttl.compute()
        def compute():
            with score.wait() as item:
                with head.reserve() as result:
                    result.store(item)
            node_x, node_y = ttl.node(dims=2)
            if node_x + node_y < 0:
                with score_fp32.wait() as _item: pass
                with max_scaler.wait() as _item: pass
                with maximum.wait() as _item: pass
                with exps.wait() as _item: pass
                with sum_scaler.wait() as _item: pass
                with reciprocal.wait() as _item: pass
                with probability.wait() as _item: pass
                with probability_saved.wait() as _item: pass
                with probability_bf16.wait() as _item: pass
                with value_full.wait() as _item: pass
                with partial.wait() as _item: pass

        @ttl.datamovement()
        def write():
            with head.wait() as item:
                tx = ttl.copy(item, arena[0:1, 0:1]); tx.wait()

    variant = hashlib.sha256(
        f"attention-forward-core-v3:{sequence_length}:{query_heads}:{kv_heads}:{head_dim}:phase={value_k_phase_tiles}".encode()
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.attention_forward_core.{variant}",
        arg_roles=("in", "in", "out"),
        version_tag=f"attention-forward-core-v1-{variant}",
    )
    def operation(scores, value, arena):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(scores, value, arena)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 13:
                    raise RuntimeError(
                        f"attention forward core expected 13 CBs, got {len(compiled.cb_configs)}"
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
                if len(reader_indices) != 1 or sorted(reader_indices[0]) != [0, 1]:
                    raise RuntimeError(f"attention forward reader ABI changed: {reader_indices!r}")
                if writer_indices != [[2]]:
                    raise RuntimeError(f"attention forward writer ABI changed: {writer_indices!r}")
                positions = {tensor: pos for pos, tensor in enumerate(reader_indices[0])}
                replacements = {
                    "@SEQ_TILES@": str(seq_tiles),
                    "@DIM_TILES@": str(dim_tiles),
                    "@QUERY_HEADS@": str(query_heads),
                    "@KV_HEADS@": str(kv_heads),
                    "@PV_PHASE_TILES@": str(value_k_phase_tiles),
                    "@FP32_CANONICAL_ATTENTION@": "0",
                }
                cpp = Path(__file__).with_name("cpp")
                sources = {
                    "read": (cpp / "attention_forward_core_reader.cpp").read_text(),
                    "compute": (cpp / "attention_forward_core_compute.cpp").read_text(),
                    "write": (cpp / "attention_forward_core_writer.cpp").read_text(),
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
                        raise RuntimeError(f"unexpanded attention forward {kind} placeholder")
                    sources[kind] = source
                for path, thread in compiled.kernel_paths:
                    kind = "compute" if thread == "compute" else (
                        "read" if "read" in Path(path).name else "write"
                    )
                    Path(path).write_text(sources[kind], encoding="utf-8")

                from ttl.dataflow_buffer import CompilerAllocatedDFBConfig
                for cb_index, blocks in (
                    (1, seq_tiles), (2, 1), (3, 1), (4, seq_tiles),
                    (5, 1), (6, 1), (7, seq_tiles), (8, seq_tiles),
                    (11, dim_tiles),
                ):
                    config = CompilerAllocatedDFBConfig(
                        dfb_index=cb_index, num_tiles=1,
                        data_format="float32", block_count=blocks,
                    )
                    config._cb_index = cb_index
                    compiled.cb_configs[cb_index] = config
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

    operation.output_shape = arena_shape
    operation.head_output_pages = query_heads * seq_tiles * dim_tiles
    operation.probability_raw_pages = query_heads * seq_tiles * seq_tiles * 2
    # Expose the compiled TT-Lang program so the exact suffix can be composed
    # into a larger artifact-v4 resident owner without a PJRT boundary.
    operation._ttl_program = _program
    return operation
