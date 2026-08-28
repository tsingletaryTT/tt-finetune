"""One-call exact resident RoPE -> causal attention forward owner."""

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

_PROJECTION_RANGE = {"start": [0, 0], "end": [4, 9]}
_ATTENTION_RANGE = {"start": [5, 0], "end": [10, 9]}
_ALL_WORKERS_RANGE = {"start": [0, 0], "end": [10, 9]}


def attention_forward_resident_arena_shape(
    *, sequence_length: int, query_heads: int, kv_heads: int, head_dim: int,
) -> tuple[int, int]:
    if sequence_length != 128 or head_dim != 64:
        raise ValueError("first resident attention forward requires S=128 and D=64")
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    seq_tiles = sequence_length // 32
    dim_tiles = head_dim // 32
    q_pages = query_heads * seq_tiles * dim_tiles
    k_pages = kv_heads * seq_tiles * dim_tiles
    head_pages = q_pages
    probability_pages = 2 * query_heads * seq_tiles * seq_tiles
    return (q_pages + k_pages + head_pages + probability_pages) * 32, 32


def _resident_compute_source(
    replacements: dict[str, str], *, projection_rope_prefix_only: bool = False,
) -> str:
    cpp = Path(__file__).with_name("cpp")
    prefix = (cpp / "attention_forward_prefix_compute.cpp").read_text()
    core_helpers = ""
    tail = ""
    if not projection_rope_prefix_only:
        core = (cpp / "attention_forward_core_compute.cpp").read_text()
        namespace_start = core.index("namespace {")
        kernel_start = core.index("void kernel_main()")
        core_helpers = core[namespace_start:kernel_start]
        tail = core[kernel_start:].replace(
            "void kernel_main()", "FORCE_INLINE void run_attention_tail()", 1
        )
        cb_mapping = {0: 18, **{index: 19 + index for index in range(1, 13)}}
        tail = re.sub(
            r"get_compile_time_arg_val\((\d+)\)",
            lambda match: str(cb_mapping[int(match.group(1))]),
            tail,
        )
    extra_includes = """
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/softmax.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
"""
    prefix_kernel = prefix.index("void kernel_main()")
    source = (
        prefix[:prefix_kernel]
        + (extra_includes if not projection_rope_prefix_only else "")
        + core_helpers
        + tail
        + "\n"
        + prefix[prefix_kernel:]
    )
    if projection_rope_prefix_only:
        attention_start = source.index("#if @FP32_CANONICAL_ATTENTION@")
        source = (
            source[:attention_start]
            + "    CircularBuffer qrot(cb_q_rotated), krot(cb_k_rotated);\n"
            + "    qrot.pop_front(dim_tiles);\n"
            + "    krot.pop_front(vector_tiles);\n"
            + "}\n"
        )
    else:
        final_brace = source.rfind("\n}")
        if final_brace < 0:
            raise RuntimeError("forward-prefix kernel body changed")
        source = (
            source[:final_brace]
            + "\n    run_attention_tail();"
            + source[final_brace:]
        )
    for marker, value in replacements.items():
        source = source.replace(marker, value)
    if re.search(r"@[A-Z0-9_]+@", source):
        raise RuntimeError("unexpanded resident attention compute placeholder")
    return source


@lru_cache(maxsize=16)
def build_tt_lang_attention_forward_resident(
    *, sequence_length: int, query_heads: int, kv_heads: int,
    head_dim: int, scale: float, projection_handoff: bool = False,
    canonical_moreh_softmax: bool = False,
    publish_score_checkpoint: bool = False,
    fp32_canonical_attention: bool = True,
) -> Callable:
    arena_shape = attention_forward_resident_arena_shape(
        sequence_length=sequence_length,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )
    seq_tiles = sequence_length // 32
    dim_tiles = head_dim // 32
    hidden_tiles = query_heads * dim_tiles
    kv_tiles = kv_heads * dim_tiles
    if query_heads * seq_tiles > 60:
        raise ValueError("resident attention forward exceeds 60 active rows")
    scale_bits = struct.unpack("<I", struct.pack("<f", float(scale)))[0]
    canonical_pow2_scale = bool(
        canonical_moreh_softmax and float(scale) == 0.125
    )

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("resident attention forward requires TT-XLA") from exc
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
    def _program(query, key, value, cosine, sine, mask, arena):
        def dfb(tensor, blocks):
            return ttl.make_dataflow_buffer_like(
                tensor, shape=(1, 1), block_count=max(1, blocks)
            )

        b0 = dfb(query, dim_tiles); b1 = dfb(cosine, dim_tiles)
        b2 = dfb(sine, dim_tiles); b3 = dfb(key, seq_tiles * dim_tiles)
        b4 = dfb(cosine, seq_tiles * dim_tiles); b5 = dfb(sine, seq_tiles * dim_tiles)
        b6 = dfb(arena, dim_tiles); b7 = dfb(arena, dim_tiles)
        b8 = dfb(arena, dim_tiles); b9 = dfb(arena, dim_tiles)
        b10 = dfb(arena, seq_tiles * dim_tiles); b11 = dfb(arena, seq_tiles * dim_tiles)
        b12 = dfb(arena, seq_tiles * dim_tiles); b13 = dfb(arena, seq_tiles * dim_tiles)
        b14 = dfb(arena, seq_tiles * dim_tiles); b15 = dfb(arena, seq_tiles)
        b16 = dfb(arena, seq_tiles); b17 = dfb(mask, seq_tiles)
        b18 = dfb(arena, seq_tiles); b19 = dfb(arena, 1)
        b20 = dfb(arena, seq_tiles); b21 = dfb(arena, 1)
        b22 = dfb(arena, 1); b23 = dfb(arena, seq_tiles)
        b24 = dfb(arena, 1); b25 = dfb(arena, 1)
        b26 = dfb(arena, seq_tiles); b27 = dfb(arena, seq_tiles)
        b28 = dfb(arena, seq_tiles); b29 = dfb(value, seq_tiles * dim_tiles)
        b30 = dfb(arena, dim_tiles); b31 = dfb(arena, dim_tiles)

        @ttl.datamovement()
        def read():
            with b0.reserve() as block:
                tx = ttl.copy(query[0:1, 0:1], block); tx.wait()
            with b3.reserve() as block:
                tx = ttl.copy(key[0:1, 0:1], block); tx.wait()
            with b29.reserve() as block:
                tx = ttl.copy(value[0:1, 0:1], block); tx.wait()
            with b1.reserve() as block:
                tx = ttl.copy(cosine[0:1, 0:1], block); tx.wait()
            with b2.reserve() as block:
                tx = ttl.copy(sine[0:1, 0:1], block); tx.wait()
            with b17.reserve() as block:
                tx = ttl.copy(mask[0:1, 0:1], block); tx.wait()

        @ttl.compute()
        def compute():
            with b0.wait() as item:
                with b31.reserve() as result:
                    result.store(item)
            x, y = ttl.node(dims=2)
            if x + y < 0:
                with b1.wait() as _: pass
                with b2.wait() as _: pass
                with b3.wait() as _: pass
                with b4.wait() as _: pass
                with b5.wait() as _: pass
                with b6.wait() as _: pass
                with b7.wait() as _: pass
                with b8.wait() as _: pass
                with b9.wait() as _: pass
                with b10.wait() as _: pass
                with b11.wait() as _: pass
                with b12.wait() as _: pass
                with b13.wait() as _: pass
                with b14.wait() as _: pass
                with b15.wait() as _: pass
                with b16.wait() as _: pass
                with b17.wait() as _: pass
                with b18.wait() as _: pass
                with b19.wait() as _: pass
                with b20.wait() as _: pass
                with b21.wait() as _: pass
                with b22.wait() as _: pass
                with b23.wait() as _: pass
                with b24.wait() as _: pass
                with b25.wait() as _: pass
                with b26.wait() as _: pass
                with b27.wait() as _: pass
                with b28.wait() as _: pass
                with b29.wait() as _: pass
                with b30.wait() as _: pass

        @ttl.datamovement()
        def write():
            with b31.wait() as item:
                tx = ttl.copy(item, arena[0:1, 0:1]); tx.wait()

    variant = hashlib.sha256(
        (
            f"attention-forward-resident-v19-fp32-pv-partial:{sequence_length}:{query_heads}:"
            f"{kv_heads}:{head_dim}:{scale:.17g}:{int(projection_handoff)}:"
            f"moreh={int(canonical_moreh_softmax)}:"
            f"score={int(publish_score_checkpoint)}:"
            f"fp32={int(fp32_canonical_attention)}"
        ).encode()
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.attention_forward_resident.{variant}",
        arg_roles=("in", "in", "in", "in", "in", "in", "out"),
        version_tag=f"attention-forward-resident-v19-{variant}",
    )
    def operation(query, key, value, cosine, sine, mask, arena):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(query, key, value, cosine, sine, mask, arena)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 32:
                    raise RuntimeError(
                        f"resident attention forward expected 32 CBs, got {len(compiled.cb_configs)}"
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
                if len(reader_indices) != 1 or sorted(reader_indices[0]) != list(range(6)):
                    raise RuntimeError(f"resident attention reader ABI changed: {reader_indices!r}")
                if writer_indices != [[6]]:
                    raise RuntimeError(f"resident attention writer ABI changed: {writer_indices!r}")
                positions = {tensor: pos for pos, tensor in enumerate(reader_indices[0])}
                replacements = {
                    "@SEQ_TILES@": str(seq_tiles), "@DIM_TILES@": str(dim_tiles),
                    "@HIDDEN_TILES@": str(hidden_tiles), "@KV_TILES@": str(kv_tiles),
                    "@QUERY_HEADS@": str(query_heads), "@KV_HEADS@": str(kv_heads),
                    "@SCALE_BITS@": str(scale_bits), "@PV_PHASE_TILES@": "1",
                    "@FP32_CANONICAL_ATTENTION@": (
                        "1" if fp32_canonical_attention else "0"
                    ),
                    "@CANONICAL_MOREH_SOFTMAX@": (
                        "1" if canonical_moreh_softmax else "0"
                    ),
                    "@CANONICAL_POW2_SCALE@": (
                        "1" if canonical_pow2_scale else "0"
                    ),
                    "@SCALE_SHIFT@": "3",
                    "@PROJECTION_ROPE_PREFIX_ONLY@": "0",
                    "@PUBLISH_RAW_SCORE_CHECKPOINT@": (
                        "1" if publish_score_checkpoint else "0"
                    ),
                    "@ATTENTION_CHECKPOINT_CB@": "27",
                    "@SOFTMAX_TAIL_MASK_CB@": "28",
                    "@PROBABILITY_BF16_CB@": (
                        "19"
                        if canonical_moreh_softmax and not fp32_canonical_attention
                        else "28"
                    ),
                    "@COMPOSED_INVOCATION_BARRIER@": "0",
                    "@PUBLISH_TARGET_MAILBOX@": "",
                    "@RESET_O_GO@": "",
                    "@HEAD_TO_O_HANDOFF@": "",
                }
                from ttl.dataflow_buffer import CompilerAllocatedDFBConfig
                capacities = (
                    (
                        max(
                            dim_tiles,
                            -(-(seq_tiles * hidden_tiles) // 50),
                        )
                        if projection_handoff else dim_tiles
                    ), dim_tiles, dim_tiles,
                    seq_tiles * dim_tiles, seq_tiles * dim_tiles, seq_tiles * dim_tiles,
                    dim_tiles, dim_tiles, dim_tiles, dim_tiles,
                    seq_tiles * dim_tiles, seq_tiles * dim_tiles,
                    seq_tiles * dim_tiles, seq_tiles * dim_tiles,
                    seq_tiles * dim_tiles, seq_tiles, seq_tiles, seq_tiles, seq_tiles,
                    1, seq_tiles, 1, 1, seq_tiles, 1, 1, seq_tiles, seq_tiles,
                    seq_tiles, seq_tiles * dim_tiles, dim_tiles, dim_tiles,
                )
                capacities = list(capacities)
                # Canonical FP32 attention reuses these local buffers across
                # sequential QK, softmax and P@V epochs. Keep one tile beyond
                # a full generation so producer modulo pointers cannot alias
                # the empty state after reuse.
                capacities[18] = max(capacities[18], seq_tiles + 1)
                # Q is promoted into this FIFO once, then the canonical simple
                # BMM schedule appends one Q tile for every (N, K) phase before
                # consuming the original resident generation.  Budget both
                # generations plus the wraparound guard explicitly.  The old
                # ``2 * dim_tiles + seq_tiles`` expression happened to fit
                # friendly shapes, but deadlocked at S=128/D=64 on the final
                # streamed Q tile (9 received credits in a 9-tile CB).
                capacities[19] = max(
                    capacities[19],
                    dim_tiles + seq_tiles * dim_tiles + 1,
                )
                capacities[20] = max(
                    capacities[20],
                    2 * seq_tiles * dim_tiles + seq_tiles + 1,
                )
                capacities[23] = max(capacities[23], 2 * seq_tiles + 1)
                if canonical_moreh_softmax and not fp32_canonical_attention:
                    # CB27 is Moreh's one-tile temporary, then the complete
                    # probability checkpoint generation. Keep both lifetimes
                    # physically non-aliasing across the partial pointer epoch.
                    capacities[27] = max(capacities[27], seq_tiles + 1)
                capacities[30] = max(
                    capacities[30],
                    (
                        seq_tiles + dim_tiles + 1
                        if canonical_moreh_softmax else dim_tiles + 1
                    ),
                )
                capacities[28] = max(capacities[28], 1)
                fp32_cbs = (
                    {
                        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30,
                    }
                    if fp32_canonical_attention
                    else (
                        {19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30}
                        if canonical_moreh_softmax
                        else {19, 20, 21, 22, 23, 24, 25, 26, 27, 30}
                    )
                )
                for cb_index, blocks in enumerate(capacities):
                    config = CompilerAllocatedDFBConfig(
                        dfb_index=cb_index, num_tiles=1,
                        data_format=(
                            "float32" if cb_index in fp32_cbs else "bfloat16"
                        ),
                        block_count=blocks,
                    )
                    config._cb_index = cb_index
                    compiled.cb_configs[cb_index] = config
                cpp = Path(__file__).with_name("cpp")
                if projection_handoff:
                    generated = Path(compiled.kernel_paths[0][0]).parent
                    attention_compute = _resident_compute_source(replacements).replace(
                        "get_absolute_logical_y() * 11 + get_absolute_logical_x()",
                        "get_absolute_logical_y() * 6 + "
                        "(get_absolute_logical_x() - 5)",
                    )
                    attention_writer = (
                        cpp / "attention_forward_resident_writer.cpp"
                    ).read_text().replace(
                        "get_absolute_logical_y() * 11 + get_absolute_logical_x()",
                        "get_absolute_logical_y() * 6 + "
                        "(get_absolute_logical_x() - 5)",
                    )
                    source_specs = (
                        (
                            "attention_forward_projection_handoff_reader.cpp",
                            "noc",
                            (cpp / "attention_forward_projection_handoff_reader.cpp").read_text(),
                        ),
                        (
                            "attention_forward_projection_handoff_compute.cpp",
                            "compute",
                            (cpp / "attention_forward_projection_handoff_compute.cpp").read_text(),
                        ),
                        (
                            "attention_forward_projection_handoff_writer.cpp",
                            "noc",
                            (cpp / "attention_forward_projection_handoff_writer.cpp").read_text(),
                        ),
                        (
                            "attention_forward_handoff_resident_reader.cpp",
                            "noc",
                            (cpp / "attention_forward_handoff_resident_reader.cpp").read_text(),
                        ),
                        (
                            "attention_forward_handoff_resident_compute.cpp",
                            "compute",
                            attention_compute,
                        ),
                        (
                            "attention_forward_handoff_resident_writer.cpp",
                            "noc",
                            attention_writer,
                        ),
                    )
                    paths: list[tuple[str, str]] = []
                    for filename, thread, source in source_specs:
                        for marker, value in replacements.items():
                            source = source.replace(marker, value)
                        if re.search(r"@[A-Z0-9_]+@", source):
                            raise RuntimeError(
                                f"unexpanded projection handoff placeholder in {filename}"
                            )
                        path = generated / filename
                        path.write_text(source, encoding="utf-8")
                        paths.append((str(path), thread))

                    import ttnn

                    compiled.kernel_paths = paths
                    compiled.kernel_configs = [
                        ttnn.ReaderConfigDescriptor(),
                        ttnn.ComputeConfigDescriptor(
                            math_fidelity=ttnn.MathFidelity.HiFi4,
                            math_approx_mode=True,
                            fp32_dest_acc_en=True,
                        ),
                        ttnn.WriterConfigDescriptor(),
                        ttnn.ReaderConfigDescriptor(),
                        ttnn.ComputeConfigDescriptor(
                            math_fidelity=ttnn.MathFidelity.HiFi4,
                            math_approx_mode=True,
                            fp32_dest_acc_en=True,
                        ),
                        ttnn.WriterConfigDescriptor(),
                    ]
                    compiled.kernel_tensor_indices = [
                        [0, 1, 2], [], [], [3, 4, 5], [], [6],
                    ]
                    compiled.kernel_core_ranges = [
                        _PROJECTION_RANGE,
                        _PROJECTION_RANGE,
                        _PROJECTION_RANGE,
                        _ATTENTION_RANGE,
                        _ATTENTION_RANGE,
                        _ATTENTION_RANGE,
                    ]
                    compiled.kernel_noc_roles = [
                        "reader", None, "writer", "reader", None, "writer",
                    ]
                    # The direct handoff uses the receiver CB's initial L1
                    # address. Give every CB one compiler allocation spanning
                    # both islands so the same local address is valid on all
                    # 110 workers.
                    compiled.cb_core_ranges = [_ALL_WORKERS_RANGE] * 32
                    compiled.semaphore_configs = [
                        {
                            "id": semaphore,
                            "core_type": "worker",
                            "core_range": _ALL_WORKERS_RANGE,
                            "initial_value": 0,
                        }
                        for semaphore in range(5)
                    ]
                    compiled.num_pipe_sync_semaphores = 5
                    compiled.kernel_arg_specs = [[] for _ in range(6)]
                    continue
                sources = {
                    "read": (cpp / "attention_forward_resident_reader.cpp").read_text(),
                    "compute": _resident_compute_source(replacements),
                    "write": (cpp / "attention_forward_resident_writer.cpp").read_text(),
                }
                sources["read"] = re.sub(
                    r"get_common_arg_val<uint32_t>\((\d+)\)",
                    lambda m: f"get_common_arg_val<uint32_t>({positions[int(m.group(1))]})",
                    sources["read"],
                )
                for kind in ("read", "write"):
                    source = sources[kind]
                    for marker, value in replacements.items():
                        source = source.replace(marker, value)
                    if re.search(r"@[A-Z0-9_]+@", source):
                        raise RuntimeError(f"unexpanded resident attention {kind} placeholder")
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

    operation.output_shape = arena_shape
    operation.query_pages = query_heads * seq_tiles * dim_tiles
    operation.key_pages = kv_heads * seq_tiles * dim_tiles
    operation.head_pages = operation.query_pages
    operation.probability_raw_pages = 2 * query_heads * seq_tiles * seq_tiles
    operation.intermediate_dram_bytes = 0
    operation.projection_handoff = projection_handoff
    operation.publish_score_checkpoint = publish_score_checkpoint
    operation.fp32_canonical_attention = fp32_canonical_attention
    return operation
