"""Exact TT-Metal RMSNorm input-gradient primitive.

This is deliberately a small ownership primitive rather than a public training
API.  It preserves the FP32/BF16 edges emitted by the accepted StableHLO path
and is intended to be embedded into the resident attention/layer owner.
"""

from __future__ import annotations

import hashlib
import os
import re
import struct
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID


ttl = None
tt_lang_operation = None


def _float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _compact_compute_source(source: str) -> str:
    """Use one streaming FP32 correction window without changing math.

    The StableHLO contract explicitly widens BF16 x/weighted values before
    multiplication. BF16 -> FP32 is exact, so full-FP32 unpack reproduces
    that edge without materializing two hidden-width conversion FIFOs. Once
    the row coefficient is known, direct/correction/final-pack are independent
    per hidden tile and can use depth-two windows instead of three full rows.
    """

    marker = "void kernel_main() {\n"
    if source.count(marker) != 1:
        raise RuntimeError("RMSNorm compact compute marker changed")
    prefix = source.split(marker, 1)[0]
    if '#include "api/compute/eltwise_binary.h"' not in prefix:
        prefix = prefix.replace(
            '#include "api/compute/eltwise_binary_sfpu.h"\n',
            '#include "api/compute/eltwise_binary.h"\n'
            '#include "api/compute/eltwise_binary_sfpu.h"\n',
        )
    prefix += r'''
FORCE_INLINE void exact_bf16_add_tile(
    uint32_t left_id, uint32_t right_id, uint32_t output_id) {
    CircularBuffer left(left_id), right(right_id), output(output_id);
    left.wait_front(1);
    right.wait_front(1);
    output.reserve_back(1);
    binary_op_init_common(left_id, right_id, output_id);
    tile_regs_acquire();
    add_tiles_init(left_id, right_id);
    add_tiles(left_id, right_id, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, output_id, 0);
    tile_regs_release();
    output.push_back(1);
    right.pop_front(1);
    left.pop_front(1);
}

'''
    body = r'''void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dy = get_compile_time_arg_val(1);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(2);
    constexpr uint32_t cb_rstd = get_compile_time_arg_val(3);
    constexpr uint32_t cb_weighted = get_compile_time_arg_val(4);
    constexpr uint32_t cb_product = get_compile_time_arg_val(5);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(6);
    constexpr uint32_t cb_correlation = get_compile_time_arg_val(7);
    constexpr uint32_t cb_coefficient = get_compile_time_arg_val(8);
    constexpr uint32_t cb_work_a = get_compile_time_arg_val(9);
    constexpr uint32_t cb_work_b = get_compile_time_arg_val(10);
    constexpr uint32_t cb_direct = get_compile_time_arg_val(11);
    constexpr uint32_t cb_correction = get_compile_time_arg_val(12);
    constexpr uint32_t cb_output = get_compile_time_arg_val(18);
    constexpr uint32_t cb_row_broadcast = get_compile_time_arg_val(19);
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@;
    constexpr uint32_t token_tiles = @TOKEN_TILES@;
    constexpr uint32_t neg_half_bits = @NEG_HALF_BITS@;
    constexpr float hidden_scalar = @HIDDEN_FLOAT@;
    constexpr uint32_t two_bits = @TWO_BITS@;
    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= token_tiles) return;

    compute_kernel_hw_startup(cb_x, cb_scaler, cb_output);

    // Preserve BF16(dY*gamma). x and weighted then enter DEST through the
    // exact BF16->FP32 unpack mode used by the materialized reference edge.
    multiply_row_broadcast(
        cb_dy, cb_weight, cb_row_broadcast, cb_weighted, hidden_tiles);
    binary_tiles<0>(cb_weighted, cb_x, cb_product, hidden_tiles);
    compute_kernel_lib::reduce<
        PoolType::SUM, ReduceDim::REDUCE_ROW,
        cb_product, cb_scaler, cb_correlation,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT,
        ReduceFp32Mode::Fast>(
            compute_kernel_lib::ReduceInputBlockShape::row(hidden_tiles));

    scalar_multiply(cb_correlation, cb_coefficient, 1, neg_half_bits);
    CircularBuffer correlation(cb_correlation);
    correlation.pop_front(1);
    cube_tiles(cb_rstd, cb_work_a, hidden_tiles);
    sfpu_multiply_col_broadcast(
        cb_work_a, cb_coefficient, cb_direct, cb_work_b, hidden_tiles);
    divide_by_scalar(
        cb_work_b, cb_work_a, hidden_tiles, hidden_scalar, true);

    // All remaining work is tile-local. Keep exactly the same FP32
    // operations and one final BF16 pack, but release each tile immediately.
    CircularBuffer x(cb_x), weighted(cb_weighted), rstd(cb_rstd);
    CircularBuffer scaled(cb_work_a);
    for (uint32_t tile = 0; tile < hidden_tiles; ++tile) {
        binary_tiles<0>(cb_weighted, cb_rstd, cb_direct, 1);
        derivative_times_scaled(
            cb_x, cb_work_a, cb_correction, 1, two_bits);
        binary_tiles<1, true>(
            cb_direct, cb_correction, cb_output, 1, true);
        weighted.pop_front(1);
        rstd.pop_front(1);
        x.pop_front(1);
        scaled.pop_front(1);
    }
    // Match the reader/writer tail fence: the next physical program must not
    // reuse this core's compute RISC before BRISC has made output visible.
    CircularBuffer compute_release(get_compile_time_arg_val(14));
    compute_release.wait_front(1);
    compute_release.pop_front(1);
}
'''
    return prefix + body


@lru_cache(maxsize=16)
def build_tt_lang_rmsnorm_backward(
    *, sequence_length: int, hidden_size: int, stage: str = "final",
    compact_l1: bool = False,
) -> Callable:
    if sequence_length <= 0 or sequence_length % 32:
        raise ValueError("RMSNorm backward requires a positive 32-aligned sequence length")
    if hidden_size <= 0 or hidden_size % 32:
        raise ValueError("RMSNorm backward requires a positive 32-aligned hidden size")
    token_tiles = sequence_length // 32
    hidden_tiles = hidden_size // 32
    if token_tiles > 110:
        raise ValueError("RMSNorm backward exceeds the 110-worker Blackhole grid")
    if stage not in {
        "weighted", "direct", "correction", "scaled", "correction_fp32", "final"
    }:
        raise ValueError(
            "invalid RMSNorm backward diagnostic stage"
        )
    if compact_l1 and stage != "final":
        raise ValueError("compact RMSNorm backward supports only the final stage")
    epoch_reset = os.environ.get("TT_FINETUNE_RMS_EPOCH_RESET") == "1"
    dst_full_sync = os.environ.get("TT_FINETUNE_RMS_DST_FULL_SYNC") == "1"
    reduce_uninit = os.environ.get("TT_FINETUNE_RMS_REDUCE_UNINIT") == "1"
    clear_dest = os.environ.get("TT_FINETUNE_RMS_CLEAR_DEST") == "1"
    accurate_reduce = os.environ.get("TT_FINETUNE_RMS_ACCURATE_REDUCE") == "1"
    pre_reduce_uninit = (
        os.environ.get("TT_FINETUNE_RMS_PRE_REDUCE_UNINIT") == "1"
    )
    zeroacc_all = os.environ.get("TT_FINETUNE_RMS_ZEROACC_ALL") == "1"

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        import ttl as ttl_module
        from tt_torch.tt_lang import tt_lang_operation as decorator

        ttl = ttl_module
        tt_lang_operation = decorator

    @ttl.operation(grid=BLACKHOLE_P150A_GRID, fp32_dest_acc_en=True)
    def _program(x, dy, weight, rstd, seed, output):
        def dfb(tensor, blocks):
            return ttl.make_dataflow_buffer_like(
                tensor, shape=(1, 1), block_count=max(1, blocks)
            )

        x_cb = dfb(x, hidden_tiles)
        dy_cb = dfb(dy, hidden_tiles)
        weight_cb = dfb(weight, hidden_tiles)
        rstd_cb = dfb(rstd, hidden_tiles)
        weighted = dfb(output, hidden_tiles)
        x_fp32 = dfb(rstd, hidden_tiles)
        weighted_fp32 = dfb(rstd, hidden_tiles)
        product = dfb(rstd, hidden_tiles)
        scaler = dfb(rstd, 1)
        correlation = dfb(rstd, 1)
        direct = dfb(rstd, hidden_tiles)
        coefficient = dfb(rstd, 1)
        rstd_cube = dfb(rstd, hidden_tiles)
        scaled = dfb(rstd, hidden_tiles)
        derivative = dfb(rstd, hidden_tiles)
        correction = dfb(rstd, hidden_tiles)
        summed = dfb(rstd, hidden_tiles)
        broadcast = dfb(rstd, 1)
        result = dfb(output, hidden_tiles)
        row_broadcast = dfb(output, 1)
        output_seed = dfb(seed, 1)

        @ttl.datamovement()
        def read():
            with x_cb.reserve() as block:
                tx = ttl.copy(x[0:1, 0:1], block)
                tx.wait()
            with dy_cb.reserve() as block:
                tx = ttl.copy(dy[0:1, 0:1], block)
                tx.wait()
            with weight_cb.reserve() as block:
                tx = ttl.copy(weight[0:1, 0:1], block)
                tx.wait()
            with rstd_cb.reserve() as block:
                tx = ttl.copy(rstd[0:1, 0:1], block)
                tx.wait()
            with output_seed.reserve() as block:
                tx = ttl.copy(seed[0:1, 0:1], block)
                tx.wait()

        @ttl.compute()
        def compute():
            with output_seed.wait() as item:
                with result.reserve() as target:
                    target.store(item)
            # Keep every otherwise-unused CB in the compiler's allocation and
            # ABI.  The generated kernels are replaced below by native C++.
            cx, cy = ttl.node(dims=2)
            if cy * 11 + cx < 0:
                with dy_cb.wait() as _: pass
                with weight_cb.wait() as _: pass
                with rstd_cb.wait() as _: pass
                with weighted.wait() as _: pass
                with x_fp32.wait() as _: pass
                with weighted_fp32.wait() as _: pass
                with product.wait() as _: pass
                with scaler.wait() as _: pass
                with correlation.wait() as _: pass
                with direct.wait() as _: pass
                with coefficient.wait() as _: pass
                with rstd_cube.wait() as _: pass
                with scaled.wait() as _: pass
                with derivative.wait() as _: pass
                with correction.wait() as _: pass
                with summed.wait() as _: pass
                with broadcast.wait() as _: pass
                with row_broadcast.wait() as _: pass

        @ttl.datamovement()
        def write():
            with result.wait() as item:
                tx = ttl.copy(item, output[0:1, 0:1])
                tx.wait()

    variant = hashlib.sha256(
        (
            f"rmsnorm-backward-v7-gamma-broadcast:"
            f"{sequence_length}:{hidden_size}:{stage}:"
            f"compact={int(compact_l1)}:epoch-reset={int(epoch_reset)}:"
            f"dst-full-sync={int(dst_full_sync)}"
            f":reduce-uninit={int(reduce_uninit)}"
            f":clear-dest={int(clear_dest)}"
            f":accurate-reduce={int(accurate_reduce)}"
            f":pre-reduce-uninit={int(pre_reduce_uninit)}"
            f":zeroacc-all={int(zeroacc_all)}"
        ).encode()
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.rmsnorm_backward.{variant}",
        arg_roles=("in", "in", "in", "in", "in", "out"),
        version_tag=f"rmsnorm-backward-v1-{variant}",
    )
    def operation(x, dy, weight, rstd, seed, output):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(x, dy, weight, rstd, seed, output)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 21:
                    raise RuntimeError(
                        f"RMSNorm backward expected 21 CBs, got {len(compiled.cb_configs)}"
                    )
                reader_indices = [
                    compiled.kernel_tensor_indices[index]
                    for index, (path, thread) in enumerate(compiled.kernel_paths)
                    if thread != "compute" and "read" in Path(path).name
                ]
                writer_indices = [
                    compiled.kernel_tensor_indices[index]
                    for index, (path, thread) in enumerate(compiled.kernel_paths)
                    if thread != "compute" and "write" in Path(path).name
                ]
                if len(reader_indices) != 1 or sorted(reader_indices[0]) != [0, 1, 2, 3, 4]:
                    raise RuntimeError(f"RMSNorm backward reader ABI changed: {reader_indices!r}")
                if writer_indices != [[5]]:
                    raise RuntimeError(f"RMSNorm backward writer ABI changed: {writer_indices!r}")
                read_positions = {
                    tensor: position
                    for position, tensor in enumerate(reader_indices[0])
                }
                replacements = {
                    "@HIDDEN_TILES@": str(hidden_tiles),
                    "@TOKEN_TILES@": str(token_tiles),
                    "@X_POSITION@": str(read_positions[0]),
                    "@DY_POSITION@": str(read_positions[1]),
                    "@WEIGHT_POSITION@": str(read_positions[2]),
                    "@RSTD_POSITION@": str(read_positions[3]),
                    "@OUTPUT_POSITION@": "0",
                    "@OUTPUT_TENSOR@": "5",
                    "@OUTPUT_TILE_BYTES@": str(
                        4096 if stage in {"scaled", "correction_fp32"} else 2048
                    ),
                    "@NEG_HALF_BITS@": str(_float_bits(-0.5)),
                    "@INVERSE_HIDDEN_BITS@": str(_float_bits(1.0 / hidden_size)),
                    "@HIDDEN_FLOAT@": f"{float(hidden_size):.1f}f",
                    "@TWO_BITS@": str(_float_bits(2.0)),
                    "@OUTPUT_STAGE@": str(
                        {
                            "weighted": 0,
                            "direct": 1,
                            "correction": 2,
                            "scaled": 3,
                            "correction_fp32": 4,
                            "final": 5,
                        }[stage]
                    ),
                }
                cpp = Path(__file__).with_name("cpp")
                sources = {
                    "read": (cpp / "rmsnorm_backward_reader.cpp").read_text(),
                    "compute": (cpp / "rmsnorm_backward_compute.cpp").read_text(),
                    "write": (cpp / "rmsnorm_backward_writer.cpp").read_text(),
                }
                if compact_l1:
                    sources["read"] = sources["read"].replace(
                        "constexpr uint32_t cb_scaler = get_compile_time_arg_val(8);",
                        "constexpr uint32_t cb_scaler = get_compile_time_arg_val(6);",
                    )
                    sources["compute"] = _compact_compute_source(
                        sources["compute"]
                    )
                    if pre_reduce_uninit:
                        startup = (
                            "    compute_kernel_hw_startup("
                            "cb_x, cb_scaler, cb_output);\n"
                        )
                        if sources["compute"].count(startup) != 1:
                            raise RuntimeError(
                                "RMSNorm pre-reduce-uninit marker changed"
                            )
                        sources["compute"] = sources["compute"].replace(
                            startup,
                            startup + "    reduce_uninit(cb_product);\n",
                            1,
                        )
                    if zeroacc_all:
                        startup = (
                            "    compute_kernel_hw_startup("
                            "cb_x, cb_scaler, cb_output);\n"
                        )
                        if sources["compute"].count(startup) != 1:
                            raise RuntimeError(
                                "RMSNorm zeroacc-all startup marker changed"
                            )
                        sources["compute"] = sources["compute"].replace(
                            startup,
                            startup
                            + "    MATH(TTI_ZEROACC("
                            + "p_zeroacc::CLR_ALL, 1, 1, "
                            + "ADDR_MOD_1, 0));\n",
                            1,
                        )
                    if accurate_reduce:
                        marker = "        ReduceFp32Mode::Fast>(\n"
                        if sources["compute"].count(marker) != 1:
                            raise RuntimeError(
                                "RMSNorm accurate-reduce marker changed"
                            )
                        sources["compute"] = sources["compute"].replace(
                            marker,
                            "        ReduceFp32Mode::Accurate>(\n",
                            1,
                        )
                    if clear_dest:
                        reduce_call = "    compute_kernel_lib::reduce<\n"
                        if sources["compute"].count(reduce_call) != 1:
                            raise RuntimeError(
                                "RMSNorm clear-dest reduction marker changed"
                            )
                        sources["compute"] = sources["compute"].replace(
                            reduce_call,
                            "    fill_tile_init();\n"
                            "    tile_regs_acquire();\n"
                            "    fill_tile(0, 0.0f);\n"
                            "    tile_regs_commit();\n"
                            "    tile_regs_wait();\n"
                            "    tile_regs_release();\n"
                            + reduce_call,
                            1,
                        )
                    if reduce_uninit:
                        next_op = "\n    scalar_multiply("
                        if sources["compute"].count(next_op) != 1:
                            raise RuntimeError(
                                "RMSNorm reduce-uninit marker changed"
                            )
                        sources["compute"] = sources["compute"].replace(
                            next_op,
                            "\n    reduce_uninit(cb_product);\n"
                            + next_op,
                            1,
                        )
                    if epoch_reset:
                        startup = (
                            "    compute_kernel_hw_startup("
                            "cb_x, cb_scaler, cb_output);\n"
                        )
                        if sources["compute"].count(startup) != 1:
                            raise RuntimeError(
                                "RMSNorm epoch-reset startup marker changed"
                            )
                        sources["compute"] = sources["compute"].replace(
                            startup,
                            startup
                            + "    disable_fp32_dest_acc();\n"
                            + "    enable_fp32_dest_acc();\n",
                            1,
                        )
                for kind, source in tuple(sources.items()):
                    for placeholder, replacement in replacements.items():
                        source = source.replace(placeholder, replacement)
                    if re.search(r"@[A-Z0-9_]+@", source):
                        raise RuntimeError(f"unexpanded RMSNorm backward {kind} placeholder")
                    sources[kind] = source
                for path, thread in compiled.kernel_paths:
                    kind = "compute" if thread == "compute" else (
                        "read" if "read" in Path(path).name else "write"
                    )
                    Path(path).write_text(sources[kind], encoding="utf-8")

                from ttl.dataflow_buffer import CompilerAllocatedDFBConfig

                float32_blocks = (
                    {
                        3: hidden_tiles,
                        5: hidden_tiles,
                        6: 1,
                        7: 1,
                        8: 1,
                        9: hidden_tiles,
                        10: hidden_tiles,
                        11: 2,
                        12: 2,
                    }
                    if compact_l1
                    else {
                        3: hidden_tiles,
                        5: hidden_tiles,
                        6: hidden_tiles,
                        7: hidden_tiles,
                        8: 1,
                        9: 1,
                        10: hidden_tiles,
                        11: 1,
                        12: hidden_tiles,
                        13: hidden_tiles,
                        14: 1,
                        15: hidden_tiles,
                        16: 1,
                        17: 1,
                    }
                )
                for cb_index, blocks in float32_blocks.items():
                    config = CompilerAllocatedDFBConfig(
                        dfb_index=cb_index,
                        num_tiles=1,
                        data_format="float32",
                        block_count=blocks,
                    )
                    config._cb_index = cb_index
                    compiled.cb_configs[cb_index] = config
                if compact_l1:
                    compact_bf16_blocks = {
                        0: hidden_tiles,
                        1: hidden_tiles,
                        2: hidden_tiles,
                        4: hidden_tiles,
                        18: hidden_tiles,
                        19: 1,
                        20: 1,
                    }
                    for cb_index, blocks in compact_bf16_blocks.items():
                        compiled.cb_configs[cb_index].shape = (1, 1)
                        compiled.cb_configs[cb_index].block_count = blocks
                    for cb_index in range(13, 18):
                        compiled.cb_configs[cb_index].shape = (1, 1)
                        compiled.cb_configs[cb_index].block_count = 1
                for config in compiled.kernel_configs:
                    if hasattr(config, "math_fidelity"):
                        enum_type = type(config.math_fidelity)
                        config.math_fidelity = getattr(enum_type, "HiFi4")
                    if hasattr(config, "math_approx_mode"):
                        config.math_approx_mode = False
                    if hasattr(config, "fp32_dest_acc_en"):
                        config.fp32_dest_acc_en = True
                    if dst_full_sync and hasattr(config, "dst_full_sync_en"):
                        config.dst_full_sync_en = True
                    if hasattr(config, "unpack_to_dest_mode"):
                        import ttnn

                        if len(config.unpack_to_dest_mode) == 0:
                            for cb_index in range(64):
                                # CB7 intentionally matches TTNN's fast
                                # FP32 REDUCE_ROW unpack path rather than
                                # full-FP32 SFPU unpacking.
                                compact_full_fp32 = {
                                    0, 3, 4, 7, 8, 9, 10, 11, 12
                                }
                                config.unpack_to_dest_mode.append(
                                    ttnn.UnpackToDestMode.UnpackToDestFp32
                                    if cb_index in (
                                        compact_full_fp32
                                        if compact_l1 else {
                                        3, 5, 6, 9, 10, 11, 12, 13, 14,
                                        15, 16, 17,
                                        }
                                    )
                                    else ttnn.UnpackToDestMode.Default
                                )
            break
        return result

    operation.sequence_length = sequence_length
    operation.hidden_size = hidden_size
    operation.local_gradient_only = True
    operation.stage = stage
    operation.compact_l1 = bool(compact_l1)
    operation.estimated_cb_l1_bytes = (
        (4 * hidden_tiles + 7) * 4096
        + (5 * hidden_tiles + 7) * 2048
        if compact_l1 else None
    )
    # Private composition hook: a larger resident owner can trigger this
    # operation's normal lowering, serialize the accepted compact child, and
    # remap its six local tensor IDs into a unified physical tensor table.
    # The public callable remains unchanged.
    operation._ttl_program = _program
    return operation
