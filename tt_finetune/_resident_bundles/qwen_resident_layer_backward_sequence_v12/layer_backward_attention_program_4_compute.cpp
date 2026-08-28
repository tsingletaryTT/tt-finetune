#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/waypoint.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace resident_layer_phase_attention_4 {
// Exact resident dV -> dP -> softmax backward -> dQ -> dK -> GQA reduce.

#define FP32_DEST_ACC_EN 1


#ifdef TRISC_MATH
namespace ckernel::sfpu {
// TTNN's BF16 FPU subtraction rounds its only possible halfway case away
// from zero.  With BF16 operands, subtraction is either already BF16 or has
// exactly one discarded bit, so add 0x8000 to the magnitude field and mask.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_bf16_sub_round() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; ++d) {
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPIADD(
            0, p_sfpu::LREG13, p_sfpu::LREG1,
            sfpi::SFPIADD_MOD1_CC_NONE);  // +0x7fff
        TTI_SFPIADD(
            0, p_sfpu::LREG12, p_sfpu::LREG1,
            sfpi::SFPIADD_MOD1_CC_NONE);  // +1 => +0x8000
        TTI_SFPAND(0, p_sfpu::LREG14, p_sfpu::LREG1, 0);
        TTI_SFPSTORE(
            p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
    }
}
}  // namespace ckernel::sfpu
#endif

namespace {

constexpr bool kBf16 = 1;

// Kept byte-for-byte structurally equivalent to the separately qualified
// attention_dv_reference primitive.  Small-looking loop/signature changes
// can alter Tensix instruction scheduling, so the resident owner calls this
// exact implementation for its BF16 dV edge.
FORCE_INLINE void transpose_probability_column_exact(
    uint32_t input_id, uint32_t output_id, uint32_t tiles) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(tiles);
    output.reserve_back(tiles);
    compute_kernel_hw_startup(input_id, output_id);
    transpose_init(input_id);
    tile_regs_acquire();
    for (uint32_t tile = 0; tile < tiles; ++tile)
        transpose_tile(input_id, tile, tile);
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(output_id)));
    for (uint32_t tile = 0; tile < tiles; ++tile)
        pack_tile<true>(tile, output_id, tile);
    tile_regs_release();
    output.push_back(tiles);
    input.pop_front(tiles);
}

FORCE_INLINE void matmul_dv_exact(
    uint32_t lhs_id, uint32_t rhs_id, uint32_t output_id,
    uint32_t partial_id, uint32_t n_tiles, uint32_t k_tiles,
    uint32_t phase_tiles, uint32_t out_subblock_tiles,
    uint32_t m_tiles) {
    CircularBuffer lhs(lhs_id), rhs(rhs_id), output(output_id), partial(partial_id);
    lhs.wait_front(m_tiles * k_tiles);
    rhs.wait_front(k_tiles * n_tiles);
    compute_kernel_hw_startup<SrcOrder::Reverse>(lhs_id, rhs_id, partial_id);
    for (uint32_t n_start = 0; n_start < n_tiles; n_start += out_subblock_tiles) {
      const uint32_t phases = k_tiles / phase_tiles;
      matmul_block_init(
          lhs_id, rhs_id, false, out_subblock_tiles, m_tiles, phase_tiles);
      for (uint32_t phase = 0; phase < phases; ++phase) {
        tile_regs_acquire();
        if (phase > 0) {
            copy_tile_to_dst_init_short_with_dt(rhs_id, partial_id);
            partial.wait_front(m_tiles * out_subblock_tiles);
            copy_block_matmul_partials(
                partial_id, 0, 0, m_tiles * out_subblock_tiles);
            partial.pop_front(m_tiles * out_subblock_tiles);
            reconfig_data_format_srca(partial_id, rhs_id);
        }
        matmul_block_init(
            lhs_id, rhs_id, false, out_subblock_tiles, m_tiles, phase_tiles);
        for (uint32_t inner = 0; inner < phase_tiles; ++inner) {
            const uint32_t k = phase * phase_tiles + inner;
            matmul_block(
                lhs_id, rhs_id, k, k * n_tiles + n_start, 0,
                false, out_subblock_tiles, m_tiles, phase_tiles);
        }
        tile_regs_commit();
        const bool last = phase + 1 == phases;
        if (last) output.reserve_back(m_tiles * out_subblock_tiles);
        else partial.reserve_back(m_tiles * out_subblock_tiles);
        tile_regs_wait();
        const uint32_t target = last ? output_id : partial_id;
        PACK((pack_reconfig_data_format(target)));
        pack_tile_block(0, target, m_tiles * out_subblock_tiles);
        tile_regs_release();
        if (last) output.push_back(m_tiles * out_subblock_tiles);
        else partial.push_back(m_tiles * out_subblock_tiles);
      }
    }
    lhs.pop_front(m_tiles * k_tiles);
    rhs.pop_front(k_tiles * n_tiles);
}

FORCE_INLINE void transpose_tiles(
    uint32_t input_id, uint32_t output_id, uint32_t rows, uint32_t cols) {
    CircularBuffer input(input_id), output(output_id);
    const uint32_t tiles = rows * cols;
    input.wait_front(tiles);
    output.reserve_back(tiles);
    compute_kernel_hw_startup(input_id, output_id);
    transpose_init(input_id);
    for (uint32_t start = 0; start < tiles; start += 4) {
        const uint32_t chunk = (tiles - start) < 4 ? tiles - start : 4;
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < chunk; ++tile)
            transpose_tile(input_id, start + tile, tile);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        for (uint32_t tile = 0; tile < chunk; ++tile) {
            const uint32_t source = start + tile;
            const uint32_t row = source / cols;
            const uint32_t col = source - row * cols;
            pack_tile<true>(tile, output_id, col * rows + row);
        }
        tile_regs_release();
    }
    output.push_back(tiles);
}

FORCE_INLINE void transpose_linear(
    uint32_t input_id, uint32_t output_id, uint32_t tiles) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(tiles);
    output.reserve_back(tiles);
    compute_kernel_hw_startup(input_id, output_id);
    transpose_init(input_id);
    for (uint32_t start = 0; start < tiles; start += 4) {
        const uint32_t chunk = (tiles - start) < 4 ? tiles - start : 4;
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < chunk; ++tile)
            transpose_tile(input_id, start + tile, tile);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        for (uint32_t tile = 0; tile < chunk; ++tile)
            pack_tile<true>(tile, output_id, start + tile);
        tile_regs_release();
    }
    output.push_back(tiles);
}

FORCE_INLINE void transpose_strided_column(
    uint32_t input_id, uint32_t output_id, uint32_t tiles,
    uint32_t stride, uint32_t offset) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(tiles * stride);
    output.reserve_back(tiles);
    compute_kernel_hw_startup(input_id, output_id);
    transpose_init(input_id);
    for (uint32_t start = 0; start < tiles; start += 4) {
        const uint32_t chunk = (tiles - start) < 4 ? tiles - start : 4;
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < chunk; ++tile)
            transpose_tile(
                input_id, (start + tile) * stride + offset, tile);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        for (uint32_t tile = 0; tile < chunk; ++tile)
            pack_tile<true>(tile, output_id, start + tile);
        tile_regs_release();
    }
    output.push_back(tiles);
}

FORCE_INLINE void matmul_row(
    uint32_t lhs_id, uint32_t rhs_id, uint32_t output_id,
    uint32_t partial_id, uint32_t n_tiles, uint32_t k_tiles,
    uint32_t n_subblock_tiles, uint32_t phase_tiles,
    bool startup_on_output = false) {
    CircularBuffer lhs(lhs_id), rhs(rhs_id), output(output_id), partial(partial_id);
    lhs.wait_front(k_tiles);
    rhs.wait_front(k_tiles * n_tiles);
    compute_kernel_hw_startup<SrcOrder::Reverse>(
        lhs_id, rhs_id, startup_on_output ? output_id : partial_id);
    for (uint32_t n_start = 0; n_start < n_tiles; n_start += n_subblock_tiles) {
        matmul_block_init(lhs_id, rhs_id, false, n_subblock_tiles, 1, 1);
        const uint32_t phases = k_tiles / phase_tiles;
        for (uint32_t phase = 0; phase < phases; ++phase) {
            tile_regs_acquire();
            if (phase > 0) {
                copy_tile_to_dst_init_short_with_dt(rhs_id, partial_id);
                partial.wait_front(n_subblock_tiles);
                copy_block_matmul_partials(partial_id, 0, 0, n_subblock_tiles);
                partial.pop_front(n_subblock_tiles);
                reconfig_data_format_srca(partial_id, rhs_id);
            }
            matmul_block_init(
                lhs_id, rhs_id, false, n_subblock_tiles, 1, phase_tiles);
            for (uint32_t inner = 0; inner < phase_tiles; ++inner) {
                const uint32_t k = phase * phase_tiles + inner;
                matmul_block(
                    lhs_id, rhs_id, k, k * n_tiles + n_start, 0,
                    false, n_subblock_tiles, 1, phase_tiles);
            }
            tile_regs_commit();
            const bool last = phase + 1 == phases;
            if (last) output.reserve_back(n_subblock_tiles);
            else partial.reserve_back(n_subblock_tiles);
            tile_regs_wait();
            const uint32_t target = last ? output_id : partial_id;
            PACK((pack_reconfig_data_format(target)));
            pack_tile_block(0, target, n_subblock_tiles);
            tile_regs_release();
            if (last) output.push_back(n_subblock_tiles);
            else partial.push_back(n_subblock_tiles);
        }
    }
}

// Exact shape used by the real autograd graph for dK: Q.T @ dS, followed by
// an output transpose.  Keeping this operand order matters at the FP32 bit
// level even though it is algebraically equivalent to dS.T @ Q.
FORCE_INLINE void matmul_column(
    uint32_t lhs_id, uint32_t rhs_id, uint32_t output_id,
    uint32_t partial_id, uint32_t m_tiles, uint32_t k_tiles,
    uint32_t phase_tiles) {
    CircularBuffer lhs(lhs_id), rhs(rhs_id), output(output_id), partial(partial_id);
    lhs.wait_front(m_tiles * k_tiles);
    rhs.wait_front(k_tiles);
    compute_kernel_hw_startup<SrcOrder::Reverse>(lhs_id, rhs_id, output_id);
    for (uint32_t m = 0; m < m_tiles; ++m) {
        const uint32_t phases = k_tiles / phase_tiles;
        for (uint32_t phase = 0; phase < phases; ++phase) {
            tile_regs_acquire();
            if (phase > 0) {
                copy_tile_to_dst_init_short_with_dt(rhs_id, partial_id);
                partial.wait_front(1);
                copy_block_matmul_partials(partial_id, 0, 0, 1);
                partial.pop_front(1);
                reconfig_data_format_srca(partial_id, rhs_id);
            }
            matmul_block_init(lhs_id, rhs_id, false, 1, 1, phase_tiles);
            for (uint32_t inner = 0; inner < phase_tiles; ++inner) {
                const uint32_t k = phase * phase_tiles + inner;
                matmul_block(
                    lhs_id, rhs_id, m * k_tiles + k, k, 0,
                    false, 1, 1, phase_tiles);
            }
            tile_regs_commit();
            const bool last = phase + 1 == phases;
            if (last) output.reserve_back(1);
            else partial.reserve_back(1);
            tile_regs_wait();
            const uint32_t target = last ? output_id : partial_id;
            PACK((pack_reconfig_data_format(target)));
            pack_tile_block(0, target, 1);
            tile_regs_release();
            if (last) output.push_back(1);
            else partial.push_back(1);
        }
    }
}

// Reproduce bmm_large_block_zm's IN0_TRANSPOSE_TILE path for a row worker.
// TTNN transposes one input tile immediately before the matching K block,
// then spills the FP32 partial before advancing to the next tile.  Performing
// all transposes up front leaves identical BF16 tiles in L1 but a different
// observable Tensix configuration/reduction schedule.
FORCE_INLINE void matmul_row_streamed_in0_transpose(
    uint32_t original_lhs_id, uint32_t transposed_lhs_id,
    uint32_t rhs_id, uint32_t output_id, uint32_t partial_id,
    uint32_t n_tiles, uint32_t k_tiles) {
    CircularBuffer original_lhs(original_lhs_id);
    CircularBuffer transposed_lhs(transposed_lhs_id);
    CircularBuffer rhs(rhs_id), output(output_id), partial(partial_id);
    original_lhs.wait_front(k_tiles);
    rhs.wait_front(k_tiles * n_tiles);
    compute_kernel_hw_startup<SrcOrder::Reverse>(
        transposed_lhs_id, rhs_id, partial_id);

    for (uint32_t n = 0; n < n_tiles; ++n) {
        bool reload = false;
        for (uint32_t block = 0; block < k_tiles; ++block) {
            reconfig_data_format_srca(rhs_id, original_lhs_id);
            transpose_init(original_lhs_id);
            PACK((pack_reconfig_data_format(transposed_lhs_id)));
            transposed_lhs.reserve_back(1);
            tile_regs_acquire();
            transpose_tile(original_lhs_id, block, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, transposed_lhs_id);
            tile_regs_release();
            transposed_lhs.push_back(1);

            reconfig_data_format_srca(original_lhs_id, rhs_id);
            matmul_block_init(
                transposed_lhs_id, rhs_id, false, 1, 1, 1);
            PACK((pack_reconfig_data_format(partial_id)));
            transposed_lhs.wait_front(1);
            tile_regs_acquire();
            if (reload) {
                copy_tile_to_dst_init_short_with_dt(rhs_id, partial_id);
                partial.wait_front(1);
                copy_block_matmul_partials(partial_id, 0, 0, 1);
                partial.pop_front(1);
                reconfig_data_format_srca(partial_id, rhs_id);
                matmul_block_init(
                    transposed_lhs_id, rhs_id, false, 1, 1, 1);
            }
            matmul_block(
                transposed_lhs_id, rhs_id, 0,
                block * n_tiles + n, 0, false, 1, 1, 1);
            tile_regs_commit();
            const bool last = block + 1 == k_tiles;
            if (last) output.reserve_back(1);
            else partial.reserve_back(1);
            tile_regs_wait();
            const uint32_t target = last ? output_id : partial_id;
            PACK((pack_reconfig_data_format(target)));
            pack_tile_block(0, target, 1);
            tile_regs_release();
            if (last) output.push_back(1);
            else partial.push_back(1);
            transposed_lhs.pop_front(1);
            reload = !last;
        }
    }
}

FORCE_INLINE void scale_tiles(
    uint32_t input_id, uint32_t output_id,
    uint32_t tiles, uint32_t scale_bits) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(tiles);
    output.reserve_back(tiles);
    init_sfpu(input_id, output_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        copy_tile_init(input_id);
        copy_tile(input_id, tile, 0);
        binop_with_scalar_tile_init();
        { volatile int32_t scalar = scale_bits; mul_unary_tile(0, scalar); }
        if constexpr (kBf16) {
            typecast_tile_init<
                (uint32_t)DataFormat::Float32,
                (uint32_t)DataFormat::Float16_b>();
            typecast_tile<
                (uint32_t)DataFormat::Float32,
                (uint32_t)DataFormat::Float16_b>(0);
        }
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, tile);
        tile_regs_release();
    }
    output.push_back(tiles);
    input.pop_front(tiles);
}

FORCE_INLINE void copy_tiles_preserve(
    uint32_t input_id, uint32_t output_id, uint32_t tiles) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(tiles);
    output.reserve_back(tiles);
    compute_kernel_hw_startup(input_id, output_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        copy_tile_init(input_id);
        copy_tile(input_id, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, tile);
        tile_regs_release();
    }
    output.push_back(tiles);
}

FORCE_INLINE void repeat_first_tile_preserve(
    uint32_t input_id, uint32_t output_id, uint32_t copies) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(1);
    output.reserve_back(copies);
    compute_kernel_hw_startup(input_id, output_id);
    for (uint32_t copy = 0; copy < copies; ++copy) {
        tile_regs_acquire();
        copy_tile_init(input_id);
        copy_tile(input_id, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, copy);
        tile_regs_release();
    }
    output.push_back(copies);
}

template <uint32_t Op, bool PackBf16 = kBf16>
FORCE_INLINE void sfpu_binary_tile_to_cb(
    uint32_t left_id, uint32_t right_id, uint32_t output_id,
    uint32_t left_tile, uint32_t right_tile) {
    CircularBuffer left(left_id), right(right_id), output(output_id);
    left.wait_front(left_tile + 1);
    right.wait_front(right_tile + 1);
    output.reserve_back(1);
    unary_op_init_common(left_id, output_id);
    if constexpr (Op == 0) mul_binary_tile_init();
    else if constexpr (Op == 1) sub_binary_tile_init();
    else add_binary_tile_init();
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(right_id, left_id);
    copy_tile(left_id, left_tile, 0);
    copy_tile_to_dst_init_short_with_dt(left_id, right_id);
    copy_tile(right_id, right_tile, 1);
    if constexpr (Op == 0) mul_binary_tile(0, 1, 0);
    else if constexpr (Op == 1) sub_binary_tile(0, 1, 0);
    else add_binary_tile(0, 1, 0);
    if constexpr (PackBf16) {
        // A combined program must keep FP32 destination accumulation for its
        // matmuls.  TTNN's standalone BF16 binary kernels instead use the
        // 16-bit destination path, whose halfway conversion is RNE.  Apply
        // that conversion explicitly before packing so both contracts can
        // coexist in one compute kernel.
        typecast_tile_init<
            (uint32_t)DataFormat::Float32,
            (uint32_t)DataFormat::Float16_b>();
        typecast_tile<
            (uint32_t)DataFormat::Float32,
            (uint32_t)DataFormat::Float16_b>(0);
    }
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(output_id)));
    pack_tile(0, output_id);
    tile_regs_release();
    output.push_back(1);
}

FORCE_INLINE void fp32_to_bf16_tiles(
    uint32_t input_id, uint32_t output_id, uint32_t tiles) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(tiles);
    output.reserve_back(tiles);
    init_sfpu(input_id, output_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        copy_tile_init(input_id);
        copy_tile(input_id, tile, 0);
        typecast_tile_init<
            (uint32_t)DataFormat::Float32,
            (uint32_t)DataFormat::Float16_b>();
        typecast_tile<
            (uint32_t)DataFormat::Float32,
            (uint32_t)DataFormat::Float16_b>(0);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, tile);
        tile_regs_release();
    }
    output.push_back(tiles);
}

#if !0
// Mirror binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp.
// TTNN processes the four score tiles owned by a row worker in one DST
// acquire/commit: lhs tiles occupy even registers and rhs tiles odd registers.
// That register schedule is numerically observable in BF16.
FORCE_INLINE void sfpu_multiply_block_to_cb(
    uint32_t left_id, uint32_t right_id, uint32_t output_id,
    uint32_t tiles) {
    CircularBuffer left(left_id), right(right_id), output(output_id);
    left.wait_front(tiles);
    right.wait_front(tiles);
    output.reserve_back(tiles);
    unary_op_init_common(left_id, output_id);
    mul_binary_tile_init();
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(right_id, left_id);
    for (uint32_t tile = 0; tile < tiles; ++tile)
        copy_tile(left_id, tile, tile * 2);
    copy_tile_to_dst_init_short_with_dt(left_id, right_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        copy_tile(right_id, tile, tile * 2 + 1);
        mul_binary_tile(tile * 2, tile * 2 + 1, tile * 2);
    }
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(output_id)));
    for (uint32_t tile = 0; tile < tiles; ++tile)
        pack_tile(tile * 2, output_id);
    tile_regs_release();
    output.push_back(tiles);
}

FORCE_INLINE void fpu_add_tile_to_cb(
    uint32_t left_id, uint32_t right_id, uint32_t output_id,
    uint32_t left_tile, uint32_t right_tile) {
    CircularBuffer left(left_id), right(right_id), output(output_id);
    left.wait_front(left_tile + 1);
    right.wait_front(right_tile + 1);
    output.reserve_back(1);
    binary_op_init_common(left_id, right_id, output_id);
    tile_regs_acquire();
    add_tiles_init(left_id, right_id);
    add_tiles(left_id, right_id, left_tile, right_tile, 0);
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(output_id)));
    pack_tile<true>(0, output_id, 0);
    tile_regs_release();
    output.push_back(1);
}

FORCE_INLINE void materialize_col_broadcast(
    uint32_t row_sum_id, uint32_t broadcast_id) {
    CircularBuffer row_sum(row_sum_id), broadcast(broadcast_id);
    row_sum.wait_front(1);
    broadcast.reserve_back(1);
    unary_op_init_common(row_sum_id, broadcast_id);
    unary_bcast_init<BroadcastType::COL>(row_sum_id, broadcast_id);
    tile_regs_acquire();
    unary_bcast<BroadcastType::COL>(row_sum_id, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(broadcast_id)));
    pack_tile<true>(0, broadcast_id, 0);
    tile_regs_release();
    broadcast.push_back(1);
}

FORCE_INLINE void fpu_sub_tile_to_cb(
    uint32_t matrix_id, uint32_t broadcast_id, uint32_t output_id,
    uint32_t matrix_tile) {
    CircularBuffer matrix(matrix_id), broadcast(broadcast_id), output(output_id);
    matrix.wait_front(matrix_tile + 1);
    broadcast.wait_front(1);
    output.reserve_back(1);
    binary_op_init_common(matrix_id, broadcast_id, output_id);
    sub_tiles_init(matrix_id, broadcast_id);
    tile_regs_acquire();
    sub_tiles(matrix_id, broadcast_id, matrix_tile, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(output_id)));
    pack_tile<true>(0, output_id, 0);
    tile_regs_release();
    output.push_back(1);
}

// Match binary_ng's broadcast-aware FP32 subtract without materializing the
// row-sum broadcast.  Keeping the broadcast value in DST is numerically
// observable: an FP32 L1 round trip changes BF16 boundary values.
FORCE_INLINE void sfpu_sub_row_sum_to_cb(
    uint32_t matrix_id, uint32_t row_sum_id, uint32_t output_id,
    uint32_t matrix_tile) {
    CircularBuffer matrix(matrix_id), row_sum(row_sum_id), output(output_id);
    matrix.wait_front(matrix_tile + 1);
    row_sum.wait_front(1);
    output.reserve_back(1);
    unary_op_init_common(matrix_id, output_id);
    unary_bcast_init<BroadcastType::COL>(row_sum_id, output_id);
    sub_binary_tile_init();
    tile_regs_acquire();
    unary_bcast<BroadcastType::COL>(row_sum_id, 0, 1);
    copy_tile_to_dst_init_short_with_dt(row_sum_id, matrix_id);
    copy_tile(matrix_id, matrix_tile, 0);
    sub_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(output_id)));
    pack_tile<true>(0, output_id, 0);
    tile_regs_release();
    output.push_back(1);
}

template <uint32_t packed_rows_id, uint32_t scaler_id, uint32_t reduced_rows_id>
FORCE_INLINE void exact_gqa_h_reduce(uint32_t tiles) {
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        compute_kernel_lib::reduce<
            PoolType::SUM, ReduceDim::REDUCE_COL,
            packed_rows_id, scaler_id, reduced_rows_id,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT,
            ReduceFp32Mode::Fast>(
                compute_kernel_lib::ReduceInputBlockShape::single(),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{});
    }
}

FORCE_INLINE void fpu_add_tile_to_fp32_cb(
    uint32_t left_id, uint32_t right_id, uint32_t output_id,
    uint32_t left_tile, uint32_t right_tile) {
    CircularBuffer left(left_id), right(right_id), output(output_id);
    output.reserve_back(1);
    binary_op_init_common(left_id, right_id, output_id);
    add_tiles_init(left_id, right_id, false);
    tile_regs_acquire();
    add_tiles(left_id, right_id, left_tile, right_tile, 0);
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(output_id)));
    pack_tile<true>(0, output_id, 0);
    tile_regs_release();
    output.push_back(1);
}

FORCE_INLINE void copy_tile_to_fp32_cb(
    uint32_t input_id, uint32_t output_id, uint32_t input_tile) {
    CircularBuffer input(input_id), output(output_id);
    output.reserve_back(1);
    compute_kernel_hw_startup(input_id, output_id);
    tile_regs_acquire();
    copy_tile_init(input_id);
    copy_tile(input_id, input_tile, 0);
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(output_id)));
    pack_tile<true>(0, output_id, 0);
    tile_regs_release();
    output.push_back(1);
}

FORCE_INLINE void rne_fp32_tile_to_bf16_cb(
    uint32_t input_id, uint32_t output_id) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(1);
    output.reserve_back(1);
    init_sfpu(input_id, output_id);
    tile_regs_acquire();
    copy_tile_init(input_id);
    copy_tile(input_id, 0, 0);
    typecast_tile_init<
        (uint32_t)DataFormat::Float32,
        (uint32_t)DataFormat::Float16_b>();
    typecast_tile<
        (uint32_t)DataFormat::Float32,
        (uint32_t)DataFormat::Float16_b>(0);
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(output_id)));
    pack_tile<true>(0, output_id, 0);
    tile_regs_release();
    output.push_back(1);
}

template <uint32_t input_id, uint32_t output_id>
FORCE_INLINE void diagnostic_gqa_passthrough(uint32_t outputs, uint32_t group) {
    CircularBuffer input(input_id), output(output_id);
    compute_kernel_hw_startup(input_id, output_id);
    for (uint32_t output_tile = 0; output_tile < outputs; ++output_tile) {
        input.wait_front(group);
        output.reserve_back(1);
        tile_regs_acquire();
        copy_tile_init(input_id);
        copy_tile(input_id, 0, 0);
        input.pop_front(group);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, 0);
        tile_regs_release();
        output.push_back(1);
    }
}

FORCE_INLINE void packed_bf16_gqa_sum(
    uint32_t input_id, uint32_t scratch_a_id, uint32_t scratch_b_id,
    uint32_t output_id, uint32_t outputs, uint32_t group) {
    CircularBuffer input(input_id);
    for (uint32_t output_tile = 0; output_tile < outputs; ++output_tile) {
        input.wait_front(group);
        sfpu_binary_tile_to_cb<2>(
            input_id, input_id, scratch_a_id, 0, 1);
        uint32_t source = scratch_a_id;
        uint32_t target = scratch_b_id;
        for (uint32_t member = 2; member < group; ++member) {
            const bool last = member + 1 == group;
            const uint32_t destination = last ? output_id : target;
            sfpu_binary_tile_to_cb<2>(
                source, input_id, destination, 0, member);
            CircularBuffer source_cb(source);
            source_cb.pop_front(1);
            const uint32_t old_source = source;
            source = target;
            target = old_source;
        }
        input.pop_front(group);
    }
}
#endif

}  // namespace

void run() {
    DeviceZoneScopedN("attention_resident_compute");
    constexpr uint32_t cb_p_row = get_compile_time_arg_val(25);
    constexpr uint32_t cb_p_column = get_compile_time_arg_val(26);
    constexpr uint32_t cb_dout_row = get_compile_time_arg_val(27);
    constexpr uint32_t cb_dout_full = get_compile_time_arg_val(28);
    constexpr uint32_t cb_q_full = get_compile_time_arg_val(29);
    constexpr uint32_t cb_k = get_compile_time_arg_val(30);
    constexpr uint32_t cb_v = get_compile_time_arg_val(31);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(32);
    constexpr uint32_t cb_pt = get_compile_time_arg_val(33);
    constexpr uint32_t cb_dv = get_compile_time_arg_val(34);
    constexpr uint32_t cb_vt = get_compile_time_arg_val(35);
    constexpr uint32_t cb_dp = get_compile_time_arg_val(36);
    constexpr uint32_t cb_product = get_compile_time_arg_val(37);
    constexpr uint32_t cb_rowsum = get_compile_time_arg_val(38);
    constexpr uint32_t cb_centered = get_compile_time_arg_val(39);
    constexpr uint32_t cb_ds = get_compile_time_arg_val(40);
    constexpr uint32_t cb_ds_scaled = get_compile_time_arg_val(41);
    constexpr uint32_t cb_dq = get_compile_time_arg_val(42);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(43);
    constexpr uint32_t cb_ds_handoff = get_compile_time_arg_val(44);
    constexpr uint32_t cb_ds_column = get_compile_time_arg_val(45);
    constexpr uint32_t cb_ds_t = get_compile_time_arg_val(46);
    constexpr uint32_t cb_dk = get_compile_time_arg_val(47);
    constexpr uint32_t cb_peer_dk = get_compile_time_arg_val(48);
    constexpr uint32_t cb_peer_dv = get_compile_time_arg_val(49);
    constexpr uint32_t cb_sum_dk_a = get_compile_time_arg_val(50);
    constexpr uint32_t cb_sum_dk_b = get_compile_time_arg_val(51);
    constexpr uint32_t cb_sum_dv_a = get_compile_time_arg_val(52);
    constexpr uint32_t cb_sum_dv_b = get_compile_time_arg_val(53);
    constexpr uint32_t cb_final_dk = get_compile_time_arg_val(54);
    constexpr uint32_t cb_final_dv = get_compile_time_arg_val(55);
    constexpr uint32_t cb_p_bf16 = get_compile_time_arg_val(56);
    constexpr uint32_t cb_rope_cos = get_compile_time_arg_val(57);
    constexpr uint32_t cb_rope_sin = get_compile_time_arg_val(58);
    constexpr uint32_t cb_rope_cos_product = get_compile_time_arg_val(59);
    constexpr uint32_t cb_rope_sin_product = get_compile_time_arg_val(60);
    constexpr uint32_t cb_rope_q = get_compile_time_arg_val(61);
    constexpr uint32_t cb_rope_k = get_compile_time_arg_val(62);
    constexpr uint32_t seq_tiles = 4;
    constexpr uint32_t dim_tiles = 2;
    constexpr uint32_t query_heads = 14;
    constexpr uint32_t kv_heads = 2;
    constexpr uint32_t group = query_heads / kv_heads;
    constexpr uint32_t jobs = query_heads * seq_tiles;
    constexpr uint32_t reducer_jobs = kv_heads * seq_tiles;
    constexpr uint32_t vector_tiles = seq_tiles * dim_tiles;
    constexpr uint32_t scale_bits = 1040187392;
    constexpr uint32_t matmul_phase_tiles = 1;
    constexpr bool diagnostic_ds = 0;
    constexpr bool diagnostic_dv = 0;
    constexpr bool diagnostic_pt = 0;
    constexpr bool diagnostic_dq = 0;
    constexpr bool diagnostic_dk = 0;
    constexpr bool diagnostic_dp = 0;
    constexpr bool diagnostic_product = 0;
    constexpr bool diagnostic_centered = 0;
    constexpr bool diagnostic_rowsum = 0;
    constexpr bool diagnostic_unscaled_ds = 0;
    constexpr bool diagnostic_dout_handoff = 0;
    constexpr bool probability_bf16_input = 0;

    const uint32_t node =
        get_absolute_logical_y() * 6 + (get_absolute_logical_x() - 5);
    if (node >= jobs) return;
    const uint32_t head = node / seq_tiles;
    const uint32_t member = head % group;
    CircularBuffer p_row(cb_p_row), p_column(cb_p_column);
    CircularBuffer dout_row(cb_dout_row), dout_full(cb_dout_full);
    CircularBuffer q_full(cb_q_full), k(cb_k), v(cb_v), pt(cb_pt), vt(cb_vt);
    CircularBuffer dp(cb_dp), rowsum(cb_rowsum), centered(cb_centered);
    CircularBuffer ds(cb_ds), ds_column(cb_ds_column), ds_t(cb_ds_t);

    if constexpr (diagnostic_dout_handoff) {
        copy_tiles_preserve(cb_dout_row, cb_dv, dim_tiles);
        return;
    }

    // Match eager autograd's execution order as well as its individual
    // matmul schedules: dV is the first backward dot, followed by dP.  The
    // order is numerically observable because dV's qualified TTNN schedule
    // starts from a fresh matmul-engine state.
    if constexpr (kBf16) {
        if constexpr (!probability_bf16_input) {
            fp32_to_bf16_tiles(cb_p_column, cb_p_bf16, seq_tiles);
            // The framework executes the FP32->BF16 cast and BMM as separate
            // programs. Re-arm FP32 destination mode here so the resident
            // BMM starts from the same hardware mode boundary.
            disable_fp32_dest_acc();
            enable_fp32_dest_acc();
        }
        constexpr uint32_t packed_p_id =
            probability_bf16_input ? cb_p_column : cb_p_bf16;
        if constexpr (diagnostic_pt) {
            transpose_tiles(packed_p_id, cb_pt, 1, seq_tiles);
            copy_tiles_preserve(cb_pt, cb_dv, dim_tiles);
            return;
        }
        transpose_probability_column_exact(packed_p_id, cb_pt, seq_tiles);
        matmul_dv_exact(
            cb_pt, cb_dout_full, cb_dv, cb_partial,
            dim_tiles, seq_tiles, 1, dim_tiles, 1);
    } else {
        transpose_tiles(cb_p_column, cb_pt, 1, seq_tiles);
        matmul_row(
            cb_pt, cb_dout_full, cb_dv, cb_partial,
            dim_tiles, seq_tiles, dim_tiles, matmul_phase_tiles);
        pt.pop_front(seq_tiles);
    }
    WAYPOINT("ADV");
    if constexpr (diagnostic_dv) return;

    transpose_linear(cb_v, cb_vt, vector_tiles);
    matmul_row(
        cb_dout_row, cb_vt, cb_dp, cb_partial,
        seq_tiles, dim_tiles, 4,
        matmul_phase_tiles == 1 ? 1 : dim_tiles);
    vt.pop_front(vector_tiles);
    if constexpr (diagnostic_dp) {
        copy_tiles_preserve(cb_dp, cb_ds_handoff, seq_tiles);
        dp.pop_front(seq_tiles);
        return;
    }

    p_row.wait_front(seq_tiles);
    dp.wait_front(seq_tiles);
    // StableHLO has an observable BF16 -> FP32 conversion between dP and the
    // softmax-backward multiply. CB18 must retain TTNN's default TF32 matmul
    // partial-reload contract, so stage full-FP32 dP in CB15. CB15 is empty
    // until dS is produced and can be reused after this staging value is
    // consumed.
    copy_tiles_preserve(cb_dp, cb_ds, seq_tiles);
    for (uint32_t tile = 0; tile < seq_tiles; ++tile)
        sfpu_binary_tile_to_cb<0, false>(
            cb_p_row, cb_ds, cb_product, tile, tile);
    if constexpr (diagnostic_product) {
        fp32_to_bf16_tiles(cb_product, cb_ds_handoff, seq_tiles);
        CircularBuffer product(cb_product);
        CircularBuffer dp_fp32(cb_ds);
        product.pop_front(seq_tiles);
        dp_fp32.pop_front(seq_tiles);
        dp.pop_front(seq_tiles);
        p_row.pop_front(seq_tiles);
        return;
    }
#if !0
    compute_kernel_lib::reduce<
        PoolType::SUM, ReduceDim::REDUCE_ROW,
        cb_product, cb_scaler, cb_rowsum,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT,
        ReduceFp32Mode::Fast>(
            compute_kernel_lib::ReduceInputBlockShape::row(seq_tiles));
    // The scaler is produced once per invocation.  The reduction helper
    // treats it as reusable and therefore does not advance its CB.  This
    // owner has exactly one row reduction, so release it now; otherwise stale
    // scaler pages accumulate and the fifth invocation blocks its reader
    // before Q/K/V are loaded.
    CircularBuffer row_reduce_scaler(cb_scaler);
    row_reduce_scaler.pop_front(1);
    rowsum.wait_front(1);
    if constexpr (diagnostic_rowsum) {
        repeat_first_tile_preserve(cb_rowsum, cb_ds_handoff, seq_tiles);
        CircularBuffer dp_fp32(cb_ds);
        dp_fp32.pop_front(seq_tiles);
        rowsum.pop_front(1);
        dp.pop_front(seq_tiles);
        p_row.pop_front(seq_tiles);
        return;
    }
    for (uint32_t tile = 0; tile < seq_tiles; ++tile)
        sfpu_sub_row_sum_to_cb(
            cb_ds, cb_rowsum, cb_centered, tile);
    CircularBuffer dp_fp32(cb_ds);
    dp_fp32.pop_front(seq_tiles);
    if constexpr (diagnostic_centered) {
        fp32_to_bf16_tiles(cb_centered, cb_ds_handoff, seq_tiles);
        centered.pop_front(seq_tiles);
        rowsum.pop_front(1);
        dp.pop_front(seq_tiles);
        p_row.pop_front(seq_tiles);
        return;
    }
    for (uint32_t tile = 0; tile < seq_tiles; ++tile)
        sfpu_binary_tile_to_cb<0, false>(
            cb_p_row, cb_centered, cb_ds, tile, tile);
    centered.pop_front(seq_tiles);
    if constexpr (diagnostic_unscaled_ds) {
        fp32_to_bf16_tiles(cb_ds, cb_ds_handoff, seq_tiles);
        CircularBuffer diagnostic_ds(cb_ds);
        diagnostic_ds.pop_front(seq_tiles);
        rowsum.pop_front(1);
        dp.pop_front(seq_tiles);
        p_row.pop_front(seq_tiles);
        return;
    }
    rowsum.pop_front(1);
    dp.pop_front(seq_tiles);
    p_row.pop_front(seq_tiles);
    // The real autograd graph applies score scale once to dS before both dQ
    // and dK. This is the observable TTNN boundary qualified by the row op.
    if constexpr (kBf16) {
        fp32_to_bf16_tiles(cb_ds, cb_p_bf16, seq_tiles);
        CircularBuffer ds_fp32(cb_ds);
        ds_fp32.pop_front(seq_tiles);
        scale_tiles(cb_p_bf16, cb_ds_scaled, seq_tiles, scale_bits);
    } else {
        scale_tiles(cb_ds, cb_ds_scaled, seq_tiles, scale_bits);
    }
    copy_tiles_preserve(cb_ds_scaled, cb_ds_handoff, seq_tiles);
    WAYPOINT("ADS");
    if constexpr (diagnostic_ds) return;
    matmul_row(
        cb_ds_scaled, cb_k, cb_dq, cb_partial,
        dim_tiles, seq_tiles, dim_tiles, matmul_phase_tiles);
    CircularBuffer ds_scaled(cb_ds_scaled);
    ds_scaled.pop_front(seq_tiles);
    WAYPOINT("ADQ");
    if constexpr (diagnostic_dq) return;

    // Match eager-autograd StableHLO exactly: Q.T @ scaled_dS followed by an
    // output transpose.  The algebraically equivalent dS.T @ Q schedule has
    // a different BF16 reduction tree and is therefore not interchangeable.
    for (uint32_t dim = 0; dim < dim_tiles; ++dim) {
        transpose_strided_column(
            cb_q_full, cb_ds_t, seq_tiles, dim_tiles, dim);
        matmul_column(
            cb_ds_t, cb_ds_column, cb_sum_dv_b, cb_partial,
            1, seq_tiles, matmul_phase_tiles);
        ds_t.pop_front(seq_tiles);
    }
    transpose_linear(cb_sum_dv_b, cb_dk, dim_tiles);
    CircularBuffer dk_pretranspose(cb_sum_dv_b);
    dk_pretranspose.pop_front(dim_tiles);
    ds_column.pop_front(seq_tiles);
    WAYPOINT("ADK");

    if constexpr (diagnostic_dk) return;

    if (member == 0) {
        // The reader packs 32 logical sequence rows for each dK tile and then
        // 32 rows for each dV tile.  Each input is exactly the single H tile
        // produced by TTNN's transpose(group, H); the writer performs the
        // inverse row scatter entirely in local L1.
        if constexpr (!kBf16) {
            exact_gqa_h_reduce<cb_sum_dk_a, cb_sum_dv_a, cb_sum_dk_b>(
                2 * dim_tiles * 32);
            CircularBuffer local_dk(cb_dk), local_dv(cb_dv);
            CircularBuffer peer_dk(cb_peer_dk), peer_dv(cb_peer_dv);
            local_dk.pop_front(dim_tiles);
            local_dv.pop_front(dim_tiles);
            peer_dk.pop_front((group - 1) * dim_tiles);
            peer_dv.pop_front((group - 1) * dim_tiles);
        } else {
            // TTNN's reduce over the GQA member dimension accumulates all
            // members in one FP32 destination register and rounds once at
            // the final BF16 pack.  Pairwise packed BF16 additions introduce
            // an extra rounding boundary and are observably nonexact.
            CircularBuffer gathered(cb_sum_dk_a);
            CircularBuffer zero(cb_sum_dv_a);
            CircularBuffer reduced(cb_sum_dk_b);
            zero.wait_front(1);
            WAYPOINT("AG0");
            // The standalone exact reducer begins from a fresh binary
            // compute configuration.  This owner reaches the same phase
            // after a transpose/matmul pipeline, so explicitly restart the
            // unpack/math/pack configuration before reusing DST register 0.
            compute_kernel_hw_startup(
                cb_sum_dk_a, cb_sum_dv_a, cb_sum_dk_b);
            binary_op_init_common(cb_sum_dk_a, cb_sum_dv_a, cb_sum_dk_b);
            for (uint32_t output_tile = 0;
                 output_tile < 2 * dim_tiles; ++output_tile) {
                gathered.wait_front(group);
                WAYPOINT("AG1");
                copy_tile_to_dst_init_short_with_dt(
                    cb_sum_dv_a, cb_sum_dk_a);
                tile_regs_acquire();
                copy_tile(cb_sum_dk_a, 0, 0);
                reconfig_data_format_srcb(cb_sum_dk_a, cb_sum_dv_a);
                add_tiles_init(cb_sum_dk_a, cb_sum_dv_a, true);
                for (uint32_t source_member = 1;
                     source_member < group; ++source_member) {
                    add_tiles(
                        cb_sum_dk_a, cb_sum_dv_a,
                        source_member, 0, 0);
                }
                gathered.pop_front(group);
                tile_regs_commit();
                reduced.reserve_back(1);
                tile_regs_wait();
                PACK((pack_reconfig_data_format(cb_sum_dk_b)));
                pack_tile(0, cb_sum_dk_b);
                tile_regs_release();
                reduced.push_back(1);
                WAYPOINT("AG2");
            }
            zero.pop_front(1);
        }
    }
    WAYPOINT("AGD");

    // Inverse RoPE remains inside the attention island. Match the framework's
    // two separately packed BF16 products before the final add/sub boundary.
    CircularBuffer rope_cos(cb_rope_cos), rope_sin(cb_rope_sin);
    CircularBuffer dq_gradient(cb_dq), rope_q(cb_rope_q);
    compute_kernel_hw_startup(cb_dq, cb_rope_cos, cb_rope_cos_product);
    for (uint32_t dim = 0; dim < dim_tiles; ++dim) {
        const uint32_t peer = dim ^ 1u;
        sfpu_binary_tile_to_cb<0, true>(
            cb_dq, cb_rope_cos, cb_rope_cos_product, dim, dim);
        sfpu_binary_tile_to_cb<0, true>(
            cb_dq, cb_rope_sin, cb_rope_sin_product, peer, peer);
        if (dim == 0)
            fpu_add_tile_to_cb(
                cb_rope_cos_product, cb_rope_sin_product, cb_rope_q, 0, 0);
        else
            fpu_sub_tile_to_cb(
                cb_rope_cos_product, cb_rope_sin_product, cb_rope_q, 0);
        CircularBuffer cos_product(cb_rope_cos_product);
        CircularBuffer sin_product(cb_rope_sin_product);
        cos_product.pop_front(1);
        sin_product.pop_front(1);
    }
    dq_gradient.pop_front(dim_tiles);

    if (member == 0) {
        CircularBuffer reduced(cb_sum_dk_b), rope_k(cb_rope_k);
        compute_kernel_hw_startup(
            cb_sum_dk_b, cb_rope_cos, cb_rope_cos_product);
        for (uint32_t dim = 0; dim < dim_tiles; ++dim) {
            const uint32_t peer = dim ^ 1u;
            sfpu_binary_tile_to_cb<0, true>(
                cb_sum_dk_b, cb_rope_cos, cb_rope_cos_product, dim, dim);
            sfpu_binary_tile_to_cb<0, true>(
                cb_sum_dk_b, cb_rope_sin, cb_rope_sin_product, peer, peer);
            if (dim == 0)
                fpu_add_tile_to_cb(
                    cb_rope_cos_product, cb_rope_sin_product, cb_rope_k, 0, 0);
            else
                fpu_sub_tile_to_cb(
                    cb_rope_cos_product, cb_rope_sin_product, cb_rope_k, 0);
            CircularBuffer cos_product(cb_rope_cos_product);
            CircularBuffer sin_product(cb_rope_sin_product);
            cos_product.pop_front(1);
            sin_product.pop_front(1);
        }
    }
    rope_cos.pop_front(dim_tiles);
    rope_sin.pop_front(dim_tiles);

    p_column.pop_front(seq_tiles);
    dout_row.pop_front(dim_tiles);
    dout_full.pop_front(vector_tiles);
    q_full.pop_front(vector_tiles);
    k.pop_front(vector_tiles);
    v.pop_front(vector_tiles);
#endif
}
}  // namespace resident_layer_phase_attention_4

void kernel_main() {
    resident_layer_phase_attention_4::run();
}
