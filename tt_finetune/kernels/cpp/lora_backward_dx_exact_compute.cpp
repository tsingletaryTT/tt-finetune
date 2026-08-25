// Hand-written TT-Metal compute kernel for the exact LoRA dX path.
//
//   scaled_dY = BF16(dY * scale)
//   dH = scaled_dY @ B
//   dX = dH @ A
//
// The long output-feature reduction mirrors the exact forward kernel:
// HiFi4, FP32 destination accumulation, two K tiles per phase, and FP32
// partial spill/reload. dH is packed once to BF16 and remains in L1 for the
// rank-reduction dX matmul.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_dy = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaled_dy = get_compile_time_arg_val(2);
    constexpr uint32_t cb_partial_fp32 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_a = get_compile_time_arg_val(4);
    constexpr uint32_t cb_dx = get_compile_time_arg_val(5);

    constexpr uint32_t reduction_block_tiles = 2;
    constexpr uint32_t num_n_blocks = @NUM_N_BLOCKS@;
    constexpr uint32_t k_tiles = @K_TILES@;
    constexpr uint32_t second_subblock_w = @SECOND_SUBBLOCK_W@;
    constexpr uint32_t m_tiles_per_node = @M_TILES_PER_NODE@;
    constexpr uint32_t total_m_tiles = @TOTAL_M_TILES@;
    constexpr uint32_t grid_cols = @GRID_COLS@;
    constexpr uint32_t scale_bits = @SCALE_BITS@;

    CircularBuffer dy_cb(cb_dy);
    CircularBuffer b_cb(cb_b);
    CircularBuffer scaled_dy_cb(cb_scaled_dy);
    CircularBuffer partial_cb(cb_partial_fp32);
    CircularBuffer a_cb(cb_a);
    CircularBuffer dx_cb(cb_dx);

    const uint32_t node_index =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();

    for (uint32_t local_m = 0; local_m < m_tiles_per_node; ++local_m) {
        const uint32_t m_tile = node_index * m_tiles_per_node + local_m;
        if (m_tile >= total_m_tiles) {
            continue;
        }

        // Preserve the generic graph's BF16 boundary after dY * scale.
        constexpr uint32_t n_tiles =
            num_n_blocks * reduction_block_tiles;
        dy_cb.wait_front(n_tiles);
        scaled_dy_cb.reserve_back(n_tiles);
        compute_kernel_hw_startup(cb_dy, cb_scaled_dy);
        uint32_t tile = 0;
        for (; tile + 1 < n_tiles; tile += 2) {
            init_sfpu(cb_dy, cb_scaled_dy);
            tile_regs_acquire();
            copy_tile_init(cb_dy);
            copy_tile(cb_dy, tile, 0);
            copy_tile(cb_dy, tile + 1, 2);
            binop_with_scalar_tile_init();
            {
                volatile int32_t scalar = scale_bits;
                mul_unary_tile(0, scalar);
                mul_unary_tile(2, scalar);
            }
            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(cb_scaled_dy)));
            pack_tile<true>(0, cb_scaled_dy, tile);
            pack_tile<true>(2, cb_scaled_dy, tile + 1);
            tile_regs_release();
        }
        dy_cb.pop_front(n_tiles);
        scaled_dy_cb.push_back(n_tiles);

        // Exact long-N reduction for dH = scaled_dY @ B.
        compute_kernel_hw_startup<SrcOrder::Reverse>(
            cb_scaled_dy, cb_b, cb_partial_fp32);
        matmul_block_init(
            cb_scaled_dy,
            cb_b,
            false,
            1,
            1,
            reduction_block_tiles);

        for (uint32_t block = 0; block < num_n_blocks; ++block) {
            scaled_dy_cb.wait_front(reduction_block_tiles);
            b_cb.wait_front(reduction_block_tiles);

            tile_regs_acquire();
            if (block > 0) {
                copy_tile_to_dst_init_short_with_dt(
                    cb_b, cb_partial_fp32);
                partial_cb.wait_front(1);
                copy_block_matmul_partials(
                    cb_partial_fp32, 0, 0, 1);
                partial_cb.pop_front(1);
                reconfig_data_format_srca(cb_partial_fp32, cb_b);
                matmul_block_init(
                    cb_scaled_dy,
                    cb_b,
                    false,
                    1,
                    1,
                    reduction_block_tiles);
            }

            for (
                uint32_t inner = 0;
                inner < reduction_block_tiles;
                ++inner
            ) {
                matmul_block(
                    cb_scaled_dy,
                    cb_b,
                    inner,
                    inner,
                    0,
                    false,
                    1,
                    1,
                    reduction_block_tiles);
            }

            tile_regs_commit();
            const bool last = block == num_n_blocks - 1;
            const uint32_t target_cb =
                last ? cb_scaled_dy : cb_partial_fp32;
            if (last) {
                scaled_dy_cb.reserve_back(1);
            } else {
                partial_cb.reserve_back(1);
            }
            tile_regs_wait();
            PACK((pack_reconfig_data_format(target_cb)));
            pack_tile_block(0, target_cb, 1);
            tile_regs_release();
            if (last) {
                scaled_dy_cb.push_back(1);
            } else {
                partial_cb.push_back(1);
            }

            b_cb.pop_front(reduction_block_tiles);
            scaled_dy_cb.pop_front(reduction_block_tiles);
        }

        // Rank-space reduction for dX = dH @ A. Blackhole exposes four FP32
        // destination tiles, so walk the input-feature width in <=4 tiles.
        scaled_dy_cb.wait_front(1);
        a_cb.wait_front(k_tiles);
        dx_cb.reserve_back(k_tiles);
        compute_kernel_hw_startup<SrcOrder::Reverse>(
            cb_scaled_dy, cb_a, cb_dx);
        for (
            uint32_t k_start = 0;
            k_start < k_tiles;
            k_start += second_subblock_w
        ) {
            matmul_block_init(
                cb_scaled_dy,
                cb_a,
                false,
                second_subblock_w,
                1,
                1);
            tile_regs_acquire();
            matmul_block(
                cb_scaled_dy,
                cb_a,
                0,
                k_start,
                0,
                false,
                second_subblock_w,
                1,
                1);
            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(cb_dx)));
            for (
                uint32_t out_tile = 0;
                out_tile < second_subblock_w;
                ++out_tile
            ) {
                pack_tile<true>(
                    out_tile,
                    cb_dx,
                    k_start + out_tile);
            }
            tile_regs_release();
        }
        dx_cb.push_back(k_tiles);
        a_cb.pop_front(k_tiles);
        scaled_dy_cb.pop_front(1);
    }
}
