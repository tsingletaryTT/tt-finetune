// Hand-written TT-Metal compute kernel for the exact LoRA dB path.
//
//   scaled_dY = BF16(dY * scale)
//   dB = transpose(transpose(H) @ scaled_dY)
//
// This mirrors Linear autograd's observable operand order instead of the
// algebraically equivalent scaled_dY.T @ H expression. H is transposed in
// L1 and the token reduction uses
// the same HiFi4/FP32-destination/two-K-tile partial-spill schedule as the
// accepted exact native LoRA forward and dX kernels.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_dy = get_compile_time_arg_val(0);
    constexpr uint32_t cb_hidden = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaled_dy = get_compile_time_arg_val(2);
    constexpr uint32_t cb_transposed_dy = get_compile_time_arg_val(3);
    constexpr uint32_t cb_partial_fp32 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_pretranspose_db = get_compile_time_arg_val(5);
    constexpr uint32_t cb_db = get_compile_time_arg_val(6);

    constexpr uint32_t reduction_tiles = @REDUCTION_TILES@;
    constexpr uint32_t reduction_block_tiles = 2;
    constexpr uint32_t reduction_blocks =
        reduction_tiles / reduction_block_tiles;
    constexpr uint32_t rank_tiles = @RANK_TILES@;
    constexpr uint32_t n_tiles_per_node = @N_TILES_PER_NODE@;
    constexpr uint32_t total_n_tiles = @TOTAL_N_TILES@;
    constexpr uint32_t grid_cols = @GRID_COLS@;
    constexpr uint32_t scale_bits = @SCALE_BITS@;

    static_assert(
        reduction_tiles % reduction_block_tiles == 0,
        "exact dB requires an even token tile count");
    static_assert(
        rank_tiles == 1,
        "initial exact dB kernel supports one physical rank tile");

    CircularBuffer dy_cb(cb_dy);
    CircularBuffer hidden_cb(cb_hidden);
    CircularBuffer scaled_dy_cb(cb_scaled_dy);
    CircularBuffer transposed_hidden_cb(cb_transposed_dy);
    CircularBuffer partial_cb(cb_partial_fp32);
    CircularBuffer pretranspose_db_cb(cb_pretranspose_db);
    CircularBuffer db_cb(cb_db);

    const uint32_t node_index =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();

    for (uint32_t local_n = 0; local_n < n_tiles_per_node; ++local_n) {
        const uint32_t n_tile = node_index * n_tiles_per_node + local_n;
        if (n_tile >= total_n_tiles) {
            continue;
        }

        // Preserve the generic graph's BF16 boundary after dY * scale.
        dy_cb.wait_front(reduction_tiles);
        scaled_dy_cb.reserve_back(reduction_tiles);
        compute_kernel_hw_startup(cb_dy, cb_scaled_dy);
        for (
            uint32_t tile = 0;
            tile < reduction_tiles;
            tile += 2
        ) {
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
        dy_cb.pop_front(reduction_tiles);
        scaled_dy_cb.push_back(reduction_tiles);

        // Linear autograd computes H.T @ scaled_dY, then transposes.
        hidden_cb.wait_front(reduction_tiles);
        transposed_hidden_cb.reserve_back(reduction_tiles);
        compute_kernel_hw_startup(cb_hidden, cb_transposed_dy);
        transpose_init(cb_hidden);
        for (
            uint32_t tile_start = 0;
            tile_start < reduction_tiles;
            tile_start += 4
        ) {
            constexpr uint32_t max_chunk = 4;
            const uint32_t remaining = reduction_tiles - tile_start;
            const uint32_t chunk =
                remaining < max_chunk ? remaining : max_chunk;
            tile_regs_acquire();
            for (uint32_t tile = 0; tile < chunk; ++tile) {
                transpose_tile(
                    cb_hidden,
                    tile_start + tile,
                    tile);
            }
            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(cb_transposed_dy)));
            for (uint32_t tile = 0; tile < chunk; ++tile) {
                pack_tile<true>(
                    tile,
                    cb_transposed_dy,
                    tile_start + tile);
            }
            tile_regs_release();
        }
        hidden_cb.pop_front(reduction_tiles);
        transposed_hidden_cb.push_back(reduction_tiles);

        // Exact token reduction. A two-tile FP32 spill/reload boundary is
        // numerically observable in BF16 and deliberately mirrors TT-NN.
        compute_kernel_hw_startup<SrcOrder::Reverse>(
            cb_transposed_dy,
            cb_scaled_dy,
            cb_partial_fp32);
        matmul_block_init(
            cb_transposed_dy,
            cb_scaled_dy,
            false,
            1,
            1,
            reduction_block_tiles);

        for (uint32_t block = 0; block < reduction_blocks; ++block) {
            transposed_hidden_cb.wait_front(reduction_block_tiles);
            scaled_dy_cb.wait_front(reduction_block_tiles);

            tile_regs_acquire();
            if (block > 0) {
                copy_tile_to_dst_init_short_with_dt(
                    cb_scaled_dy, cb_partial_fp32);
                partial_cb.wait_front(1);
                copy_block_matmul_partials(
                    cb_partial_fp32, 0, 0, 1);
                partial_cb.pop_front(1);
                reconfig_data_format_srca(
                    cb_partial_fp32, cb_scaled_dy);
                matmul_block_init(
                    cb_transposed_dy,
                    cb_scaled_dy,
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
                    cb_transposed_dy,
                    cb_scaled_dy,
                    inner,
                    inner,
                    0,
                    false,
                    1,
                    1,
                    reduction_block_tiles);
            }
            tile_regs_commit();
            const bool last = block == reduction_blocks - 1;
            // Matmul produces [R_tile, N_tile]. Stage the final tile in the
            // emptied transpose CB, then transpose to [N_tile, R_tile].
            const uint32_t target_cb =
                last ? cb_pretranspose_db : cb_partial_fp32;
            if (last) {
                pretranspose_db_cb.reserve_back(1);
            } else {
                partial_cb.reserve_back(1);
            }
            tile_regs_wait();
            PACK((pack_reconfig_data_format(target_cb)));
            pack_tile_block(0, target_cb, 1);
            tile_regs_release();
            if (last) {
                pretranspose_db_cb.push_back(1);
            } else {
                partial_cb.push_back(1);
            }

            scaled_dy_cb.pop_front(reduction_block_tiles);
            transposed_hidden_cb.pop_front(reduction_block_tiles);
        }

        pretranspose_db_cb.wait_front(1);
        db_cb.reserve_back(1);
        compute_kernel_hw_startup(cb_pretranspose_db, cb_db);
        transpose_init(cb_pretranspose_db);
        tile_regs_acquire();
        transpose_tile(cb_pretranspose_db, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(cb_db)));
        pack_tile<true>(0, cb_db, 0);
        tile_regs_release();
        db_cb.push_back(1);
        pretranspose_db_cb.pop_front(1);
    }
}
