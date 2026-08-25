// Hand-written TT-Metal compute kernel for the exact LoRA dA path.
//
//   dA = transpose(transpose(X) @ dH)
//
// This mirrors Linear autograd's observable operand order instead of the
// algebraically equivalent dH.T @ X expression. X is transposed tile-by-tile
// in L1. The complete token reduction uses HiFi4, FP32 destination
// accumulation, two K tiles per phase, and FP32 partial spill/reload.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_dh = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t cb_transposed_dh = get_compile_time_arg_val(2);
    constexpr uint32_t cb_partial_fp32 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_pretranspose_da = get_compile_time_arg_val(4);
    constexpr uint32_t cb_da = get_compile_time_arg_val(5);

    constexpr uint32_t reduction_tiles = @REDUCTION_TILES@;
    constexpr uint32_t reduction_block_tiles = 2;
    constexpr uint32_t reduction_blocks =
        reduction_tiles / reduction_block_tiles;
    constexpr uint32_t rank_tiles = @RANK_TILES@;
    constexpr uint32_t k_tiles_per_node = @K_TILES_PER_NODE@;
    constexpr uint32_t total_k_tiles = @TOTAL_K_TILES@;
    constexpr uint32_t grid_cols = @GRID_COLS@;

    static_assert(
        reduction_tiles % reduction_block_tiles == 0,
        "exact dA requires an even token tile count");
    static_assert(
        rank_tiles == 1,
        "initial exact dA kernel supports one physical rank tile");

    CircularBuffer dh_cb(cb_dh);
    CircularBuffer x_cb(cb_x);
    CircularBuffer transposed_x_cb(cb_transposed_dh);
    CircularBuffer partial_cb(cb_partial_fp32);
    CircularBuffer pretranspose_da_cb(cb_pretranspose_da);
    CircularBuffer da_cb(cb_da);

    const uint32_t node_index =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();

    for (uint32_t local_k = 0; local_k < k_tiles_per_node; ++local_k) {
        const uint32_t k_tile = node_index * k_tiles_per_node + local_k;
        if (k_tile >= total_k_tiles) {
            continue;
        }

        // Linear autograd computes X.T @ dH, then transposes the result.
        x_cb.wait_front(reduction_tiles);
        transposed_x_cb.reserve_back(reduction_tiles);
        compute_kernel_hw_startup(cb_x, cb_transposed_dh);
        transpose_init(cb_x);
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
                transpose_tile(cb_x, tile_start + tile, tile);
            }
            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(cb_transposed_dh)));
            for (uint32_t tile = 0; tile < chunk; ++tile) {
                pack_tile<true>(
                    tile,
                    cb_transposed_dh,
                    tile_start + tile);
            }
            tile_regs_release();
        }
        x_cb.pop_front(reduction_tiles);
        transposed_x_cb.push_back(reduction_tiles);

        // Exact token reduction into one rank-by-input-feature output tile.
        compute_kernel_hw_startup<SrcOrder::Reverse>(
            cb_transposed_dh,
            cb_dh,
            cb_partial_fp32);
        matmul_block_init(
            cb_transposed_dh,
            cb_dh,
            false,
            1,
            1,
            reduction_block_tiles);

        for (uint32_t block = 0; block < reduction_blocks; ++block) {
            transposed_x_cb.wait_front(reduction_block_tiles);
            dh_cb.wait_front(reduction_block_tiles);

            tile_regs_acquire();
            if (block > 0) {
                copy_tile_to_dst_init_short_with_dt(
                    cb_dh, cb_partial_fp32);
                partial_cb.wait_front(1);
                copy_block_matmul_partials(
                    cb_partial_fp32, 0, 0, 1);
                partial_cb.pop_front(1);
                reconfig_data_format_srca(cb_partial_fp32, cb_dh);
                matmul_block_init(
                    cb_transposed_dh,
                    cb_dh,
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
                    cb_transposed_dh,
                    cb_dh,
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
            // The matmul output is [K_tile, R_tile]. Stage the final tile
            // in the now-empty transpose CB, then transpose it to PEFT's
            // [R_tile, K_tile] orientation.
            const uint32_t target_cb =
                last ? cb_pretranspose_da : cb_partial_fp32;
            if (last) {
                pretranspose_da_cb.reserve_back(1);
            } else {
                partial_cb.reserve_back(1);
            }
            tile_regs_wait();
            PACK((pack_reconfig_data_format(target_cb)));
            pack_tile_block(0, target_cb, 1);
            tile_regs_release();
            if (last) {
                pretranspose_da_cb.push_back(1);
            } else {
                partial_cb.push_back(1);
            }

            dh_cb.pop_front(reduction_block_tiles);
            transposed_x_cb.pop_front(reduction_block_tiles);
        }

        pretranspose_da_cb.wait_front(1);
        da_cb.reserve_back(1);
        compute_kernel_hw_startup(cb_pretranspose_da, cb_da);
        transpose_init(cb_pretranspose_da);
        tile_regs_acquire();
        transpose_tile(cb_pretranspose_da, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(cb_da)));
        pack_tile<true>(0, cb_da, 0);
        tile_regs_release();
        da_cb.push_back(1);
        pretranspose_da_cb.pop_front(1);
    }
}
