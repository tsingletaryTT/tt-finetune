// Bit-exact frozen Linear input gradient:
//     dX[M,K] = dY[M,N] @ W[N,K]
//
// The generic Blackhole matmul contract spills its FP32 partial at a
// shape-specific K phase boundary. The kernel intentionally has no dW path.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_dy = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(1);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(2);
    constexpr uint32_t cb_result = get_compile_time_arg_val(3);

    constexpr uint32_t jobs_per_node = @JOBS_PER_NODE@;
    constexpr uint32_t total_jobs = @TOTAL_JOBS@;
    constexpr uint32_t k_tiles = @K_TILES@;
    constexpr uint32_t n_tiles = @N_TILES@;
    constexpr uint32_t reduction_tiles_per_phase =
        @REDUCTION_TILES_PER_PHASE@;
    constexpr uint32_t reduction_blocks =
        n_tiles / reduction_tiles_per_phase;
    constexpr bool retain_destination_across_reduction =
        @RETAIN_DESTINATION_ACROSS_REDUCTION@;
    constexpr bool packer_l1_accumulate_across_reduction =
        @PACKER_L1_ACCUMULATE_ACROSS_REDUCTION@;
    constexpr uint32_t grid_cols = @GRID_COLS@;

    CircularBuffer dy(cb_dy);
    CircularBuffer weight(cb_weight);
    CircularBuffer partial(cb_partial);
    CircularBuffer result(cb_result);

    const uint32_t node_index =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();

    compute_kernel_hw_startup<SrcOrder::Reverse>(
        cb_dy,
        cb_weight,
        retain_destination_across_reduction ? cb_result : cb_partial);
    if constexpr (retain_destination_across_reduction) {
        matmul_init(cb_dy, cb_weight);
    } else {
        matmul_block_init(
            cb_dy,
            cb_weight,
            false,
            1,
            1,
            reduction_tiles_per_phase);
    }

    for (uint32_t local_job = 0; local_job < jobs_per_node; ++local_job) {
        const uint32_t job = node_index * jobs_per_node + local_job;
        if (job >= total_jobs) {
            continue;
        }

        if constexpr (retain_destination_across_reduction) {
            tile_regs_acquire();
            for (
                uint32_t block = 0;
                block < reduction_blocks;
                ++block
            ) {
                dy.wait_front(reduction_tiles_per_phase);
                weight.wait_front(reduction_tiles_per_phase);
                for (
                    uint32_t inner = 0;
                    inner < reduction_tiles_per_phase;
                    ++inner
                ) {
                    matmul_tiles(
                        cb_dy, cb_weight, inner, inner, 0);
                }
                weight.pop_front(reduction_tiles_per_phase);
                dy.pop_front(reduction_tiles_per_phase);
            }
            tile_regs_commit();
            result.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, cb_result);
            tile_regs_release();
            result.push_back(1);
            continue;
        }

        // TTNN's wide, narrow-M 1D systolic matmul uses the packer's FP32
        // L1 read/modify/write path for all but the final reduction block.
        // The one-tile partial CB deliberately wraps to the same physical L1
        // address after every push/pop.  Only the penultimate partial remains
        // logically visible and is reloaded into destination registers for
        // the final block, exactly matching bmm_large_block_zm.
        if constexpr (packer_l1_accumulate_across_reduction) {
            bool enable_reload = false;
            PACK((pack_reconfig_data_format(cb_partial)));
            PACK((llk_pack_reconfig_l1_acc(0)));

            for (uint32_t block = 0; block < reduction_blocks; ++block) {
                dy.wait_front(reduction_tiles_per_phase);
                weight.wait_front(reduction_tiles_per_phase);
                tile_regs_acquire();

                if (enable_reload) {
                    copy_tile_to_dst_init_short_with_dt(
                        cb_weight, cb_partial);
                    partial.wait_front(1);
                    copy_block_matmul_partials(cb_partial, 0, 0, 1);
                    partial.pop_front(1);
                    reconfig_data_format_srca(cb_partial, cb_weight);
                    matmul_block_init(
                        cb_dy,
                        cb_weight,
                        false,
                        1,
                        1,
                        reduction_tiles_per_phase);
                }

                for (
                    uint32_t inner = 0;
                    inner < reduction_tiles_per_phase;
                    ++inner
                ) {
                    matmul_block(
                        cb_dy,
                        cb_weight,
                        inner,
                        inner,
                        0,
                        false,
                        1,
                        1,
                        reduction_tiles_per_phase);
                }
                tile_regs_commit();

                const bool last = block == reduction_blocks - 1;
                if (last) {
                    result.reserve_back(1);
                    tile_regs_wait();
                    PACK((pack_reconfig_data_format(cb_result)));
                    PACK((llk_pack_reconfig_l1_acc(0)));
                    pack_tile_block(0, cb_result, 1);
                    tile_regs_release();
                    result.push_back(1);
                } else {
                    partial.reserve_back(1);
                    tile_regs_wait();
                    if (block == 0) {
                        PACK((llk_pack_reconfig_l1_acc(0)));
                    } else if (block == 1) {
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }
                    pack_tile_block(0, cb_partial, 1);
                    tile_regs_release();
                    partial.push_back(1);

                    if (block < reduction_blocks - 2) {
                        partial.wait_front(1);
                        partial.pop_front(1);
                    }
                    if (block == reduction_blocks - 2) {
                        enable_reload = true;
                    }
                }

                weight.pop_front(reduction_tiles_per_phase);
                dy.pop_front(reduction_tiles_per_phase);
            }
            continue;
        }

        for (uint32_t block = 0; block < reduction_blocks; ++block) {
            dy.wait_front(reduction_tiles_per_phase);
            weight.wait_front(reduction_tiles_per_phase);
            tile_regs_acquire();
            if (block > 0) {
                copy_tile_to_dst_init_short_with_dt(
                    cb_weight, cb_partial);
                partial.wait_front(1);
                copy_block_matmul_partials(cb_partial, 0, 0, 1);
                partial.pop_front(1);
                reconfig_data_format_srca(cb_partial, cb_weight);
                matmul_block_init(
                    cb_dy,
                    cb_weight,
                    false,
                    1,
                    1,
                    reduction_tiles_per_phase);
            }

            for (
                uint32_t inner = 0;
                inner < reduction_tiles_per_phase;
                ++inner
            ) {
                matmul_block(
                    cb_dy,
                    cb_weight,
                    inner,
                    inner,
                    0,
                    false,
                    1,
                    1,
                    reduction_tiles_per_phase);
            }
            tile_regs_commit();

            const bool last = block == reduction_blocks - 1;
            const uint32_t target_cb = last ? cb_result : cb_partial;
            if (last) {
                result.reserve_back(1);
            } else {
                partial.reserve_back(1);
            }
            tile_regs_wait();
            PACK((pack_reconfig_data_format(target_cb)));
            pack_tile_block(0, target_cb, 1);
            tile_regs_release();
            if (last) {
                result.push_back(1);
            } else {
                partial.push_back(1);
            }
            weight.pop_front(reduction_tiles_per_phase);
            dy.pop_front(reduction_tiles_per_phase);
        }
    }
}
