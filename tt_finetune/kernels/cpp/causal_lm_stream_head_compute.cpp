// Exact frozen LM-head input gradient while dLogits remains resident:
//     dNormalized[M,K] = dLogits[M,N] @ W[N,K]
//
// Each row owns a six-core group. A core retains one FP32 partial per assigned
// hidden tile and advances it in ascending vocabulary-tile order. Multiple
// independent hidden jobs occupy separate destination tiles so the exact TTNN
// K-phase boundary is preserved without replaying resident dLogits.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_dlogits = 0;
    constexpr uint32_t cb_result = 1;
    constexpr uint32_t cb_partial = 2;
    constexpr uint32_t cb_weight = 7;
    constexpr uint32_t vocabulary_tiles = @VOCAB_TILES@;
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@;
    constexpr uint32_t head_workers = @HEAD_WORKERS@;
    constexpr uint32_t head_start_x = @HEAD_START_X@;
    constexpr uint32_t reduction_tiles_per_phase =
        @HEAD_REDUCTION_TILES_PER_PHASE@;
    constexpr uint32_t reduction_phases =
        vocabulary_tiles / reduction_tiles_per_phase;

    const uint32_t worker = get_absolute_logical_x() - head_start_x;
    const uint32_t local_jobs =
        (hidden_tiles + head_workers - 1) / head_workers;
    const uint32_t remainder = hidden_tiles % head_workers;
    const uint32_t valid_jobs =
        remainder == 0 || worker < remainder ? local_jobs : local_jobs - 1;

    CircularBuffer dlogits(cb_dlogits);
    CircularBuffer weight(cb_weight);
    CircularBuffer partial(cb_partial);
    CircularBuffer result(cb_result);

    compute_kernel_hw_startup<SrcOrder::Reverse>(
        cb_dlogits, cb_weight, cb_partial);
    matmul_block_init(cb_dlogits, cb_weight, false, 1, 1, 1);

    for (uint32_t phase = 0; phase < reduction_phases; ++phase) {
        dlogits.wait_front(reduction_tiles_per_phase);
        for (
            uint32_t local_job = 0;
            local_job < valid_jobs;
            ++local_job
        ) {
            tile_regs_acquire();
            if (phase > 0) {
                copy_tile_to_dst_init_short_with_dt(
                    cb_weight, cb_partial);
                partial.wait_front(1);
                copy_block_matmul_partials(cb_partial, 0, 0, 1);
                partial.pop_front(1);
                reconfig_data_format_srca(cb_partial, cb_weight);
                matmul_block_init(
                    cb_dlogits, cb_weight, false, 1, 1, 1);
            }

            for (
                uint32_t inner = 0;
                inner < reduction_tiles_per_phase;
                ++inner
            ) {
                weight.wait_front(1);
                matmul_block(
                    cb_dlogits,
                    cb_weight,
                    inner,
                    0,
                    0,
                    false,
                    1,
                    1,
                    1);
                weight.pop_front(1);
            }
            tile_regs_commit();

            const bool last = phase == reduction_phases - 1;
            const uint32_t target = last ? cb_result : cb_partial;
            if (last) result.reserve_back(1);
            else partial.reserve_back(1);
            tile_regs_wait();
            PACK((pack_reconfig_data_format(target)));
            pack_tile_block(0, target, 1);
            tile_regs_release();
            if (last) result.push_back(1);
            else partial.push_back(1);
        }
        dlogits.pop_front(reduction_tiles_per_phase);
    }
}
