// Exact capturable-AdamW epsilon phase:
//   FP32 denominator + FP32 scaled_epsilon.

#define FP32_DEST_ACC_EN 1

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace {
constexpr uint32_t cb_denominator = 0;
constexpr uint32_t cb_epsilon = 1;
constexpr uint32_t cb_output = 2;
}  // namespace

void kernel_main() {
    constexpr uint32_t total_blocks = @TOTAL_BLOCKS@;
    constexpr uint32_t blocks_per_node = @BLOCKS_PER_NODE@;
    constexpr uint32_t rows_per_block = @ROWS_PER_BLOCK@;
    constexpr uint32_t grid_columns = 8;

    CircularBuffer denominator(cb_denominator);
    CircularBuffer epsilon(cb_epsilon);
    CircularBuffer output(cb_output);
    const uint32_t node =
        get_absolute_logical_y() * grid_columns + get_absolute_logical_x();

    epsilon.wait_front(1);
    DeviceZoneScopedN("adamw_exact_epsilon");
    for (uint32_t local_block = 0; local_block < blocks_per_node;
         ++local_block) {
        const uint32_t block = node * blocks_per_node + local_block;
        if (block >= total_blocks) {
            continue;
        }
        denominator.wait_front(rows_per_block);
        output.reserve_back(rows_per_block);
        unary_op_init_common(cb_denominator, cb_output);
        for (uint32_t tile = 0; tile < rows_per_block; ++tile) {
            tile_regs_acquire();
            copy_tile_to_dst_init_short_with_dt(
                cb_epsilon, cb_denominator);
            copy_tile(cb_denominator, tile, 0);
            copy_tile_to_dst_init_short_with_dt(cb_denominator, cb_epsilon);
            copy_tile(cb_epsilon, 0, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(cb_output)));
            pack_tile<true>(0, cb_output, tile);
            tile_regs_release();
        }
        output.push_back(rows_per_block);
        denominator.pop_front(rows_per_block);
    }
    epsilon.pop_front(1);
}
