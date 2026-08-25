// Exact FP32 add boundary for the split capturable-AdamW parameter update.

#define FP32_DEST_ACC_EN 1

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t total_blocks = @TOTAL_BLOCKS@;
    constexpr uint32_t blocks_per_node = @BLOCKS_PER_NODE@;
    constexpr uint32_t rows_per_block = @ROWS_PER_BLOCK@;
    constexpr uint32_t grid_columns = 8;
    CircularBuffer lhs(0), rhs(1), output(2);
    const uint32_t node =
        get_absolute_logical_y() * grid_columns + get_absolute_logical_x();

    DeviceZoneScopedN("adamw_exact_parameter_add");
    for (uint32_t local_block = 0; local_block < blocks_per_node;
         ++local_block) {
        const uint32_t block = node * blocks_per_node + local_block;
        if (block >= total_blocks) {
            continue;
        }
        lhs.wait_front(rows_per_block);
        rhs.wait_front(rows_per_block);
        output.reserve_back(rows_per_block);
        unary_op_init_common(0, 2);
        for (uint32_t tile = 0; tile < rows_per_block; ++tile) {
            tile_regs_acquire();
            copy_tile_init(0);
            copy_tile(0, tile, 0);
            copy_tile_to_dst_init_short_with_dt(0, 1);
            copy_tile(1, tile, 1);
            add_tiles_init(0, 1);
            add_tiles(0, 1, tile, tile, 0);
            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(2)));
            pack_tile<true>(0, 2, tile);
            tile_regs_release();
        }
        output.push_back(rows_per_block);
        rhs.pop_front(rows_per_block);
        lhs.pop_front(rows_per_block);
    }
}
