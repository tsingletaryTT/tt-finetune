// Exact FP32-to-BF16 parameter commit while forwarding BF16 moments.

#define FP32_DEST_ACC_EN 1

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace {
ALWI void copy_one(uint32_t input_id, uint32_t input_tile,
                   uint32_t output_id, uint32_t output_tile) {
    init_sfpu(input_id, output_id);
    tile_regs_acquire();
    copy_tile_init(input_id);
    copy_tile(input_id, input_tile, 0);
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(output_id)));
    pack_tile<true>(0, output_id, output_tile);
    tile_regs_release();
}
}  // namespace

void kernel_main() {
    constexpr uint32_t total_blocks = @TOTAL_BLOCKS@;
    constexpr uint32_t blocks_per_node = @BLOCKS_PER_NODE@;
    constexpr uint32_t rows_per_block = @ROWS_PER_BLOCK@;
    constexpr uint32_t grid_columns = 8;
    CircularBuffer sum(0), exp_avg(1), exp_avg_sq(2),
        param_out(3), exp_avg_out(4), exp_avg_sq_out(5);
    const uint32_t node =
        get_absolute_logical_y() * grid_columns + get_absolute_logical_x();

    DeviceZoneScopedN("adamw_exact_parameter_commit");
    for (uint32_t local_block = 0; local_block < blocks_per_node;
         ++local_block) {
        const uint32_t block = node * blocks_per_node + local_block;
        if (block >= total_blocks) {
            continue;
        }
        sum.wait_front(rows_per_block);
        exp_avg.wait_front(rows_per_block);
        exp_avg_sq.wait_front(rows_per_block);
        param_out.reserve_back(rows_per_block);
        exp_avg_out.reserve_back(rows_per_block);
        exp_avg_sq_out.reserve_back(rows_per_block);
        for (uint32_t tile = 0; tile < rows_per_block; ++tile) {
            unary_op_init_common(0, 3);
            tile_regs_acquire();
            copy_tile_init(0);
            copy_tile(0, tile, 0);
            typecast_tile_init<
                (uint32_t)DataFormat::Float32,
                (uint32_t)DataFormat::Float16_b>();
            typecast_tile<
                (uint32_t)DataFormat::Float32,
                (uint32_t)DataFormat::Float16_b>(0);
            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(3)));
            pack_tile<true>(0, 3, tile);
            tile_regs_release();
            copy_one(1, tile, 4, tile);
            copy_one(2, tile, 5, tile);
        }
        param_out.push_back(rows_per_block);
        exp_avg_out.push_back(rows_per_block);
        exp_avg_sq_out.push_back(rows_per_block);
        exp_avg_sq.pop_front(rows_per_block);
        exp_avg.pop_front(rows_per_block);
        sum.pop_front(rows_per_block);
    }
}
