// Split exact capturable-AdamW final parameter add and BF16 conversion.

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
constexpr uint32_t cb_decayed = 0;
constexpr uint32_t cb_update = 1;
constexpr uint32_t cb_exp_avg = 2;
constexpr uint32_t cb_exp_avg_sq = 3;
constexpr uint32_t cb_param_out = 4;
constexpr uint32_t cb_exp_avg_out = 5;
constexpr uint32_t cb_exp_avg_sq_out = 6;
constexpr uint32_t cb_sum_f32 = 7;

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

    CircularBuffer decayed(cb_decayed), update(cb_update),
        exp_avg(cb_exp_avg), exp_avg_sq(cb_exp_avg_sq),
        param_out(cb_param_out), exp_avg_out(cb_exp_avg_out),
        exp_avg_sq_out(cb_exp_avg_sq_out), sum_f32(cb_sum_f32);
    const uint32_t node =
        get_absolute_logical_y() * grid_columns + get_absolute_logical_x();

    DeviceZoneScopedN("adamw_exact_parameter_finalize");
    for (uint32_t local_block = 0; local_block < blocks_per_node;
         ++local_block) {
        const uint32_t block = node * blocks_per_node + local_block;
        if (block >= total_blocks) {
            continue;
        }
        decayed.wait_front(rows_per_block);
        update.wait_front(rows_per_block);
        exp_avg.wait_front(rows_per_block);
        exp_avg_sq.wait_front(rows_per_block);
        param_out.reserve_back(rows_per_block);
        exp_avg_out.reserve_back(rows_per_block);
        exp_avg_sq_out.reserve_back(rows_per_block);

        for (uint32_t tile = 0; tile < rows_per_block; ++tile) {
            sum_f32.reserve_back(1);
            unary_op_init_common(cb_decayed, cb_sum_f32);
            tile_regs_acquire();
            copy_tile_init(cb_decayed);
            copy_tile(cb_decayed, tile, 0);
            copy_tile_to_dst_init_short_with_dt(cb_decayed, cb_update);
            copy_tile(cb_update, tile, 1);
            add_tiles_init(cb_decayed, cb_update);
            add_tiles(cb_decayed, cb_update, tile, tile, 0);
            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(cb_sum_f32)));
            pack_tile<true>(0, cb_sum_f32, 0);
            tile_regs_release();
            sum_f32.push_back(1);

            sum_f32.wait_front(1);
            unary_op_init_common(cb_sum_f32, cb_param_out);
            tile_regs_acquire();
            copy_tile_init(cb_sum_f32);
            copy_tile(cb_sum_f32, 0, 0);
            typecast_tile_init<
                (uint32_t)DataFormat::Float32,
                (uint32_t)DataFormat::Float16_b>();
            typecast_tile<
                (uint32_t)DataFormat::Float32,
                (uint32_t)DataFormat::Float16_b>(0);
            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(cb_param_out)));
            pack_tile<true>(0, cb_param_out, tile);
            tile_regs_release();
            sum_f32.pop_front(1);

            copy_one(cb_exp_avg, tile, cb_exp_avg_out, tile);
            copy_one(cb_exp_avg_sq, tile, cb_exp_avg_sq_out, tile);
        }
        param_out.push_back(rows_per_block);
        exp_avg_out.push_back(rows_per_block);
        exp_avg_sq_out.push_back(rows_per_block);
        exp_avg_sq.pop_front(rows_per_block);
        exp_avg.pop_front(rows_per_block);
        update.pop_front(rows_per_block);
        decayed.pop_front(rows_per_block);
    }
}
