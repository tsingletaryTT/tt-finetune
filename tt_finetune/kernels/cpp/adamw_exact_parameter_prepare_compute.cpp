// Split exact capturable-AdamW parameter preparation.
//
// Emits the two FP32 terms consumed by the final add:
//   term 0 = FP32(BF16(parameter * decay))
//   term 1 = FP32(BF16 first_moment) / FP32 denominator

// The BF16 decay checkpoint mirrors the reference graph.  The split is an
// exactness qualification path; once accepted, the same schedules can be
// placed in one resident program without public intermediate tensors.

#define FP32_DEST_ACC_EN 1

#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace {
constexpr uint32_t cb_param = 0;
constexpr uint32_t cb_exp_avg = 1;
constexpr uint32_t cb_denominator = 2;
constexpr uint32_t cb_decay = 3;
constexpr uint32_t cb_decay_bf16 = 4;
constexpr uint32_t cb_decayed_out = 5;
constexpr uint32_t cb_update_out = 6;
}  // namespace

void kernel_main() {
    constexpr uint32_t total_blocks = @TOTAL_BLOCKS@;
    constexpr uint32_t blocks_per_node = @BLOCKS_PER_NODE@;
    constexpr uint32_t rows_per_block = @ROWS_PER_BLOCK@;
    constexpr uint32_t grid_columns = 8;

    CircularBuffer param(cb_param), exp_avg(cb_exp_avg),
        denominator(cb_denominator), decay(cb_decay),
        decay_bf16(cb_decay_bf16), decayed_out(cb_decayed_out),
        update_out(cb_update_out);
    const uint32_t node =
        get_absolute_logical_y() * grid_columns + get_absolute_logical_x();

    decay.wait_front(1);
    DeviceZoneScopedN("adamw_exact_parameter_prepare");
    for (uint32_t local_block = 0; local_block < blocks_per_node;
         ++local_block) {
        const uint32_t block = node * blocks_per_node + local_block;
        if (block >= total_blocks) {
            continue;
        }
        param.wait_front(rows_per_block);
        exp_avg.wait_front(rows_per_block);
        denominator.wait_front(rows_per_block);
        decayed_out.reserve_back(rows_per_block);
        update_out.reserve_back(rows_per_block);

        for (uint32_t tile = 0; tile < rows_per_block; ++tile) {
            decay_bf16.reserve_back(1);
            unary_op_init_common(cb_param, cb_decay_bf16);
            tile_regs_acquire();
            copy_tile_to_dst_init_short_with_dt(cb_decay, cb_param);
            copy_tile(cb_param, tile, 0);
            copy_tile_to_dst_init_short_with_dt(cb_param, cb_decay);
            copy_tile(cb_decay, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            typecast_tile_init<
                (uint32_t)DataFormat::Float32,
                (uint32_t)DataFormat::Float16_b>();
            typecast_tile<
                (uint32_t)DataFormat::Float32,
                (uint32_t)DataFormat::Float16_b>(0);
            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(cb_decay_bf16)));
            pack_tile<true>(0, cb_decay_bf16, 0);
            tile_regs_release();
            decay_bf16.push_back(1);

            decay_bf16.wait_front(1);
            unary_op_init_common(cb_decay_bf16, cb_decayed_out);
            tile_regs_acquire();
            copy_tile_init(cb_decay_bf16);
            copy_tile(cb_decay_bf16, 0, 0);
            typecast_tile_init<
                (uint32_t)DataFormat::Float16_b,
                (uint32_t)DataFormat::Float32>();
            typecast_tile<
                (uint32_t)DataFormat::Float16_b,
                (uint32_t)DataFormat::Float32>(0);
            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(cb_decayed_out)));
            pack_tile<true>(0, cb_decayed_out, tile);
            tile_regs_release();
            decay_bf16.pop_front(1);

            unary_op_init_common(cb_exp_avg, cb_update_out);
            tile_regs_acquire();
            copy_tile_init(cb_exp_avg);
            copy_tile(cb_exp_avg, tile, 0);
            typecast_tile_init<
                (uint32_t)DataFormat::Float16_b,
                (uint32_t)DataFormat::Float32>();
            typecast_tile<
                (uint32_t)DataFormat::Float16_b,
                (uint32_t)DataFormat::Float32>(0);
            copy_tile_to_dst_init_short_with_dt(
                cb_exp_avg, cb_denominator);
            copy_tile(cb_denominator, tile, 1);
            div_binary_tile_init();
            div_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(cb_update_out)));
            pack_tile<true>(0, cb_update_out, tile);
            tile_regs_release();
        }
        decayed_out.push_back(rows_per_block);
        update_out.push_back(rows_per_block);
        denominator.pop_front(rows_per_block);
        exp_avg.pop_front(rows_per_block);
        param.pop_front(rows_per_block);
    }
    decay.pop_front(1);
}
