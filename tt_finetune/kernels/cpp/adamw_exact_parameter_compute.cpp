// Exact capturable-AdamW parameter phase.
//
// Reproduces:
//   BF16(parameter * decay) -> FP32
//   + FP32(BF16 first_moment) / FP32 denominator
//   -> BF16 parameter
// while forwarding the already-updated BF16 moments unchanged.

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
constexpr uint32_t cb_exp_avg_sq = 2;
constexpr uint32_t cb_denominator = 3;
constexpr uint32_t cb_decay = 4;
constexpr uint32_t cb_param_out = 5;
constexpr uint32_t cb_exp_avg_out = 6;
constexpr uint32_t cb_exp_avg_sq_out = 7;
constexpr uint32_t cb_decay_bf16 = 8;
constexpr uint32_t cb_update_f32 = 9;
constexpr uint32_t cb_param_f32 = 10;
constexpr uint32_t cb_m_f32 = 11;
constexpr uint32_t cb_sum_f32 = 12;

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

    CircularBuffer param(cb_param), exp_avg(cb_exp_avg),
        exp_avg_sq(cb_exp_avg_sq), denominator(cb_denominator),
        decay(cb_decay), param_out(cb_param_out),
        exp_avg_out(cb_exp_avg_out), exp_avg_sq_out(cb_exp_avg_sq_out),
        decay_bf16(cb_decay_bf16), update_f32(cb_update_f32),
        param_f32(cb_param_f32), m_f32(cb_m_f32), sum_f32(cb_sum_f32);
    const uint32_t node =
        get_absolute_logical_y() * grid_columns + get_absolute_logical_x();

    decay.wait_front(1);
    DeviceZoneScopedN("adamw_exact_parameter");
    for (uint32_t local_block = 0; local_block < blocks_per_node;
         ++local_block) {
        const uint32_t block = node * blocks_per_node + local_block;
        if (block >= total_blocks) {
            continue;
        }
        param.wait_front(rows_per_block);
        exp_avg.wait_front(rows_per_block);
        exp_avg_sq.wait_front(rows_per_block);
        denominator.wait_front(rows_per_block);
        param_out.reserve_back(rows_per_block);
        exp_avg_out.reserve_back(rows_per_block);
        exp_avg_sq_out.reserve_back(rows_per_block);

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
            // The multiplication result is a BF16 graph value before the
            // following explicit FP32 conversion.
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

            // The reference materializes both BF16->F32 conversions, divide,
            // add and F32->BF16 conversion as separate TTNN programs.  Mirror
            // every numerical edge with private L1 CBs, but keep the entire
            // sequence inside one physical program.
            decay_bf16.wait_front(1);
            param_f32.reserve_back(1);
            init_sfpu(cb_decay_bf16, cb_param_f32);
            tile_regs_acquire();
            copy_tile(cb_decay_bf16, 0, 0);
            typecast_tile_init<
                (uint32_t)DataFormat::Float16_b,
                (uint32_t)DataFormat::Float32>();
            typecast_tile<
                (uint32_t)DataFormat::Float16_b,
                (uint32_t)DataFormat::Float32>(0);
            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(cb_param_f32)));
            pack_tile<true>(0, cb_param_f32, 0);
            tile_regs_release();
            param_f32.push_back(1);
            decay_bf16.pop_front(1);

            m_f32.reserve_back(1);
            init_sfpu(cb_exp_avg, cb_m_f32);
            tile_regs_acquire();
            copy_tile(cb_exp_avg, tile, 0);
            typecast_tile_init<
                (uint32_t)DataFormat::Float16_b,
                (uint32_t)DataFormat::Float32>();
            typecast_tile<
                (uint32_t)DataFormat::Float16_b,
                (uint32_t)DataFormat::Float32>(0);
            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(cb_m_f32)));
            pack_tile<true>(0, cb_m_f32, 0);
            tile_regs_release();
            m_f32.push_back(1);

            m_f32.wait_front(1);
            update_f32.reserve_back(1);
            unary_op_init_common(cb_m_f32, cb_update_f32);
            tile_regs_acquire();
            copy_tile_to_dst_init_short_with_dt(
                cb_denominator, cb_m_f32);
            copy_tile(cb_m_f32, 0, 0);
            copy_tile_to_dst_init_short_with_dt(cb_m_f32, cb_denominator);
            copy_tile(cb_denominator, tile, 1);
            div_binary_tile_init();
            div_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(cb_update_f32)));
            pack_tile<true>(0, cb_update_f32, 0);
            tile_regs_release();
            update_f32.push_back(1);
            m_f32.pop_front(1);

            param_f32.wait_front(1);
            update_f32.wait_front(1);
            sum_f32.reserve_back(1);
            unary_op_init_common(cb_param_f32, cb_sum_f32);
            tile_regs_acquire();
            copy_tile_to_dst_init_short_with_dt(
                cb_update_f32, cb_param_f32);
            copy_tile(cb_param_f32, 0, 0);
            copy_tile_to_dst_init_short_with_dt(cb_param_f32, cb_update_f32);
            copy_tile(cb_update_f32, 0, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(cb_sum_f32)));
            pack_tile<true>(0, cb_sum_f32, 0);
            tile_regs_release();
            sum_f32.push_back(1);
            update_f32.pop_front(1);
            param_f32.pop_front(1);

            sum_f32.wait_front(1);
            init_sfpu(cb_sum_f32, cb_param_out);
            tile_regs_acquire();
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
        denominator.pop_front(rows_per_block);
        exp_avg_sq.pop_front(rows_per_block);
        exp_avg.pop_front(rows_per_block);
        param.pop_front(rows_per_block);
    }
    decay.pop_front(1);
}
