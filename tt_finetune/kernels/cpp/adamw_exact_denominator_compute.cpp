// Exact capturable-AdamW denominator rounding phase.
//
// For each tile this reproduces:
//   BF16 sqrt(v) -> FP32 / FP32 denominator_scale -> FP32
// The TTNN export confirms there is no BF16 checkpoint after the division.

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

constexpr uint32_t cb_sqrt_v = 0;
constexpr uint32_t cb_scale = 1;
constexpr uint32_t cb_output = 2;
constexpr uint32_t cb_sqrt_f32 = 3;

}  // namespace

void kernel_main() {
    constexpr uint32_t total_blocks = @TOTAL_BLOCKS@;
    constexpr uint32_t blocks_per_node = @BLOCKS_PER_NODE@;
    constexpr uint32_t rows_per_block = @ROWS_PER_BLOCK@;
    constexpr uint32_t grid_columns = 8;

    CircularBuffer sqrt_v(cb_sqrt_v);
    CircularBuffer scale(cb_scale);
    CircularBuffer output(cb_output);
    CircularBuffer sqrt_f32(cb_sqrt_f32);
    const uint32_t node =
        get_absolute_logical_y() * grid_columns + get_absolute_logical_x();

    scale.wait_front(1);
    DeviceZoneScopedN("adamw_exact_denominator");
    for (uint32_t local_block = 0; local_block < blocks_per_node;
         ++local_block) {
        const uint32_t block = node * blocks_per_node + local_block;
        if (block >= total_blocks) {
            continue;
        }
        sqrt_v.wait_front(rows_per_block);
        output.reserve_back(rows_per_block);
        for (uint32_t tile = 0; tile < rows_per_block; ++tile) {
            // TTNN materializes the BF16->FP32 typecast before its divide.
            // Preserve that exact numerical boundary in private L1.
            sqrt_f32.reserve_back(1);
            unary_op_init_common(cb_sqrt_v, cb_sqrt_f32);
            tile_regs_acquire();
            copy_tile_init(cb_sqrt_v);
            copy_tile(cb_sqrt_v, tile, 0);
            typecast_tile_init<
                (uint32_t)DataFormat::Float16_b,
                (uint32_t)DataFormat::Float32>();
            typecast_tile<
                (uint32_t)DataFormat::Float16_b,
                (uint32_t)DataFormat::Float32>(0);
            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(cb_sqrt_f32)));
            pack_tile<true>(0, cb_sqrt_f32, 0);
            tile_regs_release();
            sqrt_f32.push_back(1);

            sqrt_f32.wait_front(1);
            unary_op_init_common(cb_sqrt_f32, cb_output);
            tile_regs_acquire();
            copy_tile_to_dst_init_short_with_dt(cb_scale, cb_sqrt_f32);
            copy_tile(cb_sqrt_f32, 0, 0);
            copy_tile_to_dst_init_short_with_dt(cb_sqrt_f32, cb_scale);
            copy_tile(cb_scale, 0, 1);
            div_binary_tile_init();
            div_binary_tile(0, 1, 0);

            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(cb_output)));
            pack_tile<true>(0, cb_output, tile);
            tile_regs_release();
            sqrt_f32.pop_front(1);
        }
        output.push_back(rows_per_block);
        sqrt_v.pop_front(rows_per_block);
    }
    scale.pop_front(1);
}
