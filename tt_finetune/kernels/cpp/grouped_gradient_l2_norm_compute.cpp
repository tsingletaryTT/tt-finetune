// Exact per-region BF16 square -> FP32-destination reduce -> sqrt.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void kernel_main() {
    constexpr uint32_t jobs = @JOBS@;
    constexpr uint32_t jobs_per_layer = @JOBS_PER_LAYER@;
    constexpr uint32_t heights[jobs_per_layer] = {@HEIGHT_TILE_PATTERN@};
    constexpr uint32_t widths[jobs_per_layer] = {@WIDTH_TILE_PATTERN@};
    const uint32_t node =
        get_absolute_logical_y() * @GRID_WIDTH@ + get_absolute_logical_x();
    constexpr uint32_t cb_gradient_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scaler_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_square_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sum_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_norm_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_exponent_id = get_compile_time_arg_val(5);
    CircularBuffer gradient(cb_gradient_id), square(cb_square_id);
    CircularBuffer sum(cb_sum_id), norm(cb_norm_id), exponent(cb_exponent_id);
    compute_kernel_hw_startup(
        cb_gradient_id, cb_gradient_id, cb_square_id);

    for (uint32_t job = node; job < jobs; job += @CORE_COUNT@) {
        const uint32_t local_job = job % jobs_per_layer;
        const uint32_t height_tiles = heights[local_job];
        const uint32_t width_tiles = widths[local_job];
        const uint32_t pages = height_tiles * width_tiles;
        for (uint32_t page = 0; page < pages; ++page) {
            gradient.wait_front(1);
            square.reserve_back(1);
            tile_regs_acquire();
            copy_tile_to_dst_init_short_with_dt(
                cb_gradient_id, cb_gradient_id);
            copy_tile(cb_gradient_id, 0, 0);
            abs_tile_init();
            abs_tile(0);
            power_tile_init();
            power_tile(0, 0x40000000u);
            tile_regs_commit();
            gradient.pop_front(1);
            tile_regs_wait();
            pack_tile<true>(0, cb_square_id, 0);
            tile_regs_release();
            square.push_back(1);
        }

        compute_kernel_lib::reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_SCALAR,
        cb_square_id,
        cb_scaler_id,
        cb_sum_id,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT,
        ReduceFp32Mode::Fast>(
            compute_kernel_lib::ReduceInputBlockShape::of(
                height_tiles, width_tiles, 1),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                compute_kernel_lib::NoOp{});

        sum.wait_front(1);
        exponent.wait_front(1);
        norm.reserve_back(1);
        pack_reconfig_data_format(cb_norm_id);
        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(cb_sum_id, cb_exponent_id);
        copy_tile(cb_sum_id, 0, 0);
        copy_tile(cb_exponent_id, 0, 1);
        power_binary_tile_init();
        power_binary_tile(0, 1, 0);
        tile_regs_commit();
        sum.pop_front(1);
        exponent.pop_front(1);
        tile_regs_wait();
        pack_tile<true>(0, cb_norm_id, 0);
        tile_regs_release();
        norm.push_back(1);

        CircularBuffer scaler(cb_scaler_id);
        scaler.pop_front(1);
    }
}
