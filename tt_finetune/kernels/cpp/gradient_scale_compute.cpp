// Exact BF16 scalar-broadcast multiply used by Torch/XLA clipping.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_gradient_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_coefficient_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_output_id = get_compile_time_arg_val(2);
    constexpr uint32_t total_blocks = @TOTAL_BLOCKS@;
    constexpr uint32_t blocks_per_node = @BLOCKS_PER_NODE@;
    constexpr uint32_t rows_per_block = @ROWS_PER_BLOCK@;

    CircularBuffer gradient(cb_gradient_id);
    CircularBuffer coefficient(cb_coefficient_id);
    CircularBuffer output(cb_output_id);
    const uint32_t node =
        get_absolute_logical_y() * 8 + get_absolute_logical_x();

    binary_op_init_common(
        cb_gradient_id, cb_coefficient_id, cb_output_id);
    coefficient.wait_front(1);
    for (uint32_t local = 0; local < blocks_per_node; ++local) {
        const uint32_t block = node * blocks_per_node + local;
        if (block >= total_blocks) break;
        gradient.wait_front(rows_per_block);
        output.reserve_back(rows_per_block);
        tile_regs_acquire();
        for (uint32_t row = 0; row < rows_per_block; ++row) {
            copy_tile_to_dst_init_short_with_dt(
                cb_coefficient_id, cb_gradient_id);
            copy_tile(cb_gradient_id, row, row);
            copy_tile_to_dst_init_short_with_dt(
                cb_gradient_id, cb_coefficient_id);
            copy_tile(cb_coefficient_id, 0, rows_per_block + row);
            mul_binary_tile_init();
            mul_binary_tile(row, rows_per_block + row, row);
        }
        tile_regs_commit();
        gradient.pop_front(rows_per_block);
        tile_regs_wait();
        for (uint32_t row = 0; row < rows_per_block; ++row) {
            pack_tile(row, cb_output_id);
        }
        tile_regs_release();
        output.push_back(rows_per_block);
    }
    coefficient.pop_front(1);
}
