// Compiler-friendly one-pack BF16 GQA reduction.

#define FP32_DEST_ACC_EN 1

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_input = get_compile_time_arg_val(0);
    constexpr uint32_t cb_zero = get_compile_time_arg_val(1);
    constexpr uint32_t cb_output = get_compile_time_arg_val(2);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t dim_tiles = @DIM_TILES@;
    constexpr uint32_t kv_heads = @KV_HEADS@;
    constexpr uint32_t group = @GROUP@;
    constexpr uint32_t jobs = kv_heads * seq_tiles;
    const uint32_t node = get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= jobs) return;
    CircularBuffer input(cb_input), zero(cb_zero), output(cb_output);
    zero.wait_front(1);
    binary_op_init_common(cb_input, cb_zero, cb_output);
    for (uint32_t tile = 0; tile < dim_tiles; ++tile) {
        input.wait_front(group);
        copy_tile_to_dst_init_short_with_dt(cb_zero, cb_input);
        tile_regs_acquire();
        copy_tile(cb_input, 0, 0);
        reconfig_data_format_srcb(cb_input, cb_zero);
        add_tiles_init(cb_input, cb_zero, true);
        for (uint32_t member = 1; member < group; ++member)
            add_tiles(cb_input, cb_zero, member, 0, 0);
        input.pop_front(group);
        tile_regs_commit();
        output.reserve_back(1);
        tile_regs_wait();
        PACK((pack_reconfig_data_format(cb_output)));
        pack_tile(0, cb_output);
        tile_regs_release();
        output.push_back(1);
    }
    zero.pop_front(1);
}
