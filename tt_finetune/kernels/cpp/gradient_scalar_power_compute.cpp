// Exact BF16 StableHLO binary power for one scalar tile.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_input_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_exponent_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_output_id = get_compile_time_arg_val(2);
    CircularBuffer input(cb_input_id), exponent(cb_exponent_id), output(cb_output_id);

    compute_kernel_hw_startup(cb_input_id, cb_exponent_id, cb_output_id);
    input.wait_front(1);
    exponent.wait_front(1);
    output.reserve_back(1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(cb_input_id, cb_exponent_id);
    copy_tile(cb_input_id, 0, 0);
    copy_tile(cb_exponent_id, 0, 1);
    power_binary_tile_init();
    power_binary_tile(0, 1, 0);
    tile_regs_commit();
    input.pop_front(1);
    exponent.pop_front(1);
    tile_regs_wait();
    pack_tile<true>(0, cb_output_id, 0);
    tile_regs_release();
    output.push_back(1);
}
