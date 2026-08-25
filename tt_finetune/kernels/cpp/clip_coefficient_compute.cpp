// Exact BF16 max_norm / (total_norm + epsilon), clamped to one.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_norm_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_constants_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_denominator_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_ratio_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_scalar_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_output_id = get_compile_time_arg_val(5);
    CircularBuffer norm(cb_norm_id), constants(cb_constants_id);
    CircularBuffer denominator(cb_denominator_id), ratio(cb_ratio_id);
    CircularBuffer scalar(cb_scalar_id), output(cb_output_id);

    compute_kernel_hw_startup(cb_norm_id, cb_constants_id, cb_denominator_id);
    norm.wait_front(1);
    constants.wait_front(3);

    binary_op_init_common(cb_norm_id, cb_constants_id, cb_denominator_id);
    denominator.reserve_back(1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(cb_norm_id, cb_constants_id);
    copy_tile(cb_norm_id, 0, 0);
    copy_tile(cb_constants_id, 0, 1);  // epsilon
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    tile_regs_commit();
    norm.pop_front(1);
    tile_regs_wait();
    pack_tile<true>(0, cb_denominator_id, 0);
    tile_regs_release();
    denominator.push_back(1);

    denominator.wait_front(1);
    unary_op_init_common(cb_constants_id, cb_ratio_id);
    ratio.reserve_back(1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(cb_denominator_id, cb_constants_id);
    copy_tile(cb_constants_id, 1, 0);  // maximum
    copy_tile_to_dst_init_short_with_dt(cb_constants_id, cb_denominator_id);
    copy_tile(cb_denominator_id, 0, 1);
    div_binary_tile_init();
    div_binary_tile(0, 1, 0);
    tile_regs_commit();
    denominator.pop_front(1);
    tile_regs_wait();
    pack_tile<true>(0, cb_ratio_id, 0);
    tile_regs_release();
    ratio.push_back(1);

    ratio.wait_front(1);
    unary_op_init_common(cb_ratio_id, cb_scalar_id);
    scalar.reserve_back(1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(cb_constants_id, cb_ratio_id);
    copy_tile(cb_ratio_id, 0, 0);
    copy_tile_to_dst_init_short_with_dt(cb_ratio_id, cb_constants_id);
    copy_tile(cb_constants_id, 2, 1);  // one
    binary_min_tile_init();
    binary_min_tile(0, 1, 0);
    tile_regs_commit();
    ratio.pop_front(1);
    tile_regs_wait();
    pack_tile<true>(0, cb_scalar_id, 0);
    tile_regs_release();
    scalar.push_back(1);

    scalar.wait_front(1);
    output.reserve_back(1);
    binary_op_init_common(cb_constants_id, cb_scalar_id, cb_output_id);
    tile_regs_acquire();
    mul_tiles_bcast_scalar_init_short(cb_constants_id, cb_scalar_id);
    mul_tiles_bcast_scalar(cb_constants_id, cb_scalar_id, 2, 0, 0);
    tile_regs_commit();
    scalar.pop_front(1);
    constants.pop_front(3);
    tile_regs_wait();
    pack_tile<true>(0, cb_output_id, 0);
    tile_regs_release();
    output.push_back(1);
}
