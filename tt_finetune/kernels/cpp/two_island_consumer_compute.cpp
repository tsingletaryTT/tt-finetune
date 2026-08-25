// Exact BF16 multiply on a compute kernel compiled without FP32 destination
// accumulation. This is the numerical mode required by the accepted SwiGLU
// derivative, isolated from the producer's FP32-destination configuration.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    if (get_absolute_logical_x() != 10 || get_absolute_logical_y() != 9) {
        return;
    }

    constexpr uint32_t cb_lhs = 0;
    constexpr uint32_t cb_rhs = 1;
    constexpr uint32_t cb_output = 6;
    CircularBuffer lhs(cb_lhs);
    CircularBuffer rhs(cb_rhs);
    CircularBuffer output(cb_output);
    lhs.wait_front(1);
    rhs.wait_front(1);
    output.reserve_back(1);
#ifdef TWO_ISLAND_COPY_HANDOFF
    unary_op_init_common(cb_lhs, cb_output);
    copy_tile_init(cb_lhs);
    tile_regs_acquire();
    copy_tile(cb_lhs, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_output, 0);
    tile_regs_release();
#else
    unary_op_init_common(cb_lhs, cb_output);
    mul_binary_tile_init();
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(cb_rhs, cb_lhs);
    copy_tile(cb_lhs, 0, 0);
    copy_tile_to_dst_init_short_with_dt(cb_lhs, cb_rhs);
    copy_tile(cb_rhs, 0, 1);
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_output, 0);
    tile_regs_release();
#endif
    output.push_back(1);
    lhs.pop_front(1);
    rhs.pop_front(1);
}
