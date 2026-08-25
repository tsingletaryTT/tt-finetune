// FP32-destination island. Copying is intentional: this kernel proves that a
// program can host an FP32 matmul-style compute configuration next to an exact
// BF16 SFPU configuration without crossing a host/runtime boundary.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    if (get_absolute_logical_x() != 0 || get_absolute_logical_y() != 9) {
        return;
    }

#ifdef TWO_ISLAND_REAL_MATMUL
    constexpr uint32_t cb_lhs = 3;
    constexpr uint32_t cb_rhs = 4;
    constexpr uint32_t cb_handoff = 0;
    compute_kernel_hw_startup<SrcOrder::Reverse>(
        cb_lhs, cb_rhs, cb_handoff);
    matmul_init(cb_lhs, cb_rhs);
    cb_wait_front(cb_lhs, 1);
    cb_wait_front(cb_rhs, 1);
    tile_regs_acquire();
    matmul_tiles(cb_lhs, cb_rhs, 0, 0, 0);
    cb_pop_front(cb_lhs, 1);
    cb_pop_front(cb_rhs, 1);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_handoff, 1);
    pack_tile(0, cb_handoff, 0);
    cb_push_back(cb_handoff, 1);
    tile_regs_release();
#else
    constexpr uint32_t cb_input = 1;
    constexpr uint32_t cb_dummy_output = 2;
    CircularBuffer input(cb_input);
    CircularBuffer dummy_output(cb_dummy_output);
    input.wait_front(1);
    dummy_output.reserve_back(1);
    unary_op_init_common(cb_input, cb_dummy_output);
    copy_tile_init(cb_input);
    tile_regs_acquire();
    copy_tile(cb_input, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_dummy_output, 0);
    tile_regs_release();
    dummy_output.push_back(1);
    input.pop_front(1);
#endif
}
