// Standalone FP32 binary diagnostic using the TTNN binary-ng SFPU schedule.

#define FP32_DEST_ACC_EN 1

#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t total_tiles = @TOTAL_TILES@;
    constexpr uint32_t tiles_per_node = @TILES_PER_NODE@;
    constexpr uint32_t grid_columns = 8;
    CircularBuffer lhs(0), rhs(1), output(2);
    const uint32_t node =
        get_absolute_logical_y() * grid_columns + get_absolute_logical_x();
    unary_op_init_common(0, 2);
    @OP_INIT@
    for (uint32_t local = 0; local < tiles_per_node; ++local) {
        if (node * tiles_per_node + local >= total_tiles) continue;
        lhs.wait_front(1);
        rhs.wait_front(1);
        output.reserve_back(1);
        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(1, 0);
        copy_tile(0, 0, 0);
        copy_tile_to_dst_init_short_with_dt(0, 1);
        copy_tile(1, 0, 1);
        @OP_CALL@
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, 2);
        tile_regs_release();
        output.push_back(1);
        rhs.pop_front(1);
        lhs.pop_front(1);
    }
}
