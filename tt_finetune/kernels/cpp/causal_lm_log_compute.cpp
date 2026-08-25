// Accurate StableHLO-compatible FP32 LOG for one reduction tile per core.

#include <cstdint>
#include "ttnn/kernel/compute/moreh_common.hpp"

void kernel_main() {
    constexpr uint32_t row_tiles = @ROW_TILES@;
    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= row_tiles) return;

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_1;
    compute_kernel_hw_startup(cb_input, cb_input, cb_output);
    DataflowBuffer input(cb_input);
    DataflowBuffer output(cb_output);
    log_tile_to_cb(input, output);
}
