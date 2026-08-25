// Accurate FP32 LOG on a distinct compute role avoids reduction-state reuse.

#include <cstdint>
#include "ttnn/kernel/compute/moreh_common.hpp"

void kernel_main() {
    constexpr auto cb_input = tt::CBIndex::c_11;
    constexpr auto cb_output = tt::CBIndex::c_12;
    compute_kernel_hw_startup(cb_input, cb_input, cb_output);
    DataflowBuffer input(cb_input), output(cb_output);
    log_tile_to_cb(input, output);
}
