// Publish only the final frozen-head input gradient. dLogits and all FP32
// vocabulary-reduction partials remain private to the TT-Metal program.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t kBf16TileBytes = 2048;
}

void kernel_main() {
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@;
    constexpr uint32_t arena_tiles = @ARENA_TILES@;
    constexpr uint32_t head_workers = @HEAD_WORKERS@;
    constexpr uint32_t head_start_x = @HEAD_START_X@;
    const uint32_t row = get_absolute_logical_y();
    const uint32_t worker = get_absolute_logical_x() - head_start_x;
    const uint32_t local_jobs =
        (hidden_tiles + head_workers - 1) / head_workers;

    const auto output = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                3, 32>(),
            0>(),
        get_common_arg_val<uint32_t>(0), kBf16TileBytes);
    Noc noc;
    DataflowBuffer result(1);
    for (uint32_t local_job = 0; local_job < local_jobs; ++local_job) {
        const uint32_t k_tile = worker + local_job * head_workers;
        if (k_tile >= hidden_tiles) continue;
        result.wait_front(1);
        noc.async_write(
            result, output, kBf16TileBytes,
            {.offset_bytes = 0},
            {.page_id = row * arena_tiles + k_tile});
        noc.async_write_barrier();
        result.pop_front(1);
    }
}
