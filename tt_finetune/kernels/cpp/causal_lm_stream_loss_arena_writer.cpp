// Publish the final FP32 loss bits into the first four bytes of the arena's
// row-zero trailer tile. The arena is BF16 only to keep the public custom call
// tuple-free; consumers bitcast these two words back to one FP32 scalar.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t kBf16TileBytes = 2048;
}

void kernel_main() {
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@;
    const auto arena = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                3, 32>(),
            0>(),
        get_common_arg_val<uint32_t>(0), kBf16TileBytes);
    Noc noc;
    DataflowBuffer loss(20);
    loss.wait_front(1);
    // Page hidden_tiles is the trailer tile of token-row tile zero.
    noc.async_write(
        loss, arena, sizeof(uint32_t),
        {.offset_bytes = 0}, {.page_id = hidden_tiles});
    noc.async_write_barrier();
    loss.pop_front(1);
}
