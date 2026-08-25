// Publish one complete token-row tile from the large-W log-softmax owner.

#include <cstdint>
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_output = 16;
    constexpr uint32_t wt = @VOCAB_TILES@;
    constexpr uint32_t row_tiles = @ROW_TILES@;
    constexpr uint32_t tile_bytes = @OUTPUT_TILE_BYTES@;
    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= row_tiles) return;

    const uint32_t destination_address = get_common_arg_val<uint32_t>(0);
    const auto destination = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 29>(),
            0>(),
        destination_address,
        tile_bytes);
    Noc noc;
    DataflowBuffer output(cb_output);
    const uint32_t first_page = node * wt;
    for (uint32_t tile = 0; tile < wt; ++tile) {
        output.wait_front(1);
        noc.async_write(
            output,
            destination,
            tile_bytes,
            {.offset_bytes = 0},
            {.page_id = first_page + tile});
        noc.async_write_barrier();
        output.pop_front(1);
    }
}
