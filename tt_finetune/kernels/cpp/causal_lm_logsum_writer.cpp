// Publish the one FP32 reduction tile owned by each token-row core.

#include <cstdint>
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_logsum = @OUTPUT_CB@;
    constexpr uint32_t row_tiles = @ROW_TILES@;
    constexpr uint32_t tile_bytes = 4096;
    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= row_tiles) return;

    const uint32_t destination_address = get_common_arg_val<uint32_t>(0);
    const auto destination = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 5>(),
            0>(),
        destination_address,
        tile_bytes);
    Noc noc;
    DataflowBuffer output(cb_logsum);
    output.wait_front(1);
    noc.async_write(
        output,
        destination,
        tile_bytes,
        {.offset_bytes = 0},
        {.page_id = node});
    noc.async_write_barrier();
    output.pop_front(1);
}
