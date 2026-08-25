// Stream one FP32 row-reduction tile to its owning core.

#include <cstdint>
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_input = 0;
    constexpr uint32_t row_tiles = @ROW_TILES@;
    constexpr uint32_t tile_bytes = 4096;
    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= row_tiles) return;

    const uint32_t source_address = get_common_arg_val<uint32_t>(0);
    const auto source = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 2>(),
            0>(),
        source_address,
        tile_bytes);
    Noc noc;
    DataflowBuffer input(cb_input);
    input.reserve_back(1);
    noc.async_read(
        source,
        input,
        tile_bytes,
        {.page_id = node},
        {.offset_bytes = 0});
    noc.async_read_barrier();
    input.push_back(1);
}
