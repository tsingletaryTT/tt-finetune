// Publish only the final row log-sum tile.

#include <cstdint>
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t tile_bytes = 4096;
    const uint32_t row = get_absolute_logical_y();
    const uint32_t output_address = get_common_arg_val<uint32_t>(0);
    const auto output = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 29>(),
            0>(), output_address, tile_bytes);
    Noc noc;
    DataflowBuffer value(12);
    value.wait_front(1);
    noc.async_write(
        value, output, tile_bytes,
        {.offset_bytes = 0}, {.page_id = row});
    noc.async_write_barrier();
    value.pop_front(1);
}
