// Publish the FP32 cast tiles only for exact-schedule diagnosis.
#include <cstdint>
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t wt = @VOCAB_TILES@;
    constexpr uint32_t tile_bytes = 4096;
    const uint32_t row = get_absolute_logical_y();
    const uint32_t output_address = get_common_arg_val<uint32_t>(0);
    const auto output = TensorAccessor(
        TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 29>(), 0>(),
        output_address, tile_bytes);
    Noc noc;
    DataflowBuffer cast(24);
    const uint32_t first_page = row * wt;
    for (uint32_t tile = 0; tile < wt; ++tile) {
        cast.wait_front(1);
        noc.async_write(cast, output, tile_bytes, {.offset_bytes = 0}, {.page_id = first_page + tile});
        noc.async_write_barrier();
        cast.pop_front(1);
    }
}
