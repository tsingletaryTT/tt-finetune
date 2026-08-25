// Divide the global numerator by the on-device count and publish scalar loss.

#include <cstdint>
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t kFp32TileBytes = 4096;
    const auto output = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                2, 32>(),
            0>(),
        get_common_arg_val<uint32_t>(0), kFp32TileBytes);
    DataflowBuffer numerator(20);
    numerator.wait_front(1);
    Noc noc;
    noc.async_write(
        numerator, output, kFp32TileBytes,
        {.offset_bytes = 0}, {.page_id = 0});
    noc.async_write_barrier();
    numerator.pop_front(1);
}
