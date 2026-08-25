// Write only the final BF16-island result to DRAM.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    if (get_absolute_logical_x() != 10 || get_absolute_logical_y() != 9) {
        return;
    }

    constexpr uint32_t cb_output = 6;
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
#ifdef TWO_ISLAND_PJRT_ARTIFACT
    constexpr auto output_args = TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<3, 7>(),
        3>();
#else
    constexpr auto output_args = TensorAccessorArgs<0>();
#endif
    const auto output = TensorAccessor(output_args, output_addr);

    Noc noc(1);
    CircularBuffer result(cb_output);
    result.wait_front(1);
    noc.async_write(result, output, 2048, {}, {.page_id = 0});
    noc.async_write_barrier();
    result.pop_front(1);
}
