// Publish one scalar norm page per active worker core.

#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t jobs = @JOBS@;
    const uint32_t node =
        get_absolute_logical_y() * @GRID_WIDTH@ + get_absolute_logical_x();
    Noc noc1(1);
    CircularBuffer norm(get_compile_time_arg_val(4));
    const uint32_t output_address = get_common_arg_val<uint32_t>(0);
    auto output_args = TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 6>(),
        0>();
    TensorAccessor output_accessor(output_args, output_address, 2048);
    for (uint32_t job = node; job < jobs; job += @CORE_COUNT@) {
        norm.wait_front(1);
        noc1.async_write(
            CoreLocalMem<uint32_t>(norm.get_read_ptr()),
            output_accessor,
            output_accessor.get_aligned_page_size(), {},
            {.page_id = job});
        noc1.async_write_barrier();
        norm.pop_front(1);
    }
}
