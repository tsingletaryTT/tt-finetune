// Read FP32 tiles without arithmetic. The writer stores their raw bytes in a
// BF16 page arena; no unpack/pack conversion is allowed at this boundary.

#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc0(0);
    constexpr uint32_t cb_raw = get_compile_time_arg_val(@RAW_CB_INDEX@);
    constexpr uint32_t jobs = @JOBS@;
    constexpr uint32_t jobs_per_core = @JOBS_PER_CORE@;
    constexpr uint32_t cores = 110;
    constexpr uint32_t fp32_tile_bytes = 4096;
    const uint32_t node = get_absolute_logical_y() * 11 + get_absolute_logical_x();
    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const auto input = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 2>(), 0>(),
        input_addr, fp32_tile_bytes);
    CircularBuffer raw(cb_raw);
    for (uint32_t local = 0; local < jobs_per_core; ++local) {
        const uint32_t job = node + local * cores;
        if (job >= jobs) continue;
        raw.reserve_back(1);
        noc0.async_read(
            input, CoreLocalMem<uint32_t>(raw.get_write_ptr()),
            fp32_tile_bytes, {.page_id = job}, {});
        noc0.async_read_barrier();
        raw.push_back(1);
    }
}
