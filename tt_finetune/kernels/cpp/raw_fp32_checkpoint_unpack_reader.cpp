// Reassemble two opaque BF16 arena pages into one raw FP32 CB tile.

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
    constexpr uint32_t bf16_page_bytes = 2048;
    const uint32_t node = get_absolute_logical_y() * 11 + get_absolute_logical_x();
    const uint32_t arena_addr = get_common_arg_val<uint32_t>(0);
    const auto arena = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 2>(), 0>(),
        arena_addr, bf16_page_bytes);
    CircularBuffer raw(cb_raw);
    for (uint32_t local = 0; local < jobs_per_core; ++local) {
        const uint32_t job = node + local * cores;
        if (job >= jobs) continue;
        raw.reserve_back(1);
        const uint32_t destination = raw.get_write_ptr();
        noc0.async_read(
            arena, CoreLocalMem<uint32_t>(destination), bf16_page_bytes,
            {.page_id = @PAGE_OFFSET@ + 2 * job}, {});
        noc0.async_read(
            arena, CoreLocalMem<uint32_t>(destination + bf16_page_bytes),
            bf16_page_bytes, {.page_id = @PAGE_OFFSET@ + 2 * job + 1}, {});
        noc0.async_read_barrier();
        raw.push_back(1);
    }
}
