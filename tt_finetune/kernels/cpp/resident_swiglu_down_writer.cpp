// Final-only writer for the resident exact SwiGLU + down projection.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace {
static __attribute__((noinline, noclone)) void
rebase_empty_consumer_cb(uint32_t cb_id) {
    *get_cb_tiles_received_ptr(cb_id) = 0;
    *get_cb_tiles_acked_ptr(cb_id) = 0;
    auto& interface = get_local_cb_interface(cb_id);
    const uint32_t base = interface.fifo_limit - interface.fifo_size;
    interface.fifo_rd_ptr = base;
    interface.fifo_wr_ptr = base;
    interface.tiles_acked_received_init = 0;
    interface.fifo_wr_tile_ptr = 0;
}
}  // namespace

void kernel_main() {
    Noc noc1(1);
    constexpr uint32_t cb_h_saved = get_compile_time_arg_val(9);
    constexpr uint32_t cb_output = get_compile_time_arg_val(16);
    constexpr uint32_t h_tiles = @H_TILES@;
    constexpr uint32_t rank_tiles = @RANK_TILES@;
    constexpr uint32_t output_jobs = @OUTPUT_JOBS@;
    constexpr uint32_t output_jobs_per_core = @OUTPUT_JOBS_PER_CORE@;
    constexpr uint32_t arena_cols = h_tiles + rank_tiles;
    constexpr uint32_t kCoreCount = 110;
    constexpr uint32_t kOutputCoreCount = 72;
    constexpr uint32_t kTileBytes = 2048;

    CircularBuffer h_saved(cb_h_saved);
    CircularBuffer output(cb_output);
    rebase_empty_consumer_cb(cb_h_saved);
    rebase_empty_consumer_cb(cb_output);
    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    const uint32_t arena_addr = get_common_arg_val<uint32_t>(0);
    const auto arena = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<5, 18>(), 0>(),
        arena_addr, kTileBytes);

    for (uint32_t local = 0; local < output_jobs_per_core; ++local) {
        if (node >= kOutputCoreCount) {
            continue;
        }
        const uint32_t job = node + local * kOutputCoreCount;
        if (job >= output_jobs) {
            continue;
        }
        const uint32_t m = job / h_tiles;
        const uint32_t h = job - m * h_tiles;
        if (h == 0) {
            h_saved.wait_front(rank_tiles);
            noc1.async_write(
                CoreLocalMem<uint32_t>(h_saved.get_read_ptr()), arena,
                rank_tiles * kTileBytes, {},
                {.page_id = m * arena_cols + h_tiles});
            noc1.async_write_barrier();
            h_saved.pop_front(rank_tiles);
        }
        output.wait_front(1);
        {
            DeviceZoneScopedN("resident_mlp_final_write");
            noc1.async_write(
                CoreLocalMem<uint32_t>(output.get_read_ptr()), arena,
                kTileBytes, {}, {.page_id = m * arena_cols + h});
            noc1.async_write_barrier();
        }
        output.pop_front(1);
    }
}
