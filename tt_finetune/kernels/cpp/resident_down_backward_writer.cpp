// Final-only writer for resident down-projection backward.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace { constexpr uint32_t kTileBytes = 2048; }

void kernel_main() {
    Noc noc1(1);
    constexpr uint32_t cb_db = get_compile_time_arg_val(15);
    constexpr uint32_t cb_da = get_compile_time_arg_val(22);
    constexpr uint32_t cb_final_dx = get_compile_time_arg_val(24);
    constexpr uint32_t m_tiles = @M_TILES@;
    constexpr uint32_t k_tiles = @K_TILES@;
    constexpr uint32_t n_tiles = @N_TILES@;
    constexpr uint32_t jobs_per_node = @JOBS_PER_NODE@;
    constexpr uint32_t da_k_tiles_per_node = @DA_K_TILES_PER_NODE@;
    constexpr uint32_t n_tiles_per_node = @N_TILES_PER_NODE@;
    constexpr uint32_t total_jobs = m_tiles * k_tiles;
    constexpr uint32_t arena_cols = k_tiles;

    CircularBuffer db(cb_db);
    CircularBuffer da(cb_da);
    CircularBuffer final_dx(cb_final_dx);
    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();

    const uint32_t arena_addr = get_common_arg_val<uint32_t>(0);
    const auto arena = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<6, 25>(),
            0>(), arena_addr, kTileBytes);

    for (uint32_t local_n = 0; local_n < n_tiles_per_node; ++local_n) {
        const uint32_t n = node * n_tiles_per_node + local_n;
        if (n >= n_tiles) continue;
        db.wait_front(1);
        {
            DeviceZoneScopedN("resident_down_final_db_write");
            noc1.async_write(
                CoreLocalMem<uint32_t>(db.get_read_ptr()), arena,
                kTileBytes, {}, {.page_id = (m_tiles + 1 + n) * arena_cols});
            noc1.async_write_barrier();
        }
        db.pop_front(1);
    }

    for (uint32_t local_k = 0; local_k < da_k_tiles_per_node; ++local_k) {
        const uint32_t k = node * da_k_tiles_per_node + local_k;
        if (k >= k_tiles) continue;
        da.wait_front(1);
        {
            DeviceZoneScopedN("resident_down_final_da_write");
            noc1.async_write(
                CoreLocalMem<uint32_t>(da.get_read_ptr()), arena,
                kTileBytes, {}, {.page_id = m_tiles * arena_cols + k});
            noc1.async_write_barrier();
        }
        da.pop_front(1);
    }

    const uint32_t first_job = node * jobs_per_node;
    for (uint32_t local = 0; local < jobs_per_node; ++local) {
        const uint32_t job = first_job + local;
        if (job >= total_jobs) continue;
        final_dx.wait_front(1);
        {
            DeviceZoneScopedN("resident_down_final_dx_write");
            noc1.async_write(
                CoreLocalMem<uint32_t>(final_dx.get_read_ptr()), arena,
                kTileBytes, {}, {.page_id = job});
            noc1.async_write_barrier();
        }
        final_dx.pop_front(1);
    }
}
