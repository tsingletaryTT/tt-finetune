// Final-output writer and member1 -> member0 exact-sum L1 handoff.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace {
constexpr uint32_t kGridCols = 11;
constexpr uint32_t kMemberNodes = 55;
constexpr uint32_t kTileBytes = 2048;
constexpr uint8_t kPhysicalX[kGridCols] = {
    1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13};
FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}  // namespace

void kernel_main() {
    Noc noc1(1);
    constexpr uint32_t cb_final_dx = get_compile_time_arg_val(12);
    constexpr uint32_t cb_peer_sum = get_compile_time_arg_val(18);
    constexpr uint32_t cb_da_or_db = get_compile_time_arg_val(15);
    constexpr uint32_t m_tiles = @M_TILES@;
    constexpr uint32_t k_tiles = @K_TILES@;
    constexpr uint32_t n_tiles = @N_TILES@;
    constexpr uint32_t jobs_per_node = @JOBS_PER_NODE@;
    constexpr uint32_t n_tiles_per_node = @N_TILES_PER_NODE@;
    constexpr uint32_t total_jobs = m_tiles * k_tiles;
    constexpr uint32_t arena_cols = k_tiles;

    CircularBuffer final_dx(cb_final_dx);
    CircularBuffer peer_sum(cb_peer_sum);
    CircularBuffer da_or_db(cb_da_or_db);

    const uint32_t logical_x = get_absolute_logical_x();
    const uint32_t logical_y = get_absolute_logical_y();
    const uint32_t global_node = logical_y * kGridCols + logical_x;
    const uint32_t member = global_node / kMemberNodes;
    const uint32_t node = global_node - member * kMemberNodes;

    const uint32_t arena_addr = get_common_arg_val<uint32_t>(0);
    const auto arena = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<11, 19>(),
            0>(),
        arena_addr,
        kTileBytes);

    // dB is already a final result of phase one. Draining it first makes CB15
    // available for the resident dA output without another allocation.
    for (uint32_t local_n = 0; local_n < n_tiles_per_node; ++local_n) {
        const uint32_t n_tile = node * n_tiles_per_node + local_n;
        if (n_tile >= n_tiles) {
            continue;
        }
        da_or_db.wait_front(1);
        const uint32_t row =
            m_tiles + 2 + member * n_tiles + n_tile;
        {
            DeviceZoneScopedN("resident_final_db_write");
            noc1.async_write(
                CoreLocalMem<uint32_t>(da_or_db.get_read_ptr()),
                arena,
                kTileBytes,
                {},
                {.page_id = row * arena_cols});
            noc1.async_write_barrier();
        }
        da_or_db.pop_front(1);
    }

    // dA is independent of the inter-member exact dX tree. Drain it as soon
    // as compute publishes it so the member-one writer never holds a final
    // output behind the peer semaphore handshake.
    if (node < k_tiles) {
        da_or_db.wait_front(1);
        const uint32_t row = m_tiles + member;
        {
            DeviceZoneScopedN("resident_final_da_write");
            noc1.async_write(
                CoreLocalMem<uint32_t>(da_or_db.get_read_ptr()),
                arena,
                kTileBytes,
                {},
                {.page_id = row * arena_cols + node});
            noc1.async_write_barrier();
        }
        da_or_db.pop_front(1);
    }

    const uint32_t first_job = node * jobs_per_node;
    const uint32_t valid_jobs =
        first_job >= total_jobs
            ? 0
            : ((total_jobs - first_job) < jobs_per_node
                   ? total_jobs - first_job
                   : jobs_per_node);

    if (member == 1 && valid_jobs != 0) {
        volatile tt_l1_ptr uint32_t* request_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(10));
        noc_semaphore_wait(request_sem, 1);
        noc_semaphore_set(request_sem, 0);
        peer_sum.wait_front(valid_jobs);
        {
            DeviceZoneScopedN("resident_peer_noc");
            const uint32_t source = peer_sum.get_read_ptr();
            const uint32_t peer_logical_y = logical_y - 5;
            for (uint32_t local = 0; local < valid_jobs; ++local) {
                noc_async_write(
                    source + local * kTileBytes,
                    get_noc_addr(
                        kPhysicalX[logical_x],
                        physical_y(peer_logical_y),
                        source + local * kTileBytes),
                    kTileBytes);
            }
            noc_async_write_barrier();
            volatile tt_l1_ptr uint32_t* response_sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(9));
            noc_semaphore_set(response_sem, valid_jobs);
            noc_async_write(
                get_semaphore(9),
                get_noc_addr(
                    kPhysicalX[logical_x],
                    physical_y(peer_logical_y),
                    get_semaphore(9)),
                sizeof(uint32_t));
            noc_async_write_barrier();
            noc_semaphore_set(response_sem, 0);
        }
        peer_sum.pop_front(valid_jobs);
    }

    if (member == 0) {
        for (uint32_t local = 0; local < valid_jobs; ++local) {
            const uint32_t job = first_job + local;
            final_dx.wait_front(1);
            {
                DeviceZoneScopedN("resident_final_dx_write");
                noc1.async_write(
                    CoreLocalMem<uint32_t>(final_dx.get_read_ptr()),
                    arena,
                    kTileBytes,
                    {},
                    {.page_id = job});
                noc1.async_write_barrier();
            }
            final_dx.pop_front(1);
        }
    }

}
