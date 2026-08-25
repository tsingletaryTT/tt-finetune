// Final down dA/dB writer plus direct dProduct L1 handoff.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace {
constexpr uint32_t kGridCols = 10;
constexpr uint32_t kTileBytes = 2048;
constexpr uint8_t kActivationPhysicalX = 13;  // logical x=10
constexpr uint32_t kMailboxSemaphore = @MAILBOX_SEMAPHORE@;
constexpr uint32_t kReadySemaphore = @READY_SEMAPHORE@;
FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}  // namespace

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
        get_absolute_logical_y() * kGridCols + get_absolute_logical_x();

    const uint32_t arena_addr = get_common_arg_val<uint32_t>(0);
    const auto arena = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                8, @TOTAL_CBS@>(),
            0>(),
        arena_addr,
        kTileBytes);

    // The first 2*M rows belong to final dGate/dUp, produced by the BF16
    // island. Down dA and dB occupy the following rows.
    for (uint32_t local_n = 0; local_n < n_tiles_per_node; ++local_n) {
        const uint32_t n = node * n_tiles_per_node + local_n;
        if (n >= n_tiles) continue;
        db.wait_front(1);
        noc1.async_write(
            CoreLocalMem<uint32_t>(db.get_read_ptr()),
            arena,
            kTileBytes,
            {},
            {.page_id = (2 * m_tiles + 1 + n) * arena_cols});
        noc1.async_write_barrier();
        db.pop_front(1);
    }

    for (uint32_t local_k = 0; local_k < da_k_tiles_per_node; ++local_k) {
        const uint32_t k = node * da_k_tiles_per_node + local_k;
        if (k >= k_tiles) continue;
        da.wait_front(1);
        noc1.async_write(
            CoreLocalMem<uint32_t>(da.get_read_ptr()),
            arena,
            kTileBytes,
            {},
            {.page_id = 2 * m_tiles * arena_cols + k});
        noc1.async_write_barrier();
        da.pop_front(1);
    }

    const uint32_t first_job = node * jobs_per_node;
    for (uint32_t local = 0; local < jobs_per_node; ++local) {
        const uint32_t job = first_job + local;
        if (job >= total_jobs) continue;
        final_dx.wait_front(1);

        // Every producer is permanently paired with activation core
        // `node % 10`, so no two consumers can overwrite this mailbox.
        volatile tt_l1_ptr uint32_t* mailbox =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(kMailboxSemaphore));
        noc_semaphore_wait_min(mailbox, 1);
        const uint32_t destination_l1 = *mailbox;
        noc_semaphore_set(mailbox, 0);

        const uint32_t activation_y = node % 10;
        noc_async_write(
            final_dx.get_read_ptr(),
            get_noc_addr(
                kActivationPhysicalX, physical_y(activation_y),
                destination_l1),
            kTileBytes);
        noc_async_write_barrier();
        noc_semaphore_inc(
            get_noc_addr(
                kActivationPhysicalX, physical_y(activation_y),
                get_semaphore(kReadySemaphore)),
            1);
        // `noc_semaphore_inc` is a non-posted atomic.  Flush it before this
        // writer advances (and especially before kernel exit), otherwise the
        // activation reader can wait forever for a ready notification that
        // is still pending in the producer's NoC queue.
        noc_async_atomic_barrier();
        final_dx.pop_front(1);
    }
}
