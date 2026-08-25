// Final/checkpoint writer and rank-H L1/NoC distributor.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kWorkers = @WORKERS@;
constexpr uint32_t kColumns = @GRID_COLUMNS@;
constexpr uint8_t kPhysicalX[11] = {
    1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13};
FORCE_INLINE uint64_t worker_l1(uint32_t node, uint32_t address) {
    return get_noc_addr(
        kPhysicalX[node % kColumns],
        static_cast<uint8_t>(node / kColumns + 2), address);
}
FORCE_INLINE uint32_t minimum(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}
}  // namespace

void kernel_main() {
    Noc noc1(1);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t intermediate_tiles = @INTERMEDIATE_TILES@;
    constexpr uint32_t total_n_tiles = 2 * intermediate_tiles;
    constexpr uint32_t total_jobs = seq_tiles * total_n_tiles;
    constexpr uint32_t jobs_per_node =
        (total_jobs + kWorkers - 1) / kWorkers;
    constexpr uint32_t arena_row_tiles = total_n_tiles + 2;
    constexpr uint32_t cb_h_publish = 3;
    constexpr uint32_t cb_h_resident = 5;
    constexpr uint32_t cb_output = 12;

    const uint32_t node =
        get_absolute_logical_y() * kColumns + get_absolute_logical_x();
    const uint32_t first_job = node * jobs_per_node;
    const uint32_t stop_job = minimum(first_job + jobs_per_node, total_jobs);
    const auto arena = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<6, 15>(), 0>(),
        get_common_arg_val<uint32_t>(0), kTileBytes);
    CircularBuffer h_publish(cb_h_publish), output(cb_output);
    auto& h_interface = get_local_cb_interface(cb_h_resident);
    const uint32_t h_resident_base =
        h_interface.fifo_limit - h_interface.fifo_size;

    // Publish each once-reduced H tile to every core that owns at least one
    // output in the logical (row, member) group.
    for (uint32_t group = 0; group < 2 * seq_tiles; ++group) {
        const uint32_t group_first = group * intermediate_tiles;
        const uint32_t group_stop = group_first + intermediate_tiles;
        const uint32_t owner = group_first / jobs_per_node;
        if (node != owner) continue;
        h_publish.wait_front(1);
        const uint32_t row = group / 2;
        const uint32_t member = group & 1u;
        noc1.async_write(
            CoreLocalMem<uint32_t>(h_publish.get_read_ptr()), arena,
            kTileBytes, {},
            {.page_id = row * arena_row_tiles + total_n_tiles + member});
        const uint32_t first_target = group_first / jobs_per_node;
        const uint32_t last_target = (group_stop - 1) / jobs_per_node;
        for (uint32_t target = first_target; target <= last_target; ++target) {
            const uint32_t target_first_job = target * jobs_per_node;
            const uint32_t target_first_group =
                target_first_job / intermediate_tiles;
            const uint32_t slot = group - target_first_group;
            noc_async_write(
                h_publish.get_read_ptr(),
                worker_l1(target, h_resident_base + slot * kTileBytes),
                kTileBytes);
        }
        noc_async_write_barrier();
        for (uint32_t target = first_target; target <= last_target; ++target)
            noc_semaphore_inc(worker_l1(target, get_semaphore(0)), 1);
        noc_async_atomic_barrier();
        h_publish.pop_front(1);
    }

    for (uint32_t job = first_job; job < stop_job; ++job) {
        const uint32_t row = job / total_n_tiles;
        const uint32_t n = job - row * total_n_tiles;
        output.wait_front(1);
        noc1.async_write(
            CoreLocalMem<uint32_t>(output.get_read_ptr()), arena,
            kTileBytes, {}, {.page_id = row * arena_row_tiles + n});
        noc1.async_write_barrier();
        output.pop_front(1);
    }
}
