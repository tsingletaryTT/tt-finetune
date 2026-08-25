// 50-core row-cached grouped gate/up frozen+LoRA forward reader.

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
constexpr uint32_t cb_x_cache = 0;
constexpr uint32_t cb_a = 1;
constexpr uint32_t cb_h_resident = 5;
constexpr uint32_t cb_w = 6;
constexpr uint32_t cb_b = 9;

FORCE_INLINE uint32_t minimum(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}
}  // namespace

void kernel_main() {
    Noc noc0(0);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@;
    constexpr uint32_t intermediate_tiles = @INTERMEDIATE_TILES@;
    constexpr uint32_t total_n_tiles = 2 * intermediate_tiles;
    constexpr uint32_t total_jobs = seq_tiles * total_n_tiles;
    constexpr uint32_t jobs_per_node =
        (total_jobs + kWorkers - 1) / kWorkers;
    constexpr uint32_t base_phase = @BASE_PHASE@;
    constexpr uint32_t base_blocks = hidden_tiles / base_phase;

    const uint32_t node =
        get_absolute_logical_y() * kColumns + get_absolute_logical_x();
    const uint32_t first_job = node * jobs_per_node;
    const uint32_t stop_job = minimum(first_job + jobs_per_node, total_jobs);
    const uint32_t valid_jobs = stop_job > first_job ? stop_job - first_job : 0;
    const uint32_t first_row = valid_jobs ? first_job / total_n_tiles : 0;
    const uint32_t last_row = valid_jobs ? (stop_job - 1) / total_n_tiles : 0;
    const uint32_t cached_rows = valid_jobs ? last_row - first_row + 1 : 0;
    const uint32_t first_group = valid_jobs ? first_job / intermediate_tiles : 0;
    const uint32_t last_group = valid_jobs ? (stop_job - 1) / intermediate_tiles : 0;
    const uint32_t local_groups = valid_jobs ? last_group - first_group + 1 : 0;

    const auto input = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 15>(), 0>(),
        get_common_arg_val<uint32_t>(0), kTileBytes);
    const auto weight = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 15>(), 1>(),
        get_common_arg_val<uint32_t>(1), kTileBytes);
    const auto a0 = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 15>(), 2>(),
        get_common_arg_val<uint32_t>(2), kTileBytes);
    const auto b0 = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<3, 15>(), 3>(),
        get_common_arg_val<uint32_t>(3), kTileBytes);
    const auto a1 = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<4, 15>(), 4>(),
        get_common_arg_val<uint32_t>(4), kTileBytes);
    const auto b1 = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<5, 15>(), 5>(),
        get_common_arg_val<uint32_t>(5), kTileBytes);

    CircularBuffer x_cache(cb_x_cache), a_cb(cb_a), h_resident(cb_h_resident);
    CircularBuffer w_cb(cb_w), b_cb(cb_b);

    // Semaphore L1 can contain values from another program. Every reader
    // clears its local credit before a device-wide initialization barrier.
    auto* h_credit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(0));
    auto* init_count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(1));
    auto* init_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(2));
    noc_semaphore_set(h_credit, 0);
    noc_semaphore_set(init_ready, 0);
    noc_semaphore_inc(get_noc_addr(1, 2, get_semaphore(1)), 1);
    noc_async_atomic_barrier();
    if (node == 0) {
        noc_semaphore_wait_min(init_count, kWorkers);
        noc_semaphore_set(init_count, 0);
        noc_semaphore_set(init_ready, 1);
        for (uint32_t target = 1; target < kWorkers; ++target) {
            const uint32_t tx = target % kColumns;
            const uint32_t ty = target / kColumns;
            noc_async_write(
                get_semaphore(2),
                get_noc_addr(kPhysicalX[tx],
                             static_cast<uint8_t>(ty + 2), get_semaphore(2)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    noc_semaphore_wait(init_ready, 1);

    if (cached_rows) {
        x_cache.reserve_back(cached_rows * hidden_tiles);
        for (uint32_t row = first_row; row <= last_row; ++row) {
            const uint32_t slot = row - first_row;
            for (uint32_t k = 0; k < hidden_tiles; ++k) {
                noc0.async_read(
                    input,
                    CoreLocalMem<uint32_t>(
                        x_cache.get_write_ptr()
                        + (slot * hidden_tiles + k) * kTileBytes),
                    kTileBytes, {.page_id = row * hidden_tiles + k}, {});
            }
        }
        noc0.async_read_barrier();
        x_cache.push_back(cached_rows * hidden_tiles);
    }

    // Exactly one owner reduces X@A for each (token row, member). The owner
    // is the core containing that group's first output job and therefore
    // already owns the required cached input row.
    for (uint32_t group = 0; group < 2 * seq_tiles; ++group) {
        const uint32_t group_first_job = group * intermediate_tiles;
        const uint32_t owner = group_first_job / jobs_per_node;
        if (node != owner) continue;
        const bool up_member = (group & 1u) != 0;
        const auto& selected_a = up_member ? a1 : a0;
        a_cb.reserve_back(hidden_tiles);
        for (uint32_t k = 0; k < hidden_tiles; ++k) {
            noc0.async_read(
                selected_a,
                CoreLocalMem<uint32_t>(a_cb.get_write_ptr() + k * kTileBytes),
                kTileBytes, {.page_id = k}, {});
        }
        noc0.async_read_barrier();
        a_cb.push_back(hidden_tiles);
    }

    if (local_groups) h_resident.reserve_back(local_groups);
    if (local_groups) {
        noc_semaphore_wait_min(h_credit, local_groups);
        noc_semaphore_set(h_credit, 0);
        h_resident.push_back(local_groups);
    }

    // Only W and B stream per output. X and H remain local for every job.
    for (uint32_t job = first_job; job < stop_job; ++job) {
        const uint32_t n = job % total_n_tiles;
        const uint32_t member_n = n % intermediate_tiles;
        const bool up_member = n >= intermediate_tiles;
        for (uint32_t block = 0; block < base_blocks; ++block) {
            w_cb.reserve_back(base_phase);
            for (uint32_t inner = 0; inner < base_phase; ++inner) {
                const uint32_t k = block * base_phase + inner;
                noc0.async_read(
                    weight,
                    CoreLocalMem<uint32_t>(w_cb.get_write_ptr() + inner * kTileBytes),
                    kTileBytes, {.page_id = k * total_n_tiles + n}, {});
            }
            noc0.async_read_barrier();
            w_cb.push_back(base_phase);
        }
        b_cb.reserve_back(1);
        const auto& selected_b = up_member ? b1 : b0;
        noc0.async_read(
            selected_b, CoreLocalMem<uint32_t>(b_cb.get_write_ptr()),
            kTileBytes, {.page_id = member_n}, {});
        noc0.async_read_barrier();
        b_cb.push_back(1);
    }
}
