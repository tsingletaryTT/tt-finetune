// Reader for pair-owned gate/up projection and register-resident SwiGLU.

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
FORCE_INLINE uint64_t worker_l1(uint32_t node, uint32_t address) {
    return get_noc_addr(
        kPhysicalX[node % kColumns],
        static_cast<uint8_t>(node / kColumns + 2), address);
}
}  // namespace

void kernel_main() {
    Noc noc0(0);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@;
    constexpr uint32_t intermediate_tiles = @INTERMEDIATE_TILES@;
    constexpr uint32_t total_pairs = seq_tiles * intermediate_tiles;
    constexpr uint32_t pairs_per_node =
        (total_pairs + kWorkers - 1) / kWorkers;
    constexpr uint32_t base_phase = @BASE_PHASE@;
    constexpr uint32_t base_blocks = hidden_tiles / base_phase;

    const uint32_t node =
        get_absolute_logical_y() * kColumns + get_absolute_logical_x();
    const uint32_t first_pair = node * pairs_per_node;
    const uint32_t stop_pair = minimum(first_pair + pairs_per_node, total_pairs);
    const uint32_t valid_pairs =
        stop_pair > first_pair ? stop_pair - first_pair : 0;
    const uint32_t first_row =
        valid_pairs ? first_pair / intermediate_tiles : 0;
    const uint32_t last_row =
        valid_pairs ? (stop_pair - 1) / intermediate_tiles : 0;
    const uint32_t cached_rows =
        valid_pairs ? last_row - first_row + 1 : 0;

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
    auto* h_credit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(0));
    auto* init_count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(1));
    auto* init_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(2));
    noc_semaphore_set(h_credit, 0);
    noc_semaphore_set(init_ready, 0);
    noc_semaphore_inc(worker_l1(0, get_semaphore(1)), 1);
    noc_async_atomic_barrier();
    if (node == 0) {
        noc_semaphore_wait_min(init_count, kWorkers);
        noc_semaphore_set(init_count, 0);
        noc_semaphore_set(init_ready, 1);
        for (uint32_t target = 1; target < kWorkers; ++target)
            noc_async_write(
                get_semaphore(2), worker_l1(target, get_semaphore(2)),
                sizeof(uint32_t));
        noc_async_write_barrier();
    }
    noc_semaphore_wait(init_ready, 1);

    if (cached_rows) {
        x_cache.reserve_back(cached_rows * hidden_tiles);
        for (uint32_t row = first_row; row <= last_row; ++row) {
            const uint32_t slot = row - first_row;
            for (uint32_t k = 0; k < hidden_tiles; ++k)
                noc0.async_read(
                    input,
                    CoreLocalMem<uint32_t>(
                        x_cache.get_write_ptr()
                        + (slot * hidden_tiles + k) * kTileBytes),
                    kTileBytes, {.page_id = row * hidden_tiles + k}, {});
        }
        noc0.async_read_barrier();
        x_cache.push_back(cached_rows * hidden_tiles);
    }

    // The core containing the first pair in a row computes both member Hs.
    for (uint32_t row = 0; row < seq_tiles; ++row) {
        const uint32_t owner = (row * intermediate_tiles) / pairs_per_node;
        if (node != owner) continue;
        for (uint32_t member = 0; member < 2; ++member) {
            const auto& selected_a = member ? a1 : a0;
            a_cb.reserve_back(hidden_tiles);
            for (uint32_t k = 0; k < hidden_tiles; ++k)
                noc0.async_read(
                    selected_a,
                    CoreLocalMem<uint32_t>(
                        a_cb.get_write_ptr() + k * kTileBytes),
                    kTileBytes, {.page_id = k}, {});
            noc0.async_read_barrier();
            a_cb.push_back(hidden_tiles);
        }
    }

    if (cached_rows) {
        h_resident.reserve_back(2 * cached_rows);
        noc_semaphore_wait_min(h_credit, 2 * cached_rows);
        noc_semaphore_set(h_credit, 0);
        h_resident.push_back(2 * cached_rows);
    }

    // Pair order is gate then up for every logical output tile.
    for (uint32_t pair = first_pair; pair < stop_pair; ++pair) {
        const uint32_t n = pair % intermediate_tiles;
        for (uint32_t member = 0; member < 2; ++member) {
            const uint32_t weight_n = member * intermediate_tiles + n;
            for (uint32_t block = 0; block < base_blocks; ++block) {
                w_cb.reserve_back(base_phase);
                for (uint32_t inner = 0; inner < base_phase; ++inner) {
                    const uint32_t k = block * base_phase + inner;
                    noc0.async_read(
                        weight,
                        CoreLocalMem<uint32_t>(
                            w_cb.get_write_ptr() + inner * kTileBytes),
                        kTileBytes,
                        {.page_id = k * 2 * intermediate_tiles + weight_n}, {});
                }
                noc0.async_read_barrier();
                w_cb.push_back(base_phase);
            }
            b_cb.reserve_back(1);
            const auto& selected_b = member ? b1 : b0;
            noc0.async_read(
                selected_b, CoreLocalMem<uint32_t>(b_cb.get_write_ptr()),
                kTileBytes, {.page_id = n}, {});
            noc0.async_read_barrier();
            b_cb.push_back(1);
        }
    }
}
