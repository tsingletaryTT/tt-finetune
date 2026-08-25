// Writer and rank-H distributor for paired gate/up + resident SwiGLU.

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
    constexpr uint32_t total_pairs = seq_tiles * intermediate_tiles;
    constexpr uint32_t pairs_per_node =
        (total_pairs + kWorkers - 1) / kWorkers;
    constexpr uint32_t arena_row_tiles = 3 * intermediate_tiles + 2;
    constexpr uint32_t cb_h_publish = 3;
    constexpr uint32_t cb_h_resident = 5;
    constexpr uint32_t cb_gate_up = 12;
    constexpr uint32_t cb_product = 13;

    const uint32_t node =
        get_absolute_logical_y() * kColumns + get_absolute_logical_x();
    const uint32_t first_pair = node * pairs_per_node;
    const uint32_t stop_pair = minimum(first_pair + pairs_per_node, total_pairs);
    const auto arena = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<6, 15>(), 0>(),
        get_common_arg_val<uint32_t>(0), kTileBytes);
    CircularBuffer h_publish(cb_h_publish), h_resident(cb_h_resident);
    CircularBuffer gate_up(cb_gate_up), product(cb_product);
    const uint32_t h_base =
        get_local_cb_interface(cb_h_resident).fifo_limit
        - get_local_cb_interface(cb_h_resident).fifo_size;

    for (uint32_t row = 0; row < seq_tiles; ++row) {
        const uint32_t row_first = row * intermediate_tiles;
        const uint32_t row_stop = row_first + intermediate_tiles;
        const uint32_t owner = row_first / pairs_per_node;
        if (node != owner) continue;
        const uint32_t first_target = row_first / pairs_per_node;
        const uint32_t last_target = (row_stop - 1) / pairs_per_node;
        for (uint32_t member = 0; member < 2; ++member) {
            h_publish.wait_front(1);
            noc1.async_write(
                CoreLocalMem<uint32_t>(h_publish.get_read_ptr()), arena,
                kTileBytes, {},
                {.page_id = row * arena_row_tiles
                    + 3 * intermediate_tiles + member});
            for (uint32_t target = first_target; target <= last_target; ++target) {
                const uint32_t target_first = target * pairs_per_node;
                const uint32_t target_row = target_first / intermediate_tiles;
                const uint32_t slot = (row - target_row) * 2 + member;
                noc_async_write(
                    h_publish.get_read_ptr(),
                    worker_l1(target, h_base + slot * kTileBytes),
                    kTileBytes);
            }
            noc_async_write_barrier();
            for (uint32_t target = first_target; target <= last_target; ++target)
                noc_semaphore_inc(worker_l1(target, get_semaphore(0)), 1);
            noc_async_atomic_barrier();
            h_publish.pop_front(1);
        }
    }

    const uint32_t valid_pairs = stop_pair - first_pair;
    if (valid_pairs) {
        gate_up.wait_front(2 * valid_pairs);
        product.wait_front(valid_pairs);
    }
    for (uint32_t pair = first_pair; pair < stop_pair; ++pair) {
        const uint32_t row = pair / intermediate_tiles;
        const uint32_t n = pair - row * intermediate_tiles;
        const uint32_t local = pair - first_pair;
        noc1.async_write(
            CoreLocalMem<uint32_t>(
                gate_up.get_read_ptr() + 2 * local * kTileBytes), arena,
            kTileBytes, {},
            {.page_id = row * arena_row_tiles + n});
        noc1.async_write(
            CoreLocalMem<uint32_t>(
                gate_up.get_read_ptr() + (2 * local + 1) * kTileBytes), arena,
            kTileBytes, {},
            {.page_id = row * arena_row_tiles + intermediate_tiles + n});
        noc1.async_write(
            CoreLocalMem<uint32_t>(
                product.get_read_ptr() + local * kTileBytes), arena,
            kTileBytes, {},
            {.page_id = row * arena_row_tiles + 2 * intermediate_tiles + n});
        noc1.async_write_barrier();
    }
    if (valid_pairs) {
        gate_up.pop_front(2 * valid_pairs);
        product.pop_front(valid_pairs);
    }
}
