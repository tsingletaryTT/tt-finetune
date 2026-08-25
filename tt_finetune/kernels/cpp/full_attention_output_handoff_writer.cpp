// Output-projection final writer for the first fused attention-block phase.
//
// dA/dB are final gradients and are written to the projection arena. dO is
// not an output: its token-major tiles are sent directly to the 60-core
// attention island in the head-major layout consumed by the resident core.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/debug/waypoint.h"
#include "api/tensor/noc_traits.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace {
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kProjectionColumns = 5;
constexpr uint32_t kProjectionWorkers = 50;
constexpr uint32_t kAttentionColumns = 6;
constexpr uint32_t kAttentionXOffset = 5;
constexpr uint8_t kPhysicalX[11] = {
    1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13};
FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
FORCE_INLINE uint64_t projection_core_l1(
    uint32_t node, uint32_t address) {
    return get_noc_addr(
        kPhysicalX[node % kProjectionColumns],
        physical_y(node / kProjectionColumns), address);
}
FORCE_INLINE uint64_t attention_core_l1(
    uint32_t node, uint32_t address) {
    const uint32_t logical_x =
        kAttentionXOffset + node % kAttentionColumns;
    return get_noc_addr(
        kPhysicalX[logical_x], physical_y(node / kAttentionColumns), address);
}
}  // namespace

void kernel_main() {
    WAYPOINT("OP0");
    Noc noc1(1);
    constexpr uint32_t cb_db = get_compile_time_arg_val(15);
    constexpr uint32_t cb_da = get_compile_time_arg_val(22);
    constexpr uint32_t cb_final_dx = get_compile_time_arg_val(24);
    constexpr uint32_t m_tiles = @M_TILES@;
    constexpr uint32_t k_tiles = @K_TILES@;
    constexpr uint32_t n_tiles = @N_TILES@;
    constexpr uint32_t dim_tiles = @DIM_TILES@;
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t query_heads = @QUERY_HEADS@;
    constexpr uint32_t jobs_per_node = @JOBS_PER_NODE@;
    constexpr uint32_t da_k_tiles_per_node = @DA_K_TILES_PER_NODE@;
    constexpr uint32_t n_tiles_per_node = @N_TILES_PER_NODE@;
    constexpr uint32_t total_jobs = m_tiles * k_tiles;
    constexpr uint32_t active_attention_nodes = query_heads * seq_tiles;
    constexpr uint32_t arena_cols = k_tiles;
    constexpr bool diagnostic_write_dout = @DIAGNOSTIC_WRITE_DOUT@;
    constexpr uint32_t credit_semaphore = 12;
    constexpr uint32_t consumer_ready_semaphore = 13;
    constexpr uint32_t producer_go_semaphore = 14;
    // Projection's resident-down protocol uses 0..10 (9 is the phase count
    // and 10 is the phase-go word).  Keep receiver addresses outside that
    // map: 11 and 12 are otherwise idle on the projection island during this
    // phase.  (Semaphore 12 names attention-side credits below, but its
    // projection-local word is not part of that protocol.)  Attention itself
    // uses its local slot 15 only as the temporary source word.
    constexpr uint32_t receiver_row_address_semaphore = 11;
    constexpr uint32_t receiver_full_address_semaphore = 12;

    static_assert(m_tiles == seq_tiles, "projection tokens must equal attention S");
    static_assert(k_tiles == query_heads * dim_tiles, "hidden/head tile mismatch");
    static_assert(active_attention_nodes <= 60, "attention island overflow");

    CircularBuffer db(cb_db);
    CircularBuffer da(cb_da);
    CircularBuffer final_dx(cb_final_dx);
    const uint32_t logical_x = get_absolute_logical_x();
    const uint32_t logical_y = get_absolute_logical_y();
    const uint32_t node = logical_y * kProjectionColumns + logical_x;
    const uint32_t arena_addr = get_common_arg_val<uint32_t>(0);
    const auto arena = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                @OUTPUT_TENSOR_INDEX@, @TOTAL_CBS@>(), 0>(),
        arena_addr, kTileBytes);

    // Final output-projection adapter gradients retain the accepted arena
    // locations. The dO rows at the start of the old arena are never written.
    for (uint32_t local_n = 0; local_n < n_tiles_per_node; ++local_n) {
        const uint32_t n = node * n_tiles_per_node + local_n;
        if (n >= n_tiles) continue;
        db.wait_front(1);
        noc1.async_write(
            CoreLocalMem<uint32_t>(db.get_read_ptr()), arena, kTileBytes, {},
            {.page_id = (m_tiles + 1 + n) * arena_cols});
        noc1.async_write_barrier();
        db.pop_front(1);
    }
    for (uint32_t local_k = 0; local_k < da_k_tiles_per_node; ++local_k) {
        const uint32_t k = node * da_k_tiles_per_node + local_k;
        if (k >= k_tiles) continue;
        da.wait_front(1);
        noc1.async_write(
            CoreLocalMem<uint32_t>(da.get_read_ptr()), arena, kTileBytes, {},
            {.page_id = m_tiles * arena_cols + k});
        noc1.async_write_barrier();
        da.pop_front(1);
    }
    WAYPOINT("OP1");

    // Consumer readers reset/reserve their windows first and atomically
    // publish readiness to projection node zero. Do not send a payload until
    // all active attention workers have done so; otherwise a late reset could
    // erase an early credit from a fast producer.
    volatile tt_l1_ptr uint32_t* go =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(producer_go_semaphore));
    if (node == 0) {
        volatile tt_l1_ptr uint32_t* ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(consumer_ready_semaphore));
        noc_semaphore_wait_min(ready, active_attention_nodes);
        WAYPOINT("OP2");
        noc_semaphore_set(ready, 0);
        // Attention CBs are allocated in a different per-core layout from
        // projection CBs. The attention coordinator publishes its actual L1
        // receiver addresses; distribute those addresses before the go flag.
        for (uint32_t target = 1; target < kProjectionWorkers; ++target) {
            noc_async_write(
                get_semaphore(receiver_row_address_semaphore),
                projection_core_l1(
                    target, get_semaphore(receiver_row_address_semaphore)),
                sizeof(uint32_t));
            noc_async_write(
                get_semaphore(receiver_full_address_semaphore),
                projection_core_l1(
                    target, get_semaphore(receiver_full_address_semaphore)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
        noc_semaphore_set(go, 1);
        for (uint32_t target = 1; target < kProjectionWorkers; ++target) {
            noc_async_write(
                get_semaphore(producer_go_semaphore),
                projection_core_l1(
                    target, get_semaphore(producer_go_semaphore)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    noc_semaphore_wait(go, 1);
    noc_semaphore_set(go, 0);
    WAYPOINT("OP3");
    const uint32_t attention_dout_row_address =
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(receiver_row_address_semaphore));
    const uint32_t attention_dout_full_address =
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(receiver_full_address_semaphore));
    const uint32_t first_job = node * jobs_per_node;
    for (uint32_t local = 0; local < jobs_per_node; ++local) {
        const uint32_t job = first_job + local;
        if (job >= total_jobs) continue;
        const uint32_t token_tile = job / k_tiles;
        const uint32_t hidden_tile = job - token_tile * k_tiles;
        const uint32_t head = hidden_tile / dim_tiles;
        const uint32_t dim = hidden_tile - head * dim_tiles;
        final_dx.wait_front(1);
        const uint32_t source = final_dx.get_read_ptr();
        if constexpr (diagnostic_write_dout) {
            noc1.async_write(
                CoreLocalMem<uint32_t>(source), arena, kTileBytes, {},
                {.page_id = token_tile * arena_cols + hidden_tile});
            noc1.async_write_barrier();
        }
        const uint32_t full_slot = token_tile * dim_tiles + dim;
        for (uint32_t row = 0; row < seq_tiles; ++row) {
            const uint32_t target = head * seq_tiles + row;
            noc_async_write(
                source,
                attention_core_l1(
                    target,
                    attention_dout_full_address + full_slot * kTileBytes),
                kTileBytes);
        }
        const uint32_t row_target = head * seq_tiles + token_tile;
        noc_async_write(
            source,
            attention_core_l1(
                row_target,
                attention_dout_row_address + dim * kTileBytes),
            kTileBytes);
        // Payload completion precedes every credit. One credit represents one
        // source tile; each attention worker waits for S_tiles*D_tiles.
        noc_async_write_barrier();
        for (uint32_t row = 0; row < seq_tiles; ++row) {
            const uint32_t target = head * seq_tiles + row;
            noc_semaphore_inc(
                attention_core_l1(
                    target, get_semaphore(credit_semaphore)), 1);
        }
        noc_async_atomic_barrier();
        final_dx.pop_front(1);
    }
    WAYPOINT("OP4");
}
