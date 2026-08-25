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

namespace resident_layer_phase_attention {
// Output-projection final writer for the first fused attention-block phase.
//
// dA/dB are final gradients and are written to the projection arena. dO is
// not an output: its token-major tiles are sent directly to the 60-core
// attention island in the head-major layout consumed by the resident core.



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

// Final-only writer for resident down-projection backward.



namespace { constexpr uint32_t v_kTileBytes = 2048; }

FORCE_INLINE void v_writer_phase() {
    Noc noc1(1);
    constexpr uint32_t cb_db = get_compile_time_arg_val(31);
    constexpr uint32_t cb_da = get_compile_time_arg_val(33);
    constexpr uint32_t cb_final_dx = get_compile_time_arg_val(35);
    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t k_tiles = 18;
    constexpr uint32_t n_tiles = 6;
    constexpr uint32_t jobs_per_node = 2;
    constexpr uint32_t da_k_tiles_per_node = 1;
    constexpr uint32_t n_tiles_per_node = 1;
    constexpr uint32_t total_jobs = m_tiles * k_tiles;
    constexpr uint32_t arena_cols = k_tiles;

    CircularBuffer db(cb_db);
    CircularBuffer da(cb_da);
    CircularBuffer final_dx(cb_final_dx);
    const uint32_t node =
        get_absolute_logical_y() * 5 + get_absolute_logical_x();

    const uint32_t arena_addr = get_common_arg_val<uint32_t>(0);
    const auto arena = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<41, 142>(), 0>(), arena_addr, v_kTileBytes);

    for (uint32_t local_n = 0; local_n < n_tiles_per_node; ++local_n) {
        const uint32_t n = node * n_tiles_per_node + local_n;
        if (n >= n_tiles) continue;
        db.wait_front(1);
        {
            DeviceZoneScopedN("resident_down_final_db_write");
            noc1.async_write(
                CoreLocalMem<uint32_t>(db.get_read_ptr()), arena,
                v_kTileBytes, {}, {.page_id = 19 * arena_cols + ((31 + n) * arena_cols)});
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
                v_kTileBytes, {}, {.page_id = 19 * arena_cols + (6 * arena_cols + k)});
            noc1.async_write_barrier();
        }
        da.pop_front(1);
    }


    // A count-only CB fence guarantees compute has completed the phase.
    CircularBuffer phase_done_cb(25);
    volatile tt_l1_ptr uint32_t* phase_done_word =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            phase_done_cb.get_read_ptr());
    while (*phase_done_word != 1) {}
    *phase_done_word = 0;
    // Clear this core's complete resident-down semaphore protocol while all
    // readers are still held behind the previous phase.  The global writer
    // barrier below proves every core has completed this reset before any
    // reader is released into the next phase.
    constexpr uint32_t resident_m_tiles = 4;
    for (uint32_t semaphore = 0;
         semaphore < 2 * resident_m_tiles + 1; ++semaphore) {
        volatile tt_l1_ptr uint32_t* flag =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(semaphore));
        noc_semaphore_set(flag, 0);
    }
    // A complete device-side barrier is required here. A fast core must not
    // enter the next reader and clear semaphores while a slow core is still
    // finishing this phase's cross-core dH multicast.
    constexpr uint32_t done_semaphore = 15;
    constexpr uint32_t reader_go_semaphore = 10;
    volatile tt_l1_ptr uint32_t* local_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(done_semaphore));
    // Reader-side phase initialization publishes semaphore 14 state two only
    // after every reader has reset 0..10.  Semaphore 15 remains at output
    // completion state two, so V advances it monotonically to token/release
    // states three/four without a cross-RISC clear race.
    volatile tt_l1_ptr uint32_t* readers_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(14));
    noc_semaphore_wait_min(readers_ready, 2);
    if (node == 0) {
        noc_semaphore_inc(
            get_noc_addr(2, 2, get_semaphore(done_semaphore)), 1);
        noc_async_atomic_barrier();
        noc_semaphore_wait_min(local_done, 3);
    } else {
        noc_semaphore_wait_min(local_done, 3);
        const uint32_t next = node + 1 == 50 ? 0 : node + 1;
        noc_semaphore_inc(
            get_noc_addr(
                1 + next % 5, 2 + next / 5,
                get_semaphore(done_semaphore)),
            1);
        noc_async_atomic_barrier();
    }
    if (node == 0) {
        noc_semaphore_set(local_done, 4);
        for (uint32_t target = 1; target < 50; ++target) {
            noc_async_write(
                get_semaphore(done_semaphore),
                get_noc_addr(
                    1 + target % 5, 2 + target / 5,
                    get_semaphore(done_semaphore)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    noc_semaphore_wait_min(local_done, 4);
    // Every writer has now retired.  Release the co-located reader locally;
    // this avoids a root-to-49 fan-out whose remote stores were observed to
    // disappear under full-program traffic even after the NoC barrier.
    volatile tt_l1_ptr uint32_t* reader_go =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(reader_go_semaphore));
    noc_semaphore_set(reader_go, 1);
}

// Final-only writer for resident down-projection backward.



namespace { constexpr uint32_t k_kTileBytes = 2048; }

FORCE_INLINE void k_writer_phase() {
    Noc noc1(1);
    constexpr uint32_t cb_db = get_compile_time_arg_val(31);
    constexpr uint32_t cb_da = get_compile_time_arg_val(33);
    constexpr uint32_t cb_final_dx = get_compile_time_arg_val(35);
    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t k_tiles = 18;
    constexpr uint32_t n_tiles = 6;
    constexpr uint32_t jobs_per_node = 2;
    constexpr uint32_t da_k_tiles_per_node = 1;
    constexpr uint32_t n_tiles_per_node = 1;
    constexpr uint32_t total_jobs = m_tiles * k_tiles;
    constexpr uint32_t arena_cols = k_tiles;

    CircularBuffer db(cb_db);
    CircularBuffer da(cb_da);
    CircularBuffer final_dx(cb_final_dx);
    const uint32_t node =
        get_absolute_logical_y() * 5 + get_absolute_logical_x();

    const uint32_t arena_addr = get_common_arg_val<uint32_t>(0);
    const auto arena = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<41, 142>(), 0>(), arena_addr, k_kTileBytes);

    for (uint32_t local_n = 0; local_n < n_tiles_per_node; ++local_n) {
        const uint32_t n = node * n_tiles_per_node + local_n;
        if (n >= n_tiles) continue;
        db.wait_front(1);
        {
            DeviceZoneScopedN("resident_down_final_db_write");
            noc1.async_write(
                CoreLocalMem<uint32_t>(db.get_read_ptr()), arena,
                k_kTileBytes, {}, {.page_id = 19 * arena_cols + ((25 + n) * arena_cols)});
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
                k_kTileBytes, {}, {.page_id = 19 * arena_cols + (5 * arena_cols + k)});
            noc1.async_write_barrier();
        }
        da.pop_front(1);
    }


    // A count-only CB fence guarantees compute has completed the phase.
    CircularBuffer phase_done_cb(28);
    volatile tt_l1_ptr uint32_t* phase_done_word =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            phase_done_cb.get_read_ptr());
    while (*phase_done_word != 1) {}
    *phase_done_word = 0;
    // Clear this core's complete resident-down semaphore protocol while all
    // readers are still held behind the previous phase.  The global writer
    // barrier below proves every core has completed this reset before any
    // reader is released into the next phase.
    constexpr uint32_t resident_m_tiles = 4;
    for (uint32_t semaphore = 0;
         semaphore < 2 * resident_m_tiles + 1; ++semaphore) {
        volatile tt_l1_ptr uint32_t* flag =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(semaphore));
        noc_semaphore_set(flag, 0);
    }
    // A complete device-side barrier is required here. A fast core must not
    // enter the next reader and clear semaphores while a slow core is still
    // finishing this phase's cross-core dH multicast.
    constexpr uint32_t done_semaphore = 15;
    constexpr uint32_t reader_go_semaphore = 10;
    volatile tt_l1_ptr uint32_t* local_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(done_semaphore));
    // V left semaphore 15 at release state four.  K advances the same epoch
    // monotonically to token/release states five/six.
    if (node == 0) {
        noc_semaphore_inc(
            get_noc_addr(2, 2, get_semaphore(done_semaphore)), 1);
        noc_async_atomic_barrier();
        noc_semaphore_wait_min(local_done, 5);
    } else {
        noc_semaphore_wait_min(local_done, 5);
        const uint32_t next = node + 1 == 50 ? 0 : node + 1;
        noc_semaphore_inc(
            get_noc_addr(
                1 + next % 5, 2 + next / 5,
                get_semaphore(done_semaphore)),
            1);
        noc_async_atomic_barrier();
    }
    if (node == 0) {
        noc_semaphore_set(local_done, 6);
        for (uint32_t target = 1; target < 50; ++target) {
            noc_async_write(
                get_semaphore(done_semaphore),
                get_noc_addr(
                    1 + target % 5, 2 + target / 5,
                    get_semaphore(done_semaphore)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    noc_semaphore_wait_min(local_done, 6);
    volatile tt_l1_ptr uint32_t* reader_go =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(reader_go_semaphore));
    noc_semaphore_set(reader_go, 1);
    noc_semaphore_set(local_done, 0);
}

// Final-only writer for resident down-projection backward.



namespace { constexpr uint32_t q_kTileBytes = 2048; }

FORCE_INLINE void q_writer_phase() {
    Noc noc1(1);
    constexpr uint32_t cb_db = 34;
    constexpr uint32_t cb_da = 35;
    constexpr uint32_t cb_final_dx = 56;
    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t k_tiles = 18;
    constexpr uint32_t n_tiles = 18;
    constexpr uint32_t jobs_per_node = 2;
    constexpr uint32_t da_k_tiles_per_node = 1;
    constexpr uint32_t n_tiles_per_node = 1;
    constexpr uint32_t total_jobs = m_tiles * k_tiles;
    constexpr uint32_t arena_cols = k_tiles;

    CircularBuffer db(cb_db);
    CircularBuffer da(cb_da);
    CircularBuffer final_dx(cb_final_dx);
    const uint32_t node =
        get_absolute_logical_y() * 5 + get_absolute_logical_x();

    const uint32_t arena_addr = get_common_arg_val<uint32_t>(0);
    const auto arena = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<41, 142>(), 0>(), arena_addr, q_kTileBytes);

    for (uint32_t local_n = 0; local_n < n_tiles_per_node; ++local_n) {
        const uint32_t n = node * n_tiles_per_node + local_n;
        if (n >= n_tiles) continue;
        db.wait_front(1);
        {
            DeviceZoneScopedN("resident_down_final_db_write");
            noc1.async_write(
                CoreLocalMem<uint32_t>(db.get_read_ptr()), arena,
                q_kTileBytes, {}, {.page_id = 19 * arena_cols + ((7 + n) * arena_cols)});
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
                q_kTileBytes, {}, {.page_id = 19 * arena_cols + (4 * arena_cols + k)});
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
                q_kTileBytes, {}, {.page_id = job});
            noc1.async_write_barrier();
        }
        final_dx.pop_front(1);
    }
    // A count-only CB fence guarantees compute has completed the phase.
    CircularBuffer phase_done_cb(29);
    volatile tt_l1_ptr uint32_t* phase_done_word =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            phase_done_cb.get_read_ptr());
    while (*phase_done_word != 1) {}
    *phase_done_word = 0;
    // Clear this core's complete resident-down semaphore protocol while all
    // readers are still held behind the previous phase.  The global writer
    // barrier below proves every core has completed this reset before any
    // reader is released into the next phase.
    constexpr uint32_t resident_m_tiles = 4;
    for (uint32_t semaphore = 0;
         semaphore < 2 * resident_m_tiles + 1; ++semaphore) {
        volatile tt_l1_ptr uint32_t* flag =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(semaphore));
        noc_semaphore_set(flag, 0);
    }
    // A complete device-side barrier is required here. A fast core must not
    // enter the next reader and clear semaphores while a slow core is still
    // finishing this phase's cross-core dH multicast.
    constexpr uint32_t phase_count_semaphore = 9;
    constexpr uint32_t phase_go_semaphore = 10;
    noc_semaphore_inc(
        get_noc_addr(1, 2, get_semaphore(phase_count_semaphore)), 1);
    noc_async_atomic_barrier();
    if (node == 0) {
        volatile tt_l1_ptr uint32_t* count =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(phase_count_semaphore));
        noc_semaphore_wait_min(count, 50);
        noc_semaphore_set(count, 0);
        volatile tt_l1_ptr uint32_t* go =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(phase_go_semaphore));
        noc_semaphore_set(go, 1);
        for (uint32_t target = 1; target < 50; ++target) {
            noc_semaphore_inc(
                get_noc_addr(
                    1 + target % 5, 2 + target / 5,
                    get_semaphore(phase_go_semaphore)),
                1);
        }
        noc_async_atomic_barrier();
    }

}

void run() {
    {
    // Establish the output-writer ring mailbox before any writer can reach
    // the final handoff.  Every writer executes this as its first action;
    // the long output phase follows, so the token cannot race initialization.
    volatile tt_l1_ptr uint32_t* output_phase_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(15));
    noc_semaphore_set(output_phase_done, 0);
    WAYPOINT("OP0");
    Noc noc1(1);
    constexpr uint32_t cb_db = get_compile_time_arg_val(31);
    constexpr uint32_t cb_da = get_compile_time_arg_val(33);
    constexpr uint32_t cb_final_dx = get_compile_time_arg_val(35);
    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t k_tiles = 18;
    constexpr uint32_t n_tiles = 18;
    constexpr uint32_t dim_tiles = 2;
    constexpr uint32_t seq_tiles = 4;
    constexpr uint32_t query_heads = 9;
    constexpr uint32_t jobs_per_node = 2;
    constexpr uint32_t da_k_tiles_per_node = 1;
    constexpr uint32_t n_tiles_per_node = 1;
    constexpr uint32_t total_jobs = m_tiles * k_tiles;
    constexpr uint32_t active_attention_nodes = query_heads * seq_tiles;
    constexpr uint32_t arena_cols = k_tiles;
    constexpr bool diagnostic_write_dout = 0;
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
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<41, 142>(), 0>(),
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

    if (node == 0) {
        noc_semaphore_inc(
            projection_core_l1(1, get_semaphore(15)), 1);
        noc_async_atomic_barrier();
        noc_semaphore_wait(output_phase_done, 1);
        noc_semaphore_set(output_phase_done, 2);
        for (uint32_t target = 1; target < kProjectionWorkers; ++target) {
            noc_async_write(
                get_semaphore(15),
                projection_core_l1(target, get_semaphore(15)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    } else {
        noc_semaphore_wait(output_phase_done, 1);
        const uint32_t next =
            node + 1 == kProjectionWorkers ? 0 : node + 1;
        noc_semaphore_inc(
            projection_core_l1(next, get_semaphore(15)), 1);
        noc_async_atomic_barrier();
    }
    noc_semaphore_wait(output_phase_done, 2);
    }
    {
    v_writer_phase();
    k_writer_phase();
    q_writer_phase();
    // Release the attention owners only after every final Q/K/V projection
    // output on this core has reached the output arena.
    noc_semaphore_inc(get_noc_addr(6, 2, get_semaphore(10)), 1);
    noc_async_atomic_barrier();
    }
}
}  // namespace resident_layer_phase_attention

void kernel_main() {
    resident_layer_phase_attention::run();
}
