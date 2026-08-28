#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "api/dataflow/endpoints.h"
#include "api/debug/waypoint.h"

namespace resident_layer_phase_mlp {
// Final down dA/dB writer plus direct dProduct L1 handoff.



namespace {
constexpr uint32_t kGridCols = 10;
constexpr uint32_t kTileBytes = 2048;
constexpr uint8_t kActivationPhysicalX = 13;  // logical x=10
constexpr uint32_t kMailboxSemaphore = 9;
constexpr uint32_t kReadySemaphore = 10;
FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}  // namespace

FORCE_INLINE void full_down_writer_phase() {
    Noc noc1(1);
    constexpr uint32_t cb_db = get_compile_time_arg_val(20);
    constexpr uint32_t cb_da = get_compile_time_arg_val(30);
    constexpr uint32_t cb_final_dx = get_compile_time_arg_val(12);
    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t k_tiles = 152;
    constexpr uint32_t n_tiles = 28;
    constexpr uint32_t jobs_per_node = 7;
    constexpr uint32_t da_k_tiles_per_node = 2;
    constexpr uint32_t n_tiles_per_node = 1;
    constexpr uint32_t total_jobs = m_tiles * k_tiles;
    constexpr uint32_t arena_cols = k_tiles;

    CircularBuffer db(cb_db);
    CircularBuffer da(cb_da);
    CircularBuffer final_dx(cb_final_dx);
    const uint32_t node =
        get_absolute_logical_y() * kGridCols + get_absolute_logical_x();

    const uint32_t shared_output_addr =
        get_common_arg_val<uint32_t>(0);
    const auto shared_output = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<34, 142>(), 0>(),
        shared_output_addr,
        kTileBytes);
    const uint32_t down_a_output_addr =
        get_common_arg_val<uint32_t>(5);
    const auto down_a_output = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<39, 142>(), 5>(),
        down_a_output_addr,
        kTileBytes);
    const uint32_t down_b_output_addr =
        get_common_arg_val<uint32_t>(6);
    const auto down_b_output = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<40, 142>(), 6>(),
        down_b_output_addr,
        kTileBytes);

    // The first 2*M rows belong to final dGate/dUp, produced by the BF16
    // island. Down dA and dB occupy the following rows.
    for (uint32_t local_n = 0; local_n < n_tiles_per_node; ++local_n) {
        const uint32_t n = node * n_tiles_per_node + local_n;
        if (n >= n_tiles) continue;
        db.wait_front(1);
        noc1.async_write(
            CoreLocalMem<uint32_t>(db.get_read_ptr()),
            down_b_output,
            kTileBytes,
            {},
            {.page_id = n});
        noc1.async_write_barrier();
        db.pop_front(1);
    }

    for (uint32_t local_k = 0; local_k < da_k_tiles_per_node; ++local_k) {
        const uint32_t k = node * da_k_tiles_per_node + local_k;
        if (k >= k_tiles) continue;
        da.wait_front(1);
        noc1.async_write(
            CoreLocalMem<uint32_t>(da.get_read_ptr()),
            down_a_output,
            kTileBytes,
            {},
            {.page_id = k});
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

// Final-output writer and member1 -> member0 exact-sum L1 handoff.



namespace {
constexpr uint32_t kGroupedWriterGridCols = 10;
constexpr uint32_t kGroupedWriterMemberNodes = 50;
constexpr uint32_t kGroupedWriterTileBytes = 2048;
constexpr uint8_t kGroupedWriterPhysicalX[kGroupedWriterGridCols] = {
    1, 2, 3, 4, 5, 6, 7, 10, 11, 12};
FORCE_INLINE uint8_t grouped_writer_physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}  // namespace

FORCE_INLINE void full_grouped_writer_phase() {
    Noc noc1(1);
    constexpr uint32_t cb_final_dx = get_compile_time_arg_val(19);
    constexpr uint32_t cb_peer_sum = get_compile_time_arg_val(27);
    constexpr uint32_t cb_da_or_db = get_compile_time_arg_val(20);
    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t k_tiles = 28;
    constexpr uint32_t n_tiles = 152;
    constexpr uint32_t jobs_per_node = 3;
    constexpr uint32_t n_tiles_per_node = 4;
    constexpr uint32_t total_jobs = m_tiles * k_tiles;
    constexpr uint32_t arena_cols = k_tiles;

    CircularBuffer final_dx(cb_final_dx);
    CircularBuffer peer_sum(cb_peer_sum);
    CircularBuffer da_or_db(cb_da_or_db);

    const uint32_t logical_x = get_absolute_logical_x();
    const uint32_t logical_y = get_absolute_logical_y();
    const uint32_t global_node = logical_y * kGroupedWriterGridCols + logical_x;
    const uint32_t member = global_node / kGroupedWriterMemberNodes;
    const uint32_t node = global_node - member * kGroupedWriterMemberNodes;

    const uint32_t input_gradient_output_addr =
        get_common_arg_val<uint32_t>(0);
    const auto input_gradient_output = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<34, 142>(), 0>(),
        input_gradient_output_addr,
        kGroupedWriterTileBytes);
    const uint32_t gate_a_output_addr =
        get_common_arg_val<uint32_t>(1);
    const auto gate_a_output = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<35, 142>(), 1>(),
        gate_a_output_addr,
        kGroupedWriterTileBytes);
    const uint32_t gate_b_output_addr =
        get_common_arg_val<uint32_t>(2);
    const auto gate_b_output = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<36, 142>(), 2>(),
        gate_b_output_addr,
        kGroupedWriterTileBytes);
    const uint32_t up_a_output_addr =
        get_common_arg_val<uint32_t>(3);
    const auto up_a_output = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<37, 142>(), 3>(),
        up_a_output_addr,
        kGroupedWriterTileBytes);
    const uint32_t up_b_output_addr =
        get_common_arg_val<uint32_t>(4);
    const auto up_b_output = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<38, 142>(), 4>(),
        up_b_output_addr,
        kGroupedWriterTileBytes);

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
            if (member == 0) {
                noc1.async_write(
                    CoreLocalMem<uint32_t>(da_or_db.get_read_ptr()),
                    gate_b_output, kGroupedWriterTileBytes, {},
                    {.page_id = n_tile});
            } else {
                noc1.async_write(
                    CoreLocalMem<uint32_t>(da_or_db.get_read_ptr()),
                    up_b_output, kGroupedWriterTileBytes, {},
                    {.page_id = n_tile});
            }
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
            if (member == 0) {
                noc1.async_write(
                    CoreLocalMem<uint32_t>(da_or_db.get_read_ptr()),
                    gate_a_output, kGroupedWriterTileBytes, {},
                    {.page_id = node});
            } else {
                noc1.async_write(
                    CoreLocalMem<uint32_t>(da_or_db.get_read_ptr()),
                    up_a_output, kGroupedWriterTileBytes, {},
                    {.page_id = node});
            }
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
                    source + local * kGroupedWriterTileBytes,
                    get_noc_addr(
                        kGroupedWriterPhysicalX[logical_x],
                        grouped_writer_physical_y(peer_logical_y),
                        source + local * kGroupedWriterTileBytes),
                    kGroupedWriterTileBytes);
            }
            noc_async_write_barrier();
            volatile tt_l1_ptr uint32_t* response_sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(9));
            noc_semaphore_set(response_sem, valid_jobs);
            noc_async_write(
                get_semaphore(9),
                get_noc_addr(
                    kGroupedWriterPhysicalX[logical_x],
                    grouped_writer_physical_y(peer_logical_y),
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
                    input_gradient_output,
                    kGroupedWriterTileBytes,
                    {},
                    {.page_id = job});
                noc1.async_write_barrier();
            }
            final_dx.pop_front(1);
        }
    }

}

void run() {
    full_down_writer_phase();
    full_grouped_writer_phase();
}
}  // namespace resident_layer_phase_mlp

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
    constexpr uint32_t cb_db = get_compile_time_arg_val(42);
    constexpr uint32_t cb_da = get_compile_time_arg_val(43);
    constexpr uint32_t cb_final_dx = get_compile_time_arg_val(22);
    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t k_tiles = 28;
    constexpr uint32_t n_tiles = 4;
    constexpr uint32_t jobs_per_node = 4;
    constexpr uint32_t da_k_tiles_per_node = 1;
    constexpr uint32_t n_tiles_per_node = 1;
    constexpr uint32_t total_jobs = m_tiles * k_tiles;
    constexpr uint32_t arena_cols = k_tiles;

    CircularBuffer db(cb_db);
    CircularBuffer da(cb_da);
    CircularBuffer final_dx(cb_final_dx);
    const uint32_t node =
        get_absolute_logical_y() * 5 + get_absolute_logical_x();

    const uint32_t arena_addr = get_common_arg_val<uint32_t>(7);
    const auto arena = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<41, 142>(), 7>(), arena_addr, v_kTileBytes);

    for (uint32_t local_n = 0; local_n < n_tiles_per_node; ++local_n) {
        const uint32_t n = node * n_tiles_per_node + local_n;
        if (n >= n_tiles) continue;
        db.wait_front(1);
        {
            DeviceZoneScopedN("resident_down_final_db_write");
            noc1.async_write(
                CoreLocalMem<uint32_t>(db.get_read_ptr()), arena,
                v_kTileBytes, {}, {.page_id = 29 * arena_cols + ((39 + n) * arena_cols)});
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
                v_kTileBytes, {}, {.page_id = 29 * arena_cols + (6 * arena_cols + k)});
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
    constexpr uint32_t cb_db = get_compile_time_arg_val(42);
    constexpr uint32_t cb_da = get_compile_time_arg_val(43);
    constexpr uint32_t cb_final_dx = get_compile_time_arg_val(22);
    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t k_tiles = 28;
    constexpr uint32_t n_tiles = 4;
    constexpr uint32_t jobs_per_node = 4;
    constexpr uint32_t da_k_tiles_per_node = 1;
    constexpr uint32_t n_tiles_per_node = 1;
    constexpr uint32_t total_jobs = m_tiles * k_tiles;
    constexpr uint32_t arena_cols = k_tiles;

    CircularBuffer db(cb_db);
    CircularBuffer da(cb_da);
    CircularBuffer final_dx(cb_final_dx);
    const uint32_t node =
        get_absolute_logical_y() * 5 + get_absolute_logical_x();

    const uint32_t arena_addr = get_common_arg_val<uint32_t>(7);
    const auto arena = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<41, 142>(), 7>(), arena_addr, k_kTileBytes);

    for (uint32_t local_n = 0; local_n < n_tiles_per_node; ++local_n) {
        const uint32_t n = node * n_tiles_per_node + local_n;
        if (n >= n_tiles) continue;
        db.wait_front(1);
        {
            DeviceZoneScopedN("resident_down_final_db_write");
            noc1.async_write(
                CoreLocalMem<uint32_t>(db.get_read_ptr()), arena,
                k_kTileBytes, {}, {.page_id = 29 * arena_cols + ((35 + n) * arena_cols)});
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
                k_kTileBytes, {}, {.page_id = 29 * arena_cols + (5 * arena_cols + k)});
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
    constexpr uint32_t k_tiles = 28;
    constexpr uint32_t n_tiles = 28;
    constexpr uint32_t jobs_per_node = 4;
    constexpr uint32_t da_k_tiles_per_node = 1;
    constexpr uint32_t n_tiles_per_node = 1;
    constexpr uint32_t total_jobs = m_tiles * k_tiles;
    constexpr uint32_t arena_cols = k_tiles;

    CircularBuffer db(cb_db);
    CircularBuffer da(cb_da);
    CircularBuffer final_dx(cb_final_dx);
    const uint32_t node =
        get_absolute_logical_y() * 5 + get_absolute_logical_x();

    const uint32_t arena_addr = get_common_arg_val<uint32_t>(7);
    const auto arena = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<41, 142>(), 7>(), arena_addr, q_kTileBytes);

    for (uint32_t local_n = 0; local_n < n_tiles_per_node; ++local_n) {
        const uint32_t n = node * n_tiles_per_node + local_n;
        if (n >= n_tiles) continue;
        db.wait_front(1);
        {
            DeviceZoneScopedN("resident_down_final_db_write");
            noc1.async_write(
                CoreLocalMem<uint32_t>(db.get_read_ptr()), arena,
                q_kTileBytes, {}, {.page_id = 29 * arena_cols + ((7 + n) * arena_cols)});
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
                q_kTileBytes, {}, {.page_id = 29 * arena_cols + (4 * arena_cols + k)});
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
    constexpr uint32_t cb_db = get_compile_time_arg_val(42);
    constexpr uint32_t cb_da = get_compile_time_arg_val(43);
    constexpr uint32_t cb_final_dx = get_compile_time_arg_val(22);
    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t k_tiles = 28;
    constexpr uint32_t n_tiles = 28;
    constexpr uint32_t dim_tiles = 2;
    constexpr uint32_t seq_tiles = 4;
    constexpr uint32_t query_heads = 14;
    constexpr uint32_t jobs_per_node = 4;
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
    const uint32_t arena_addr = get_common_arg_val<uint32_t>(7);
    const auto arena = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<41, 142>(), 7>(),
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
    resident_layer_phase_mlp::run();
    constexpr uint8_t layer_physical_x[11] = {
        1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13};
    const uint32_t layer_node =
        get_absolute_logical_y() * 5 +
        (get_absolute_logical_x() - 0);
    noc_semaphore_inc(
        get_noc_addr(layer_physical_x[0], 2, get_semaphore(13)), 1);
    noc_async_atomic_barrier();
    if (layer_node == 0) {
        volatile tt_l1_ptr uint32_t* layer_arrivals =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(13));
        noc_semaphore_wait_min(layer_arrivals, 50);
        noc_semaphore_set(layer_arrivals, 0);
        volatile tt_l1_ptr uint32_t* layer_release_word =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(14));
        noc_semaphore_set(layer_release_word, 1);
        for (uint32_t target = 1; target < 50; ++target) {
            noc_async_write(
                get_semaphore(14),
                get_noc_addr(
                    layer_physical_x[0 + target % 5],
                    static_cast<uint8_t>(target / 5 + 2),
                    get_semaphore(14)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
        noc_semaphore_set(layer_release_word, 0);
    } else {
        volatile tt_l1_ptr uint32_t* layer_release_word =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(14));
        noc_semaphore_wait(layer_release_word, 1);
        noc_semaphore_set(layer_release_word, 0);
    }
    volatile tt_l1_ptr uint32_t* layer_reader_release =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(12));
    noc_semaphore_set(layer_reader_release, 1);
    CircularBuffer layer_compute_release(get_compile_time_arg_val(61));
    layer_compute_release.reserve_back(1);
    layer_compute_release.push_back(1);
    resident_layer_phase_attention::run();
}
