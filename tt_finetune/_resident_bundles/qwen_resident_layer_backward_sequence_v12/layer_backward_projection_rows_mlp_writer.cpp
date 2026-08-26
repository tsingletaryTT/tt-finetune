#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "api/dataflow/endpoints.h"

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

void kernel_main() {
    resident_layer_phase_mlp::run();
}
