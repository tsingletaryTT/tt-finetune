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
    constexpr uint32_t cb_db = get_compile_time_arg_val(87);
    constexpr uint32_t cb_da = get_compile_time_arg_val(88);
    constexpr uint32_t cb_final_dx = get_compile_time_arg_val(90);
    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t k_tiles = 48;
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
    constexpr uint32_t cb_final_dx = get_compile_time_arg_val(81);
    constexpr uint32_t cb_peer_sum = get_compile_time_arg_val(91);
    constexpr uint32_t cb_da_or_db = get_compile_time_arg_val(87);
    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t k_tiles = 18;
    constexpr uint32_t n_tiles = 48;
    constexpr uint32_t jobs_per_node = 2;
    constexpr uint32_t n_tiles_per_node = 1;
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
// L1/NoC dS transpose, GQA handoff, and final-only gradient writer.


namespace {
constexpr uint32_t kGridCols = 6;
constexpr uint32_t kTileBytes = 2048;
constexpr uint8_t kPhysicalX[kGridCols] = {6, 7, 10, 11, 12, 13};
FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
FORCE_INLINE uint64_t core_l1(uint32_t node, uint32_t address) {
    return get_noc_addr(
        kPhysicalX[node % kGridCols], physical_y(node / kGridCols), address);
}
}  // namespace

void run() {
    DeviceZoneScopedN("attention_resident_writer");
    Noc noc1(1);
    constexpr uint32_t cb_dv = get_compile_time_arg_val(98);
    constexpr uint32_t cb_dq = get_compile_time_arg_val(86);
    constexpr uint32_t cb_ds_handoff = get_compile_time_arg_val(79);
    constexpr uint32_t cb_ds_column = get_compile_time_arg_val(80);
    constexpr uint32_t cb_dk = get_compile_time_arg_val(87);
    constexpr uint32_t cb_peer_dk = get_compile_time_arg_val(81);
    constexpr uint32_t cb_peer_dv = get_compile_time_arg_val(82);
    constexpr uint32_t cb_gqa_reduced = get_compile_time_arg_val(83);
    constexpr uint32_t cb_gqa_bf16 = get_compile_time_arg_val(89);
    constexpr uint32_t cb_final_dk = get_compile_time_arg_val(90);
    constexpr uint32_t cb_final_dv = get_compile_time_arg_val(91);
    constexpr uint32_t cb_rope_q = get_compile_time_arg_val(96);
    constexpr uint32_t cb_rope_k = get_compile_time_arg_val(97);
    constexpr uint32_t seq_tiles = 4;
    constexpr uint32_t dim_tiles = 2;
    constexpr uint32_t query_heads = 9;
    constexpr uint32_t kv_heads = 3;
    constexpr uint32_t group = query_heads / kv_heads;
    constexpr bool kBf16 = 1;
    constexpr uint32_t vector_tiles = seq_tiles * dim_tiles;
    constexpr uint32_t output_row_offset = 23;
    constexpr uint32_t output_arena_col_tiles = 18;
    auto output_page = [](uint32_t packed_page) {
        return output_row_offset * output_arena_col_tiles +
               (packed_page / dim_tiles) * output_arena_col_tiles +
               packed_page % dim_tiles;
    };
    constexpr uint32_t jobs = query_heads * seq_tiles;
    constexpr uint32_t reducer_jobs = kv_heads * seq_tiles;
    constexpr bool diagnostic_ds = 0;
    constexpr bool diagnostic_dv = 0;
    constexpr bool diagnostic_pt = 0;
    constexpr bool diagnostic_dq = 0;
    constexpr bool diagnostic_dk = 0;
    constexpr bool diagnostic_dp = 0;
    constexpr bool diagnostic_product = 0;
    constexpr bool diagnostic_centered = 0;
    constexpr bool diagnostic_rowsum = 0;
    constexpr bool diagnostic_unscaled_ds = 0;
    constexpr bool diagnostic_prefix =
        diagnostic_ds || diagnostic_dv || diagnostic_pt || diagnostic_dq || diagnostic_dp || diagnostic_product || diagnostic_centered ||
        diagnostic_rowsum || diagnostic_unscaled_ds;

    const uint32_t node =
        get_absolute_logical_y() * kGridCols + (get_absolute_logical_x() - 5);
    const uint32_t output_addr = get_common_arg_val<uint32_t>(7);
    const auto output = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<41, 142>(), 7>(),
        output_addr, kTileBytes);

    CircularBuffer gqa_reduced(cb_gqa_reduced);
    if (node >= jobs) return;
    const uint32_t head = node / seq_tiles;
    const uint32_t row = node - head * seq_tiles;
    const uint32_t kv_head = head / group;
    const uint32_t member = head - kv_head * group;
    const uint32_t owner_node = kv_head * group * seq_tiles + row;

    CircularBuffer dq(cb_dq), dv(cb_dv), dk(cb_dk);
    CircularBuffer rope_q(cb_rope_q), rope_k(cb_rope_k);
    CircularBuffer ds_handoff(cb_ds_handoff), ds_column(cb_ds_column);
    CircularBuffer peer_dk(cb_peer_dk), peer_dv(cb_peer_dv);
    CircularBuffer final_dk(cb_final_dk), final_dv(cb_final_dv);

    if constexpr (diagnostic_dv || diagnostic_pt)
        dv.wait_front(dim_tiles);
    else if constexpr (diagnostic_ds || (diagnostic_prefix && !diagnostic_dq))
        ds_handoff.wait_front(seq_tiles);
    else rope_q.wait_front(dim_tiles);
    WAYPOINT("AW0");
    // rope_q remains at the CB front for the projection island.
    if constexpr (diagnostic_prefix) {
        if constexpr (!diagnostic_dq)
            ds_handoff.pop_front(seq_tiles);
        return;
    }

    if constexpr (diagnostic_dk) {
        dk.wait_front(dim_tiles);
        for (uint32_t tile = 0; tile < dim_tiles; ++tile) {
            noc1.async_write(
                CoreLocalMem<uint32_t>(
                    dk.get_read_ptr() + tile * kTileBytes),
                output, kTileBytes, {},
                {.page_id = output_page(node * dim_tiles + tile)});
        }
        noc1.async_write_barrier();
        dk.pop_front(dim_tiles);
        return;
    }

    if (member != 0) {
        DeviceZoneScopedN("attention_gqa_peer_noc");
        dk.wait_front(dim_tiles);
        dv.wait_front(dim_tiles);
        volatile tt_l1_ptr uint32_t* dk_mailbox =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(8));
        volatile tt_l1_ptr uint32_t* dv_mailbox =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(9));
        noc_semaphore_wait_min(dk_mailbox, 1);
        noc_semaphore_wait_min(dv_mailbox, 1);
        WAYPOINT("AW1");
        const uint32_t peer_dk_destination = *dk_mailbox;
        const uint32_t peer_dv_destination = *dv_mailbox;
        const uint32_t peer_offset = (member - 1) * dim_tiles * kTileBytes;
        for (uint32_t tile = 0; tile < dim_tiles; ++tile) {
            noc_async_write(
                dk.get_read_ptr() + tile * kTileBytes,
                core_l1(
                    owner_node,
                    peer_dk_destination + peer_offset + tile * kTileBytes),
                kTileBytes);
            noc_async_write(
                dv.get_read_ptr() + tile * kTileBytes,
                core_l1(
                    owner_node,
                    peer_dv_destination + peer_offset + tile * kTileBytes),
                kTileBytes);
        }
        noc_async_write_barrier();
        noc_semaphore_inc(core_l1(owner_node, get_semaphore(5)), 1);
        noc_async_atomic_barrier();
        WAYPOINT("AW2");
        dk.pop_front(dim_tiles);
        dv.pop_front(dim_tiles);
        // Projection readers publish this only after every output-projection
        // writer has stopped touching the shared semaphore map.
        volatile tt_l1_ptr uint32_t* qkv_publish_go =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(14));
        noc_semaphore_wait(qkv_publish_go, 1);
        noc_semaphore_set(qkv_publish_go, 0);
        // Publish only after this worker has finished all GQA peer writes.
        noc_semaphore_inc(
            get_noc_addr(1, physical_y(0), get_semaphore(0)), 1);
        noc_async_atomic_barrier();
        volatile tt_l1_ptr uint32_t* release =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(11));
        noc_semaphore_wait(release, 1);
        noc_semaphore_set(release, 0);
        rope_q.pop_front(dim_tiles);
        return;
    }

    if constexpr (kBf16) {
        DeviceZoneScopedN("attention_final_dkdv_write");
        const uint32_t dk_base = query_heads * vector_tiles;
        const uint32_t dv_base = dk_base + kv_heads * vector_tiles;
        // Do not publish Q/K/V addresses into the
        // projection-local 0..4 map until resident-down has retired there.
        volatile tt_l1_ptr uint32_t* qkv_publish_go =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(14));
        noc_semaphore_wait(qkv_publish_go, 1);
        noc_semaphore_set(qkv_publish_go, 0);
        // Semaphore 10 is the projection-writer completion counter on
        // attention node zero.  It may contain state from an unrelated
        // program (or an earlier invocation), so establish a clean epoch
        // before this node publishes the last attention-ready credit.  The
        // projection readers cannot start until every attention worker,
        // including node zero, has published that credit; consequently no
        // projection writer can race this reset.
        if (node == 0) {
            noc_semaphore_set(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(10)),
                0);
        }
        WAYPOINT("AH0");
        rope_k.wait_front(dim_tiles);
        // Skip the unrotated dK window.  The reduced dV window remains at
        // the front beside post-RoPE K until the projection readers finish.
        gqa_reduced.pop_front(dim_tiles);
        gqa_reduced.wait_front(dim_tiles);
        WAYPOINT("AH1");
        if (node == 0) {
            volatile tt_l1_ptr uint32_t* address_word =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(15));
            const uint32_t addresses[3] = {
                rope_q.get_read_ptr(), rope_k.get_read_ptr(),
                gqa_reduced.get_read_ptr()};
            for (uint32_t index = 0; index < 3; ++index) {
                noc_semaphore_set(address_word, addresses[index]);
                noc_async_write(
                    get_semaphore(15),
                    get_noc_addr(
                        1, physical_y(0), get_semaphore(11 + index)),
                    sizeof(uint32_t));
                noc_async_write_barrier();
            }
        }
        noc_semaphore_inc(
            get_noc_addr(1, physical_y(0), get_semaphore(0)), 1);
        noc_async_atomic_barrier();
        WAYPOINT("AH2");
        volatile tt_l1_ptr uint32_t* release =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(11));
        if (node == 0) {
            WAYPOINT("AH3");
            volatile tt_l1_ptr uint32_t* projection_done =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(10));
            noc_semaphore_wait_min(projection_done, 50);
            noc_semaphore_set(projection_done, 0);
            noc_semaphore_set(release, 1);
            for (uint32_t target = 1; target < jobs; ++target) {
                noc_async_write(
                    get_semaphore(11),
                    core_l1(target, get_semaphore(11)), sizeof(uint32_t));
            }
            noc_async_write_barrier();
        }
        noc_semaphore_wait(release, 1);
        noc_semaphore_set(release, 0);
        rope_q.pop_front(dim_tiles);
        rope_k.pop_front(dim_tiles);
        gqa_reduced.pop_front(dim_tiles);
        WAYPOINT("AW4");
        return;
    }

    // Inverse of the reader's transpose(group, H): each reduced tile carries
    // one logical sequence row in row zero.  Assemble the 32 rows directly in
    // the final L1 tiles, so no intermediate tensor is visible to PJRT/DRAM.
    constexpr uint32_t face_bytes = 16 * 16 * 2;
    constexpr uint32_t face_row_bytes = 16 * 2;
    auto row_half_offset = [](uint32_t logical_row, uint32_t right) {
        const uint32_t face_row = logical_row & 15;
        const uint32_t face_band = logical_row >> 4;
        return (face_band * 2 + right) * face_bytes +
               face_row * face_row_bytes;
    };
    final_dk.reserve_back(dim_tiles);
    final_dv.reserve_back(dim_tiles);
    noc1.async_write_zeros(final_dk, dim_tiles * kTileBytes);
    noc1.write_zeros_l1_barrier();
    noc1.async_write_zeros(final_dv, dim_tiles * kTileBytes);
    noc1.write_zeros_l1_barrier();
    for (uint32_t kind = 0; kind < 2; ++kind) {
        const uint32_t destination = kind == 0
            ? final_dk.get_write_ptr() : final_dv.get_write_ptr();
        for (uint32_t tile = 0; tile < dim_tiles; ++tile) {
            for (uint32_t logical_row = 0; logical_row < 32; ++logical_row) {
                gqa_reduced.wait_front(1);
                const uint32_t source = gqa_reduced.get_read_ptr();
                for (uint32_t right = 0; right < 2; ++right) {
                    noc_async_write(
                        source + row_half_offset(0, right),
                        get_noc_addr(
                            destination + tile * kTileBytes +
                            row_half_offset(logical_row, right)),
                        face_row_bytes);
                }
                noc_async_write_barrier();
                gqa_reduced.pop_front(1);
            }
        }
    }
    final_dk.push_back(dim_tiles);
    final_dv.push_back(dim_tiles);
    final_dk.wait_front(dim_tiles);
    final_dv.wait_front(dim_tiles);
    const uint32_t dk_base = query_heads * vector_tiles;
    const uint32_t dv_base = dk_base + kv_heads * vector_tiles;
    {
        DeviceZoneScopedN("attention_final_dkdv_write");
        for (uint32_t tile = 0; tile < dim_tiles; ++tile) {
            const uint32_t kv_page = kv_head * vector_tiles + row * dim_tiles + tile;
            noc1.async_write(
                CoreLocalMem<uint32_t>(final_dk.get_read_ptr() + tile * kTileBytes),
                output, kTileBytes, {}, {.page_id = output_page(dk_base + kv_page)});
            noc1.async_write(
                CoreLocalMem<uint32_t>(final_dv.get_read_ptr() + tile * kTileBytes),
                output, kTileBytes, {}, {.page_id = output_page(dv_base + kv_page)});
        }
        noc1.async_write_barrier();
    }
    final_dk.pop_front(dim_tiles);
    final_dv.pop_front(dim_tiles);
}
}  // namespace resident_layer_phase_attention

void kernel_main() {
    resident_layer_phase_mlp::run();
    constexpr uint8_t layer_physical_x[11] = {
        1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13};
    const uint32_t layer_node =
        get_absolute_logical_y() * 5 +
        (get_absolute_logical_x() - 5);
    noc_semaphore_inc(
        get_noc_addr(layer_physical_x[5], 2, get_semaphore(13)), 1);
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
                    layer_physical_x[5 + target % 5],
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
    CircularBuffer layer_compute_release(get_compile_time_arg_val(102));
    layer_compute_release.reserve_back(1);
    layer_compute_release.push_back(1);
    resident_layer_phase_attention::run();
}
