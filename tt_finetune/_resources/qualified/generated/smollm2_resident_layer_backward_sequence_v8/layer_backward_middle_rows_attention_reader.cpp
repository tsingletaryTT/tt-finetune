#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/debug/waypoint.h"
#include "api/tensor/noc_traits.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace resident_layer_phase_attention {
// External input reader plus device-side dS transpose/GQA receive queues.



namespace {
constexpr uint32_t kGridCols = 6;
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kProbabilityTileBytes = 4096;
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
    DeviceZoneScopedN("attention_resident_reader");
    Noc noc0(0);
    constexpr uint32_t cb_p_row = get_compile_time_arg_val(71);
    constexpr uint32_t cb_p_column = get_compile_time_arg_val(72);
    constexpr uint32_t cb_dout_row = get_compile_time_arg_val(88);
    constexpr uint32_t cb_dout_full = get_compile_time_arg_val(66);
    constexpr uint32_t cb_q_full = get_compile_time_arg_val(67);
    constexpr uint32_t cb_k = get_compile_time_arg_val(68);
    constexpr uint32_t cb_v = get_compile_time_arg_val(69);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(101);
    constexpr uint32_t cb_dv = get_compile_time_arg_val(98);
    constexpr uint32_t cb_ds_handoff = get_compile_time_arg_val(79);
    constexpr uint32_t cb_ds_column = get_compile_time_arg_val(80);
    constexpr uint32_t cb_dk = get_compile_time_arg_val(87);
    constexpr uint32_t cb_peer_dk = get_compile_time_arg_val(81);
    constexpr uint32_t cb_peer_dv = get_compile_time_arg_val(82);
    constexpr uint32_t cb_gqa_rows = get_compile_time_arg_val(63);
    constexpr uint32_t cb_gqa_scaler = get_compile_time_arg_val(100);
    constexpr uint32_t cb_rope_cos = get_compile_time_arg_val(92);
    constexpr uint32_t cb_rope_sin = get_compile_time_arg_val(93);
    constexpr uint32_t seq_tiles = 4;
    constexpr uint32_t dim_tiles = 2;
    constexpr uint32_t query_heads = 9;
    constexpr uint32_t kv_heads = 3;
    constexpr uint32_t group = query_heads / kv_heads;
    constexpr bool kBf16 = 1;
    constexpr uint32_t score_tiles = seq_tiles * seq_tiles;
    constexpr uint32_t vector_tiles = seq_tiles * dim_tiles;
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
    constexpr uint32_t active_nodes = jobs;

    const uint32_t node =
        get_absolute_logical_y() * kGridCols + (get_absolute_logical_x() - 5);
    if (node >= active_nodes) return;

    // Every invocation owns fresh data-ready state.  Reset locally in the
    // reader, as in the accepted resident MLP owner.  The first cross-core
    // signal is the dS transpose credit and cannot be produced until this
    // reader has supplied the complete Q/K/V/dO/P prefix to compute, so no
    // producer can race these resets.  A coordinator broadcast here is not
    // safe: a slower core can clear its phase word after receiving the
    // broadcast and wait forever on the next invocation.
    for (uint32_t semaphore = 0; semaphore < 13; ++semaphore) {
        volatile tt_l1_ptr uint32_t* flag =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(semaphore));
        noc_semaphore_set(flag, 0);
    }

    CircularBuffer p_row(cb_p_row), p_column(cb_p_column);
    CircularBuffer dout_row(cb_dout_row), dout_full(cb_dout_full);
    CircularBuffer q_full(cb_q_full), k(cb_k), v(cb_v);
    CircularBuffer ds_column(cb_ds_column), dk(cb_dk), dv(cb_dv);
    CircularBuffer local_ds_handoff(cb_ds_handoff);
    CircularBuffer peer_dk(cb_peer_dk), peer_dv(cb_peer_dv);
    CircularBuffer gqa_rows(cb_gqa_rows);
    const uint32_t head = node / seq_tiles;
    const uint32_t row = node - head * seq_tiles;
    const uint32_t kv_head = head / group;
    const uint32_t member = head - kv_head * group;
    if constexpr (kBf16) {
        if (member == 0) {
            CircularBuffer gqa_zero(cb_gqa_scaler);
            gqa_zero.reserve_back(1);
            noc0.async_write_zeros(gqa_zero, kTileBytes);
            noc0.write_zeros_l1_barrier();
            gqa_zero.push_back(1);
        }
    }
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    // Match generic_reductions.cpp exactly: after transpose(group, H), the
    // padded group dimension is one full 32-row tile and reduce.cpp uses the
    // normal (TF32/FPU) REDUCE_COL path with a full-tile SUM scaler.
    if constexpr (!kBf16) {
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            cb_gqa_scaler, ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_COL>();
    }

    const uint32_t cos_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t sin_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t checkpoint_arena_addr = get_common_arg_val<uint32_t>(2);
    constexpr uint32_t checkpoint_query_page_base = 72;
    constexpr uint32_t checkpoint_key_page_base = 144;
    constexpr uint32_t checkpoint_value_page_base = 168;
    constexpr uint32_t checkpoint_probability_page_base = 280;
    const auto cos_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<21, 142>(), 0>(),
        cos_addr, kTileBytes);
    const auto sin_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<22, 142>(), 1>(),
        sin_addr, kTileBytes);
    const auto checkpoint_arena_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<33, 142>(), 2>(),
        checkpoint_arena_addr, kTileBytes);
    CircularBuffer rope_cos(cb_rope_cos), rope_sin(cb_rope_sin);
    rope_cos.reserve_back(dim_tiles);
    rope_sin.reserve_back(dim_tiles);
    for (uint32_t tile = 0; tile < dim_tiles; ++tile) {
        const uint32_t page = row * dim_tiles + tile;
        noc0.async_read(
            cos_tensor,
            CoreLocalMem<uint32_t>(rope_cos.get_write_ptr() + tile * kTileBytes),
            kTileBytes, {.page_id = page}, {});
        noc0.async_read(
            sin_tensor,
            CoreLocalMem<uint32_t>(rope_sin.get_write_ptr() + tile * kTileBytes),
            kTileBytes, {.page_id = page}, {});
    }
    noc0.async_read_barrier();
    rope_cos.push_back(dim_tiles);
    rope_sin.push_back(dim_tiles);

    p_row.reserve_back(seq_tiles);
    p_column.reserve_back(seq_tiles);
    for (uint32_t tile = 0; tile < seq_tiles; ++tile) {
        const uint32_t p_row_logical_page = head * score_tiles + row * seq_tiles + tile;
        const uint32_t p_row_physical_page =
            checkpoint_probability_page_base + 2 * p_row_logical_page;
        noc0.async_read(
            checkpoint_arena_tensor,
            CoreLocalMem<uint32_t>(p_row.get_write_ptr() + tile * kProbabilityTileBytes),
            kTileBytes, {.page_id = p_row_physical_page}, {});
        noc0.async_read(
            checkpoint_arena_tensor,
            CoreLocalMem<uint32_t>(p_row.get_write_ptr() + tile * kProbabilityTileBytes + kTileBytes),
            kTileBytes, {.page_id = p_row_physical_page + 1}, {});
        const uint32_t p_column_logical_page = head * score_tiles + tile * seq_tiles + row;
        const uint32_t p_column_physical_page =
            checkpoint_probability_page_base + 2 * p_column_logical_page;
        noc0.async_read(
            checkpoint_arena_tensor,
            CoreLocalMem<uint32_t>(p_column.get_write_ptr() + tile * kProbabilityTileBytes),
            kTileBytes, {.page_id = p_column_physical_page}, {});
        noc0.async_read(
            checkpoint_arena_tensor,
            CoreLocalMem<uint32_t>(p_column.get_write_ptr() + tile * kProbabilityTileBytes + kTileBytes),
            kTileBytes, {.page_id = p_column_physical_page + 1}, {});
    }
    dout_row.reserve_back(dim_tiles);
    dout_full.reserve_back(vector_tiles);
    WAYPOINT("OR0");
    // Publish reservation only after clearing the credit semaphore. Projection
    // node zero waits for every active consumer before releasing producers.
    volatile tt_l1_ptr uint32_t* dout_credit =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(12));
    noc_semaphore_set(dout_credit, 0);

    // Establish a clean epoch for projection-node-zero's consumer-ready
    // counter before publishing any credits.  A program launch does not
    // guarantee that semaphore 13 is zero: unrelated programs can leave a
    // value at the same L1 address.  If the projection writer observes that
    // stale value, it can send dO before a slow attention reader clears its
    // local semaphore 12, and that clear permanently erases early credits.
    //
    // Semaphore 11 is otherwise unused on the attention island until the
    // final Q/K/V lifetime release, and semaphore 10 is unused until the
    // final projection-completion count.  Reuse them here for a two-stage
    // reader barrier, then leave both at zero for their later owners.
    constexpr uint32_t active_attention_readers = query_heads * seq_tiles;
    volatile tt_l1_ptr uint32_t* ready_epoch_count =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(10));
    volatile tt_l1_ptr uint32_t* ready_epoch_go =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(11));
    if (node == 0) {
        noc_semaphore_set(ready_epoch_count, 0);
        noc_semaphore_set(ready_epoch_go, 1);
        for (uint32_t target = 1;
             target < active_attention_readers; ++target) {
            noc_async_write(
                get_semaphore(11), core_l1(target, get_semaphore(11)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    noc_semaphore_wait(ready_epoch_go, 1);
    noc_semaphore_set(ready_epoch_go, 0);
    noc_semaphore_inc(core_l1(0, get_semaphore(10)), 1);
    noc_async_atomic_barrier();
    if (node == 0) {
        noc_semaphore_wait_min(
            ready_epoch_count, active_attention_readers);
        noc_semaphore_set(ready_epoch_count, 0);
        // Reset the remote counter only after every attention reader has
        // cleared its own dO credit word.
        noc_async_write(
            get_semaphore(10),
            get_noc_addr(1, physical_y(0), get_semaphore(13)),
            sizeof(uint32_t));
        noc_async_write_barrier();
        noc_semaphore_set(ready_epoch_go, 1);
        for (uint32_t target = 1;
             target < active_attention_readers; ++target) {
            noc_async_write(
                get_semaphore(11), core_l1(target, get_semaphore(11)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    noc_semaphore_wait(ready_epoch_go, 1);
    noc_semaphore_set(ready_epoch_go, 0);
    if (node == 0) {
        volatile tt_l1_ptr uint32_t* published_address =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(15));
        noc_semaphore_set(published_address, dout_row.get_write_ptr());
        noc_async_write(
            get_semaphore(15),
            get_noc_addr(1, physical_y(0), get_semaphore(11)),
            sizeof(uint32_t));
        noc_async_write_barrier();
        noc_semaphore_set(published_address, dout_full.get_write_ptr());
        noc_async_write(
            get_semaphore(15),
            get_noc_addr(1, physical_y(0), get_semaphore(12)),
            sizeof(uint32_t));
        noc_async_write_barrier();
    }
    noc_semaphore_inc(get_noc_addr(1, physical_y(0), get_semaphore(13)), 1);
    noc_async_atomic_barrier();
    WAYPOINT("OR1");
    q_full.reserve_back(vector_tiles);
    for (uint32_t tile = 0; tile < vector_tiles; ++tile) {
        noc0.async_read(
            checkpoint_arena_tensor,
            CoreLocalMem<uint32_t>(q_full.get_write_ptr() + tile * kTileBytes),
            kTileBytes, {.page_id = checkpoint_query_page_base +
                head * vector_tiles + tile}, {});
    }
    k.reserve_back(vector_tiles);
    v.reserve_back(vector_tiles);
    for (uint32_t tile = 0; tile < vector_tiles; ++tile) {
        noc0.async_read(
            checkpoint_arena_tensor,
            CoreLocalMem<uint32_t>(k.get_write_ptr() + tile * kTileBytes),
            kTileBytes, {.page_id = checkpoint_key_page_base +
                kv_head * vector_tiles + tile}, {});
    }
    for (uint32_t dim = 0; dim < dim_tiles; ++dim) {
        for (uint32_t seq = 0; seq < seq_tiles; ++seq) {
            const uint32_t target = dim * seq_tiles + seq;
            const uint32_t source = seq * dim_tiles + dim;
            noc0.async_read(
                checkpoint_arena_tensor,
                CoreLocalMem<uint32_t>(v.get_write_ptr() + target * kTileBytes),
                kTileBytes, {.page_id = checkpoint_value_page_base +
                    kv_head * vector_tiles + source}, {});
        }
    }
    noc0.async_read_barrier();
    WAYPOINT("OR2");
    noc_semaphore_wait_min(dout_credit, vector_tiles);
    noc_semaphore_set(dout_credit, 0);
    p_row.push_back(seq_tiles);
    p_column.push_back(seq_tiles);
    dout_row.push_back(dim_tiles);
    dout_full.push_back(vector_tiles);
    q_full.push_back(vector_tiles);
    k.push_back(vector_tiles);
    v.push_back(vector_tiles);
    WAYPOINT("AR0");

    if constexpr (
        diagnostic_ds || diagnostic_dv || diagnostic_pt || diagnostic_dq || diagnostic_dp || diagnostic_product || diagnostic_centered ||
        diagnostic_rowsum || diagnostic_unscaled_ds) return;

    // Every query-row owner pushes its resident dS row into the correct slot
    // of all four column owners.  This is the same payload-before-credit
    // protocol used by the proven resident MLP path: the data reader owns both
    // the NoC transfer and its completion semaphore, while dS itself crosses
    // from compute through the local CB without touching DRAM or PJRT.
    ds_column.reserve_back(seq_tiles);
    local_ds_handoff.wait_front(seq_tiles);
    {
        DeviceZoneScopedN("attention_ds_noc_transpose");
        for (uint32_t target_row = 0; target_row < seq_tiles; ++target_row) {
            const uint32_t target_node = head * seq_tiles + target_row;
            noc_async_write(
                local_ds_handoff.get_read_ptr() + target_row * kTileBytes,
                core_l1(
                    target_node,
                    ds_column.get_write_ptr() + row * kTileBytes),
                kTileBytes);
        }
        noc_async_write_barrier();
        for (uint32_t target_row = 0; target_row < seq_tiles; ++target_row) {
            const uint32_t target_node = head * seq_tiles + target_row;
            noc_semaphore_inc(core_l1(target_node, get_semaphore(0)), 1);
        }
        noc_async_atomic_barrier();
    }
    volatile tt_l1_ptr uint32_t* ds_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    {
        DeviceZoneScopedN("attention_ds_device_barrier");
        noc_semaphore_wait_min(ds_ready, seq_tiles);
    }
    noc_semaphore_set(ds_ready, 0);
    local_ds_handoff.pop_front(seq_tiles);
    ds_column.push_back(seq_tiles);
    WAYPOINT("AR1");

    if constexpr (diagnostic_dk) return;

    if (member == 0) {
        constexpr uint32_t peer_tiles = (group - 1) * dim_tiles;
        peer_dk.reserve_back(peer_tiles);
        peer_dv.reserve_back(peer_tiles);
        volatile tt_l1_ptr uint32_t* dk_mailbox =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(8));
        volatile tt_l1_ptr uint32_t* dv_mailbox =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(9));
        noc_semaphore_set(dk_mailbox, peer_dk.get_write_ptr());
        noc_semaphore_set(dv_mailbox, peer_dv.get_write_ptr());
        for (uint32_t source_member = 1; source_member < group; ++source_member) {
            const uint32_t source_node =
                (kv_head * group + source_member) * seq_tiles + row;
            noc_async_write(
                get_semaphore(8), core_l1(source_node, get_semaphore(8)),
                sizeof(uint32_t));
            noc_async_write(
                get_semaphore(9), core_l1(source_node, get_semaphore(9)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
        WAYPOINT("AR2");
        volatile tt_l1_ptr uint32_t* peers_ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(5));
        {
            DeviceZoneScopedN("attention_gqa_peer_barrier");
            noc_semaphore_wait_min(peers_ready, group - 1);
        }
        noc_semaphore_set(peers_ready, 0);
        WAYPOINT("AR3");
        peer_dk.push_back(peer_tiles);
        peer_dv.push_back(peer_tiles);

        if constexpr (kBf16) {
            // Pack complete member tiles locally. Compute explicitly clears
            // its destination register before the ordered add chain, so the
            // owner core can reduce without a second ownership domain.
            dk.wait_front(dim_tiles);
            dv.wait_front(dim_tiles);
            WAYPOINT("AR4");
            constexpr uint32_t gathered_tiles = 2 * dim_tiles * group;
            gqa_rows.reserve_back(gathered_tiles);
            uint32_t linear = 0;
            DeviceZoneScopedN("attention_gqa_l1_gather");
            for (uint32_t kind = 0; kind < 2; ++kind) {
                const uint32_t local_base =
                    kind == 0 ? dk.get_read_ptr() : dv.get_read_ptr();
                const uint32_t peer_base =
                    kind == 0 ? peer_dk.get_read_ptr() : peer_dv.get_read_ptr();
                for (uint32_t tile = 0; tile < dim_tiles; ++tile) {
                    for (uint32_t source_member = 0;
                         source_member < group; ++source_member) {
                        const uint32_t source = source_member == 0
                            ? local_base + tile * kTileBytes
                            : peer_base +
                                  ((source_member - 1) * dim_tiles + tile) *
                                      kTileBytes;
                        noc_async_write(
                            source,
                            get_noc_addr(
                                gqa_rows.get_write_ptr() +
                                linear * kTileBytes),
                            kTileBytes);
                        ++linear;
                    }
                }
            }
            noc_async_write_barrier();
            gqa_rows.push_back(gathered_tiles);
            WAYPOINT("AR5");
            dk.pop_front(dim_tiles);
            dv.pop_front(dim_tiles);
            peer_dk.pop_front((group - 1) * dim_tiles);
            peer_dv.pop_front((group - 1) * dim_tiles);
            return;
        }

        // Recreate TTNN's logical transpose(group, H) without a tensor or a
        // DRAM boundary.  For each logical sequence row, gather that row from
        // every GQA member into rows [0, group) of one local FP32 tile.  The
        // compute kernel consumes this stream with the same hardware H reduce
        // used by ttnn.sum.  Tiled FP32 is four 16x16 faces, each face row is
        // 64 bytes.
        constexpr uint32_t face_bytes = 16 * 16 * 2;
        constexpr uint32_t face_row_bytes = 16 * 2;
        auto row_half_offset = [](uint32_t logical_row, uint32_t right) {
            const uint32_t face_row = logical_row & 15;
            const uint32_t face_band = logical_row >> 4;
            return (face_band * 2 + right) * face_bytes +
                   face_row * face_row_bytes;
        };
        dk.wait_front(dim_tiles);
        dv.wait_front(dim_tiles);
        for (uint32_t kind = 0; kind < 2; ++kind) {
            const uint32_t local_base =
                kind == 0 ? dk.get_read_ptr() : dv.get_read_ptr();
            const uint32_t peer_base =
                kind == 0 ? peer_dk.get_read_ptr() : peer_dv.get_read_ptr();
            for (uint32_t tile = 0; tile < dim_tiles; ++tile) {
                for (uint32_t logical_row = 0; logical_row < 32; ++logical_row) {
                    gqa_rows.reserve_back(1);
                    noc0.async_write_zeros(gqa_rows, kTileBytes);
                    noc0.write_zeros_l1_barrier();
                    const uint32_t destination = gqa_rows.get_write_ptr();
                    for (uint32_t source_member = 0; source_member < group; ++source_member) {
                        const uint32_t source = source_member == 0
                            ? local_base + tile * kTileBytes
                            : peer_base +
                                  ((source_member - 1) * dim_tiles + tile) * kTileBytes;
                        for (uint32_t right = 0; right < 2; ++right) {
                            noc_async_write(
                                source + row_half_offset(logical_row, right),
                                get_noc_addr(
                                    destination + row_half_offset(source_member, right)),
                                face_row_bytes);
                        }
                    }
                    noc_async_write_barrier();
                    gqa_rows.push_back(1);
                }
            }
        }
    }
}
}  // namespace resident_layer_phase_attention

void kernel_main() {
    resident_layer_phase_attention::run();
}
