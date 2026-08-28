// Direct exact Q/K/V output publication into resident attention, followed by
// final O output/Ho publication. No projection or head intermediate tensor is
// materialized between the two islands.

#include <cstdint>
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t kTileBytes = 2048, kProjectionColumns = 5;
constexpr uint32_t kProjectionWorkers = @PROJECTION_WORKERS@, kAttentionColumns = 6;
constexpr uint32_t kAttentionXOffset = 5;
constexpr bool kInputRMSNorm = @INPUT_RMSNORM@;
constexpr bool kPrefixOnly = @PROJECTION_ROPE_PREFIX_ONLY@;
constexpr bool kDedicatedORank = @DEDICATED_O_RANK@;
constexpr bool kPostAttentionBlock = @POST_ATTENTION_BLOCK@;
constexpr bool kFullMLPForward = @FULL_MLP_FORWARD@;
constexpr uint32_t kPostDiagnosticStage = @POST_DIAGNOSTIC_STAGE@;
constexpr uint32_t kNormalizedHidden = 22;
constexpr uint32_t kNormReadyCount = 8, kNormReady = 9;
constexpr uint8_t kPhysicalX[11] = {1,2,3,4,5,6,7,10,11,12,13};
FORCE_INLINE uint8_t physical_y(uint32_t y) { return static_cast<uint8_t>(y + 2); }
FORCE_INLINE uint64_t projection_l1(uint32_t node, uint32_t address) {
    return get_noc_addr(kPhysicalX[node % kProjectionColumns],
                        physical_y(node / kProjectionColumns), address);
}
FORCE_INLINE uint64_t attention_l1(uint32_t node, uint32_t address) {
    const uint32_t x = kAttentionXOffset + node % kAttentionColumns;
    return get_noc_addr(kPhysicalX[x], physical_y(node / kAttentionColumns), address);
}

static __attribute__((noinline, noclone)) void
rebase_empty_consumer_cb(uint32_t cb_id) {
    volatile uint32_t* received = get_cb_tiles_received_ptr(cb_id);
    const uintptr_t ack_address =
        reinterpret_cast<uintptr_t>(get_cb_tiles_acked_ptr(cb_id));
    const uint16_t pushed = static_cast<uint16_t>(received[0]);
    if (cb_id == 22) {
        WAYPOINT("R220");
        // CB22's blocked RMSNorm producer includes physical K padding that
        // has no logical consumer. The producer is complete at this phase
        // boundary, so acknowledge the dead tail explicitly before resetting
        // the epoch instead of waiting for a logical pop that will never run.
        *get_cb_tiles_acked_ptr(cb_id) = pushed;
    }
    while (true) {
        invalidate_l1_cache();
        if (static_cast<uint16_t>(reg_read(ack_address)) == pushed) break;
    }
    if (cb_id == 22) WAYPOINT("R22D");
    *get_cb_tiles_received_ptr(cb_id) = 0;
    *get_cb_tiles_acked_ptr(cb_id) = 0;
    auto& interface = get_local_cb_interface(cb_id);
    const uint32_t base = interface.fifo_limit - interface.fifo_size;
    interface.fifo_rd_ptr = base;
    interface.fifo_wr_ptr = base;
    interface.tiles_acked_received_init = 0;
    interface.fifo_wr_tile_ptr = 0;
}

// Temporary lifecycle probes: identical to cb_wait_front(), but retain a
// phase-specific waypoint while blocked so Watcher identifies the missing
// producer instead of reporting the generic CWFW marker.
FORCE_INLINE void wait_post_output(uint32_t cb_id) {
    const uint32_t acked = get_cb_tiles_acked_ptr(cb_id)[0];
    const uintptr_t received =
        reinterpret_cast<uintptr_t>(get_cb_tiles_received_ptr(cb_id));
    WAYPOINT("POW0");
    while (static_cast<uint16_t>(reg_read(received)) - acked < 1) {}
    WAYPOINT("POW1");
}
FORCE_INLINE void wait_post_rstd(uint32_t cb_id) {
    const uint32_t acked = get_cb_tiles_acked_ptr(cb_id)[0];
    const uintptr_t received =
        reinterpret_cast<uintptr_t>(get_cb_tiles_received_ptr(cb_id));
    WAYPOINT("PRW0");
    while (static_cast<uint16_t>(reg_read(received)) - acked < 1) {}
    WAYPOINT("PRW1");
}
FORCE_INLINE void wait_post_normalized(uint32_t cb_id) {
    const uint32_t acked = get_cb_tiles_acked_ptr(cb_id)[0];
    const uintptr_t received =
        reinterpret_cast<uintptr_t>(get_cb_tiles_received_ptr(cb_id));
    WAYPOINT("PNW0");
    while (static_cast<uint16_t>(reg_read(received)) - acked < 1) {}
    WAYPOINT("PNW1");
}
FORCE_INLINE void wait_qkv_output(uint32_t cb_id) {
    const uint32_t acked = get_cb_tiles_acked_ptr(cb_id)[0];
    const uintptr_t received =
        reinterpret_cast<uintptr_t>(get_cb_tiles_received_ptr(cb_id));
    WAYPOINT("QWO0");
    while (static_cast<uint16_t>(reg_read(received)) - acked < 1) {}
    WAYPOINT("QWO1");
}
FORCE_INLINE void wait_qkv_h(uint32_t cb_id, uint32_t count) {
    const uint32_t acked = get_cb_tiles_acked_ptr(cb_id)[0];
    const uintptr_t received =
        reinterpret_cast<uintptr_t>(get_cb_tiles_received_ptr(cb_id));
    WAYPOINT("QWH0");
    while (static_cast<uint16_t>(reg_read(received)) - acked < count) {}
    WAYPOINT("QWH1");
}
FORCE_INLINE void wait_qkv_ack(uint32_t cb_id) {
    const uint32_t acked = get_cb_tiles_acked_ptr(cb_id)[0];
    const uintptr_t received =
        reinterpret_cast<uintptr_t>(get_cb_tiles_received_ptr(cb_id));
    WAYPOINT("QWA0");
    while (static_cast<uint16_t>(reg_read(received)) - acked < 1) {}
    WAYPOINT("QWA1");
}
FORCE_INLINE void wait_o_output(uint32_t cb_id) {
    const uint32_t acked = get_cb_tiles_acked_ptr(cb_id)[0];
    const uintptr_t received =
        reinterpret_cast<uintptr_t>(get_cb_tiles_received_ptr(cb_id));
    WAYPOINT("OWO0");
    while (static_cast<uint16_t>(reg_read(received)) - acked < 1) {}
    WAYPOINT("OWO1");
}
FORCE_INLINE void wait_o_h(uint32_t cb_id, uint32_t count) {
    const uint32_t acked = get_cb_tiles_acked_ptr(cb_id)[0];
    const uintptr_t received =
        reinterpret_cast<uintptr_t>(get_cb_tiles_received_ptr(cb_id));
    WAYPOINT("OWH0");
    while (static_cast<uint16_t>(reg_read(received)) - acked < count) {}
    WAYPOINT("OWH1");
}
}

__attribute__((optimize("Os"))) void kernel_main() {
    constexpr uint32_t cb_saved_h = 4, cb_output = 12;
    constexpr uint32_t cb_o_saved_h = @O_SAVED_H_CB@;
    constexpr uint32_t cb_o_output = @O_OUTPUT_CB@;
    constexpr uint32_t seq_tiles = @SEQ_TILES@, dim_tiles = @DIM_TILES@;
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@, kv_tiles = @KV_TILES@;
    constexpr uint32_t qkv_rank_tiles = @QKV_RANK_TILES@;
    constexpr bool grouped_qkv_lora = @GROUPED_QKV_LORA@;
    constexpr bool smol_grouped_qkv_lora = @SMOL_GROUPED_QKV_LORA@;
    constexpr bool precomputed_qkv_h = @PRECOMPUTED_QKV_H@;
    constexpr uint32_t query_heads = @QUERY_HEADS@, kv_heads = @KV_HEADS@;
    constexpr uint32_t group = query_heads / kv_heads;
    constexpr uint32_t active_attention = query_heads * seq_tiles;
    constexpr uint32_t q_ready_sem = 0, k_ready_sem = 1, v_ready_sem = 2;
    constexpr uint32_t consumer_ready_sem = 3, producer_go_sem = 4;
    constexpr uint32_t invocation_writer_count_sem = 14;
    constexpr uint32_t invocation_done_sem = 15;
    const uint32_t x = get_absolute_logical_x(), y = get_absolute_logical_y();
    if (x >= kProjectionColumns) return;
    const uint32_t node = y * kProjectionColumns + x;
    const auto arena = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<@ARENA_TENSOR_INDEX@, @TOTAL_CBS@>(), 0>(),
        get_common_arg_val<uint32_t>(0), kTileBytes);
    Noc noc1(1);
    if (node == 0) {
        // Reader cores increment this only after both O input windows are
        // reserved.  Reset here, before the much longer Q/K/V epochs, so a
        // previous program invocation cannot satisfy the next readiness wait.
        volatile tt_l1_ptr uint32_t* o_ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(13));
        noc_semaphore_set(o_ready, 0);
    }
    if constexpr (kInputRMSNorm) {
        // RMSNorm compute publishes CB22 only on the sequence row owners.
        // Consume that CB through the supported compute -> writer edge, then
        // expose its stable L1 read address to every Q/K/V reader.  The CB is
        // deliberately retained until all three projection epochs finish.
        if (node < seq_tiles) {
            CircularBuffer normalized(kNormalizedHidden);
            // The input and gamma FIFOs are block-padded, but the accepted
            // RMSNorm suffix publishes only the logical hidden tiles.
            normalized.wait_front(hidden_tiles);
            if (node == 0) {
                noc1.async_write(
                    CoreLocalMem<uint32_t>(normalized.get_read_ptr()), arena,
                    kTileBytes, {}, {.page_id = @LAYER_PROBE_PAGE@});
                noc1.async_write_barrier();
            }
            noc_semaphore_inc<true>(
                projection_l1(0, get_semaphore(kNormReadyCount)), 1);
            // Node zero waits for all row arrivals before publishing the
            // norm-ready generation. That generation is the completion
            // proof; the individual sender needs no return acknowledgement.
        }
        if (node == 0) {
            volatile tt_l1_ptr uint32_t* count =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(kNormReadyCount));
            noc_semaphore_wait_min(count, seq_tiles);
            noc_semaphore_set(count, 0);
            volatile tt_l1_ptr uint32_t* ready =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(kNormReady));
            noc_semaphore_set(ready, 2);
            for (uint32_t target = 1; target < kProjectionWorkers; ++target)
                noc_async_write(
                    get_semaphore(kNormReady),
                    projection_l1(target, get_semaphore(kNormReady)),
                    sizeof(uint32_t));
            noc_async_write_barrier();
        }
    }
    CircularBuffer output(cb_output), saved_h(cb_saved_h);
    volatile tt_l1_ptr uint32_t* go =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(producer_go_sem));
    if (node == 0) {
        volatile tt_l1_ptr uint32_t* ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(consumer_ready_sem));
        noc_semaphore_wait_min(ready, active_attention);
        noc_semaphore_set(ready, 0);
        for (uint32_t target = 1; target < kProjectionWorkers; ++target) {
            noc_async_write(get_semaphore(5),
                            projection_l1(target, get_semaphore(5)), sizeof(uint32_t));
            noc_async_write(get_semaphore(6),
                            projection_l1(target, get_semaphore(6)), sizeof(uint32_t));
            noc_async_write(get_semaphore(7),
                            projection_l1(target, get_semaphore(7)), sizeof(uint32_t));
        }
        noc_semaphore_set(go, 1);
        for (uint32_t target = 1; target < kProjectionWorkers; ++target)
            noc_async_write(get_semaphore(producer_go_sem),
                            projection_l1(target, get_semaphore(producer_go_sem)),
                            sizeof(uint32_t));
        noc_async_write_barrier();
    }
    noc_semaphore_wait(go, 1); noc_semaphore_set(go, 0);
    const uint32_t q_target_base =
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(5));
    const uint32_t k_target_base =
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(6));
    const uint32_t v_target_base =
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(7));

    // The projection writer receives post-norm gamma first and the owner
    // arena second.  The safe path ignores gamma, but its ABI is deliberately
    // identical to the larger attention-block owner so cache entries cannot
    // reinterpret gamma as the arena after an experimental invocation.
    for (uint32_t phase = 0; phase < 3; ++phase) {
        const uint32_t output_tiles = phase == 0 ? hidden_tiles : kv_tiles;
        const uint32_t total_jobs = seq_tiles * output_tiles;
        const uint32_t jobs_per_node =
            (total_jobs + kProjectionWorkers - 1) / kProjectionWorkers;
        for (uint32_t local = 0; local < jobs_per_node; ++local) {
            const uint32_t job = node * jobs_per_node + local;
            if (job >= total_jobs) continue;
            const uint32_t token = job / output_tiles;
            const uint32_t output_index = job - token * output_tiles;
            const uint32_t head = output_index / dim_tiles;
            const uint32_t dim = output_index - head * dim_tiles;
            wait_qkv_output(cb_output);
            if (phase == 0) {
                const uint32_t target = head * seq_tiles + token;
                noc_async_write(output.get_read_ptr(),
                    attention_l1(target, q_target_base + dim * kTileBytes), kTileBytes);
                noc_async_write_barrier();
                noc_semaphore_inc<true>(
                    attention_l1(target, get_semaphore(q_ready_sem)), 1);
                // The attention reader waits on q_ready after the data-write
                // barrier above; no sender-side acknowledgement is required.
            } else {
                const uint32_t target_base = phase == 1 ? k_target_base : v_target_base;
                const uint32_t ready_sem = phase == 1 ? k_ready_sem : v_ready_sem;
                const uint32_t slot = token * dim_tiles + dim;
                const uint32_t first_head = head * group;
                for (uint32_t member = 0; member < group; ++member)
                    for (uint32_t row = 0; row < seq_tiles; ++row) {
                        const uint32_t target = (first_head + member) * seq_tiles + row;
                        noc_async_write(output.get_read_ptr(),
                            attention_l1(target, target_base + slot * kTileBytes), kTileBytes);
                    }
                if (phase == 2)
                    noc1.async_write(CoreLocalMem<uint32_t>(output.get_read_ptr()),
                                     arena, kTileBytes, {},
                                     {.page_id = @V_BASE@ +
                                         head * seq_tiles * dim_tiles +
                                         token * dim_tiles + dim});
                noc_async_write_barrier();
                for (uint32_t member = 0; member < group; ++member)
                    for (uint32_t row = 0; row < seq_tiles; ++row) {
                        const uint32_t target = (first_head + member) * seq_tiles + row;
                        noc_semaphore_inc<true>(
                            attention_l1(target, get_semaphore(ready_sem)), 1);
                    }
                // Each replicated consumer waits for its complete K/V credit
                // count after the data-write barrier above.
            }
            output.pop_front(1);
            if (output_index == 0) {
                wait_qkv_h(cb_saved_h, qkv_rank_tiles);
                const uint32_t h_base =
                    phase == 0 ? @HQ_BASE@ : (phase == 1 ? @HK_BASE@ : @HV_BASE@);
                if constexpr (smol_grouped_qkv_lora) {
                    // SmolLM2's untouched graph computes one logical
                    // [Q|V|K] rank-24 matmul.  All three rank-8 members share
                    // one physical tile. Repack the selected member in place
                    // only after delta compute has consumed H, then emit one
                    // aligned full-tile write. Sub-row NoC writes are not safe
                    // here: Blackhole's minimum aligned transaction can leave
                    // alternating face rows stale.
                    const uint32_t member = phase == 0 ? 0 : (phase == 2 ? 1 : 2);
                    const uint32_t logical_column = member * 8;
                    volatile tt_l1_ptr uint16_t* tile =
                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                            saved_h.get_read_ptr());
                    for (uint32_t row_index = 0; row_index < 32; ++row_index) {
                        const uint32_t row_face = row_index / 16;
                        const uint32_t face_row = row_index % 16;
                        for (uint32_t column = 0; column < 8; ++column) {
                            const uint32_t source_column = logical_column + column;
                            const uint32_t source_face = source_column / 16;
                            const uint32_t source_offset =
                                (row_face * 2 + source_face) * 16 * 16
                                + face_row * 16 + source_column % 16;
                            const uint32_t target_offset =
                                (row_face * 2) * 16 * 16
                                + face_row * 16 + column;
                            tile[target_offset] = tile[source_offset];
                        }
                        for (uint32_t column = 8; column < 32; ++column) {
                            const uint32_t column_face = column / 16;
                            const uint32_t target_offset =
                                (row_face * 2 + column_face) * 16 * 16
                                + face_row * 16 + column % 16;
                            tile[target_offset] = 0;
                        }
                    }
                    const uint32_t page = h_base + token;
                    noc1.async_write(
                        CoreLocalMem<uint32_t>(saved_h.get_read_ptr()), arena,
                        kTileBytes, {}, {.page_id = page});
                } else if constexpr (grouped_qkv_lora) {
                    // The reference Qwen graph owns one logical [Q|K|V]
                    // rank-48 matmul. Its first physical tile contains Q in
                    // the left faces and K in the right faces; the second
                    // contains V in the left faces. Publish ordinary padded
                    // rank-16 checkpoints without ever materializing a
                    // framework tensor. The arena is zero-initialized, so
                    // its untouched right faces remain exact padding.
                    constexpr uint32_t face_bytes = 512;
                    const uint32_t source_tile =
                        precomputed_qkv_h ? 0 : (phase == 2 ? 1 : 0);
                    const uint32_t source_face = phase == 1 ? 1 : 0;
                    const uint32_t source = saved_h.get_read_ptr() +
                        source_tile * kTileBytes + source_face * face_bytes;
                    const uint32_t page = h_base + token;
                    noc1.async_write(
                        CoreLocalMem<uint32_t>(source), arena, face_bytes, {},
                        {.page_id = page, .offset_bytes = 0});
                    noc1.async_write(
                        CoreLocalMem<uint32_t>(source + 2 * face_bytes),
                        arena, face_bytes, {},
                        {.page_id = page, .offset_bytes = 2 * face_bytes});
                } else {
                    for (uint32_t rank_tile = 0;
                         rank_tile < qkv_rank_tiles; ++rank_tile) {
                        noc1.async_write(
                            CoreLocalMem<uint32_t>(
                                saved_h.get_read_ptr()
                                + rank_tile * kTileBytes),
                            arena, kTileBytes, {},
                            {.page_id = h_base
                                + token * qkv_rank_tiles + rank_tile});
                    }
                }
                noc1.async_write_barrier();
                saved_h.pop_front(qkv_rank_tiles);
            }
        }
    }
    if constexpr (kInputRMSNorm) {
        // Completion of all Q/K/V output jobs proves compute consumed every
        // reader tile, hence no reader can still reference the normalized
        // row. Release the retained CB without another semaphore epoch.
        if (node < seq_tiles) {
            CircularBuffer normalized(kNormalizedHidden);
            normalized.pop_front(hidden_tiles);
            rebase_empty_consumer_cb(kNormalizedHidden);
        }

        if constexpr (!kPrefixOnly) {
        // The Q/K/V output FIFOs are also O output FIFOs in the compact
        // layer ABI.  Their consumer pointers can end on a partial physical
        // cycle when a tail core owns an odd number of jobs.  Rebase only
        // after this writer has drained every Q/K/V output, then rendezvous
        // with compute before either side begins the O epoch.
        rebase_empty_consumer_cb(cb_saved_h);
        rebase_empty_consumer_cb(cb_output);
        CircularBuffer qkv_outputs_drained(28), qkv_outputs_reuse_ack(29);
        qkv_outputs_drained.reserve_back(1);
        qkv_outputs_drained.push_back(1);
        wait_qkv_ack(29);
        qkv_outputs_reuse_ack.pop_front(1);
        }
        if constexpr (kPrefixOnly) {
            // Prefix mode has no O-projection epoch and therefore no compute
            // reuse acknowledgement.  The writer has nevertheless consumed
            // every Q/K/V tile at this point, so restore the aliased output
            // descriptors explicitly before the next layer invocation.
            rebase_empty_consumer_cb(cb_saved_h);
            rebase_empty_consumer_cb(cb_output);
        }
    }

    if constexpr (!kPrefixOnly) {
    uint32_t post_output_base = 0;
    if constexpr (kPostAttentionBlock) {
        auto& post_interface = get_local_cb_interface(13);
        post_output_base = post_interface.fifo_limit - post_interface.fifo_size;
    }

    CircularBuffer o_output(cb_o_output), o_saved_h(cb_o_saved_h);
    constexpr uint32_t o_jobs = seq_tiles * hidden_tiles;
    constexpr uint32_t o_jobs_per_node =
        (o_jobs + @O_WORKERS@ - 1) / @O_WORKERS@;
    for (uint32_t local = 0; local < o_jobs_per_node; ++local) {
        const uint32_t job = node * o_jobs_per_node + local;
        if (job >= o_jobs) continue;
        const uint32_t token = job / hidden_tiles;
        const uint32_t output_index = job - token * hidden_tiles;
        wait_o_output(cb_o_output);
        if constexpr (kPostAttentionBlock && kPostDiagnosticStage < 5) {
            // Route the exact O tile directly to its token-row owner.  The
            // owner reserved the complete padded CB13 row before publishing
            // this common L1 base, so only logical tiles need NoC traffic.
            noc_async_write(
                o_output.get_read_ptr(),
                projection_l1(
                    token, post_output_base + output_index * kTileBytes),
                kTileBytes);
            noc_async_write_barrier();
            noc_semaphore_inc<true>(
                projection_l1(token, get_semaphore(7)), 1);
            // The O projection reader's credit wait is the completion proof.
        }
        noc1.async_write(CoreLocalMem<uint32_t>(o_output.get_read_ptr()),
                         arena, kTileBytes, {},
                         {.page_id = @OUTPUT_BASE@ + job});
        noc1.async_write_barrier();
        o_output.pop_front(1);
        if constexpr (!kDedicatedORank) if (output_index == 0) {
            wait_o_h(cb_o_saved_h, qkv_rank_tiles);
            for (uint32_t rank_tile = 0;
                 rank_tile < qkv_rank_tiles; ++rank_tile) {
                noc1.async_write(
                    CoreLocalMem<uint32_t>(
                        o_saved_h.get_read_ptr()
                        + rank_tile * kTileBytes),
                    arena, kTileBytes, {},
                    {.page_id = @HO_BASE@
                        + token * qkv_rank_tiles + rank_tile});
            }
            noc1.async_write_barrier();
            o_saved_h.pop_front(qkv_rank_tiles);
        }
    }

    if constexpr (kPostAttentionBlock) {
        constexpr uint32_t token_tiles = @POST_SEQ_TILES@;
        if constexpr (kPostDiagnosticStage < 4 ||
                      kPostDiagnosticStage == 9 ||
                      kPostDiagnosticStage == 10 ||
                      kPostDiagnosticStage == 11 ||
                      kPostDiagnosticStage == 12) if (node < token_tiles) {
            // CB12 is a continuous compute->writer stream: O outputs first,
            // then the exact attention residual, then post-RMSNorm output.
            // The divisor-based O scheduler makes its first epoch end on the
            // physical CB boundary for every active row owner.
            CircularBuffer post_output(cb_o_output);
            CircularBuffer post_normalized(kNormalizedHidden);
            for (uint32_t tile = 0; tile < hidden_tiles; ++tile) {
                wait_post_output(cb_o_output);
                noc1.async_write(
                    CoreLocalMem<uint32_t>(post_output.get_read_ptr()),
                    arena, kTileBytes, {},
                    {.page_id = @POST_RESIDUAL_BASE@ +
                        node * hidden_tiles + tile});
                if constexpr (kFullMLPForward) {
                    // Preserve the residual source in the now-drained O
                    // input window.  The later MLP output owners fetch it
                    // directly over NoC; the arena write remains only as the
                    // backward-lifetime checkpoint.
                    noc_async_write(
                        post_output.get_read_ptr(),
                        projection_l1(
                            node, post_output_base + tile * kTileBytes),
                        kTileBytes);
                }
                noc1.async_write_barrier();
                if constexpr (kFullMLPForward)
                    noc_async_write_barrier();
                post_output.pop_front(1);
            }
            if constexpr (kFullMLPForward) if (node == 0) {
                // Every projection core has the same CB13 base.  Publish the
                // address once; the enclosing writer-completion barrier
                // proves that all four row caches are populated before the
                // MLP reader consumes the mailbox.
                volatile tt_l1_ptr uint32_t* residual_base =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        get_semaphore(8));
                noc_semaphore_set(residual_base, post_output_base);
                for (uint32_t target = 0; target < 60; ++target) {
                    noc_async_write(
                        get_semaphore(8),
                        attention_l1(target, get_semaphore(8)),
                        sizeof(uint32_t));
                    if (target % kAttentionColumns ==
                        kAttentionColumns - 1)
                        noc_async_write_barrier();
                }
                noc_async_write_barrier();
            }
            // Gamma may overwrite the now-consumed O input CB only after the
            // residual checkpoint is visible.  This semaphore is local to
            // the row owner and never crosses PJRT.
            volatile tt_l1_ptr uint32_t* residual_written =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(12));
            noc_semaphore_set(residual_written, 1);

            if constexpr (kPostDiagnosticStage == 0 ||
                          kPostDiagnosticStage == 3) {
                // Preserve the exact raw FP32 rstd bits for layer backward
                // only on the complete production path. Diagnostic stage 3
                // isolates the norm suffix without this handshake.
                if constexpr (kPostDiagnosticStage == 0) {
                    CircularBuffer rstd_ready(29), rstd_copied(28), rstd(27);
                    wait_post_rstd(29);
                    rstd_ready.pop_front(1);
                    auto& rstd_interface = get_local_cb_interface(27);
                    const uint32_t rstd_base =
                        rstd_interface.fifo_limit - rstd_interface.fifo_size;
                    noc1.async_write(
                        CoreLocalMem<uint32_t>(rstd_base),
                        arena, kTileBytes, {},
                        {.page_id = @POST_RSTD_BASE@ + node * 2});
                    noc1.async_write(
                        CoreLocalMem<uint32_t>(rstd_base + kTileBytes),
                        arena, kTileBytes, {},
                        {.page_id = @POST_RSTD_BASE@ + node * 2 + 1});
                    noc1.async_write_barrier();
                    rstd_copied.reserve_back(1);
                    rstd_copied.push_back(1);
                }

                for (uint32_t tile = 0; tile < hidden_tiles; ++tile) {
                    wait_post_normalized(kNormalizedHidden);
                    noc1.async_write(
                        CoreLocalMem<uint32_t>(post_normalized.get_read_ptr()),
                        arena, kTileBytes, {},
                        {.page_id = @POST_NORMALIZED_BASE@ +
                            node * hidden_tiles + tile});
                    noc1.async_write_barrier();
                    post_normalized.pop_front(1);
                }
                // RMSNorm's accepted blocked kernel pads the hidden K extent
                // to its compute block (SmolLM2: 18 logical tiles, 20
                // produced).  These padding tiles are not routed or saved,
                // but they still belong to this CB epoch and must be
                // acknowledged before the descriptor can be rebased.
                rebase_empty_consumer_cb(kNormalizedHidden);
            }
        }
    }
    }

    // One completion epoch covers every projection and active attention
    // writer.  Only after the last final write is visible may a subsequent
    // resident owner reuse the fixed per-core L1 control addresses.
    noc_semaphore_inc<true>(
        projection_l1(0, get_semaphore(invocation_writer_count_sem)), 1);
    // Every sender subsequently waits for invocation_done, which node zero
    // publishes only after observing all writer-count arrivals.
    volatile tt_l1_ptr uint32_t* invocation_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(invocation_done_sem));
    if (node == 0) {
        volatile tt_l1_ptr uint32_t* writer_count =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(invocation_writer_count_sem));
        noc_semaphore_wait_min(
            writer_count, kProjectionWorkers + active_attention);
        noc_semaphore_set(writer_count, 0);
        noc_semaphore_set(invocation_done, 1);
        for (uint32_t target = 1; target < kProjectionWorkers; ++target) {
            noc_async_write(
                get_semaphore(invocation_done_sem),
                projection_l1(target, get_semaphore(invocation_done_sem)),
                sizeof(uint32_t));
            if (target % kProjectionColumns == kProjectionColumns - 1)
                noc_async_write_barrier();
        }
        noc_async_write_barrier();
        for (uint32_t target = 0; target < active_attention; ++target) {
            noc_async_write(
                get_semaphore(invocation_done_sem),
                attention_l1(target, get_semaphore(invocation_done_sem)),
                sizeof(uint32_t));
            if (target % kAttentionColumns == kAttentionColumns - 1)
                noc_async_write_barrier();
        }
        noc_async_write_barrier();
    }
    noc_semaphore_wait(invocation_done, 1);
    if constexpr (kInputRMSNorm && !kPrefixOnly) {
        rebase_empty_consumer_cb(cb_saved_h);
        rebase_empty_consumer_cb(cb_output);
    }
    CircularBuffer invocation_release(28), invocation_acknowledgement(29);
    invocation_release.reserve_back(1);
    invocation_release.push_back(1);
    invocation_acknowledgement.wait_front(1);
    invocation_acknowledgement.pop_front(1);
    noc_semaphore_set(invocation_done, 2);
}
