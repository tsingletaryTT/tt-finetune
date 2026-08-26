#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/debug/waypoint.h"
#include "api/tensor/noc_traits.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace resident_layer_phase_attention_5 {
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
    constexpr uint32_t cb_dv = get_compile_time_arg_val(34);
    constexpr uint32_t cb_dq = get_compile_time_arg_val(42);
    constexpr uint32_t cb_ds_handoff = get_compile_time_arg_val(44);
    constexpr uint32_t cb_ds_column = get_compile_time_arg_val(45);
    constexpr uint32_t cb_dk = get_compile_time_arg_val(47);
    constexpr uint32_t cb_peer_dk = get_compile_time_arg_val(48);
    constexpr uint32_t cb_peer_dv = get_compile_time_arg_val(49);
    constexpr uint32_t cb_gqa_reduced = get_compile_time_arg_val(51);
    constexpr uint32_t cb_gqa_bf16 = get_compile_time_arg_val(53);
    constexpr uint32_t cb_final_dk = get_compile_time_arg_val(54);
    constexpr uint32_t cb_final_dv = get_compile_time_arg_val(55);
    constexpr uint32_t cb_rope_q = get_compile_time_arg_val(61);
    constexpr uint32_t cb_rope_k = get_compile_time_arg_val(62);
    constexpr uint32_t seq_tiles = 4;
    constexpr uint32_t dim_tiles = 2;
    constexpr uint32_t query_heads = 14;
    constexpr uint32_t kv_heads = 2;
    constexpr uint32_t group = query_heads / kv_heads;
    constexpr bool kBf16 = 1;
    constexpr uint32_t vector_tiles = seq_tiles * dim_tiles;
    constexpr uint32_t output_row_offset = 33;
    constexpr uint32_t output_arena_col_tiles = 28;
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
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const auto output = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<41, 95>(), 0>(),
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
}  // namespace resident_layer_phase_attention_5

void kernel_main() {
    resident_layer_phase_attention_5::run();
}
