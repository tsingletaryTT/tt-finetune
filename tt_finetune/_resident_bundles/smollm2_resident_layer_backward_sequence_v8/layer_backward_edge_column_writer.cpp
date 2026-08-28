#include <cstdint>
#include "api/compile_time_args.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "api/dataflow/noc.h"
#include "api/debug/waypoint.h"
#include "api/tensor/noc_traits.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace resident_layer_phase_mlp {
FORCE_INLINE uint8_t full_product_physical_y(uint32_t y) {
    return static_cast<uint8_t>(y + 2);
}

FORCE_INLINE void full_activation_product_writer_phase() {
    constexpr uint32_t cb_product = get_compile_time_arg_val(122);
    constexpr uint32_t product_jobs = 192;
    constexpr uint32_t jobs_per_node = 20;
    constexpr uint32_t i_tiles = 48;
    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t da_tiles = 1;
    constexpr uint32_t kTileBytes = 2048;
    constexpr uint8_t px[10] = {1,2,3,4,5,6,7,10,11,12};
    const uint32_t node = get_absolute_logical_y();
    CircularBuffer product(cb_product);

    volatile tt_l1_ptr uint32_t* destinations =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(13));
    if (node == 0) {
        noc_semaphore_wait_min(destinations, 100);
        noc_semaphore_set(destinations, 0);
        volatile tt_l1_ptr uint32_t* h_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(14));
        volatile tt_l1_ptr uint32_t* da_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(15));
        noc_semaphore_wait_min(h_addr, 1);
        noc_semaphore_wait_min(da_addr, 1);
        for (uint32_t target = 1; target < 10; ++target) {
            noc_async_write(
                get_semaphore(14),
                get_noc_addr(13, full_product_physical_y(target), get_semaphore(14)),
                sizeof(uint32_t));
            noc_async_write(
                get_semaphore(15),
                get_noc_addr(13, full_product_physical_y(target), get_semaphore(15)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    volatile tt_l1_ptr uint32_t* h_addr_flag =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(14));
    volatile tt_l1_ptr uint32_t* da_addr_flag =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(15));
    noc_semaphore_wait_min(h_addr_flag, 1);
    noc_semaphore_wait_min(da_addr_flag, 1);
    const uint32_t h_base = *h_addr_flag;
    const uint32_t da_base = *da_addr_flag;
    const uint32_t first = node * jobs_per_node;
    for (uint32_t local = 0; local < jobs_per_node; ++local) {
        const uint32_t job = first + local;
        if (job >= product_jobs) continue;
        const uint32_t m = job / i_tiles;
        const uint32_t k = job - m * i_tiles;
        product.wait_front(1);
        const uint32_t source = product.get_read_ptr();
        noc_async_write(
            source,
            get_noc_addr(px[m % 10], full_product_physical_y(m / 10),
                         h_base + k * kTileBytes),
            kTileBytes);
        const uint32_t owner = k / da_tiles;
        const uint32_t slot = (k % da_tiles) * m_tiles + m;
        noc_async_write(
            source,
            get_noc_addr(px[owner % 10], full_product_physical_y(owner / 10),
                         da_base + slot * kTileBytes),
            kTileBytes);
        noc_async_write_barrier();
        product.pop_front(1);
    }
    noc_semaphore_inc(
        get_noc_addr(13, full_product_physical_y(0), get_semaphore(13)), 1);
    noc_async_atomic_barrier();
    if (node == 0) {
        noc_semaphore_wait_min(destinations, 10);
        noc_semaphore_set(destinations, 0);
        volatile tt_l1_ptr uint32_t* ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(14));
        noc_semaphore_set(ready, 1);
        for (uint32_t target = 0; target < 100; ++target) {
            noc_async_write(
                get_semaphore(14),
                get_noc_addr(px[target % 10], full_product_physical_y(target / 10),
                             get_semaphore(14)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
}

// Keep exact dGate/dUp outputs banked in activation-core L1 and publish the
// bank addresses to the matmul island. The matmul readers consume the tiles
// directly over NoC and acknowledge completion before these CBs are released.



namespace {
constexpr uint32_t kMatmulCols = 10;
constexpr uint32_t kMatmulNodes = 100;
constexpr uint32_t kActivationNodes = 10;
constexpr uint8_t kMatmulPhysicalX[kMatmulCols] = {
    1, 2, 3, 4, 5, 6, 7, 10, 11, 12};
FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}  // namespace

FORCE_INLINE void full_activation_backward_writer_phase() {
    constexpr uint32_t cb_dgate0 = get_compile_time_arg_val(103);
    constexpr uint32_t cb_dup0 = get_compile_time_arg_val(104);
    constexpr uint32_t total_jobs = 192;
    constexpr uint32_t producer_jobs = 2;
    constexpr uint32_t bank_count = 1;
    constexpr uint32_t address_start = 0;
    constexpr uint32_t output_ready = 4;
    constexpr uint32_t consumer_done = 5;
    constexpr uint32_t release = 6;
    const uint32_t node = get_absolute_logical_y();
    uint32_t activation_jobs = 0;
    for (uint32_t producer = node; producer < kMatmulNodes;
         producer += kActivationNodes) {
        const uint32_t first = producer * producer_jobs;
        if (first >= total_jobs) continue;
        const uint32_t remaining = total_jobs - first;
        activation_jobs += remaining < producer_jobs
            ? remaining : producer_jobs;
    }
    const uint32_t bank0_jobs =
        activation_jobs < 32 ? activation_jobs : 32;
    const uint32_t bank1_jobs =
        activation_jobs > 32 ? activation_jobs - 32 : 0;

    CircularBuffer dgate0(cb_dgate0);
    CircularBuffer dup0(cb_dup0);
    dgate0.wait_front(bank0_jobs);
    dup0.wait_front(bank0_jobs);
    uint32_t dgate_addresses[2] = {dgate0.get_read_ptr(), 0};
    uint32_t dup_addresses[2] = {dup0.get_read_ptr(), 0};
    CircularBuffer scaled_dgate0(
        get_compile_time_arg_val(105));
    CircularBuffer scaled_dup0(
        get_compile_time_arg_val(106));
    scaled_dgate0.wait_front(bank0_jobs);
    scaled_dup0.wait_front(bank0_jobs);
    uint32_t scaled_dgate_addresses[2] = {scaled_dgate0.get_read_ptr(), 0};
    uint32_t scaled_dup_addresses[2] = {scaled_dup0.get_read_ptr(), 0};


    noc_semaphore_inc(
        get_noc_addr(13, physical_y(0), get_semaphore(output_ready)), 1);
    noc_async_atomic_barrier();

    if (node == 0) {
        volatile tt_l1_ptr uint32_t* ready_flag =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(output_ready));
        noc_semaphore_wait_min(ready_flag, kActivationNodes);
        noc_semaphore_set(ready_flag, 0);

        // CB allocations are identical on every core in one descriptor
        // range, so core zero publishes one address per bank and output.
        for (uint32_t target = 0; target < kMatmulNodes; ++target) {
            const uint32_t target_x = target % kMatmulCols;
            const uint32_t target_y = target / kMatmulCols;
            for (uint32_t bank = 0; bank < bank_count; ++bank) {
                volatile tt_l1_ptr uint32_t* dgate_mailbox =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        get_semaphore(address_start + 2 * bank));
                volatile tt_l1_ptr uint32_t* dup_mailbox =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        get_semaphore(address_start + 2 * bank + 1));
                noc_semaphore_set(dgate_mailbox, dgate_addresses[bank]);
                noc_semaphore_set(dup_mailbox, dup_addresses[bank]);
                noc_async_write(
                    get_semaphore(address_start + 2 * bank),
                    get_noc_addr(
                        kMatmulPhysicalX[target_x], physical_y(target_y),
                        get_semaphore(address_start + 2 * bank)),
                    sizeof(uint32_t));
                noc_async_write(
                    get_semaphore(address_start + 2 * bank + 1),
                    get_noc_addr(
                        kMatmulPhysicalX[target_x], physical_y(target_y),
                        get_semaphore(address_start + 2 * bank + 1)),
                    sizeof(uint32_t));
                constexpr uint32_t scaled_address_start = 7;
                volatile tt_l1_ptr uint32_t* scaled_dgate_mailbox =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        get_semaphore(scaled_address_start + 2 * bank));
                volatile tt_l1_ptr uint32_t* scaled_dup_mailbox =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        get_semaphore(scaled_address_start + 2 * bank + 1));
                noc_semaphore_set(
                    scaled_dgate_mailbox, scaled_dgate_addresses[bank]);
                noc_semaphore_set(
                    scaled_dup_mailbox, scaled_dup_addresses[bank]);
                noc_async_write(
                    get_semaphore(scaled_address_start + 2 * bank),
                    get_noc_addr(
                        kMatmulPhysicalX[target_x], physical_y(target_y),
                        get_semaphore(scaled_address_start + 2 * bank)),
                    sizeof(uint32_t));
                noc_async_write(
                    get_semaphore(scaled_address_start + 2 * bank + 1),
                    get_noc_addr(
                        kMatmulPhysicalX[target_x], physical_y(target_y),
                        get_semaphore(scaled_address_start + 2 * bank + 1)),
                    sizeof(uint32_t));
            }
            volatile tt_l1_ptr uint32_t* target_ready =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(output_ready));
            noc_semaphore_set(target_ready, 1);
            noc_async_write(
                get_semaphore(output_ready),
                get_noc_addr(
                    kMatmulPhysicalX[target_x], physical_y(target_y),
                    get_semaphore(output_ready)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();

        volatile tt_l1_ptr uint32_t* done_flag =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(consumer_done));
        noc_semaphore_wait_min(done_flag, kMatmulNodes);
        noc_semaphore_set(done_flag, 0);
        volatile tt_l1_ptr uint32_t* release_flag =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(release));
        noc_semaphore_set(release_flag, 1);
        for (uint32_t target = 1; target < kActivationNodes; ++target) {
            noc_async_write(
                get_semaphore(release),
                get_noc_addr(13, physical_y(target), get_semaphore(release)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }

    volatile tt_l1_ptr uint32_t* release_flag =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(release));
    noc_semaphore_wait(release_flag, 1);
    noc_semaphore_set(release_flag, 0);
    dgate0.pop_front(bank0_jobs);
    dup0.pop_front(bank0_jobs);
    scaled_dgate0.pop_front(bank0_jobs);
    scaled_dup0.pop_front(bank0_jobs);
}

void run() {
    full_activation_product_writer_phase();
    full_activation_backward_writer_phase();
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
    constexpr uint32_t cb_dv = get_compile_time_arg_val(137);
    constexpr uint32_t cb_dq = get_compile_time_arg_val(125);
    constexpr uint32_t cb_ds_handoff = get_compile_time_arg_val(118);
    constexpr uint32_t cb_ds_column = get_compile_time_arg_val(119);
    constexpr uint32_t cb_dk = get_compile_time_arg_val(126);
    constexpr uint32_t cb_peer_dk = get_compile_time_arg_val(120);
    constexpr uint32_t cb_peer_dv = get_compile_time_arg_val(121);
    constexpr uint32_t cb_gqa_reduced = get_compile_time_arg_val(122);
    constexpr uint32_t cb_gqa_bf16 = get_compile_time_arg_val(128);
    constexpr uint32_t cb_final_dk = get_compile_time_arg_val(129);
    constexpr uint32_t cb_final_dv = get_compile_time_arg_val(130);
    constexpr uint32_t cb_rope_q = get_compile_time_arg_val(135);
    constexpr uint32_t cb_rope_k = get_compile_time_arg_val(136);
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
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const auto output = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<41, 142>(), 0>(),
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
        get_absolute_logical_y() * 1 +
        (get_absolute_logical_x() - 10);
    noc_semaphore_inc(
        get_noc_addr(layer_physical_x[10], 2, get_semaphore(13)), 1);
    noc_async_atomic_barrier();
    if (layer_node == 0) {
        volatile tt_l1_ptr uint32_t* layer_arrivals =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(13));
        noc_semaphore_wait_min(layer_arrivals, 10);
        noc_semaphore_set(layer_arrivals, 0);
        volatile tt_l1_ptr uint32_t* layer_release_word =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(14));
        noc_semaphore_set(layer_release_word, 1);
        for (uint32_t target = 1; target < 10; ++target) {
            noc_async_write(
                get_semaphore(14),
                get_noc_addr(
                    layer_physical_x[10 + target % 1],
                    static_cast<uint8_t>(target / 1 + 2),
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
    CircularBuffer layer_compute_release(get_compile_time_arg_val(141));
    layer_compute_release.reserve_back(1);
    layer_compute_release.push_back(1);
    resident_layer_phase_attention::run();
}
