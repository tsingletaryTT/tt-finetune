#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "api/debug/waypoint.h"

namespace resident_layer_phase_mlp {
// Reader for one-call resident down-projection backward.
//
// The exact phase-one inputs are streamed as before.  The M dH tiles are
// then multicast from their owner cores into a dedicated L1 window on every
// worker.  Both dA and LoRA dProduct consume that window without a DRAM
// round-trip.



namespace {
constexpr uint32_t kGridCols = 10;
constexpr uint32_t kWorkerNodes = 100;
constexpr uint32_t kTileBytes = 2048;
constexpr uint8_t kPhysicalX[kGridCols] = {
    1, 2, 3, 4, 5, 6, 7, 10, 11, 12};
constexpr uint8_t kActivationPhysicalX = 13;
FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}  // namespace

FORCE_INLINE void full_down_reader_phase() {
    Noc noc0(0);

    constexpr uint32_t cb_base_dy = get_compile_time_arg_val(79);
    constexpr uint32_t cb_base_weight = get_compile_time_arg_val(80);
    constexpr uint32_t cb_dh_dy = get_compile_time_arg_val(64);
    constexpr uint32_t cb_dh_b = get_compile_time_arg_val(65);
    constexpr uint32_t cb_dh_output = get_compile_time_arg_val(73);
    constexpr uint32_t cb_db_dy = get_compile_time_arg_val(87);
    constexpr uint32_t cb_db_hidden = get_compile_time_arg_val(81);
    constexpr uint32_t cb_dh_resident = get_compile_time_arg_val(85);
    constexpr uint32_t cb_tail_a = get_compile_time_arg_val(63);
    constexpr uint32_t cb_tail_x = get_compile_time_arg_val(67);
    constexpr uint32_t cb_recompute_product = get_compile_time_arg_val(62);
    constexpr uint32_t cb_recompute_a = get_compile_time_arg_val(92);
    constexpr uint32_t cb_recompute_h = get_compile_time_arg_val(89);
    constexpr uint32_t cb_recompute_gate = get_compile_time_arg_val(62);
    constexpr uint32_t cb_recompute_up = get_compile_time_arg_val(92);
    constexpr uint32_t cb_recompute_gate_saved = get_compile_time_arg_val(89);

    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t k_tiles = 152;
    constexpr uint32_t n_tiles = 28;
    constexpr uint32_t base_reduction_tiles = 2;
    constexpr uint32_t base_reduction_blocks =
        n_tiles / base_reduction_tiles;
    constexpr uint32_t jobs_per_node = 7;
    constexpr uint32_t da_k_tiles_per_node = 2;
    constexpr uint32_t n_tiles_per_node = 1;
    constexpr uint32_t total_jobs = m_tiles * k_tiles;

    static_assert(m_tiles <= 8, "resident dH semaphore map supports M <= 8");

    CircularBuffer base_dy(cb_base_dy);
    CircularBuffer base_weight(cb_base_weight);
    CircularBuffer dh_dy(cb_dh_dy);
    CircularBuffer dh_b(cb_dh_b);
    CircularBuffer dh_output(cb_dh_output);
    CircularBuffer db_dy(cb_db_dy);
    CircularBuffer db_hidden(cb_db_hidden);
    CircularBuffer dh_resident(cb_dh_resident);
    CircularBuffer tail_a(cb_tail_a);
    CircularBuffer tail_x(cb_tail_x);
    CircularBuffer recompute_product(cb_recompute_product);
    CircularBuffer recompute_a(cb_recompute_a);
    CircularBuffer recompute_h(cb_recompute_h);
    CircularBuffer recompute_gate(cb_recompute_gate);
    CircularBuffer recompute_up(cb_recompute_up);
    CircularBuffer recompute_gate_saved(cb_recompute_gate_saved);

    const uint32_t logical_x = get_absolute_logical_x();
    const uint32_t logical_y = get_absolute_logical_y();
    const uint32_t node = logical_y * kGridCols + logical_x;
    constexpr uint32_t go_semaphore = m_tiles;
    constexpr uint32_t ready_semaphore_start = m_tiles + 1;

    // Semaphore addresses may contain state from an unrelated prior program.
    // Reset every flag on every invocation before any producer can publish.
    for (uint32_t semaphore = 0; semaphore < 12; ++semaphore) {
        volatile tt_l1_ptr uint32_t* flag =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(semaphore));
        noc_semaphore_set(flag, 0);
    }

    // Tensor order: dY, W, B, H, A, X.
    const uint32_t dy_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t weight_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t hidden_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t a_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(5);

    const auto dy = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 142>(), 0>(), dy_addr, kTileBytes);
    const auto weight = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 142>(), 1>(), weight_addr, kTileBytes);
    const auto b = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 142>(), 2>(), b_addr, kTileBytes);
    const auto hidden = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<3, 142>(), 3>(), hidden_addr, kTileBytes);
    const auto a = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<4, 142>(), 4>(), a_addr, kTileBytes);
    const auto x = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<5, 142>(), 5>(), x_addr, kTileBytes);

    // The 10-core BF16 activation island computes every product tile
    // once and writes it into these exact H-row / dA-column L1 windows.
    volatile tt_l1_ptr uint32_t* product_reset_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(15));
    noc_semaphore_wait(product_reset_ready, 1);
    noc_semaphore_set(product_reset_ready, 0);

    if (node < m_tiles) {
        recompute_product.reserve_back(k_tiles);
    }
    uint32_t valid_da_tiles = 0;
    for (uint32_t local_k = 0; local_k < da_k_tiles_per_node; ++local_k) {
        if (node * da_k_tiles_per_node + local_k < k_tiles) {
            ++valid_da_tiles;
        }
    }
    if (valid_da_tiles != 0) {
        tail_x.reserve_back(valid_da_tiles * m_tiles);
    }

    // All destinations must be reserved before any activation writer starts.
    noc_semaphore_inc(
        get_noc_addr(kActivationPhysicalX, physical_y(0), get_semaphore(13)),
        1);
    noc_async_atomic_barrier();
    if (node == 0) {
        volatile tt_l1_ptr uint32_t* h_address =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(14));
        volatile tt_l1_ptr uint32_t* da_address =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(15));
        noc_semaphore_set(h_address, recompute_product.get_write_ptr());
        noc_semaphore_set(da_address, tail_x.get_write_ptr());
        noc_async_write(
            get_semaphore(14),
            get_noc_addr(kActivationPhysicalX, physical_y(0), get_semaphore(14)),
            sizeof(uint32_t));
        noc_async_write(
            get_semaphore(15),
            get_noc_addr(kActivationPhysicalX, physical_y(0), get_semaphore(15)),
            sizeof(uint32_t));
        noc_async_write_barrier();
        noc_semaphore_set(h_address, 0);
        noc_semaphore_set(da_address, 0);
    }

    volatile tt_l1_ptr uint32_t* products_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(14));
    noc_semaphore_wait(products_ready, 1);
    noc_semaphore_set(products_ready, 0);
    if (node < m_tiles) {
        recompute_product.push_back(k_tiles);
    }
    if (valid_da_tiles != 0) {
        tail_x.push_back(valid_da_tiles * m_tiles);
    }

    // Stream down A only after the product handoff so its small CB cannot
    // block the reader before activation writers observe all destinations.
    if (node < m_tiles) {
        for (uint32_t i = 0; i < k_tiles; i += 2) {
            recompute_a.reserve_back(2);
            for (uint32_t inner = 0; inner < 2; ++inner) {
                noc0.async_read(
                    a,
                    CoreLocalMem<uint32_t>(
                        recompute_a.get_write_ptr() + inner * kTileBytes),
                    kTileBytes, {.page_id = i + inner}, {});
            }
            noc0.async_read_barrier();
            recompute_a.push_back(2);
        }
    }

    // H = product @ A.T: token-row owners reproduce the accepted two-K-tile
    // spill schedule, then multicast H into the dB consumers' L1 windows.
    if (node < m_tiles) {
        recompute_h.wait_front(1);
        for (uint32_t target = 0; target < n_tiles; ++target) {
            noc_async_write(
                recompute_h.get_read_ptr(),
                get_noc_addr(
                    kPhysicalX[target % kGridCols],
                    physical_y(target / kGridCols),
                    db_hidden.get_write_ptr() + node * kTileBytes),
                kTileBytes);
        }
        noc_async_write_barrier();
        for (uint32_t target = 0; target < n_tiles; ++target) {
            noc_semaphore_inc(
                get_noc_addr(
                    kPhysicalX[target % kGridCols],
                    physical_y(target / kGridCols), get_semaphore(14)),
                1);
        }
        noc_async_atomic_barrier();
        recompute_h.pop_front(1);
    }
    if (node < n_tiles) {
        volatile tt_l1_ptr uint32_t* ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(14));
        noc_semaphore_wait_min(ready, m_tiles);
        noc_semaphore_set(ready, 0);
        db_hidden.push_back(m_tiles);
    }

    // Frozen-base dProduct input stream.
    for (uint32_t local = 0; local < jobs_per_node; ++local) {
        const uint32_t job = node * jobs_per_node + local;
        if (job >= total_jobs) continue;
        const uint32_t m = job / k_tiles;
        const uint32_t k = job % k_tiles;
        for (uint32_t block = 0; block < base_reduction_blocks; ++block) {
            const uint32_t n0 = block * base_reduction_tiles;
            base_dy.reserve_back(base_reduction_tiles);
            base_weight.reserve_back(base_reduction_tiles);
            for (uint32_t inner = 0; inner < base_reduction_tiles; ++inner) {
                noc0.async_read(
                    dy,
                    CoreLocalMem<uint32_t>(
                        base_dy.get_write_ptr() + inner * kTileBytes),
                    kTileBytes,
                    {.page_id = m * n_tiles + n0 + inner}, {});
                noc0.async_read(
                    weight,
                    CoreLocalMem<uint32_t>(
                        base_weight.get_write_ptr() + inner * kTileBytes),
                    kTileBytes,
                    {.page_id = (n0 + inner) * k_tiles + k}, {});
            }
            noc0.async_read_barrier();
            base_weight.push_back(base_reduction_tiles);
            base_dy.push_back(base_reduction_tiles);
        }
    }

    // Exact dH owner stream.
    if (node < m_tiles) {
        dh_dy.reserve_back(n_tiles);
        dh_b.reserve_back(n_tiles);
        for (uint32_t n = 0; n < n_tiles; ++n) {
            noc0.async_read(
                dy,
                CoreLocalMem<uint32_t>(dh_dy.get_write_ptr() + n * kTileBytes),
                kTileBytes, {.page_id = node * n_tiles + n}, {});
            noc0.async_read(
                b,
                CoreLocalMem<uint32_t>(dh_b.get_write_ptr() + n * kTileBytes),
                kTileBytes, {.page_id = n}, {});
        }
        noc0.async_read_barrier();
        dh_b.push_back(n_tiles);
        dh_dy.push_back(n_tiles);
    }

    // Exact dB streams.
    for (uint32_t local_n = 0; local_n < n_tiles_per_node; ++local_n) {
        const uint32_t n = node * n_tiles_per_node + local_n;
        if (n >= n_tiles) continue;
        db_dy.reserve_back(m_tiles);
        for (uint32_t m = 0; m < m_tiles; ++m) {
            noc0.async_read(
                dy,
                CoreLocalMem<uint32_t>(db_dy.get_write_ptr() + m * kTileBytes),
                kTileBytes, {.page_id = m * n_tiles + n}, {});
        }
        noc0.async_read_barrier();
        db_dy.push_back(m_tiles);
    }

    // Wait for every dH producer, then multicast its tile to the same fixed
    // offset in each worker's resident CB window.
    dh_resident.reserve_back(m_tiles);
    if (node < m_tiles) {
        dh_output.wait_front(1);
        volatile tt_l1_ptr uint32_t* owner_flag =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(node));
        noc_semaphore_set(owner_flag, 1);
        if (node != 0) {
            noc_async_write(
                get_semaphore(node),
                get_noc_addr(kPhysicalX[0], physical_y(0), get_semaphore(node)),
                sizeof(uint32_t));
            noc_async_write_barrier();
            noc_semaphore_set(owner_flag, 0);
        }
    }
    if (node == 0) {
        DeviceZoneScopedN("resident_down_dh_barrier");
        volatile tt_l1_ptr uint32_t* go =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(go_semaphore));
        noc_semaphore_set(go, 1);
        for (uint32_t owner = 0; owner < m_tiles; ++owner) {
            volatile tt_l1_ptr uint32_t* ready =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(owner));
            noc_semaphore_wait(ready, 1);
            noc_semaphore_set(ready, 0);
        }
        for (uint32_t owner = 1; owner < m_tiles; ++owner) {
            noc_async_write(
                get_semaphore(go_semaphore),
                get_noc_addr(
                    kPhysicalX[owner], physical_y(0),
                    get_semaphore(go_semaphore)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    if (node < m_tiles) {
        volatile tt_l1_ptr uint32_t* go =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(go_semaphore));
        noc_semaphore_wait(go, 1);
        noc_semaphore_set(go, 0);
        {
            DeviceZoneScopedN("resident_down_dh_noc");
            const uint32_t source = dh_output.get_read_ptr();
            const uint32_t destination =
                dh_resident.get_write_ptr() + node * kTileBytes;
            for (uint32_t target = 0; target < kWorkerNodes; ++target) {
                noc_async_write(
                    source,
                    get_noc_addr(
                        kPhysicalX[target % kGridCols],
                        physical_y(target / kGridCols), destination),
                    kTileBytes);
            }
            noc_async_write_barrier();
            volatile tt_l1_ptr uint32_t* local_ready =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(ready_semaphore_start + node));
            noc_semaphore_set(local_ready, 1);
            for (uint32_t target = 0; target < kWorkerNodes; ++target) {
                if (target == node) continue;
                noc_async_write(
                    get_semaphore(ready_semaphore_start + node),
                    get_noc_addr(
                        kPhysicalX[target % kGridCols],
                        physical_y(target / kGridCols),
                        get_semaphore(ready_semaphore_start + node)),
                    sizeof(uint32_t));
            }
            noc_async_write_barrier();
        }
        dh_output.pop_front(1);
    }
    for (uint32_t owner = 0; owner < m_tiles; ++owner) {
        volatile tt_l1_ptr uint32_t* ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(ready_semaphore_start + owner));
        noc_semaphore_wait(ready, 1);
        noc_semaphore_set(ready, 0);
    }
    dh_resident.push_back(m_tiles);


    // Cross-island initialization barrier.  Send this after dH is resident,
    // but before filling the tail input CBs.  Sending it at kernel exit can
    // deadlock: the tail reader may wait for CB space while the down writer
    // waits for the activation island that is still waiting for this ack.
    noc_semaphore_inc(
        get_noc_addr(
            kActivationPhysicalX, physical_y(node % 10),
            get_semaphore(11)),
        1);
    // Blackhole does not implicitly flush non-posted NoC atomics when a
    // kernel exits.  The activation island must observe this ack before this
    // reader can complete or move on to unrelated traffic.
    noc_async_atomic_barrier();

    // LoRA dProduct uses A once per output tile job.
    for (uint32_t local = 0; local < jobs_per_node; ++local) {
        const uint32_t job = node * jobs_per_node + local;
        if (job >= total_jobs) continue;
        const uint32_t k = job % k_tiles;
        tail_a.reserve_back(1);
        noc0.async_read(
            a, CoreLocalMem<uint32_t>(tail_a.get_write_ptr()),
            kTileBytes, {.page_id = k}, {});
        noc0.async_read_barrier();
        tail_a.push_back(1);
    }

    // dA owns complete token reductions and therefore reads M product tiles
    // for each of its (at most two on Qwen) K tiles.
    for (uint32_t local_k = 0; local_k < da_k_tiles_per_node; ++local_k) {
        const uint32_t k = node * da_k_tiles_per_node + local_k;
        if (k >= k_tiles) continue;
        // Product tiles are recreated by the compute RISC in CB18.
    }
}

// Device-resident reader for two equal-shape frozen Linear + LoRA members.
//
// Phase one is loaded exactly as in the accepted concurrent 55+55-core path.
// Its dH result is then distributed from the four owning cores to every core
// in the same 55-core half through NoC writes into CB4.  No phase-one result
// is materialized in DRAM.



namespace {

constexpr uint32_t kGroupedGridCols = 10;
constexpr uint32_t kGroupedMemberNodes = 50;
constexpr uint32_t kGroupedTileBytes = 2048;
constexpr uint8_t kGroupedPhysicalX[kGroupedGridCols] = {
    1, 2, 3, 4, 5, 6, 7, 10, 11, 12};

FORCE_INLINE uint8_t grouped_physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}

}  // namespace

FORCE_INLINE void full_grouped_reader_phase() {
    Noc noc0(0);

    constexpr uint32_t cb_base_dy = get_compile_time_arg_val(79);
    constexpr uint32_t cb_base_weight = get_compile_time_arg_val(80);
    constexpr uint32_t cb_base_result = get_compile_time_arg_val(71);
    constexpr uint32_t cb_dh_dy = get_compile_time_arg_val(62);
    constexpr uint32_t cb_dh_b = get_compile_time_arg_val(92);
    constexpr uint32_t cb_dh_scaled = get_compile_time_arg_val(93);
    constexpr uint32_t cb_tail_a = get_compile_time_arg_val(63);
    // Dedicated M-tile L1 window: reader is its only producer and tail
    // compute is its only consumer. Reusing a phase CB would either mismatch
    // geometry (CB4 is N-wide) or cross RISC producer ownership (CB12).
    constexpr uint32_t cb_dh_resident = get_compile_time_arg_val(85);
    constexpr uint32_t cb_dh_output = get_compile_time_arg_val(89);
    constexpr uint32_t cb_peer_sum = get_compile_time_arg_val(88);
    constexpr uint32_t cb_db_dy = get_compile_time_arg_val(87);
    constexpr uint32_t cb_db_hidden = get_compile_time_arg_val(81);
    constexpr uint32_t cb_tail_x = get_compile_time_arg_val(87);

    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t k_tiles = 28;
    constexpr uint32_t n_tiles = 152;
    constexpr uint32_t base_reduction_tiles = 1;
    constexpr uint32_t base_reduction_blocks =
        n_tiles / base_reduction_tiles;
    constexpr uint32_t jobs_per_node = 3;
    constexpr uint32_t n_tiles_per_node = 4;
    constexpr uint32_t total_jobs = m_tiles * k_tiles;

    CircularBuffer base_dy(cb_base_dy);
    CircularBuffer base_weight(cb_base_weight);
    CircularBuffer dh_dy(cb_dh_dy);
    CircularBuffer dh_b(cb_dh_b);
    CircularBuffer dh_scaled(cb_dh_scaled);
    CircularBuffer dh_resident(cb_dh_resident);
    CircularBuffer dh_output(cb_dh_output);
    CircularBuffer tail_a(cb_tail_a);
    CircularBuffer tail_x(cb_tail_x);
    CircularBuffer peer_sum(cb_peer_sum);
    CircularBuffer db_dy(cb_db_dy);
    CircularBuffer db_hidden(cb_db_hidden);

    const uint32_t logical_x = get_absolute_logical_x();
    const uint32_t logical_y = get_absolute_logical_y();
    const uint32_t global_node = logical_y * kGroupedGridCols + logical_x;
    const uint32_t member = global_node / kGroupedMemberNodes;
    const uint32_t node = global_node - member * kGroupedMemberNodes;
    const uint32_t group_y = member * 5;
    constexpr uint32_t dh_owner_start = 46;
    const bool is_dh_owner = node >= dh_owner_start;
    const uint32_t dh_owner_index = node - dh_owner_start;
    // Snapshot the activation-bank addresses before IDs 0..10 are
    // reused by the grouped dH and peer-sum schedule.
    volatile tt_l1_ptr uint32_t* activation_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(4));
    noc_semaphore_wait(activation_ready, 1);
    const uint32_t dgate_base0 = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(0));
    const uint32_t dup_base0 = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(1));
    uint32_t dgate_base1 = 0;
    uint32_t dup_base1 = 0;
    if constexpr (2 == 2) {
        dgate_base1 = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(2));
        dup_base1 = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(3));
    }

    // Exact BF16(dY * scale) is produced by the activation island.
    // Snapshot those bank addresses before semaphore IDs 7..10 are reused.
    const uint32_t scaled_dgate_base0 =
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(7));
    const uint32_t scaled_dup_base0 =
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(8));
    uint32_t scaled_dgate_base1 = 0;
    uint32_t scaled_dup_base1 = 0;
    if constexpr (2 == 2) {
        scaled_dgate_base1 =
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(9));
        scaled_dup_base1 =
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(10));
    }

    // Every matmul core must snapshot the bank mailboxes before any core
    // reuses IDs 0..10 for grouped dH and peer-sum synchronization.
    constexpr uint32_t snapshot_count_sem = 12;
    constexpr uint32_t snapshot_go_sem = 13;
    noc_semaphore_inc(
        get_noc_addr(kGroupedPhysicalX[0], grouped_physical_y(0),
                     get_semaphore(snapshot_count_sem)), 1);
    noc_async_atomic_barrier();
    if (global_node == 0) {
        volatile tt_l1_ptr uint32_t* count =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(snapshot_count_sem));
        noc_semaphore_wait_min(count, 100);
        noc_semaphore_set(count, 0);
        volatile tt_l1_ptr uint32_t* go =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(snapshot_go_sem));
        noc_semaphore_set(go, 1);
        for (uint32_t target = 1; target < 100; ++target) {
            noc_async_write(
                get_semaphore(snapshot_go_sem),
                get_noc_addr(
                    kGroupedPhysicalX[target % 10],
                    grouped_physical_y(target / 10),
                    get_semaphore(snapshot_go_sem)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    volatile tt_l1_ptr uint32_t* snapshot_go =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(snapshot_go_sem));
    noc_semaphore_wait(snapshot_go, 1);
    noc_semaphore_set(snapshot_go, 0);

#define FULL_READ_RESIDENT_DY(page_expr, destination_expr) do { \
    const uint32_t full_page = (page_expr); \
    const uint32_t producer = full_page / 7; \
    const uint32_t activation_node = producer % 10; \
    uint32_t local_index = full_page - producer * 7; \
    for (uint32_t prior = activation_node; prior < producer; prior += 10) { \
        const uint32_t first = prior * 7; \
        const uint32_t remaining = 608 - first; \
        local_index += remaining < 7 \
            ? remaining : 7; \
    } \
    const uint32_t bank = local_index / 32; \
    const uint32_t slot = local_index - bank * 32; \
    const uint32_t dgate_base = bank == 0 ? dgate_base0 : dgate_base1; \
    const uint32_t dup_base = bank == 0 ? dup_base0 : dup_base1; \
    const uint32_t base = member == 0 ? dgate_base : dup_base; \
    noc_async_read( \
        get_noc_addr(13, grouped_physical_y(activation_node), \
        base + slot * kGroupedTileBytes), \
        (destination_expr), kGroupedTileBytes); \
} while (0)

#define FULL_READ_RESIDENT_SCALED_DY(page_expr, destination_expr) do { \
    const uint32_t full_page = (page_expr); \
    const uint32_t producer = full_page / 7; \
    const uint32_t activation_node = producer % 10; \
    uint32_t local_index = full_page - producer * 7; \
    for (uint32_t prior = activation_node; prior < producer; prior += 10) { \
        const uint32_t first = prior * 7; \
        const uint32_t remaining = 608 - first; \
        local_index += remaining < 7 \
            ? remaining : 7; \
    } \
    const uint32_t bank = local_index / 32; \
    const uint32_t slot = local_index - bank * 32; \
    const uint32_t dgate_base = bank == 0 \
        ? scaled_dgate_base0 : scaled_dgate_base1; \
    const uint32_t dup_base = bank == 0 \
        ? scaled_dup_base0 : scaled_dup_base1; \
    const uint32_t base = member == 0 ? dgate_base : dup_base; \
    noc_async_read( \
        get_noc_addr(13, grouped_physical_y(activation_node), \
        base + slot * kGroupedTileBytes), \
        (destination_expr), kGroupedTileBytes); \
} while (0)

    volatile tt_l1_ptr uint32_t* go_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(4));

    // A program's semaphore allocation can reuse L1 addresses touched by an
    // unrelated program that ran earlier in the full transformer graph.
    // Initialize every local flag on every invocation instead of relying on
    // creation-time values. The writer cannot reach its request wait until
    // reader-fed dB and dA have been produced, so the phase-one CB pipeline
    // provides the cross-RISC ordering needed for this reader-owned reset.
    for (uint32_t semaphore = 0; semaphore < 11; ++semaphore) {
        volatile tt_l1_ptr uint32_t* flag =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(semaphore));
        noc_semaphore_set(flag, 0);
    }

    // Reader common args are deliberately asserted by the Python builder to
    // be the eleven input tensor addresses in this exact order.
    const uint32_t dy0_addr = get_common_arg_val<uint32_t>(6);
    const uint32_t w0_addr = get_common_arg_val<uint32_t>(8);
    const uint32_t b0_addr = get_common_arg_val<uint32_t>(9);
    const uint32_t h0_addr = get_common_arg_val<uint32_t>(10);
    const uint32_t a0_addr = get_common_arg_val<uint32_t>(11);
    const uint32_t dy1_addr = get_common_arg_val<uint32_t>(7);
    const uint32_t w1_addr = get_common_arg_val<uint32_t>(12);
    const uint32_t b1_addr = get_common_arg_val<uint32_t>(13);
    const uint32_t h1_addr = get_common_arg_val<uint32_t>(14);
    const uint32_t a1_addr = get_common_arg_val<uint32_t>(15);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(16);

    const auto dy0 = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<6, 142>(), 6>(),
        dy0_addr,
        kGroupedTileBytes);
    const auto w0 = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<8, 142>(), 8>(),
        w0_addr,
        kGroupedTileBytes);
    const auto b0 = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<9, 142>(), 9>(),
        b0_addr,
        kGroupedTileBytes);
    const auto h0 = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<10, 142>(), 10>(),
        h0_addr,
        kGroupedTileBytes);
    const auto a0 = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<11, 142>(), 11>(),
        a0_addr,
        kGroupedTileBytes);
    const auto dy1 = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<7, 142>(), 7>(),
        dy1_addr,
        kGroupedTileBytes);
    const auto w1 = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<12, 142>(), 12>(),
        w1_addr,
        kGroupedTileBytes);
    const auto b1 = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<13, 142>(), 13>(),
        b1_addr,
        kGroupedTileBytes);
    const auto h1 = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<14, 142>(), 14>(),
        h1_addr,
        kGroupedTileBytes);
    const auto a1 = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<15, 142>(), 15>(),
        a1_addr,
        kGroupedTileBytes);
    const auto x = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<16, 142>(), 16>(),
        x_addr,
        kGroupedTileBytes);

    const auto& dy = member == 0 ? dy0 : dy1;
    const auto& weight = member == 0 ? w0 : w1;
    const auto& b = member == 0 ? b0 : b1;
    const auto& hidden = member == 0 ? h0 : h1;
    const auto& a = member == 0 ? a0 : a1;

    // Accepted phase-one DRAM reads.
    for (uint32_t local = 0; local < jobs_per_node; ++local) {
        const uint32_t job = node * jobs_per_node + local;
        if (job >= total_jobs) {
            continue;
        }
        const uint32_t m = job / k_tiles;
        const uint32_t k = job - m * k_tiles;
        for (uint32_t block = 0; block < base_reduction_blocks; ++block) {
            const uint32_t n0 = block * base_reduction_tiles;
            base_dy.reserve_back(base_reduction_tiles);
            base_weight.reserve_back(base_reduction_tiles);
            for (uint32_t inner = 0; inner < base_reduction_tiles; ++inner) {
                FULL_READ_RESIDENT_DY(
                    m * n_tiles + n0 + inner,
                    base_dy.get_write_ptr() + inner * kGroupedTileBytes);
                noc0.async_read(
                    weight,
                    CoreLocalMem<uint32_t>(
                        base_weight.get_write_ptr() + inner * kGroupedTileBytes),
                    kGroupedTileBytes,
                    {.page_id = (n0 + inner) * k_tiles + k},
                    {});
            }
            noc0.async_read_barrier();
            base_weight.push_back(base_reduction_tiles);
            base_dy.push_back(base_reduction_tiles);
        }
    }

    if (is_dh_owner) {
        static_assert(n_tiles % 2 == 0, "grouped dH K must be /2");
        // Qwen has 76 blocks. Keep this as a runtime streaming loop: fully
        // unrolling repeated compute-format transitions creates enormous
        // BRISC code and pathological first-use compilation.
#pragma GCC unroll 1
        for (uint32_t block = 0; block < n_tiles / 2; ++block) {
            const uint32_t n0 = block * 2;
            dh_scaled.reserve_back(2);
            dh_b.reserve_back(2);
            for (uint32_t inner = 0; inner < 2; ++inner) {
                FULL_READ_RESIDENT_SCALED_DY(
                    dh_owner_index * n_tiles + n0 + inner,
                    dh_scaled.get_write_ptr() + inner * kGroupedTileBytes);
                noc0.async_read(
                    b,
                    CoreLocalMem<uint32_t>(
                        dh_b.get_write_ptr() + inner * kGroupedTileBytes),
                    kGroupedTileBytes,
                    {.page_id = n0 + inner},
                    {});
            }
            noc0.async_read_barrier();
            dh_b.push_back(2);
            dh_scaled.push_back(2);
        }
    }

    for (uint32_t local_n = 0; local_n < n_tiles_per_node; ++local_n) {
        const uint32_t n_tile = node * n_tiles_per_node + local_n;
        if (n_tile >= n_tiles) {
            continue;
        }
        db_dy.reserve_back(m_tiles);
        db_hidden.reserve_back(m_tiles);
        for (uint32_t m = 0; m < m_tiles; ++m) {
            FULL_READ_RESIDENT_DY(
                m * n_tiles + n_tile,
                db_dy.get_write_ptr() + m * kGroupedTileBytes);
            noc0.async_read(
                hidden,
                CoreLocalMem<uint32_t>(
                    db_hidden.get_write_ptr() + m * kGroupedTileBytes),
                kGroupedTileBytes,
                {.page_id = m * 334 + 304 + member},
                {});
        }
        noc0.async_read_barrier();
        db_hidden.push_back(m_tiles);
        db_dy.push_back(m_tiles);
    }

    // Reserve the complete local destination before any owner publishes.
    dh_resident.reserve_back(m_tiles);
    if (is_dh_owner) {
        dh_output.wait_front(1);
        const uint32_t source = dh_output.get_read_ptr();
        const uint32_t destination =
            dh_resident.get_write_ptr()
            + dh_owner_index * kGroupedTileBytes;
        for (uint32_t target = 0; target < kGroupedMemberNodes; ++target) {
            const uint32_t tx = target % kGroupedGridCols;
            const uint32_t ty = group_y + target / kGroupedGridCols;
            noc_async_write(
                source,
                get_noc_addr(
                    kGroupedPhysicalX[tx], grouped_physical_y(ty), destination),
                kGroupedTileBytes);
        }
        // The completion atomic becomes visible only after every dH payload.
        noc_async_write_barrier();
        noc_semaphore_inc(
            get_noc_addr(
                kGroupedPhysicalX[0], grouped_physical_y(group_y),
                get_semaphore(0)),
            1);
        noc_async_atomic_barrier();
        dh_output.pop_front(1);
    }

    if (node == 0) {
        volatile tt_l1_ptr uint32_t* completed =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
        noc_semaphore_wait_min(completed, m_tiles);
        noc_semaphore_set(completed, 0);
        volatile tt_l1_ptr uint32_t* local_go =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
        noc_semaphore_set(local_go, 1);
        for (uint32_t target = 1; target < kGroupedMemberNodes; ++target) {
            const uint32_t tx = target % kGroupedGridCols;
            const uint32_t ty = group_y + target / kGroupedGridCols;
            noc_async_write(
                get_semaphore(1),
                get_noc_addr(
                    kGroupedPhysicalX[tx], grouped_physical_y(ty),
                    get_semaphore(1)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    volatile tt_l1_ptr uint32_t* resident_go =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    noc_semaphore_wait(resident_go, 1);
    noc_semaphore_set(resident_go, 0);
    dh_resident.push_back(m_tiles);

    // Original model inputs and trainable A are ordinary inputs, not
    // phase-one intermediates. They are streamed only after residency is set.
    for (uint32_t local = 0; local < jobs_per_node; ++local) {
        const uint32_t job = node * jobs_per_node + local;
        if (job >= total_jobs) {
            continue;
        }
        const uint32_t k = job % k_tiles;
        tail_a.reserve_back(1);
        noc0.async_read(
            a,
            CoreLocalMem<uint32_t>(tail_a.get_write_ptr()),
            kGroupedTileBytes,
            {.page_id = k},
            {});
        noc0.async_read_barrier();
        tail_a.push_back(1);
    }

    if (node < k_tiles) {
        tail_x.reserve_back(m_tiles);
        for (uint32_t m = 0; m < m_tiles; ++m) {
            noc0.async_read(
                x,
                CoreLocalMem<uint32_t>(
                    tail_x.get_write_ptr() + m * kGroupedTileBytes),
                kGroupedTileBytes,
                {.page_id = m * 334 + 306 + node},
                {});
        }
        noc0.async_read_barrier();
        tail_x.push_back(m_tiles);
    }

    // Member-zero readers reserve their peer-sum slots and tell the matching
    // member-one writer that the exact BF16 values may be sent directly.
    if (member == 0) {
        const uint32_t first_job = node * jobs_per_node;
        const uint32_t valid_jobs =
            first_job >= total_jobs
                ? 0
                : ((total_jobs - first_job) < jobs_per_node
                       ? total_jobs - first_job
                       : jobs_per_node);
        if (valid_jobs != 0) {
            peer_sum.reserve_back(valid_jobs);
            const uint32_t peer_y = logical_y + 5;
            volatile tt_l1_ptr uint32_t* request_sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(10));
            noc_semaphore_set(request_sem, 1);
            noc_async_write(
                get_semaphore(10),
                get_noc_addr(
                    kGroupedPhysicalX[logical_x],
                    grouped_physical_y(peer_y),
                    get_semaphore(10)),
                sizeof(uint32_t));
            noc_async_write_barrier();
            noc_semaphore_set(request_sem, 0);
            volatile tt_l1_ptr uint32_t* peer_sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(9));
            {
                DeviceZoneScopedN("resident_peer_wait");
                noc_semaphore_wait_min(peer_sem, valid_jobs);
            }
            noc_semaphore_set(peer_sem, 0);
            peer_sum.push_back(valid_jobs);
        }
    }

    noc_async_read_barrier();
    noc_semaphore_inc(
        get_noc_addr(13, grouped_physical_y(0), get_semaphore(5)), 1);
    noc_async_atomic_barrier();
#undef FULL_READ_RESIDENT_DY
#undef FULL_READ_RESIDENT_SCALED_DY

}

void run() {
    full_down_reader_phase();
    full_grouped_reader_phase();
}
}  // namespace resident_layer_phase_mlp

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
    constexpr uint32_t cb_p_row = get_compile_time_arg_val(72);
    constexpr uint32_t cb_p_column = get_compile_time_arg_val(74);
    constexpr uint32_t cb_dout_row = get_compile_time_arg_val(88);
    constexpr uint32_t cb_dout_full = get_compile_time_arg_val(67);
    constexpr uint32_t cb_q_full = get_compile_time_arg_val(68);
    constexpr uint32_t cb_k = get_compile_time_arg_val(69);
    constexpr uint32_t cb_v = get_compile_time_arg_val(70);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(101);
    constexpr uint32_t cb_dv = get_compile_time_arg_val(98);
    constexpr uint32_t cb_ds_handoff = get_compile_time_arg_val(81);
    constexpr uint32_t cb_ds_column = get_compile_time_arg_val(82);
    constexpr uint32_t cb_dk = get_compile_time_arg_val(87);
    constexpr uint32_t cb_peer_dk = get_compile_time_arg_val(63);
    constexpr uint32_t cb_peer_dv = get_compile_time_arg_val(64);
    constexpr uint32_t cb_gqa_rows = get_compile_time_arg_val(62);
    constexpr uint32_t cb_gqa_scaler = get_compile_time_arg_val(100);
    constexpr uint32_t cb_rope_cos = get_compile_time_arg_val(92);
    constexpr uint32_t cb_rope_sin = get_compile_time_arg_val(93);
    constexpr uint32_t seq_tiles = 4;
    constexpr uint32_t dim_tiles = 2;
    constexpr uint32_t query_heads = 14;
    constexpr uint32_t kv_heads = 2;
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

    const uint32_t cos_addr = get_common_arg_val<uint32_t>(17);
    const uint32_t sin_addr = get_common_arg_val<uint32_t>(18);
    const uint32_t checkpoint_arena_addr = get_common_arg_val<uint32_t>(19);
    constexpr uint32_t checkpoint_query_page_base = 112;
    constexpr uint32_t checkpoint_key_page_base = 224;
    constexpr uint32_t checkpoint_value_page_base = 240;
    constexpr uint32_t checkpoint_probability_page_base = 384;
    const auto cos_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<21, 142>(), 17>(),
        cos_addr, kTileBytes);
    const auto sin_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<22, 142>(), 18>(),
        sin_addr, kTileBytes);
    const auto checkpoint_arena_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<33, 142>(), 19>(),
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
    resident_layer_phase_mlp::run();
    volatile tt_l1_ptr uint32_t* layer_reader_release =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(12));
    noc_semaphore_wait(layer_reader_release, 1);
    noc_semaphore_set(layer_reader_release, 0);
    resident_layer_phase_attention::run();
}
