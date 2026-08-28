#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace resident_layer_phase_attention_0 {
// Reader for one-call resident down-projection backward.
//
// The exact phase-one inputs are streamed as before.  The M dH tiles are
// then multicast from their owner cores into a dedicated L1 window on every
// worker.  Both dA and LoRA dProduct consume that window without a DRAM
// round-trip.



namespace {
constexpr uint32_t kGridCols = 5;
constexpr uint32_t kWorkerNodes = 50;
constexpr uint32_t kTileBytes = 2048;
constexpr uint8_t kPhysicalX[kGridCols] = {1, 2, 3, 4, 5};
FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}  // namespace

// Reader for one-call resident down-projection backward.
//
// The exact phase-one inputs are streamed as before.  The M dH tiles are
// then multicast from their owner cores into a dedicated L1 window on every
// worker.  Both dA and LoRA dProduct consume that window without a DRAM
// round-trip.



namespace {
constexpr uint32_t v_kGridCols = 5;
constexpr uint32_t v_kWorkerNodes = 50;
constexpr uint32_t v_kTileBytes = 2048;
constexpr uint8_t v_kPhysicalX[v_kGridCols] = {1, 2, 3, 4, 5};
FORCE_INLINE uint8_t v_physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}  // namespace


namespace {
constexpr uint8_t v_kAttentionPhysicalX[11] = {
    1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13};
FORCE_INLINE uint64_t v_resident_dy_noc_addr(uint32_t page_id) {
    constexpr uint32_t sequence_tiles = 4;
    constexpr uint32_t output_tiles = 4;
    constexpr uint32_t dimension_tiles = 2;
    constexpr uint32_t query_heads = 14;
    constexpr uint32_t kv_heads = 2;
    constexpr uint32_t group = query_heads / kv_heads;
    const uint32_t token = page_id / output_tiles;
    const uint32_t output = page_id - token * output_tiles;
    const uint32_t owner_head = output / dimension_tiles;
    const uint32_t source_node =
        owner_head * group * sequence_tiles + token;
    const uint32_t source_tile = output - owner_head * dimension_tiles;
    const uint32_t base =
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(13));
    const uint32_t logical_x = 5 + source_node % 6;
    return get_noc_addr(
        v_kAttentionPhysicalX[logical_x],
        v_physical_y(source_node / 6),
        base + source_tile * v_kTileBytes);
}
}  // namespace


static __attribute__((noinline)) void qkv_projection_phase_init() {
    constexpr uint8_t phase_physical_x[5] = {1, 2, 3, 4, 5};
    constexpr uint32_t phase_workers = 50;
    constexpr uint32_t phase_ready_semaphore = 14;
    const uint32_t phase_node =
        get_absolute_logical_y() * 5 + get_absolute_logical_x();
    for (uint32_t semaphore = 0; semaphore < 11; ++semaphore) {
        noc_semaphore_set(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(semaphore)), 0);
    }
    volatile tt_l1_ptr uint32_t* phase_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(phase_ready_semaphore));
    // Semaphore 14 was the output-to-QKV gate (state one).  Every reader
    // cleared its local word before entering here.  A one-producer ring now
    // proves that all 50 reader RISCs finished clearing the short-lived
    // protocol before any writer may consume the V phase (state two).
    if (phase_node == 0) {
        noc_semaphore_inc(
            get_noc_addr(
                phase_physical_x[1], 2,
                get_semaphore(phase_ready_semaphore)), 1);
        noc_async_atomic_barrier();
        noc_semaphore_wait_min(phase_ready, 1);
        noc_semaphore_set(phase_ready, 2);
        for (uint32_t target = 1; target < phase_workers; ++target) {
            noc_async_write(
                get_semaphore(phase_ready_semaphore),
                get_noc_addr(
                    phase_physical_x[target % 5], 2 + target / 5,
                    get_semaphore(phase_ready_semaphore)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    } else {
        noc_semaphore_wait_min(phase_ready, 1);
        const uint32_t next =
            phase_node + 1 == phase_workers ? 0 : phase_node + 1;
        noc_semaphore_inc(
            get_noc_addr(
                phase_physical_x[next % 5], 2 + next / 5,
                get_semaphore(phase_ready_semaphore)), 1);
        noc_async_atomic_barrier();
    }
    noc_semaphore_wait_min(phase_ready, 2);
}

FORCE_INLINE void v_reader_phase() {
    qkv_projection_phase_init();
    Noc noc0(0);

    constexpr uint32_t cb_base_dy = get_compile_time_arg_val(0);
    constexpr uint32_t cb_base_weight = get_compile_time_arg_val(1);
    constexpr uint32_t cb_dh_dy = get_compile_time_arg_val(4);
    constexpr uint32_t cb_dh_b = get_compile_time_arg_val(5);
    constexpr uint32_t cb_dh_output = get_compile_time_arg_val(8);
    constexpr uint32_t cb_db_dy = get_compile_time_arg_val(9);
    constexpr uint32_t cb_db_hidden = get_compile_time_arg_val(10);
    constexpr uint32_t cb_dh_resident = get_compile_time_arg_val(16);
    constexpr uint32_t cb_tail_a = get_compile_time_arg_val(17);
    constexpr uint32_t cb_tail_x = get_compile_time_arg_val(18);

    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t k_tiles = 28;
    constexpr uint32_t n_tiles = 4;
    constexpr uint32_t base_reduction_tiles = 1;
    constexpr uint32_t base_reduction_blocks =
        n_tiles / base_reduction_tiles;
    constexpr uint32_t jobs_per_node = 4;
    constexpr uint32_t da_k_tiles_per_node = 1;
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

    const uint32_t logical_x = get_absolute_logical_x();
    const uint32_t logical_y = get_absolute_logical_y();
    const uint32_t node = logical_y * v_kGridCols + logical_x;
    constexpr uint32_t go_semaphore = m_tiles;
    constexpr uint32_t ready_semaphore_start = m_tiles + 1;

    // Tensor order: dY, W, B, H, A, X.
    const uint32_t weight_addr = get_common_arg_val<uint32_t>(11);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(12);
    const uint32_t checkpoint_arena_addr = get_common_arg_val<uint32_t>(14);
    constexpr uint32_t checkpoint_hidden_page_base = 376;
    const uint32_t a_addr = get_common_arg_val<uint32_t>(13);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(4);

    const auto weight = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<30, 95>(), 11>(), weight_addr, v_kTileBytes);
    const auto b = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<31, 95>(), 12>(), b_addr, v_kTileBytes);
    const auto hidden = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<33, 95>(), 14>(), checkpoint_arena_addr, v_kTileBytes);
    const auto a = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<32, 95>(), 13>(), a_addr, v_kTileBytes);
    const auto x = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<23, 95>(), 4>(), x_addr, v_kTileBytes);

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
                noc_async_read(
                    v_resident_dy_noc_addr(m * n_tiles + n0 + inner),
                    base_dy.get_write_ptr() + inner * v_kTileBytes,
                    v_kTileBytes);
                noc0.async_read(
                    weight,
                    CoreLocalMem<uint32_t>(
                        base_weight.get_write_ptr() + inner * v_kTileBytes),
                    v_kTileBytes,
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
            noc_async_read(
                    v_resident_dy_noc_addr(node * n_tiles + n),
                    dh_dy.get_write_ptr() + n * v_kTileBytes,
                    v_kTileBytes);
            noc0.async_read(
                b,
                CoreLocalMem<uint32_t>(dh_b.get_write_ptr() + n * v_kTileBytes),
                v_kTileBytes, {.page_id = n}, {});
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
        db_hidden.reserve_back(m_tiles);
        for (uint32_t m = 0; m < m_tiles; ++m) {
            noc_async_read(
                    v_resident_dy_noc_addr(m * n_tiles + n),
                    db_dy.get_write_ptr() + m * v_kTileBytes,
                    v_kTileBytes);
            noc0.async_read(
                hidden,
                CoreLocalMem<uint32_t>(db_hidden.get_write_ptr() + m * v_kTileBytes),
                v_kTileBytes, {.page_id = checkpoint_hidden_page_base + m}, {});
        }
        noc0.async_read_barrier();
        db_hidden.push_back(m_tiles);
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
                get_noc_addr(v_kPhysicalX[0], v_physical_y(0), get_semaphore(node)),
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
                    v_kPhysicalX[owner], v_physical_y(0),
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
                dh_resident.get_write_ptr() + node * v_kTileBytes;
            for (uint32_t target = 0; target < v_kWorkerNodes; ++target) {
                noc_async_write(
                    source,
                    get_noc_addr(
                        v_kPhysicalX[target % v_kGridCols],
                        v_physical_y(target / v_kGridCols), destination),
                    v_kTileBytes);
            }
            noc_async_write_barrier();
            volatile tt_l1_ptr uint32_t* local_ready =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(ready_semaphore_start + node));
            noc_semaphore_set(local_ready, 1);
            for (uint32_t target = 0; target < v_kWorkerNodes; ++target) {
                if (target == node) continue;
                noc_async_write(
                    get_semaphore(ready_semaphore_start + node),
                    get_noc_addr(
                        v_kPhysicalX[target % v_kGridCols],
                        v_physical_y(target / v_kGridCols),
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

    // LoRA dProduct uses A once per output tile job.
    for (uint32_t local = 0; local < jobs_per_node; ++local) {
        const uint32_t job = node * jobs_per_node + local;
        if (job >= total_jobs) continue;
        const uint32_t k = job % k_tiles;
        tail_a.reserve_back(1);
        noc0.async_read(
            a, CoreLocalMem<uint32_t>(tail_a.get_write_ptr()),
            v_kTileBytes, {.page_id = k}, {});
        noc0.async_read_barrier();
        tail_a.push_back(1);
    }

    // dA owns complete token reductions and therefore reads M product tiles
    // for each of its (at most two on Qwen) K tiles.
    for (uint32_t local_k = 0; local_k < da_k_tiles_per_node; ++local_k) {
        const uint32_t k = node * da_k_tiles_per_node + local_k;
        if (k >= k_tiles) continue;
        tail_x.reserve_back(m_tiles);
        for (uint32_t m = 0; m < m_tiles; ++m) {
            noc0.async_read(
                x,
                CoreLocalMem<uint32_t>(tail_x.get_write_ptr() + m * v_kTileBytes),
                v_kTileBytes, {.page_id = m * k_tiles + k}, {});
        }
        noc0.async_read_barrier();
        tail_x.push_back(m_tiles);
    }
    volatile tt_l1_ptr uint32_t* phase_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(10));
    noc_semaphore_wait(phase_done, 1);
    noc_semaphore_set(phase_done, 0);

}

// Reader for one-call resident down-projection backward.
//
// The exact phase-one inputs are streamed as before.  The M dH tiles are
// then multicast from their owner cores into a dedicated L1 window on every
// worker.  Both dA and LoRA dProduct consume that window without a DRAM
// round-trip.



namespace {
constexpr uint32_t k_kGridCols = 5;
constexpr uint32_t k_kWorkerNodes = 50;
constexpr uint32_t k_kTileBytes = 2048;
constexpr uint8_t k_kPhysicalX[k_kGridCols] = {1, 2, 3, 4, 5};
FORCE_INLINE uint8_t k_physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}  // namespace


namespace {
constexpr uint8_t k_kAttentionPhysicalX[11] = {
    1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13};
FORCE_INLINE uint64_t k_resident_dy_noc_addr(uint32_t page_id) {
    constexpr uint32_t sequence_tiles = 4;
    constexpr uint32_t output_tiles = 4;
    constexpr uint32_t dimension_tiles = 2;
    constexpr uint32_t query_heads = 14;
    constexpr uint32_t kv_heads = 2;
    constexpr uint32_t group = query_heads / kv_heads;
    const uint32_t token = page_id / output_tiles;
    const uint32_t output = page_id - token * output_tiles;
    const uint32_t owner_head = output / dimension_tiles;
    const uint32_t source_node =
        owner_head * group * sequence_tiles + token;
    const uint32_t source_tile = output - owner_head * dimension_tiles;
    const uint32_t base =
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(12));
    const uint32_t logical_x = 5 + source_node % 6;
    return get_noc_addr(
        k_kAttentionPhysicalX[logical_x],
        k_physical_y(source_node / 6),
        base + source_tile * k_kTileBytes);
}
}  // namespace

FORCE_INLINE void k_reader_phase() {
    Noc noc0(0);

    constexpr uint32_t cb_base_dy = get_compile_time_arg_val(0);
    constexpr uint32_t cb_base_weight = get_compile_time_arg_val(1);
    constexpr uint32_t cb_dh_dy = get_compile_time_arg_val(4);
    constexpr uint32_t cb_dh_b = get_compile_time_arg_val(5);
    constexpr uint32_t cb_dh_output = get_compile_time_arg_val(8);
    constexpr uint32_t cb_db_dy = get_compile_time_arg_val(9);
    constexpr uint32_t cb_db_hidden = get_compile_time_arg_val(10);
    constexpr uint32_t cb_dh_resident = get_compile_time_arg_val(16);
    constexpr uint32_t cb_tail_a = get_compile_time_arg_val(17);
    constexpr uint32_t cb_tail_x = get_compile_time_arg_val(18);

    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t k_tiles = 28;
    constexpr uint32_t n_tiles = 4;
    constexpr uint32_t base_reduction_tiles = 1;
    constexpr uint32_t base_reduction_blocks =
        n_tiles / base_reduction_tiles;
    constexpr uint32_t jobs_per_node = 4;
    constexpr uint32_t da_k_tiles_per_node = 1;
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

    const uint32_t logical_x = get_absolute_logical_x();
    const uint32_t logical_y = get_absolute_logical_y();
    const uint32_t node = logical_y * k_kGridCols + logical_x;
    constexpr uint32_t go_semaphore = m_tiles;
    constexpr uint32_t ready_semaphore_start = m_tiles + 1;

    // Tensor order: dY, W, B, H, A, X.
    const uint32_t weight_addr = get_common_arg_val<uint32_t>(8);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(9);
    const uint32_t checkpoint_arena_addr = get_common_arg_val<uint32_t>(14);
    constexpr uint32_t checkpoint_hidden_page_base = 372;
    const uint32_t a_addr = get_common_arg_val<uint32_t>(10);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(4);

    const auto weight = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<27, 95>(), 8>(), weight_addr, k_kTileBytes);
    const auto b = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<28, 95>(), 9>(), b_addr, k_kTileBytes);
    const auto hidden = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<33, 95>(), 14>(), checkpoint_arena_addr, k_kTileBytes);
    const auto a = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<29, 95>(), 10>(), a_addr, k_kTileBytes);
    const auto x = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<23, 95>(), 4>(), x_addr, k_kTileBytes);

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
                noc_async_read(
                    k_resident_dy_noc_addr(m * n_tiles + n0 + inner),
                    base_dy.get_write_ptr() + inner * k_kTileBytes,
                    k_kTileBytes);
                noc0.async_read(
                    weight,
                    CoreLocalMem<uint32_t>(
                        base_weight.get_write_ptr() + inner * k_kTileBytes),
                    k_kTileBytes,
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
            noc_async_read(
                    k_resident_dy_noc_addr(node * n_tiles + n),
                    dh_dy.get_write_ptr() + n * k_kTileBytes,
                    k_kTileBytes);
            noc0.async_read(
                b,
                CoreLocalMem<uint32_t>(dh_b.get_write_ptr() + n * k_kTileBytes),
                k_kTileBytes, {.page_id = n}, {});
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
        db_hidden.reserve_back(m_tiles);
        for (uint32_t m = 0; m < m_tiles; ++m) {
            noc_async_read(
                    k_resident_dy_noc_addr(m * n_tiles + n),
                    db_dy.get_write_ptr() + m * k_kTileBytes,
                    k_kTileBytes);
            noc0.async_read(
                hidden,
                CoreLocalMem<uint32_t>(db_hidden.get_write_ptr() + m * k_kTileBytes),
                k_kTileBytes, {.page_id = checkpoint_hidden_page_base + m}, {});
        }
        noc0.async_read_barrier();
        db_hidden.push_back(m_tiles);
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
                get_noc_addr(k_kPhysicalX[0], k_physical_y(0), get_semaphore(node)),
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
                    k_kPhysicalX[owner], k_physical_y(0),
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
                dh_resident.get_write_ptr() + node * k_kTileBytes;
            for (uint32_t target = 0; target < k_kWorkerNodes; ++target) {
                noc_async_write(
                    source,
                    get_noc_addr(
                        k_kPhysicalX[target % k_kGridCols],
                        k_physical_y(target / k_kGridCols), destination),
                    k_kTileBytes);
            }
            noc_async_write_barrier();
            volatile tt_l1_ptr uint32_t* local_ready =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(ready_semaphore_start + node));
            noc_semaphore_set(local_ready, 1);
            for (uint32_t target = 0; target < k_kWorkerNodes; ++target) {
                if (target == node) continue;
                noc_async_write(
                    get_semaphore(ready_semaphore_start + node),
                    get_noc_addr(
                        k_kPhysicalX[target % k_kGridCols],
                        k_physical_y(target / k_kGridCols),
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

    // LoRA dProduct uses A once per output tile job.
    for (uint32_t local = 0; local < jobs_per_node; ++local) {
        const uint32_t job = node * jobs_per_node + local;
        if (job >= total_jobs) continue;
        const uint32_t k = job % k_tiles;
        tail_a.reserve_back(1);
        noc0.async_read(
            a, CoreLocalMem<uint32_t>(tail_a.get_write_ptr()),
            k_kTileBytes, {.page_id = k}, {});
        noc0.async_read_barrier();
        tail_a.push_back(1);
    }

    // dA owns complete token reductions and therefore reads M product tiles
    // for each of its (at most two on Qwen) K tiles.
    for (uint32_t local_k = 0; local_k < da_k_tiles_per_node; ++local_k) {
        const uint32_t k = node * da_k_tiles_per_node + local_k;
        if (k >= k_tiles) continue;
        tail_x.reserve_back(m_tiles);
        for (uint32_t m = 0; m < m_tiles; ++m) {
            noc0.async_read(
                x,
                CoreLocalMem<uint32_t>(tail_x.get_write_ptr() + m * k_kTileBytes),
                k_kTileBytes, {.page_id = m * k_tiles + k}, {});
        }
        noc0.async_read_barrier();
        tail_x.push_back(m_tiles);
    }
    volatile tt_l1_ptr uint32_t* phase_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(10));
    noc_semaphore_wait(phase_done, 1);
    noc_semaphore_set(phase_done, 0);

}

// Reader for one-call resident down-projection backward.
//
// The exact phase-one inputs are streamed as before.  The M dH tiles are
// then multicast from their owner cores into a dedicated L1 window on every
// worker.  Both dA and LoRA dProduct consume that window without a DRAM
// round-trip.



namespace {
constexpr uint32_t q_kGridCols = 5;
constexpr uint32_t q_kWorkerNodes = 50;
constexpr uint32_t q_kTileBytes = 2048;
constexpr uint8_t q_kPhysicalX[q_kGridCols] = {1, 2, 3, 4, 5};
FORCE_INLINE uint8_t q_physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}  // namespace


namespace {
constexpr uint8_t q_kAttentionPhysicalX[11] = {
    1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13};
FORCE_INLINE uint64_t q_resident_dy_noc_addr(uint32_t page_id) {
    constexpr uint32_t sequence_tiles = 4;
    constexpr uint32_t output_tiles = 28;
    constexpr uint32_t dimension_tiles = 2;
    constexpr uint32_t query_heads = 14;
    constexpr uint32_t kv_heads = 2;
    constexpr uint32_t group = query_heads / kv_heads;
    const uint32_t token = page_id / output_tiles;
    const uint32_t output = page_id - token * output_tiles;
    const uint32_t owner_head = output / dimension_tiles;
    const uint32_t source_node =
        owner_head * sequence_tiles + token;
    const uint32_t source_tile = output - owner_head * dimension_tiles;
    const uint32_t base =
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(11));
    const uint32_t logical_x = 5 + source_node % 6;
    return get_noc_addr(
        q_kAttentionPhysicalX[logical_x],
        q_physical_y(source_node / 6),
        base + source_tile * q_kTileBytes);
}
}  // namespace

FORCE_INLINE void q_reader_phase() {
    Noc noc0(0);

    constexpr uint32_t cb_base_dy = 45;
    constexpr uint32_t cb_base_weight = 46;
    constexpr uint32_t cb_dh_dy = 32;
    constexpr uint32_t cb_dh_b = 36;
    constexpr uint32_t cb_dh_output = 40;
    constexpr uint32_t cb_db_dy = 33;
    constexpr uint32_t cb_db_hidden = 37;
    constexpr uint32_t cb_dh_resident = 49;
    constexpr uint32_t cb_tail_a = 50;
    constexpr uint32_t cb_tail_x = 51;

    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t k_tiles = 28;
    constexpr uint32_t n_tiles = 28;
    constexpr uint32_t base_reduction_tiles = 1;
    constexpr uint32_t base_reduction_blocks =
        n_tiles / base_reduction_tiles;
    constexpr uint32_t jobs_per_node = 4;
    constexpr uint32_t da_k_tiles_per_node = 1;
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

    const uint32_t logical_x = get_absolute_logical_x();
    const uint32_t logical_y = get_absolute_logical_y();
    const uint32_t node = logical_y * q_kGridCols + logical_x;
    if (node == 0) {
    }
    constexpr uint32_t go_semaphore = m_tiles;
    constexpr uint32_t ready_semaphore_start = m_tiles + 1;

    // Tensor order: dY, W, B, H, A, X.
    const uint32_t weight_addr = get_common_arg_val<uint32_t>(5);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(6);
    const uint32_t checkpoint_arena_addr = get_common_arg_val<uint32_t>(14);
    constexpr uint32_t checkpoint_hidden_page_base = 368;
    const uint32_t a_addr = get_common_arg_val<uint32_t>(7);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(4);

    const auto weight = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<24, 95>(), 5>(), weight_addr, q_kTileBytes);
    const auto b = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<25, 95>(), 6>(), b_addr, q_kTileBytes);
    const auto hidden = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<33, 95>(), 14>(), checkpoint_arena_addr, q_kTileBytes);
    const auto a = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<26, 95>(), 7>(), a_addr, q_kTileBytes);
    const auto x = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<23, 95>(), 4>(), x_addr, q_kTileBytes);

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
                noc_async_read(
                    q_resident_dy_noc_addr(m * n_tiles + n0 + inner),
                    base_dy.get_write_ptr() + inner * q_kTileBytes,
                    q_kTileBytes);
                noc0.async_read(
                    weight,
                    CoreLocalMem<uint32_t>(
                        base_weight.get_write_ptr() + inner * q_kTileBytes),
                    q_kTileBytes,
                    {.page_id = (n0 + inner) * k_tiles + k}, {});
            }
            noc0.async_read_barrier();
            if (node == 0 && local == 0 && block == 0) {
                const volatile tt_l1_ptr uint32_t* q_words =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        base_dy.get_write_ptr());
            }
            base_weight.push_back(base_reduction_tiles);
            base_dy.push_back(base_reduction_tiles);
        }
    }

    // Exact dH owner stream.
    if (node < m_tiles) {
        dh_dy.reserve_back(n_tiles);
        dh_b.reserve_back(n_tiles);
        for (uint32_t n = 0; n < n_tiles; ++n) {
            noc_async_read(
                    q_resident_dy_noc_addr(node * n_tiles + n),
                    dh_dy.get_write_ptr() + n * q_kTileBytes,
                    q_kTileBytes);
            noc0.async_read(
                b,
                CoreLocalMem<uint32_t>(dh_b.get_write_ptr() + n * q_kTileBytes),
                q_kTileBytes, {.page_id = n}, {});
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
        db_hidden.reserve_back(m_tiles);
        for (uint32_t m = 0; m < m_tiles; ++m) {
            noc_async_read(
                    q_resident_dy_noc_addr(m * n_tiles + n),
                    db_dy.get_write_ptr() + m * q_kTileBytes,
                    q_kTileBytes);
            noc0.async_read(
                hidden,
                CoreLocalMem<uint32_t>(db_hidden.get_write_ptr() + m * q_kTileBytes),
                q_kTileBytes, {.page_id = checkpoint_hidden_page_base + m}, {});
        }
        noc0.async_read_barrier();
        db_hidden.push_back(m_tiles);
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
                get_noc_addr(q_kPhysicalX[0], q_physical_y(0), get_semaphore(node)),
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
                    q_kPhysicalX[owner], q_physical_y(0),
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
                dh_resident.get_write_ptr() + node * q_kTileBytes;
            for (uint32_t target = 0; target < q_kWorkerNodes; ++target) {
                noc_async_write(
                    source,
                    get_noc_addr(
                        q_kPhysicalX[target % q_kGridCols],
                        q_physical_y(target / q_kGridCols), destination),
                    q_kTileBytes);
            }
            noc_async_write_barrier();
            volatile tt_l1_ptr uint32_t* local_ready =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(ready_semaphore_start + node));
            noc_semaphore_set(local_ready, 1);
            for (uint32_t target = 0; target < q_kWorkerNodes; ++target) {
                if (target == node) continue;
                noc_async_write(
                    get_semaphore(ready_semaphore_start + node),
                    get_noc_addr(
                        q_kPhysicalX[target % q_kGridCols],
                        q_physical_y(target / q_kGridCols),
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

    // LoRA dProduct uses A once per output tile job.
    for (uint32_t local = 0; local < jobs_per_node; ++local) {
        const uint32_t job = node * jobs_per_node + local;
        if (job >= total_jobs) continue;
        const uint32_t k = job % k_tiles;
        tail_a.reserve_back(1);
        noc0.async_read(
            a, CoreLocalMem<uint32_t>(tail_a.get_write_ptr()),
            q_kTileBytes, {.page_id = k}, {});
        noc0.async_read_barrier();
        tail_a.push_back(1);
    }

    // dA owns complete token reductions and therefore reads M product tiles
    // for each of its (at most two on Qwen) K tiles.
    for (uint32_t local_k = 0; local_k < da_k_tiles_per_node; ++local_k) {
        const uint32_t k = node * da_k_tiles_per_node + local_k;
        if (k >= k_tiles) continue;
        tail_x.reserve_back(m_tiles);
        for (uint32_t m = 0; m < m_tiles; ++m) {
            noc0.async_read(
                x,
                CoreLocalMem<uint32_t>(tail_x.get_write_ptr() + m * q_kTileBytes),
                q_kTileBytes, {.page_id = m * k_tiles + k}, {});
        }
        noc0.async_read_barrier();
        tail_x.push_back(m_tiles);
    }
    volatile tt_l1_ptr uint32_t* phase_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(10));
    noc_semaphore_wait(phase_done, 1);
    noc_semaphore_set(phase_done, 0);

}

void run() {
    {
    Noc noc0(0);

    constexpr uint32_t cb_base_dy = get_compile_time_arg_val(0);
    constexpr uint32_t cb_base_weight = get_compile_time_arg_val(1);
    constexpr uint32_t cb_dh_dy = get_compile_time_arg_val(4);
    constexpr uint32_t cb_dh_b = get_compile_time_arg_val(5);
    constexpr uint32_t cb_dh_output = get_compile_time_arg_val(8);
    constexpr uint32_t cb_db_dy = get_compile_time_arg_val(9);
    constexpr uint32_t cb_db_hidden = get_compile_time_arg_val(10);
    constexpr uint32_t cb_dh_resident = get_compile_time_arg_val(16);
    constexpr uint32_t cb_tail_a = get_compile_time_arg_val(17);
    constexpr uint32_t cb_tail_x = get_compile_time_arg_val(18);

    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t k_tiles = 28;
    constexpr uint32_t n_tiles = 28;
    constexpr uint32_t base_reduction_tiles = 1;
    constexpr uint32_t base_reduction_blocks =
        n_tiles / base_reduction_tiles;
    constexpr uint32_t jobs_per_node = 4;
    constexpr uint32_t da_k_tiles_per_node = 1;
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

    const uint32_t logical_x = get_absolute_logical_x();
    const uint32_t logical_y = get_absolute_logical_y();
    const uint32_t node = logical_y * kGridCols + logical_x;
    constexpr uint32_t go_semaphore = m_tiles;
    constexpr uint32_t ready_semaphore_start = m_tiles + 1;

    // Semaphore addresses may contain state from an unrelated prior program.
    // Reset every flag on every invocation before any producer can publish.
    for (uint32_t semaphore = 0; semaphore < 2 * m_tiles + 1; ++semaphore) {
        volatile tt_l1_ptr uint32_t* flag =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(semaphore));
        noc_semaphore_set(flag, 0);
    }

    // Tensor order: dY, W, B, H, A, X.
    const uint32_t dy_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t weight_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t checkpoint_arena_addr = get_common_arg_val<uint32_t>(14);
    constexpr uint32_t checkpoint_hidden_page_base = 380;
    const uint32_t a_addr = get_common_arg_val<uint32_t>(3);
    constexpr uint32_t checkpoint_head_page_base = 256;
    constexpr uint32_t checkpoint_sequence_tiles = 4;
    constexpr uint32_t checkpoint_dimension_tiles = 2;

    const auto dy = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<17, 95>(), 0>(), dy_addr, kTileBytes);
    const auto weight = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<18, 95>(), 1>(), weight_addr, kTileBytes);
    const auto b = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<19, 95>(), 2>(), b_addr, kTileBytes);
    const auto hidden = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<33, 95>(), 14>(), checkpoint_arena_addr, kTileBytes);
    const auto a = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<20, 95>(), 3>(), a_addr, kTileBytes);
    const auto x = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<33, 95>(), 14>(), checkpoint_arena_addr, kTileBytes);

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
        db_hidden.reserve_back(m_tiles);
        for (uint32_t m = 0; m < m_tiles; ++m) {
            noc0.async_read(
                dy,
                CoreLocalMem<uint32_t>(db_dy.get_write_ptr() + m * kTileBytes),
                kTileBytes, {.page_id = m * n_tiles + n}, {});
            noc0.async_read(
                hidden,
                CoreLocalMem<uint32_t>(db_hidden.get_write_ptr() + m * kTileBytes),
                kTileBytes, {.page_id = checkpoint_hidden_page_base + m}, {});
        }
        noc0.async_read_barrier();
        db_hidden.push_back(m_tiles);
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
        tail_x.reserve_back(m_tiles);
        for (uint32_t m = 0; m < m_tiles; ++m) {
            noc0.async_read(
                x,
                CoreLocalMem<uint32_t>(tail_x.get_write_ptr() + m * kTileBytes),
                kTileBytes, {.page_id = checkpoint_head_page_base +
                    (k / checkpoint_dimension_tiles) *
                        checkpoint_sequence_tiles *
                        checkpoint_dimension_tiles +
                    m * checkpoint_dimension_tiles +
                    (k % checkpoint_dimension_tiles)}, {});
        }
        noc0.async_read_barrier();
        tail_x.push_back(m_tiles);
    }
    }
    {
    volatile tt_l1_ptr uint32_t* output_phase_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(15));
    noc_semaphore_wait_min(output_phase_done, 2);

    // The output writers have already completed a 50-core producer-side ring
    // and left semaphore 15 at state two on every projection core.  Node zero
    // can therefore publish the next-phase gate without polling remote state
    // or racing a writer that still owns semaphore 14.
    constexpr uint8_t entry_physical_x[11] = {
        1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13};
    constexpr uint32_t entry_projection_workers = 50;
    constexpr uint32_t entry_attention_workers = 56;
    const uint32_t entry_node =
        get_absolute_logical_y() * 5 + get_absolute_logical_x();
    constexpr uint32_t entry_start_semaphore = 14;
    volatile tt_l1_ptr uint32_t* entry_start =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(entry_start_semaphore));
    if (entry_node == 0) {
        noc_semaphore_set(entry_start, 1);
        for (uint32_t target = 1; target < entry_projection_workers; ++target) {
            noc_async_write(
                get_semaphore(entry_start_semaphore),
                get_noc_addr(
                    entry_physical_x[target % 5], 2 + target / 5,
                    get_semaphore(entry_start_semaphore)),
                sizeof(uint32_t));
        }
        // Attention semaphore 14 belonged to the completed dO producer-go
        // protocol and is now the Q/K/V publication gate.
        for (uint32_t target = 0; target < entry_attention_workers; ++target) {
            const uint32_t logical_x = 5 + target % 6;
            noc_async_write(
                get_semaphore(entry_start_semaphore),
                get_noc_addr(
                    entry_physical_x[logical_x], 2 + target / 6,
                    get_semaphore(14)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    noc_semaphore_wait(entry_start, 1);
    noc_semaphore_set(entry_start, 0);
    constexpr uint32_t kProjectionWorkers = 50;
    constexpr uint32_t kActiveAttentionNodes =
        14 * 4;
    constexpr uint8_t kProjectionPhysicalX[5] = {1, 2, 3, 4, 5};
    const uint32_t node =
        get_absolute_logical_y() * 5 + get_absolute_logical_x();
    volatile tt_l1_ptr uint32_t* attention_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    volatile tt_l1_ptr uint32_t* handoff_go =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(4));
    if (node == 0) {
        noc_semaphore_wait_min(attention_ready, kActiveAttentionNodes);
        noc_semaphore_set(attention_ready, 0);
        for (uint32_t target = 1; target < kProjectionWorkers; ++target) {
            for (uint32_t word = 11; word <= 13; ++word) {
                noc_async_write(
                    get_semaphore(word),
                    get_noc_addr(
                        kProjectionPhysicalX[target % 5],
                        2 + target / 5, get_semaphore(word)),
                    sizeof(uint32_t));
            }
        }
        noc_async_write_barrier();
        noc_semaphore_set(handoff_go, 1);
        for (uint32_t target = 1; target < kProjectionWorkers; ++target) {
            noc_async_write(
                get_semaphore(4),
                get_noc_addr(
                    kProjectionPhysicalX[target % 5],
                    2 + target / 5, get_semaphore(4)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    noc_semaphore_wait(handoff_go, 1);
    noc_semaphore_set(handoff_go, 0);
    v_reader_phase();
    k_reader_phase();
    q_reader_phase();
    }
}
}  // namespace resident_layer_phase_attention_0

void kernel_main() {
    resident_layer_phase_attention_0::run();
}
