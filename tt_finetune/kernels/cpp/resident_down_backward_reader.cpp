// Reader for one-call resident down-projection backward.
//
// The exact phase-one inputs are streamed as before.  The M dH tiles are
// then multicast from their owner cores into a dedicated L1 window on every
// worker.  Both dA and LoRA dProduct consume that window without a DRAM
// round-trip.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace {
constexpr uint32_t kGridCols = 11;
constexpr uint32_t kWorkerNodes = 110;
constexpr uint32_t kTileBytes = 2048;
constexpr uint8_t kPhysicalX[kGridCols] = {
    1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13};
FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}  // namespace

void kernel_main() {
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

    constexpr uint32_t m_tiles = @M_TILES@;
    constexpr uint32_t k_tiles = @K_TILES@;
    constexpr uint32_t n_tiles = @N_TILES@;
    constexpr uint32_t base_reduction_tiles = @BASE_REDUCTION_TILES@;
    constexpr uint32_t base_reduction_blocks =
        n_tiles / base_reduction_tiles;
    constexpr uint32_t jobs_per_node = @JOBS_PER_NODE@;
    constexpr uint32_t da_k_tiles_per_node = @DA_K_TILES_PER_NODE@;
    constexpr uint32_t n_tiles_per_node = @N_TILES_PER_NODE@;
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
    const uint32_t hidden_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t a_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(5);

    const auto dy = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 25>(),
            0>(), dy_addr, kTileBytes);
    const auto weight = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 25>(),
            1>(), weight_addr, kTileBytes);
    const auto b = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 25>(),
            2>(), b_addr, kTileBytes);
    const auto hidden = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<3, 25>(),
            3>(), hidden_addr, kTileBytes);
    const auto a = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<4, 25>(),
            4>(), a_addr, kTileBytes);
    const auto x = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<5, 25>(),
            5>(), x_addr, kTileBytes);

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
                kTileBytes, {.page_id = m}, {});
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
                kTileBytes, {.page_id = m * k_tiles + k}, {});
        }
        noc0.async_read_barrier();
        tail_x.push_back(m_tiles);
    }
}
