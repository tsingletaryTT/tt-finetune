// Device-resident reader for two equal-shape frozen Linear + LoRA members.
//
// Phase one is loaded exactly as in the accepted concurrent 55+55-core path.
// Its dH result is then distributed from the four owning cores to every core
// in the same 55-core half through NoC writes into CB4.  No phase-one result
// is materialized in DRAM.

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
constexpr uint32_t kMemberNodes = 55;
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
    constexpr uint32_t cb_base_result = get_compile_time_arg_val(3);
    constexpr uint32_t cb_dh_dy = get_compile_time_arg_val(4);
    constexpr uint32_t cb_dh_b = get_compile_time_arg_val(5);
    constexpr uint32_t cb_tail_a = get_compile_time_arg_val(5);
    // Dedicated M-tile L1 window: reader is its only producer and tail
    // compute is its only consumer. Reusing a phase CB would either mismatch
    // geometry (CB4 is N-wide) or cross RISC producer ownership (CB12).
    constexpr uint32_t cb_dh_resident = get_compile_time_arg_val(16);
    constexpr uint32_t cb_dh_output = get_compile_time_arg_val(8);
    constexpr uint32_t cb_peer_sum = get_compile_time_arg_val(18);
    constexpr uint32_t cb_db_dy = get_compile_time_arg_val(9);
    constexpr uint32_t cb_db_hidden = get_compile_time_arg_val(10);
    constexpr uint32_t cb_tail_x = get_compile_time_arg_val(9);

    constexpr uint32_t m_tiles = @M_TILES@;
    constexpr uint32_t k_tiles = @K_TILES@;
    constexpr uint32_t n_tiles = @N_TILES@;
    constexpr uint32_t base_reduction_tiles = @BASE_REDUCTION_TILES@;
    constexpr uint32_t base_reduction_blocks =
        n_tiles / base_reduction_tiles;
    constexpr uint32_t jobs_per_node = @JOBS_PER_NODE@;
    constexpr uint32_t n_tiles_per_node = @N_TILES_PER_NODE@;
    constexpr uint32_t total_jobs = m_tiles * k_tiles;

    CircularBuffer base_dy(cb_base_dy);
    CircularBuffer base_weight(cb_base_weight);
    CircularBuffer dh_dy(cb_dh_dy);
    CircularBuffer dh_b(cb_dh_b);
    CircularBuffer dh_resident(cb_dh_resident);
    CircularBuffer dh_output(cb_dh_output);
    CircularBuffer tail_a(cb_tail_a);
    CircularBuffer tail_x(cb_tail_x);
    CircularBuffer peer_sum(cb_peer_sum);
    CircularBuffer db_dy(cb_db_dy);
    CircularBuffer db_hidden(cb_db_hidden);

    const uint32_t logical_x = get_absolute_logical_x();
    const uint32_t logical_y = get_absolute_logical_y();
    const uint32_t global_node = logical_y * kGridCols + logical_x;
    const uint32_t member = global_node / kMemberNodes;
    const uint32_t node = global_node - member * kMemberNodes;
    const uint32_t group_y = member * 5;
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
    const uint32_t dy0_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w0_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t b0_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t h0_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t a0_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t dy1_addr = get_common_arg_val<uint32_t>(5);
    const uint32_t w1_addr = get_common_arg_val<uint32_t>(6);
    const uint32_t b1_addr = get_common_arg_val<uint32_t>(7);
    const uint32_t h1_addr = get_common_arg_val<uint32_t>(8);
    const uint32_t a1_addr = get_common_arg_val<uint32_t>(9);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(10);

    const auto dy0 = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 19>(),
            0>(),
        dy0_addr,
        kTileBytes);
    const auto w0 = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 19>(),
            1>(),
        w0_addr,
        kTileBytes);
    const auto b0 = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 19>(),
            2>(),
        b0_addr,
        kTileBytes);
    const auto h0 = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<3, 19>(),
            3>(),
        h0_addr,
        kTileBytes);
    const auto a0 = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<4, 19>(),
            4>(),
        a0_addr,
        kTileBytes);
    const auto dy1 = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<5, 19>(),
            5>(),
        dy1_addr,
        kTileBytes);
    const auto w1 = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<6, 19>(),
            6>(),
        w1_addr,
        kTileBytes);
    const auto b1 = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<7, 19>(),
            7>(),
        b1_addr,
        kTileBytes);
    const auto h1 = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<8, 19>(),
            8>(),
        h1_addr,
        kTileBytes);
    const auto a1 = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<9, 19>(),
            9>(),
        a1_addr,
        kTileBytes);
    const auto x = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<10, 19>(),
            10>(),
        x_addr,
        kTileBytes);

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
                noc0.async_read(
                    dy,
                    CoreLocalMem<uint32_t>(
                        base_dy.get_write_ptr() + inner * kTileBytes),
                    kTileBytes,
                    {.page_id = m * n_tiles + n0 + inner},
                    {});
                noc0.async_read(
                    weight,
                    CoreLocalMem<uint32_t>(
                        base_weight.get_write_ptr() + inner * kTileBytes),
                    kTileBytes,
                    {.page_id = (n0 + inner) * k_tiles + k},
                    {});
            }
            noc0.async_read_barrier();
            base_weight.push_back(base_reduction_tiles);
            base_dy.push_back(base_reduction_tiles);
        }
    }

    if (node < m_tiles) {
        dh_dy.reserve_back(n_tiles);
        dh_b.reserve_back(n_tiles);
        for (uint32_t n = 0; n < n_tiles; ++n) {
            noc0.async_read(
                dy,
                CoreLocalMem<uint32_t>(
                    dh_dy.get_write_ptr() + n * kTileBytes),
                kTileBytes,
                {.page_id = node * n_tiles + n},
                {});
            noc0.async_read(
                b,
                CoreLocalMem<uint32_t>(
                    dh_b.get_write_ptr() + n * kTileBytes),
                kTileBytes,
                {.page_id = n},
                {});
        }
        noc0.async_read_barrier();
        dh_b.push_back(n_tiles);
        dh_dy.push_back(n_tiles);
    }

    for (uint32_t local_n = 0; local_n < n_tiles_per_node; ++local_n) {
        const uint32_t n_tile = node * n_tiles_per_node + local_n;
        if (n_tile >= n_tiles) {
            continue;
        }
        db_dy.reserve_back(m_tiles);
        db_hidden.reserve_back(m_tiles);
        for (uint32_t m = 0; m < m_tiles; ++m) {
            noc0.async_read(
                dy,
                CoreLocalMem<uint32_t>(
                    db_dy.get_write_ptr() + m * kTileBytes),
                kTileBytes,
                {.page_id = m * n_tiles + n_tile},
                {});
            noc0.async_read(
                hidden,
                CoreLocalMem<uint32_t>(
                    db_hidden.get_write_ptr() + m * kTileBytes),
                kTileBytes,
                {.page_id = m},
                {});
        }
        noc0.async_read_barrier();
        db_hidden.push_back(m_tiles);
        db_dy.push_back(m_tiles);
    }

    // Reserve the dedicated M-tile resident dH window.
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
                get_noc_addr(
                kPhysicalX[0],
                physical_y(group_y),
                    get_semaphore(node)),
                sizeof(uint32_t));
            noc_async_write_barrier();
            noc_semaphore_set(owner_flag, 0);
        }
    }
    if (node == 0) {
        DeviceZoneScopedN("resident_dh_barrier");
        noc_semaphore_set(go_sem, 1);
        for (uint32_t owner = 0; owner < m_tiles; ++owner) {
            volatile tt_l1_ptr uint32_t* ready_flag =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(owner));
            noc_semaphore_wait(ready_flag, 1);
            noc_semaphore_set(ready_flag, 0);
        }
        for (uint32_t owner = 1; owner < m_tiles; ++owner) {
            noc_async_write(
                get_semaphore(4),
                get_noc_addr(
                    kPhysicalX[owner],
                    physical_y(group_y),
                    get_semaphore(4)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }

    if (node < m_tiles) {
        noc_semaphore_wait(go_sem, 1);
        noc_semaphore_set(go_sem, 0);
        {
            DeviceZoneScopedN("resident_dh_noc");
            const uint32_t source = dh_output.get_read_ptr();
            const uint32_t destination =
                dh_resident.get_write_ptr() + node * kTileBytes;
            for (uint32_t target = 0; target < kMemberNodes; ++target) {
                const uint32_t tx = target % kGridCols;
                const uint32_t ty = group_y + target / kGridCols;
                noc_async_write(
                    source,
                    get_noc_addr(
                        kPhysicalX[tx],
                        physical_y(ty),
                        destination),
                    kTileBytes);
            }
            // Publish readiness only after every target's tile is complete.
            // A flush merely guarantees issue; it does not prevent the
            // following semaphore atomic from becoming visible first.
            noc_async_write_barrier();
            volatile tt_l1_ptr uint32_t* owner_flag =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(5 + node));
            noc_semaphore_set(owner_flag, 1);
            for (uint32_t target = 0; target < kMemberNodes; ++target) {
                if (target == node) {
                    continue;
                }
                const uint32_t tx = target % kGridCols;
                const uint32_t ty = group_y + target / kGridCols;
                noc_async_write(
                    get_semaphore(5 + node),
                    get_noc_addr(
                        kPhysicalX[tx],
                        physical_y(ty),
                        get_semaphore(5 + node)),
                    sizeof(uint32_t));
            }
            noc_async_write_barrier();
        }
        dh_output.pop_front(1);
    }

    {
        DeviceZoneScopedN("resident_dh_wait");
        for (uint32_t owner = 0; owner < m_tiles; ++owner) {
            volatile tt_l1_ptr uint32_t* ready_flag =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(5 + owner));
            noc_semaphore_wait(ready_flag, 1);
            noc_semaphore_set(ready_flag, 0);
        }
    }
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
            kTileBytes,
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
                    tail_x.get_write_ptr() + m * kTileBytes),
                kTileBytes,
                {.page_id = m * k_tiles + node},
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
                    kPhysicalX[logical_x],
                    physical_y(peer_y),
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
}
