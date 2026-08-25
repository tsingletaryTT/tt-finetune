// Reader/NoC scheduler for the resident exact SwiGLU + down projection.

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
constexpr uint32_t kCoreCount = 110;
constexpr uint32_t kOutputCoreCount = 72;
constexpr uint32_t kProductCoreStart = kOutputCoreCount;
constexpr uint32_t kProductCoreCount = kCoreCount - kOutputCoreCount;
constexpr uint32_t kTileBytes = 2048;
constexpr uint8_t kPhysicalX[kGridCols] = {
    1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13};
FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
FORCE_INLINE uint64_t core_l1_addr(uint32_t node, uint32_t local_addr) {
    return get_noc_addr(
        kPhysicalX[node % kGridCols],
        physical_y(node / kGridCols),
        local_addr);
}

// Artifact-v4 executes several physical programs without recreating the CB
// hardware state between them.  This reader owns the producer view of all
// input/relay FIFOs below.  Rebase an empty inherited epoch before publishing
// the first tile so the new CB allocation's base and depth are authoritative.
static __attribute__((noinline, noclone)) void
rebase_empty_producer_cb(uint32_t cb_id) {
    volatile uint32_t* received = get_cb_tiles_received_ptr(cb_id);
    const uintptr_t ack_address =
        reinterpret_cast<uintptr_t>(get_cb_tiles_acked_ptr(cb_id));
    const uint16_t pushed = static_cast<uint16_t>(received[0]);
    while (true) {
        invalidate_l1_cache();
        if (static_cast<uint16_t>(reg_read(ack_address)) == pushed) break;
    }
    *get_cb_tiles_received_ptr(cb_id) = 0;
    *get_cb_tiles_acked_ptr(cb_id) = 0;
    auto& interface = get_local_cb_interface(cb_id);
    const uint32_t base = interface.fifo_limit - interface.fifo_size;
    interface.fifo_rd_ptr = base;
    interface.fifo_wr_ptr = base;
    interface.tiles_acked_received_init = 0;
    interface.fifo_wr_tile_ptr = 0;
}
}  // namespace

void kernel_main() {
    Noc noc0(0);
    UnicastEndpoint self_endpoint;
    constexpr uint32_t cb_gate = get_compile_time_arg_val(0);
    constexpr uint32_t cb_up = get_compile_time_arg_val(1);
    constexpr uint32_t cb_product_out = get_compile_time_arg_val(3);
    constexpr uint32_t cb_product_resident = get_compile_time_arg_val(4);
    constexpr uint32_t cb_a = get_compile_time_arg_val(5);
    constexpr uint32_t cb_h_mcast = get_compile_time_arg_val(7);
    constexpr uint32_t cb_h_resident = get_compile_time_arg_val(8);
    constexpr uint32_t cb_w = get_compile_time_arg_val(10);
    constexpr uint32_t cb_b = get_compile_time_arg_val(13);

    constexpr uint32_t m_tiles = @M_TILES@;
    constexpr uint32_t h_tiles = @H_TILES@;
    constexpr uint32_t i_tiles = @I_TILES@;
    constexpr uint32_t rank_tiles = @RANK_TILES@;
    constexpr uint32_t h_k_block_tiles = @H_K_BLOCK_TILES@;
    constexpr uint32_t product_jobs = @PRODUCT_JOBS@;
    constexpr uint32_t product_jobs_per_core = @PRODUCT_JOBS_PER_CORE@;
    constexpr uint32_t output_jobs = @OUTPUT_JOBS@;
    constexpr uint32_t output_jobs_per_core = @OUTPUT_JOBS_PER_CORE@;
    constexpr uint32_t base_k_block_tiles = @BASE_K_BLOCK_TILES@;
    constexpr bool decomposed_activation = @DECOMPOSED_ACTIVATION@;

    CircularBuffer gate_cb(cb_gate);
    CircularBuffer up_cb(cb_up);
    CircularBuffer product_out(cb_product_out);
    CircularBuffer product_resident(cb_product_resident);
    CircularBuffer a_cb(cb_a);
    CircularBuffer h_mcast(cb_h_mcast);
    CircularBuffer h_resident(cb_h_resident);
    CircularBuffer w_cb(cb_w);
    CircularBuffer b_cb(cb_b);

    constexpr uint32_t producer_cbs[] = {
        cb_gate, cb_up, cb_product_resident, cb_a,
        cb_h_resident, cb_w, cb_b};
    for (uint32_t cb_id : producer_cbs) rebase_empty_producer_cb(cb_id);

    const uint32_t node =
        get_absolute_logical_y() * kGridCols + get_absolute_logical_x();
    volatile tt_l1_ptr uint32_t* product_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    volatile tt_l1_ptr uint32_t* h_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    // Semaphores are L1 allocations and can contain state left by a previous
    // program. Producer signals cannot occur until the compute phase has
    // consumed reader-fed gate/up tiles, so this reader-owned reset precedes
    // every signal in the invocation.
    noc_semaphore_set(product_sem, 0);
    noc_semaphore_set(h_sem, 0);

    const uint32_t gate_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t up_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t weight_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t a_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(4);
    const auto gate = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 18>(), 0>(),
        gate_addr, kTileBytes);
    const auto up = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 18>(), 1>(),
        up_addr, kTileBytes);
    const auto weight = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 18>(), 2>(),
        weight_addr, kTileBytes);
    const auto a = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<3, 18>(), 3>(),
        a_addr, kTileBytes);
    const auto b = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<4, 18>(), 4>(),
        b_addr, kTileBytes);

    uint32_t valid_products = 0;
    for (uint32_t local = 0; local < product_jobs_per_core; ++local) {
        if (node < kProductCoreStart) {
            continue;
        }
        const uint32_t job =
            (node - kProductCoreStart) + local * kProductCoreCount;
        if (job >= product_jobs) {
            continue;
        }
        ++valid_products;
        constexpr uint32_t gate_copies = decomposed_activation ? 2 : 1;
        gate_cb.reserve_back(gate_copies);
        up_cb.reserve_back(1);
        noc0.async_read(
            gate, CoreLocalMem<uint32_t>(gate_cb.get_write_ptr()),
            kTileBytes, {.page_id = @GATE_PAGE_ID@}, {});
        if constexpr (decomposed_activation) {
            noc0.async_read(
                gate,
                CoreLocalMem<uint32_t>(
                    gate_cb.get_write_ptr() + kTileBytes),
                kTileBytes, {.page_id = @GATE_PAGE_ID@}, {});
        }
        noc0.async_read(
            up, CoreLocalMem<uint32_t>(up_cb.get_write_ptr()),
            kTileBytes, {.page_id = @UP_PAGE_ID@}, {});
        noc0.async_read_barrier();
        up_cb.push_back(1);
        gate_cb.push_back(gate_copies);
    }

    uint32_t valid_outputs = 0;
    for (uint32_t local = 0; local < output_jobs_per_core; ++local) {
        if (node < kOutputCoreCount &&
            node + local * kOutputCoreCount < output_jobs) {
            ++valid_outputs;
        }
    }
    if (valid_outputs != 0) {
        product_resident.reserve_back(valid_outputs * i_tiles);
        h_resident.reserve_back(valid_outputs * rank_tiles);
    }

    // Each product producer sends its exact BF16 tile directly into the row
    // window of every down-output owner. Payload writes are flushed before
    // the atomic credit so consumers never observe a partially written tile.
    product_out.wait_front(valid_products);
    uint32_t row_counts[m_tiles] = {};
    for (uint32_t local = 0; local < product_jobs_per_core; ++local) {
        if (node < kProductCoreStart) {
            continue;
        }
        const uint32_t job =
            (node - kProductCoreStart) + local * kProductCoreCount;
        if (job >= product_jobs) {
            continue;
        }
        const uint32_t m = job / i_tiles;
        const uint32_t i = job - m * i_tiles;
        ++row_counts[m];
        const uint32_t source = product_out.get_read_ptr() + local * kTileBytes;
        for (uint32_t h = 0; h < h_tiles; ++h) {
            const uint32_t output_job = m * h_tiles + h;
            const uint32_t target = output_job % kOutputCoreCount;
            const uint32_t slot = output_job / kOutputCoreCount;
            const uint32_t destination =
                product_resident.get_write_ptr()
                + (slot * i_tiles + i) * kTileBytes;
                noc_async_write(
                    source, core_l1_addr(target, destination), kTileBytes);
        }
    }
    if (valid_products != 0) {
        DeviceZoneScopedN("resident_mlp_product_noc");
        noc_async_write_barrier();
        for (uint32_t m = 0; m < m_tiles; ++m) {
            if (row_counts[m] == 0) {
                continue;
            }
            for (uint32_t h = 0; h < h_tiles; ++h) {
                const uint32_t output_job = m * h_tiles + h;
                const uint32_t target = output_job % kOutputCoreCount;
                noc_semaphore_inc(
                    core_l1_addr(target, get_semaphore(0)), row_counts[m]);
            }
        }
        noc_async_atomic_barrier();
        product_out.pop_front(valid_products);
    }
    if (valid_outputs != 0) {
        noc_semaphore_wait_min(product_sem, valid_outputs * i_tiles);
        noc_semaphore_set(product_sem, 0);
        product_resident.push_back(valid_outputs * i_tiles);
    }

    // The n=0 owner computes H once for each token row. Feed its exact
    // two-tile reduction, then distribute the one rank tile to all output
    // owners instead of repeating the long-I reduction H times.  Reuse the
    // now-empty gate FIFO as the canonical streaming X operand.  Addressing
    // the large resident product CB in-place changes Blackhole's matmul
    // unpack schedule for data-sensitive inputs; the accepted standalone
    // kernel instead advances a phase-sized FIFO and indexes each phase from
    // zero.  This local L1 copy preserves that exact schedule without a DRAM
    // round trip.
    for (uint32_t local = 0; local < output_jobs_per_core; ++local) {
        if (node >= kOutputCoreCount) {
            continue;
        }
        const uint32_t job = node + local * kOutputCoreCount;
        if (job >= output_jobs || job % h_tiles != 0) {
            continue;
        }
        for (uint32_t i = 0; i < i_tiles; i += h_k_block_tiles) {
            gate_cb.reserve_back(h_k_block_tiles);
            noc0.async_read(
                self_endpoint,
                CoreLocalMem<uint32_t>(gate_cb.get_write_ptr()),
                h_k_block_tiles * kTileBytes,
                {
                    .noc_x = kPhysicalX[node % kGridCols],
                    .noc_y = physical_y(node / kGridCols),
                    .addr = product_resident.get_read_ptr()
                        + (local * i_tiles + i) * kTileBytes,
                },
                {});
            a_cb.reserve_back(h_k_block_tiles * rank_tiles);
            for (uint32_t inner = 0; inner < h_k_block_tiles; ++inner) {
                noc0.async_read(
                    a,
                    CoreLocalMem<uint32_t>(
                        a_cb.get_write_ptr() + inner * kTileBytes),
                    kTileBytes, {.page_id = i + inner}, {});
            }
            noc0.async_read_barrier();
            gate_cb.push_back(h_k_block_tiles);
            a_cb.push_back(h_k_block_tiles * rank_tiles);
        }
        h_mcast.wait_front(rank_tiles);
        const uint32_t m = job / h_tiles;
        for (uint32_t h = 0; h < h_tiles; ++h) {
            const uint32_t output_job = m * h_tiles + h;
            const uint32_t target = output_job % kOutputCoreCount;
            const uint32_t slot = output_job / kOutputCoreCount;
            noc_async_write(
                h_mcast.get_read_ptr(),
                core_l1_addr(
                    target,
                    h_resident.get_write_ptr()
                        + slot * rank_tiles * kTileBytes),
                rank_tiles * kTileBytes);
        }
        noc_async_write_barrier();
        for (uint32_t h = 0; h < h_tiles; ++h) {
            const uint32_t target =
                (m * h_tiles + h) % kOutputCoreCount;
            noc_semaphore_inc(core_l1_addr(target, get_semaphore(1)), 1);
        }
        noc_async_atomic_barrier();
        h_mcast.pop_front(rank_tiles);
    }
    if (valid_outputs != 0) {
        noc_semaphore_wait_min(h_sem, valid_outputs);
        noc_semaphore_set(h_sem, 0);
        h_resident.push_back(valid_outputs * rank_tiles);
    }

    // Stream the resident product through the now-empty up FIFO alongside
    // weights.  As above, the phase FIFO is part of the exact TTNN numerical
    // contract: each K phase begins at tile index zero and is popped before
    // the next phase.
    for (uint32_t local = 0; local < output_jobs_per_core; ++local) {
        if (node >= kOutputCoreCount) {
            continue;
        }
        const uint32_t job = node + local * kOutputCoreCount;
        if (job >= output_jobs) {
            continue;
        }
        const uint32_t h = job % h_tiles;
        for (uint32_t i = 0; i < i_tiles; i += base_k_block_tiles) {
            up_cb.reserve_back(base_k_block_tiles);
            noc0.async_read(
                self_endpoint,
                CoreLocalMem<uint32_t>(up_cb.get_write_ptr()),
                base_k_block_tiles * kTileBytes,
                {
                    .noc_x = kPhysicalX[node % kGridCols],
                    .noc_y = physical_y(node / kGridCols),
                    .addr = product_resident.get_read_ptr()
                        + (local * i_tiles + i) * kTileBytes,
                },
                {});
            w_cb.reserve_back(base_k_block_tiles);
            for (uint32_t inner = 0; inner < base_k_block_tiles; ++inner) {
                noc0.async_read(
                    weight,
                    CoreLocalMem<uint32_t>(
                        w_cb.get_write_ptr() + inner * kTileBytes),
                    kTileBytes, {.page_id = (i + inner) * h_tiles + h}, {});
            }
            noc0.async_read_barrier();
            up_cb.push_back(base_k_block_tiles);
            w_cb.push_back(base_k_block_tiles);
        }
        b_cb.reserve_back(rank_tiles);
        noc0.async_read(
            b, CoreLocalMem<uint32_t>(b_cb.get_write_ptr()),
            rank_tiles * kTileBytes, {.page_id = h}, {});
        noc0.async_read_barrier();
        b_cb.push_back(rank_tiles);
    }
}
