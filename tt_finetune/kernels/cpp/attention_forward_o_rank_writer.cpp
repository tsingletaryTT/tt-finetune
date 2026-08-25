// Publish the exact O-LoRA H tile directly to O workers and to the arena
// checkpoint required by backward. No rank-hidden PJRT tensor is created.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kHiddenTiles = @HIDDEN_TILES@;
constexpr uint32_t kSequenceTiles = @SEQ_TILES@;
constexpr uint32_t kOWorkers = @O_WORKERS@;
constexpr uint32_t kOJobs = kSequenceTiles * kHiddenTiles;
constexpr uint32_t kJobsPerWorker = (kOJobs + kOWorkers - 1) / kOWorkers;
constexpr uint32_t kProjectionColumns = 5;
constexpr uint8_t kPhysicalX[11] = {1,2,3,4,5,6,7,10,11,12,13};

FORCE_INLINE uint8_t physical_y(uint32_t y) {
    return static_cast<uint8_t>(y + 2);
}

FORCE_INLINE uint64_t projection_l1(uint32_t node, uint32_t address) {
    return get_noc_addr(
        kPhysicalX[node % kProjectionColumns],
        physical_y(node / kProjectionColumns), address);
}
}

void kernel_main() {
    const uint32_t row = get_absolute_logical_x();
    if (row >= kSequenceTiles) return;

    volatile tt_l1_ptr uint32_t* destination_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
    noc_semaphore_wait(destination_ready, 1);
    noc_semaphore_set(destination_ready, 0);
    const uint32_t destination_base =
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));

    // Match TTNN's canonical mcast-1D matmul output port. The paired FP32
    // partial port is CB5 and is consumed entirely by the compute thread.
    CircularBuffer h(4);
    h.wait_front(1);
    Noc noc1(1);
    for (uint32_t output = 0; output < kHiddenTiles; ++output) {
        const uint32_t job = row * kHiddenTiles + output;
        const uint32_t target = job / kJobsPerWorker;
        const uint32_t slot = job - target * kJobsPerWorker;
        // The typed Noc API intentionally accepts tensor/core-local endpoint
        // wrappers, not a pre-composed raw NoC address.  This direct L1 edge
        // therefore uses Metalium's raw-address dataflow primitive.
        noc_async_write(
            h.get_read_ptr(),
            projection_l1(
                target, destination_base + slot * kTileBytes),
            kTileBytes);
    }
    noc1.async_write_barrier();
    for (uint32_t output = 0; output < kHiddenTiles; ++output) {
        const uint32_t target =
            (row * kHiddenTiles + output) / kJobsPerWorker;
        noc_semaphore_inc(
            projection_l1(target, get_semaphore(11)), 1);
    }
    noc_async_atomic_barrier();

    const auto arena = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                0, @TOTAL_CBS@>(), 0>(),
        get_common_arg_val<uint32_t>(0), kTileBytes);
    noc1.async_write(
        CoreLocalMem<uint32_t>(h.get_read_ptr()), arena, kTileBytes,
        {}, {.page_id = @HO_BASE@ + row});
    noc1.async_write_barrier();
    h.pop_front(1);
}
