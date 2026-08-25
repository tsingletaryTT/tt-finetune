// Dedicated O-LoRA H producer reader for the resident attention owner.
//
// Four cores own the four sequence-row tiles. Attention writers place the
// complete head row directly in each producer's L1 CB0. This reader streams
// the canonical, untransposed LoRA A tile row into CB1 and publishes the
// resident head-input address only after every producer has reserved its
// complete input window.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kHiddenTiles = @HIDDEN_TILES@;
constexpr uint32_t kSequenceTiles = @SEQ_TILES@;
constexpr uint32_t kAttentionJobs = @QUERY_HEADS@ * kSequenceTiles;
constexpr uint32_t kProjectionColumns = 5;
constexpr uint8_t kPhysicalX[11] = {1,2,3,4,5,6,7,10,11,12,13};

FORCE_INLINE uint8_t physical_y(uint32_t y) {
    return static_cast<uint8_t>(y + 2);
}

FORCE_INLINE uint64_t attention_l1(uint32_t node, uint32_t address) {
    constexpr uint32_t columns = 6;
    constexpr uint32_t x_offset = 5;
    const uint32_t x = x_offset + node % columns;
    return get_noc_addr(
        kPhysicalX[x], physical_y(node / columns), address);
}

FORCE_INLINE uint64_t producer_l1(uint32_t row, uint32_t address) {
    return get_noc_addr(
        kPhysicalX[row], physical_y(9), address);
}
}

void kernel_main() {
    const uint32_t row = get_absolute_logical_x();
    if (row >= kSequenceTiles) return;

    CircularBuffer head(0), a(1);
    volatile tt_l1_ptr uint32_t* head_credits =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    volatile tt_l1_ptr uint32_t* start =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(4));
    noc_semaphore_set(head_credits, 0);

    // A generation token makes the repeated-invocation root reset explicit.
    if (row == 0) {
        volatile tt_l1_ptr uint32_t* ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(3));
        noc_semaphore_set(ready, 0);
        noc_semaphore_set(start, 1);
        for (uint32_t target = 1; target < kSequenceTiles; ++target)
            noc_async_write(
                get_semaphore(4), producer_l1(target, get_semaphore(4)),
                sizeof(uint32_t));
        noc_async_write_barrier();
    }
    noc_semaphore_wait(start, 1);
    noc_semaphore_set(start, 0);

    head.reserve_back(kHiddenTiles);
    a.reserve_back(kHiddenTiles);
    noc_semaphore_inc(producer_l1(0, get_semaphore(3)), 1);
    noc_async_atomic_barrier();

    if (row == 0) {
        volatile tt_l1_ptr uint32_t* ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(3));
        noc_semaphore_wait_min(ready, kSequenceTiles);
        noc_semaphore_set(ready, 0);
        volatile tt_l1_ptr uint32_t* mailbox =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(8));
        noc_semaphore_set(mailbox, head.get_write_ptr());
        for (uint32_t target = 0; target < kAttentionJobs; ++target) {
            noc_async_write(
                get_semaphore(8), attention_l1(target, get_semaphore(8)),
                sizeof(uint32_t));
        }
        // The destination must observe the published CB address before its
        // ready semaphore can become visible.  Keeping the writes and atomics
        // in separate epochs makes that producer/consumer ordering explicit
        // on both NOCs instead of relying on issue order across engines.
        noc_async_write_barrier();
        for (uint32_t target = 0; target < kAttentionJobs; ++target) {
            noc_semaphore_inc(
                attention_l1(target, get_semaphore(12)), 1);
        }
        noc_async_atomic_barrier();
    }

    Noc noc0(0);
    const auto lora_a = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                0, @TOTAL_CBS@>(), 0>(),
        get_common_arg_val<uint32_t>(0), kTileBytes);
    for (uint32_t tile = 0; tile < kHiddenTiles; ++tile)
        noc0.async_read(
            lora_a,
            CoreLocalMem<uint32_t>(
                a.get_write_ptr() + tile * kTileBytes),
            kTileBytes, {.page_id = tile}, {});
    noc0.async_read_barrier();
    a.push_back(kHiddenTiles);

    noc_semaphore_wait_min(head_credits, kHiddenTiles);
    noc_semaphore_set(head_credits, 0);
    head.push_back(kHiddenTiles);
}
