// Send final BF16 dLogits directly to row-local frozen-head workers while
// continuing to route loss terms to the resident loss reduction island.

#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"

namespace {
constexpr uint8_t kMaxPhysicalX = 1;
constexpr uint8_t kLossPhysicalX = 4;
// Logical x=5..10 on this P150A. Physical worker columns 8/9 are not Tensix.
constexpr uint8_t kHeadPhysicalX[6] = {6, 7, 10, 11, 12, 13};
constexpr uint32_t kFp32TileBytes = 4096;
constexpr uint32_t kBf16TileBytes = 2048;
FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t wt = @VOCAB_TILES@;
    constexpr uint32_t head_workers = @HEAD_WORKERS@;
    const uint32_t row = get_absolute_logical_y();
    auto* mailbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(2));
    noc_semaphore_set(mailbox, 0);

    CircularBuffer exponential(16);
    for (uint32_t tile = 0; tile < wt; ++tile) {
        exponential.wait_front(1);
        noc_semaphore_wait_min(mailbox, 1);
        const uint32_t destination = *mailbox;
        noc_semaphore_set(mailbox, 0);
        noc_async_write(
            exponential.get_read_ptr(),
            get_noc_addr(kMaxPhysicalX, physical_y(row), destination),
            kFp32TileBytes);
        noc_async_write_barrier();
        noc_semaphore_inc(
            get_noc_addr(
                kMaxPhysicalX, physical_y(row), get_semaphore(3)), 1);
        noc_async_atomic_barrier();
        exponential.pop_front(1);
    }

    auto* loss_mailbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(11));
    auto* loss_permission = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(13));
    noc_semaphore_set(loss_mailbox, 0);
    noc_semaphore_set(loss_permission, 1);
    noc_async_write(
        get_semaphore(13),
        get_noc_addr(
            kLossPhysicalX, physical_y(row), get_semaphore(13)),
        sizeof(uint32_t));
    noc_async_write_barrier();
    noc_semaphore_set(loss_permission, 0);

    // Producer-owned reset avoids a launch-order race: every receiver first
    // observes zero, then the start epoch. The same protocol is repeat-safe
    // even if another program left values at these L1 semaphore addresses.
    auto* head_arrivals = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(14));
    noc_semaphore_set(head_arrivals, 0);
    for (uint32_t worker = 0; worker < head_workers; ++worker) {
        noc_async_write(
            get_semaphore(14),
            get_noc_addr(
                kHeadPhysicalX[worker], physical_y(row), get_semaphore(10)),
            sizeof(uint32_t));
        noc_async_write(
            get_semaphore(14),
            get_noc_addr(
                kHeadPhysicalX[worker], physical_y(row), get_semaphore(15)),
            sizeof(uint32_t));
    }
    noc_async_write_barrier();
    for (uint32_t worker = 0; worker < head_workers; ++worker) {
        noc_semaphore_inc(
            get_noc_addr(
                kHeadPhysicalX[worker], physical_y(row), get_semaphore(10)),
            1);
    }
    noc_async_atomic_barrier();

    CircularBuffer loss_contribution(21);
    CircularBuffer dlogits_fp32(20);
    DataflowBuffer bf16_scratch(0);
    for (uint32_t tile = 0; tile < wt; ++tile) {
        loss_contribution.wait_front(1);
        noc_semaphore_wait_min(loss_mailbox, 1);
        const uint32_t loss_destination = *loss_mailbox;
        noc_semaphore_set(loss_mailbox, 0);
        noc_async_write(
            loss_contribution.get_read_ptr(),
            get_noc_addr(
                kLossPhysicalX, physical_y(row), loss_destination),
            kFp32TileBytes);
        noc_async_write_barrier();
        noc_semaphore_inc(
            get_noc_addr(
                kLossPhysicalX, physical_y(row), get_semaphore(12)), 1);
        noc_async_atomic_barrier();
        loss_contribution.pop_front(1);

        dlogits_fp32.wait_front(1);
        bf16_scratch.reserve_back(1);
        const auto* source =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                dlogits_fp32.get_read_ptr());
        auto* destination =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                bf16_scratch.get_write_ptr());
        for (uint32_t element = 0; element < 1024; ++element) {
            const uint32_t bits = source[element];
            const uint32_t rounded =
                bits + 0x7fffu + ((bits >> 16) & 1u);
            destination[element] = static_cast<uint16_t>(rounded >> 16);
        }
        bf16_scratch.push_back(1);

        noc_semaphore_wait(head_arrivals, head_workers);
        noc_semaphore_set(head_arrivals, 0);
        const uint32_t source_address = bf16_scratch.get_read_ptr();
        // c0 is allocated at the same address on every worker and every
        // receiver has reserved its current slot before announcing arrival.
        for (uint32_t worker = 0; worker < head_workers; ++worker) {
            noc_async_write(
                source_address,
                get_noc_addr(
                    kHeadPhysicalX[worker], physical_y(row),
                    source_address),
                kBf16TileBytes);
        }
        noc_async_write_barrier();
        for (uint32_t worker = 0; worker < head_workers; ++worker) {
            noc_semaphore_inc(
                get_noc_addr(
                    kHeadPhysicalX[worker], physical_y(row),
                    get_semaphore(15)),
                1);
        }
        noc_async_atomic_barrier();
        bf16_scratch.pop_front(1);
        dlogits_fp32.pop_front(1);
    }
}
