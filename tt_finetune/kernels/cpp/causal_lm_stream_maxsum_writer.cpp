// Send max to the exp core and the final sum to the unary-log core.

#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint8_t kExpPhysicalX = 2;
constexpr uint8_t kLogPhysicalX = 3;
constexpr uint32_t kTileBytes = 4096;
FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}

void kernel_main() {
    const uint32_t row = get_absolute_logical_y();
    volatile tt_l1_ptr uint32_t* max_mailbox =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    volatile tt_l1_ptr uint32_t* sum_mailbox =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(4));
    noc_semaphore_set(max_mailbox, 0);
    noc_semaphore_set(sum_mailbox, 0);

    CircularBuffer maximum(4);
    maximum.wait_front(1);
    // Max is ready only after the exp reader has had ample time to clear its
    // permission word, making repeated invocations deterministic.
    volatile tt_l1_ptr uint32_t* permission =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(9));
    noc_semaphore_set(permission, 1);
    noc_async_write(
        get_semaphore(9),
        get_noc_addr(kExpPhysicalX, physical_y(row), get_semaphore(9)),
        sizeof(uint32_t));
    noc_async_write_barrier();
    noc_semaphore_set(permission, 0);
    noc_semaphore_wait_min(max_mailbox, 1);
    const uint32_t max_destination = *max_mailbox;
    noc_semaphore_set(max_mailbox, 0);
    noc_async_write(
        maximum.get_read_ptr(),
        get_noc_addr(kExpPhysicalX, physical_y(row), max_destination),
        kTileBytes);
    noc_async_write_barrier();
    noc_semaphore_inc(
        get_noc_addr(kExpPhysicalX, physical_y(row), get_semaphore(1)), 1);
    noc_async_atomic_barrier();
    maximum.pop_front(1);

    CircularBuffer sum(5);
    sum.wait_front(1);
    volatile tt_l1_ptr uint32_t* log_permission =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(10));
    noc_semaphore_set(log_permission, 1);
    noc_async_write(
        get_semaphore(10),
        get_noc_addr(kLogPhysicalX, physical_y(row), get_semaphore(10)),
        sizeof(uint32_t));
    noc_async_write_barrier();
    noc_semaphore_set(log_permission, 0);
    noc_semaphore_wait_min(sum_mailbox, 1);
    const uint32_t sum_destination = *sum_mailbox;
    noc_semaphore_set(sum_mailbox, 0);
    noc_async_write(
        sum.get_read_ptr(),
        get_noc_addr(kLogPhysicalX, physical_y(row), sum_destination),
        kTileBytes);
    noc_async_write_barrier();
    noc_semaphore_inc(
        get_noc_addr(kLogPhysicalX, physical_y(row), get_semaphore(5)), 1);
    noc_async_atomic_barrier();
    sum.pop_front(1);
}
