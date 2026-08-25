// Receive the one-tile FP32 SUM directly from the reduction core.

#include <cstdint>
#include "api/dataflow/dataflow_buffer.h"

namespace {
constexpr uint8_t kMaxPhysicalX = 1;
FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}

void kernel_main() {
    const uint32_t row = get_absolute_logical_y();
    volatile tt_l1_ptr uint32_t* ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(5));
    volatile tt_l1_ptr uint32_t* permission =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(10));
    noc_semaphore_set(ready, 0);
    noc_semaphore_set(permission, 0);
    noc_semaphore_wait(permission, 1);
    noc_semaphore_set(permission, 0);
    DataflowBuffer sum(11);
    sum.reserve_back(1);
    volatile tt_l1_ptr uint32_t* mailbox =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(4));
    noc_semaphore_set(mailbox, sum.get_write_ptr());
    noc_async_write(
        get_semaphore(4),
        get_noc_addr(kMaxPhysicalX, physical_y(row), get_semaphore(4)),
        sizeof(uint32_t));
    noc_async_write_barrier();
    noc_semaphore_set(mailbox, 0);
    noc_semaphore_wait(ready, 1);
    noc_semaphore_set(ready, 0);
    sum.push_back(1);
}
