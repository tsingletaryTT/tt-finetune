// Return log(sum(exp)) to the exponential island without DRAM/PJRT exposure.

#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/debug/waypoint.h"

namespace {
constexpr uint8_t kExpPhysicalX = 2;
FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}

void kernel_main() {
    WAYPOINT("LS00");
    const uint32_t row = get_absolute_logical_y();
    volatile tt_l1_ptr uint32_t* mailbox =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(6));
    volatile tt_l1_ptr uint32_t* permission =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(8));
    noc_semaphore_set(mailbox, 0);
    CircularBuffer logarithm(12);
    logarithm.wait_front(1);
    WAYPOINT("LS10");
    noc_semaphore_set(permission, 1);
    noc_async_write(
        get_semaphore(8),
        get_noc_addr(kExpPhysicalX, physical_y(row), get_semaphore(8)),
        sizeof(uint32_t));
    noc_async_write_barrier();
    WAYPOINT("LS11");
    noc_semaphore_set(permission, 0);
    WAYPOINT("LS20");
    noc_semaphore_wait_min(mailbox, 1);
    WAYPOINT("LS21");
    const uint32_t destination = *mailbox;
    noc_semaphore_set(mailbox, 0);
    noc_async_write(
        logarithm.get_read_ptr(),
        get_noc_addr(kExpPhysicalX, physical_y(row), destination),
        4096);
    noc_async_write_barrier();
    WAYPOINT("LS22");
    noc_semaphore_inc(
        get_noc_addr(kExpPhysicalX, physical_y(row), get_semaphore(7)), 1);
    noc_async_atomic_barrier();
    WAYPOINT("LS23");
    logarithm.pop_front(1);
}
