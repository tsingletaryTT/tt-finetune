// Send the exact maximum to the exponential island and stop.
#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
namespace {
constexpr uint8_t kExpPhysicalX = 2;
constexpr uint32_t kTileBytes = 4096;
FORCE_INLINE uint8_t physical_y(uint32_t y) { return uint8_t(y + 2); }
}
void kernel_main() {
    const uint32_t row = get_absolute_logical_y();
    auto* mailbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    noc_semaphore_set(mailbox, 0);
    CircularBuffer maximum(4);
    maximum.wait_front(1);
    auto* permission = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(9));
    noc_semaphore_set(permission, 1);
    noc_async_write(get_semaphore(9), get_noc_addr(kExpPhysicalX, physical_y(row), get_semaphore(9)), sizeof(uint32_t));
    noc_async_write_barrier();
    noc_semaphore_set(permission, 0);
    noc_semaphore_wait_min(mailbox, 1);
    const uint32_t destination = *mailbox;
    noc_semaphore_set(mailbox, 0);
    noc_async_write(maximum.get_read_ptr(), get_noc_addr(kExpPhysicalX, physical_y(row), destination), kTileBytes);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(kExpPhysicalX, physical_y(row), get_semaphore(1)), 1);
    noc_async_atomic_barrier();
    maximum.pop_front(1);
}
