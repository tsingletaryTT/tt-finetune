// Hold one row-tile loss partial in L1 while the aggregation island reads it.

#include <cstdint>
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"

namespace {
constexpr uint8_t kAggregatePhysicalX = 5;
constexpr uint8_t kAggregatePhysicalY = 2;
}

void kernel_main() {
    DataflowBuffer numerator(15);
    numerator.wait_front(1);
    auto* acknowledgement = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(14));
    noc_semaphore_set(acknowledgement, 0);
    noc_semaphore_inc(
        get_noc_addr(
            kAggregatePhysicalX, kAggregatePhysicalY, get_semaphore(15)),
        1);
    noc_async_atomic_barrier();
    noc_semaphore_wait(acknowledgement, 1);
    noc_semaphore_set(acknowledgement, 0);
    numerator.pop_front(1);
}
