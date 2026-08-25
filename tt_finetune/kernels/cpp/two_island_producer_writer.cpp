// Send the FP32-island result directly into the BF16 island's matching L1 CB.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t kTileBytes = 2048;
constexpr uint8_t kConsumerPhysicalX = 13; // logical x=10 on this P150A
constexpr uint8_t kConsumerPhysicalY = 11; // logical y=9 on this P150A
}

void kernel_main() {
    if (get_absolute_logical_x() != 0 || get_absolute_logical_y() != 9) {
        return;
    }

    constexpr uint32_t cb_handoff = 0;
#ifndef TWO_ISLAND_REAL_MATMUL
    constexpr uint32_t cb_dummy_output = 2;
#endif
    volatile tt_l1_ptr uint32_t* destination_mailbox =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    noc_semaphore_wait_min(destination_mailbox, 1);
    const uint32_t destination_l1 = *destination_mailbox;
    noc_semaphore_set(destination_mailbox, 0);

    cb_wait_front(cb_handoff, 1);
#ifndef TWO_ISLAND_REAL_MATMUL
    cb_wait_front(cb_dummy_output, 1);
#endif
    const uint32_t source = get_read_ptr(cb_handoff);
    const uint64_t destination = get_noc_addr(
        kConsumerPhysicalX, kConsumerPhysicalY, destination_l1);
    noc_async_write(source, destination, kTileBytes);
    noc_async_write_barrier();

    // The payload must be globally visible before the readiness flag.
    noc_semaphore_inc(
        get_noc_addr(
            kConsumerPhysicalX,
            kConsumerPhysicalY,
            get_semaphore(0)),
        1);
    cb_pop_front(cb_handoff, 1);
#ifndef TWO_ISLAND_REAL_MATMUL
    cb_pop_front(cb_dummy_output, 1);
#endif
}
