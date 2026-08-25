// Receive streamed sparse loss terms on a dedicated reduction island.

#include <cstdint>
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/debug/waypoint.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {
constexpr uint8_t kExpPhysicalX = 2;
FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}

void kernel_main() {
    WAYPOINT("LR00");
    constexpr uint32_t wt = @VOCAB_TILES@;
    const uint32_t row = get_absolute_logical_y();
    auto* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(12));
    auto* permission = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(13));
    noc_semaphore_set(ready, 0);
    noc_semaphore_set(permission, 0);

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        9, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        10, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL>();

    noc_semaphore_wait(permission, 1);
    noc_semaphore_set(permission, 0);
    WAYPOINT("LR10");
    DataflowBuffer loss_terms(13);
    auto* mailbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(11));
    for (uint32_t tile = 0; tile < wt; ++tile) {
        loss_terms.reserve_back(1);
        noc_semaphore_set(mailbox, loss_terms.get_write_ptr());
        noc_async_write(
            get_semaphore(11),
            get_noc_addr(kExpPhysicalX, physical_y(row), get_semaphore(11)),
            sizeof(uint32_t));
        noc_async_write_barrier();
        noc_semaphore_set(mailbox, 0);
        noc_semaphore_wait(ready, 1);
        noc_semaphore_set(ready, 0);
        loss_terms.push_back(1);
    }
    WAYPOINT("LR99");
}
