// Receive max, stream the reduction pass, receive log(sum(exp)), then stream
// the logits replay used by the exact top-gradient schedule.

#include <cstdint>
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/waypoint.h"

namespace {
constexpr uint8_t kMaxPhysicalX = 1;
constexpr uint8_t kLogPhysicalX = 3;
constexpr uint32_t kTileBytes = 2048;
FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}

void kernel_main() {
    WAYPOINT("PR00");
    constexpr uint32_t wt = @VOCAB_TILES@;
    const uint32_t row = get_absolute_logical_y();

    volatile tt_l1_ptr uint32_t* max_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    volatile tt_l1_ptr uint32_t* max_permission =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(9));
    volatile tt_l1_ptr uint32_t* log_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(7));
    volatile tt_l1_ptr uint32_t* log_permission =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(8));
    noc_semaphore_set(max_ready, 0);
    noc_semaphore_set(max_permission, 0);
    // Clear the later return channel before the LOG island can possibly send
    // its permission token.  Delaying this until after the vocabulary pass
    // loses an early grant and deadlocks repeated/asymmetric executions.
    noc_semaphore_set(log_ready, 0);
    noc_semaphore_set(log_permission, 0);
    noc_semaphore_wait(max_permission, 1);
    noc_semaphore_set(max_permission, 0);

    DataflowBuffer maximum(27);
    maximum.reserve_back(1);
    volatile tt_l1_ptr uint32_t* max_mailbox =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    noc_semaphore_set(max_mailbox, maximum.get_write_ptr());
    noc_async_write(
        get_semaphore(0),
        get_noc_addr(kMaxPhysicalX, physical_y(row), get_semaphore(0)),
        sizeof(uint32_t));
    noc_async_write_barrier();
    noc_semaphore_set(max_mailbox, 0);
    noc_semaphore_wait(max_ready, 1);
    noc_semaphore_set(max_ready, 0);
    maximum.push_back(1);

    const uint32_t logits_address = get_common_arg_val<uint32_t>(0);
    const auto logits = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 29>(),
            0>(),
        logits_address,
        kTileBytes);
    Noc noc;
    DataflowBuffer input(7);
    const uint32_t first_page = row * wt;

    // Pass two: x -> FP32 -> x-max -> exp, streamed to the SUM island.
    for (uint32_t tile = 0; tile < wt; ++tile) {
        input.reserve_back(1);
        noc.async_read(
            logits, input, kTileBytes,
            {.page_id = first_page + tile}, {.offset_bytes = 0});
        noc.async_read_barrier();
        input.push_back(1);
    }

    // The LOG sender grants permission only after c12 is ready, which is late
    // enough that this kernel-entry clear cannot erase the token.
    WAYPOINT("PR10");
    noc_semaphore_wait(log_permission, 1);
    WAYPOINT("PR11");
    noc_semaphore_set(log_permission, 0);
    DataflowBuffer logsum(25);
    logsum.reserve_back(1);
    volatile tt_l1_ptr uint32_t* log_mailbox =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(6));
    noc_semaphore_set(log_mailbox, logsum.get_write_ptr());
    noc_async_write(
        get_semaphore(6),
        get_noc_addr(kLogPhysicalX, physical_y(row), get_semaphore(6)),
        sizeof(uint32_t));
    noc_async_write_barrier();
    WAYPOINT("PR12");
    noc_semaphore_set(log_mailbox, 0);
    noc_semaphore_wait(log_ready, 1);
    WAYPOINT("PR13");
    noc_semaphore_set(log_ready, 0);
    logsum.push_back(1);
    WAYPOINT("PR14");

    // Pass three: replay logits only after logsum is resident, avoiding a
    // reader-side deadlock on the two-tile input CB.
    for (uint32_t tile = 0; tile < wt; ++tile) {
        input.reserve_back(1);
        noc.async_read(
            logits, input, kTileBytes,
            {.page_id = first_page + tile}, {.offset_bytes = 0});
        noc.async_read_barrier();
        input.push_back(1);
    }
}
