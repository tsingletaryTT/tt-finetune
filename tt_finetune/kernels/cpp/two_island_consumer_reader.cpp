// Make the remote L1 tile and one DRAM operand visible to BF16 compute.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint8_t kProducerPhysicalX = 1;  // logical x=0 on this P150A
constexpr uint8_t kProducerPhysicalY = 11; // logical y=9 on this P150A
}

void kernel_main() {
    if (get_absolute_logical_x() != 10 || get_absolute_logical_y() != 9) {
        return;
    }

    constexpr uint32_t cb_handoff = 0;
    constexpr uint32_t cb_rhs = 1;
    const uint32_t rhs_addr = get_common_arg_val<uint32_t>(0);
#ifdef TWO_ISLAND_PJRT_ARTIFACT
    constexpr auto rhs_args = TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 7>(),
        2>();
#else
    constexpr auto rhs_args = TensorAccessorArgs<0>();
#endif
    const auto rhs = TensorAccessor(rhs_args, rhs_addr);

    // The descriptor initializes these flags for the first dispatch and each
    // completed dispatch returns both to zero. Reserve the receive slot before
    // allowing the producer to target its L1 address.
    cb_reserve_back(cb_handoff, 1);
    cb_reserve_back(cb_rhs, 1);
    // Static CB allocation is core-local and the two descriptors are allowed
    // to receive different addresses. Send the actual reserved destination
    // pointer to the producer rather than assuming address symmetry.
    volatile tt_l1_ptr uint32_t* destination_mailbox =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    noc_semaphore_set(destination_mailbox, get_write_ptr(cb_handoff));
    noc_async_write(
        get_semaphore(1),
        get_noc_addr(
            kProducerPhysicalX,
            kProducerPhysicalY,
            get_semaphore(1)),
        sizeof(uint32_t));
    noc_async_write_barrier();
    noc_semaphore_set(destination_mailbox, 0);

    Noc noc;
    CircularBuffer local_rhs(cb_rhs);
    noc.async_read(rhs, local_rhs, 2048, {.page_id = 0}, {});
    noc.async_read_barrier();

    volatile tt_l1_ptr uint32_t* ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    noc_semaphore_wait(ready, 1);
    noc_semaphore_set(ready, 0);
    cb_push_back(cb_handoff, 1);
    cb_push_back(cb_rhs, 1);
}
