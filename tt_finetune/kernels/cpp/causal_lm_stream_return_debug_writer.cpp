// Stream first-pass exponentials, then publish the returned logsum tile.
#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/waypoint.h"

namespace {
constexpr uint8_t kMaxPhysicalX = 1;
FORCE_INLINE uint8_t physical_y(uint32_t y) { return uint8_t(y + 2); }
}

void kernel_main() {
    WAYPOINT("PW00");
    constexpr uint32_t wt = @VOCAB_TILES@;
    const uint32_t row = get_absolute_logical_y();
    volatile tt_l1_ptr uint32_t* mailbox =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
    noc_semaphore_set(mailbox, 0);
    CircularBuffer value(16);
    for (uint32_t tile = 0; tile < wt; ++tile) {
        value.wait_front(1);
        noc_semaphore_wait_min(mailbox, 1);
        const uint32_t destination = *mailbox;
        noc_semaphore_set(mailbox, 0);
        noc_async_write(
            value.get_read_ptr(),
            get_noc_addr(kMaxPhysicalX, physical_y(row), destination), 4096);
        noc_async_write_barrier();
        noc_semaphore_inc(
            get_noc_addr(kMaxPhysicalX, physical_y(row), get_semaphore(3)), 1);
        noc_async_atomic_barrier();
        value.pop_front(1);
    }
    WAYPOINT("PW10");

    const uint32_t output_address = get_common_arg_val<uint32_t>(0);
    const auto output = TensorAccessor(
        TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 29>(), 0>(),
        output_address, 4096);
    Noc noc;
    DataflowBuffer returned(16);
    WAYPOINT("PW20");
    returned.wait_front(1);
    WAYPOINT("PW21");
    noc.async_write(
        returned, output, 4096, {.offset_bytes = 0}, {.page_id = row});
    noc.async_write_barrier();
    WAYPOINT("PW22");
    returned.pop_front(1);
    WAYPOINT("PW23");
}
