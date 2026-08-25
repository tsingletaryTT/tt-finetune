// Diagnostic writer for the exact streamed maximum or pre-log SUM boundary.

#include <cstdint>
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint8_t kExpPhysicalX = 2;
constexpr uint32_t kTileBytes = 4096;
FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}

void write_output(DataflowBuffer value, uint32_t row) {
    const uint32_t output_address = get_common_arg_val<uint32_t>(0);
    const auto output = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 29>(),
            0>(), output_address, kTileBytes);
    Noc noc;
    value.wait_front(1);
    noc.async_write(
        value, output, kTileBytes, {.offset_bytes = 0}, {.page_id = row});
    noc.async_write_barrier();
    value.pop_front(1);
}

void kernel_main() {
    const uint32_t row = get_absolute_logical_y();
    DataflowBuffer maximum(4);
#if @DIAGNOSTIC_MAXIMUM@
    write_output(maximum, row);
    return;
#else
    volatile tt_l1_ptr uint32_t* max_mailbox =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    noc_semaphore_set(max_mailbox, 0);
    maximum.wait_front(1);
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
    const uint32_t destination = *max_mailbox;
    noc_semaphore_set(max_mailbox, 0);
    noc_async_write(
        maximum.get_read_ptr(),
        get_noc_addr(kExpPhysicalX, physical_y(row), destination), kTileBytes);
    noc_async_write_barrier();
    noc_semaphore_inc(
        get_noc_addr(kExpPhysicalX, physical_y(row), get_semaphore(1)), 1);
    noc_async_atomic_barrier();
    maximum.pop_front(1);
    write_output(DataflowBuffer(5), row);
#endif
}
