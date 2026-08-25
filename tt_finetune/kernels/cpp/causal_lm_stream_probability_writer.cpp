// Stream first-pass exponentials to SUM, then publish replayed probabilities
// only for the exact top-gradient diagnostic boundary.

#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint8_t kMaxPhysicalX = 1;
constexpr uint32_t kTileBytes = 4096;
FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}

void kernel_main() {
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
            get_noc_addr(kMaxPhysicalX, physical_y(row), destination),
            kTileBytes);
        noc_async_write_barrier();
        noc_semaphore_inc(
            get_noc_addr(kMaxPhysicalX, physical_y(row), get_semaphore(3)), 1);
        noc_async_atomic_barrier();
        value.pop_front(1);
    }

    const uint32_t output_address = get_common_arg_val<uint32_t>(0);
    const auto output = TensorAccessor(
        TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 29>(), 0>(),
        output_address,
        kTileBytes);
    Noc noc;
    DataflowBuffer published(16);
    const uint32_t first_page = row * wt;
    for (uint32_t tile = 0; tile < wt; ++tile) {
        published.wait_front(1);
        noc.async_write(
            published, output, kTileBytes,
            {.offset_bytes = 0}, {.page_id = first_page + tile});
        noc.async_write_barrier();
        published.pop_front(1);
    }
}
