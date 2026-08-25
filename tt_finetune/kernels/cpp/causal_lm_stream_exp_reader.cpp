// Receive the exact max tile, then stream the second logits pass.

#include <cstdint>
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint8_t kMaxPhysicalX = 1;
constexpr uint32_t kTileBytes = 2048;
FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}

void kernel_main() {
    constexpr uint32_t wt = @VOCAB_TILES@;
    const uint32_t row = get_absolute_logical_y();
    volatile tt_l1_ptr uint32_t* max_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    volatile tt_l1_ptr uint32_t* permission =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(9));
    noc_semaphore_set(max_ready, 0);
    noc_semaphore_set(permission, 0);
    noc_semaphore_wait(permission, 1);
    noc_semaphore_set(permission, 0);

    DataflowBuffer maximum(27);
    maximum.reserve_back(1);
    volatile tt_l1_ptr uint32_t* mailbox =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    noc_semaphore_set(mailbox, maximum.get_write_ptr());
    noc_async_write(
        get_semaphore(0),
        get_noc_addr(kMaxPhysicalX, physical_y(row), get_semaphore(0)),
        sizeof(uint32_t));
    noc_async_write_barrier();
    noc_semaphore_set(mailbox, 0);
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
    for (uint32_t tile = 0; tile < wt; ++tile) {
        input.reserve_back(1);
        noc.async_read(
            logits, input, kTileBytes,
            {.page_id = first_page + tile}, {.offset_bytes = 0});
        noc.async_read_barrier();
        input.push_back(1);
    }
}
