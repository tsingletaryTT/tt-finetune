// Stream first-pass exponentials to SUM, then publish only final BF16 dLogits.

#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/waypoint.h"

namespace {
constexpr uint8_t kMaxPhysicalX = 1;
constexpr uint8_t kLossPhysicalX = 4;
constexpr uint32_t kFp32TileBytes = 4096;
constexpr uint32_t kBf16TileBytes = 2048;
FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}  // namespace

void kernel_main() {
    WAYPOINT("DW00");
    constexpr uint32_t wt = @VOCAB_TILES@;
    const uint32_t row = get_absolute_logical_y();
    auto* mailbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(2));
    noc_semaphore_set(mailbox, 0);

    CircularBuffer exponential(16);
    for (uint32_t tile = 0; tile < wt; ++tile) {
        exponential.wait_front(1);
        noc_semaphore_wait_min(mailbox, 1);
        const uint32_t destination = *mailbox;
        noc_semaphore_set(mailbox, 0);
        noc_async_write(
            exponential.get_read_ptr(),
            get_noc_addr(kMaxPhysicalX, physical_y(row), destination),
            kFp32TileBytes);
        noc_async_write_barrier();
        noc_semaphore_inc(
            get_noc_addr(
                kMaxPhysicalX, physical_y(row), get_semaphore(3)),
            1);
        noc_async_atomic_barrier();
        exponential.pop_front(1);
    }
    WAYPOINT("DW10");

    auto* loss_mailbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(11));
    auto* loss_permission = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(13));
    noc_semaphore_set(loss_mailbox, 0);
    // The exponential stream above gives the log reader time to initialize
    // its permission word before this reusable handshake begins.
    noc_semaphore_set(loss_permission, 1);
    noc_async_write(
        get_semaphore(13),
        get_noc_addr(kLossPhysicalX, physical_y(row), get_semaphore(13)),
        sizeof(uint32_t));
    noc_async_write_barrier();
    noc_semaphore_set(loss_permission, 0);
    WAYPOINT("DW11");

    const auto output = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                3, 32>(),
            0>(),
        get_common_arg_val<uint32_t>(0), kBf16TileBytes);
    Noc noc;
    CircularBuffer loss_contribution(21);
    CircularBuffer dlogits_fp32(20);
    DataflowBuffer bf16_scratch(0);
    const uint32_t first_page = row * wt;
    for (uint32_t tile = 0; tile < wt; ++tile) {
        loss_contribution.wait_front(1);
        noc_semaphore_wait_min(loss_mailbox, 1);
        const uint32_t loss_destination = *loss_mailbox;
        noc_semaphore_set(loss_mailbox, 0);
        noc_async_write(
            loss_contribution.get_read_ptr(),
            get_noc_addr(
                kLossPhysicalX, physical_y(row), loss_destination),
            kFp32TileBytes);
        noc_async_write_barrier();
        noc_semaphore_inc(
            get_noc_addr(
                kLossPhysicalX, physical_y(row), get_semaphore(12)), 1);
        noc_async_atomic_barrier();
        loss_contribution.pop_front(1);

        dlogits_fp32.wait_front(1);
        bf16_scratch.reserve_back(1);
        const auto* source =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                dlogits_fp32.get_read_ptr());
        auto* destination = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
            bf16_scratch.get_write_ptr());
        for (uint32_t element = 0; element < 1024; ++element) {
            const uint32_t bits = source[element];
            const uint32_t rounded =
                bits + 0x7fffu + ((bits >> 16) & 1u);
            destination[element] = static_cast<uint16_t>(rounded >> 16);
        }
        bf16_scratch.push_back(1);
        noc.async_write(
            bf16_scratch, output, kBf16TileBytes,
            {.offset_bytes = 0}, {.page_id = first_page + tile});
        noc.async_write_barrier();
        bf16_scratch.pop_front(1);
        dlogits_fp32.pop_front(1);
    }
    WAYPOINT("DW99");
}
