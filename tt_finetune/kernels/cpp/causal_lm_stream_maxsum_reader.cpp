// Read the exact max pass, then expose a one-tile NoC receive window for exps.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint8_t kExpPhysicalX = 2;
constexpr uint32_t kTileBytesBF16 = 2048;
constexpr uint32_t kTileBytesFP32 = 4096;
FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}

void kernel_main() {
    constexpr uint32_t wt = @VOCAB_TILES@;
    const uint32_t row = get_absolute_logical_y();
    volatile tt_l1_ptr uint32_t* exp_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(3));
    noc_semaphore_set(exp_ready, 0);

    const uint32_t logits_address = get_common_arg_val<uint32_t>(0);
    const auto logits = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 29>(),
            0>(),
        logits_address,
        kTileBytesBF16);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        1, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        2, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    Noc noc;
    DataflowBuffer maximum_input(0);
    const uint32_t first_page = row * wt;
    for (uint32_t tile = 0; tile < wt; ++tile) {
        maximum_input.reserve_back(1);
        noc.async_read(
            logits,
            maximum_input,
            kTileBytesBF16,
            {.page_id = first_page + tile},
            {.offset_bytes = 0});
        noc.async_read_barrier();
        maximum_input.push_back(1);
    }

    DataflowBuffer exponentials(3);
    for (uint32_t tile = 0; tile < wt; ++tile) {
        exponentials.reserve_back(1);
        volatile tt_l1_ptr uint32_t* mailbox =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
        noc_semaphore_set(mailbox, exponentials.get_write_ptr());
        noc_async_write(
            get_semaphore(2),
            get_noc_addr(kExpPhysicalX, physical_y(row), get_semaphore(2)),
            sizeof(uint32_t));
        noc_async_write_barrier();
        noc_semaphore_set(mailbox, 0);
        noc_semaphore_wait(exp_ready, 1);
        noc_semaphore_set(exp_ready, 0);
        exponentials.push_back(1);
    }
}
