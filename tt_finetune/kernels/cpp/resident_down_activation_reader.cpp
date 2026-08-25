// Activation-island reader for the resident down -> SwiGLU backward edge.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t kProducerCols = 10;
constexpr uint32_t kProducerNodes = 100;
constexpr uint32_t kActivationNodes = 10;
constexpr uint32_t kTileBytes = 2048;
constexpr uint8_t kProducerPhysicalX[kProducerCols] = {
    1, 2, 3, 4, 5, 6, 7, 10, 11, 12};
constexpr uint32_t kMailboxSemaphore = @MAILBOX_SEMAPHORE@;
constexpr uint32_t kReadySemaphore = @READY_SEMAPHORE@;
constexpr uint32_t kInitAckSemaphore = @INIT_ACK_SEMAPHORE@;
FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}  // namespace

void kernel_main() {
    Noc noc0(0);
    constexpr uint32_t cb_dproduct = get_compile_time_arg_val(25);
    constexpr uint32_t cb_gate = get_compile_time_arg_val(26);
    constexpr uint32_t cb_up = get_compile_time_arg_val(27);
    constexpr uint32_t total_jobs = @TOTAL_JOBS@;
    constexpr uint32_t producer_jobs = @JOBS_PER_NODE@;

    CircularBuffer dproduct(cb_dproduct);
    CircularBuffer gate_cb(cb_gate);
    CircularBuffer up_cb(cb_up);
    const uint32_t activation_node = get_absolute_logical_y();

    // All semaphores are reset in both islands on every invocation. Producer
    // readers send their acknowledgement only after their long phase-one
    // input schedule, so it cannot race this activation-side reset.
    for (uint32_t semaphore = 0; semaphore < @NUM_SEMAPHORES@; ++semaphore) {
        volatile tt_l1_ptr uint32_t* flag =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(semaphore));
        noc_semaphore_set(flag, 0);
    }
    volatile tt_l1_ptr uint32_t* init_ack =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(kInitAckSemaphore));
    noc_semaphore_wait_min(init_ack, kProducerNodes / kActivationNodes);
    noc_semaphore_set(init_ack, 0);

    const uint32_t gate_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t up_addr = get_common_arg_val<uint32_t>(1);
    const auto gate = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                6, @TOTAL_CBS@>(),
            0>(), gate_addr, kTileBytes);
    const auto up = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                7, @TOTAL_CBS@>(),
            1>(), up_addr, kTileBytes);

    // Producer `p` is owned only by activation core `p % 10`. This avoids
    // two consumers racing on one producer's destination mailbox.
    for (uint32_t producer = activation_node;
         producer < kProducerNodes;
         producer += kActivationNodes) {
        for (uint32_t local = 0; local < producer_jobs; ++local) {
            const uint32_t job = producer * producer_jobs + local;
            if (job >= total_jobs) continue;

            dproduct.reserve_back(1);
            gate_cb.reserve_back(1);
            up_cb.reserve_back(1);
            noc0.async_read(
                gate,
                CoreLocalMem<uint32_t>(gate_cb.get_write_ptr()),
                kTileBytes,
                {.page_id = job},
                {});
            noc0.async_read(
                up,
                CoreLocalMem<uint32_t>(up_cb.get_write_ptr()),
                kTileBytes,
                {.page_id = job},
                {});

            volatile tt_l1_ptr uint32_t* mailbox =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(kMailboxSemaphore));
            noc_semaphore_set(mailbox, dproduct.get_write_ptr());
            const uint32_t producer_x = producer % kProducerCols;
            const uint32_t producer_y = producer / kProducerCols;
            noc_async_write(
                get_semaphore(kMailboxSemaphore),
                get_noc_addr(
                    kProducerPhysicalX[producer_x], physical_y(producer_y),
                    get_semaphore(kMailboxSemaphore)),
                sizeof(uint32_t));
            noc_async_write_barrier();
            noc_semaphore_set(mailbox, 0);

            volatile tt_l1_ptr uint32_t* ready =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(kReadySemaphore));
            noc_semaphore_wait_min(ready, 1);
            noc_semaphore_set(ready, 0);
            noc0.async_read_barrier();
            up_cb.push_back(1);
            gate_cb.push_back(1);
            dproduct.push_back(1);
        }
    }
}
