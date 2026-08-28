#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace resident_layer_phase_mlp_3 {
// Activation-island reader for the resident down -> SwiGLU backward edge.



namespace {
constexpr uint32_t kProducerCols = 10;
constexpr uint32_t kProducerNodes = 100;
constexpr uint32_t kActivationNodes = 10;
constexpr uint32_t kTileBytes = 2048;
constexpr uint8_t kProducerPhysicalX[kProducerCols] = {
    1, 2, 3, 4, 5, 6, 7, 10, 11, 12};
constexpr uint32_t kMailboxSemaphore = 9;
constexpr uint32_t kReadySemaphore = 10;
constexpr uint32_t kInitAckSemaphore = 11;
FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}  // namespace

void run() {
    Noc noc0(0);
    constexpr uint32_t cb_dproduct = get_compile_time_arg_val(25);
    constexpr uint32_t cb_gate = get_compile_time_arg_val(26);
    constexpr uint32_t cb_up = get_compile_time_arg_val(27);
    constexpr uint32_t total_jobs = 192;
    constexpr uint32_t producer_jobs = 2;

    CircularBuffer dproduct(cb_dproduct);
    CircularBuffer gate_cb(cb_gate);
    CircularBuffer up_cb(cb_up);
    const uint32_t activation_node = get_absolute_logical_y();

    // All semaphores are reset in both islands on every invocation. Producer
    // readers send their acknowledgement only after their long phase-one
    // input schedule, so it cannot race this activation-side reset.
    for (uint32_t semaphore = 0; semaphore < 12; ++semaphore) {
        volatile tt_l1_ptr uint32_t* flag =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(semaphore));
        noc_semaphore_set(flag, 0);
    }
    // One coordinator owns cross-island reset. Other readers never
    // clear IDs 12..15, so a published go token cannot be erased by a race.
    if (activation_node == 0) {
        noc_semaphore_set(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(12)),
            0);
        for (uint32_t target = 0; target < kActivationNodes; ++target) {
            for (uint32_t semaphore = 12; semaphore < 16; ++semaphore) {
                noc_async_write(
                    get_semaphore(12),
                    get_noc_addr(13, physical_y(target),
                                 get_semaphore(semaphore)),
                    sizeof(uint32_t));
            }
        }
        for (uint32_t target = 0; target < kProducerNodes; ++target) {
            for (uint32_t semaphore = 12; semaphore < 16; ++semaphore) {
                noc_async_write(
                    get_semaphore(12),
                    get_noc_addr(
                        kProducerPhysicalX[target % kProducerCols],
                        physical_y(target / kProducerCols),
                        get_semaphore(semaphore)),
                    sizeof(uint32_t));
            }
        }
        noc_async_write_barrier();
        volatile tt_l1_ptr uint32_t* go =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(12));
        noc_semaphore_set(go, 1);
        for (uint32_t target = 0; target < kActivationNodes; ++target) {
            noc_async_write(
                get_semaphore(12),
                get_noc_addr(13, physical_y(target), get_semaphore(12)),
                sizeof(uint32_t));
        }
        for (uint32_t target = 0; target < kProducerNodes; ++target) {
            noc_async_write(
                get_semaphore(12),
                get_noc_addr(
                    kProducerPhysicalX[target % kProducerCols],
                    physical_y(target / kProducerCols), get_semaphore(15)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    volatile tt_l1_ptr uint32_t* activation_go =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(12));
    noc_semaphore_wait(activation_go, 1);
    noc_semaphore_set(activation_go, 0);

    constexpr uint32_t cb_product_gate = get_compile_time_arg_val(45);
    constexpr uint32_t cb_product_gate_saved =
        get_compile_time_arg_val(46);
    constexpr uint32_t cb_product_up = get_compile_time_arg_val(47);
    CircularBuffer product_gate(cb_product_gate);
    CircularBuffer product_gate_saved(cb_product_gate_saved);
    CircularBuffer product_up(cb_product_up);
    const uint32_t checkpoint_addr = get_common_arg_val<uint32_t>(0);
    const auto checkpoint = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<5, 53>(), 0>(), checkpoint_addr, kTileBytes);
    constexpr uint32_t product_jobs = 192;
    constexpr uint32_t product_jobs_per_node = 20;
    constexpr uint32_t product_i_tiles = 48;
    constexpr uint32_t product_row_tiles = 116;
    const uint32_t product_first = activation_node * product_jobs_per_node;
    for (uint32_t local = 0; local < product_jobs_per_node; ++local) {
        const uint32_t job = product_first + local;
        if (job >= product_jobs) continue;
        const uint32_t m = job / product_i_tiles;
        const uint32_t k = job - m * product_i_tiles;
        product_gate.reserve_back(1);
        product_gate_saved.reserve_back(1);
        product_up.reserve_back(1);
        noc0.async_read(
            checkpoint, CoreLocalMem<uint32_t>(product_gate.get_write_ptr()),
            kTileBytes, {.page_id = m * product_row_tiles + k}, {});
        noc0.async_read(
            checkpoint,
            CoreLocalMem<uint32_t>(product_gate_saved.get_write_ptr()),
            kTileBytes, {.page_id = m * product_row_tiles + k}, {});
        noc0.async_read(
            checkpoint, CoreLocalMem<uint32_t>(product_up.get_write_ptr()),
            kTileBytes,
            {.page_id = m * product_row_tiles + product_i_tiles + k}, {});
        noc0.async_read_barrier();
        product_gate.push_back(1);
        product_gate_saved.push_back(1);
        product_up.push_back(1);
    }

    volatile tt_l1_ptr uint32_t* init_ack =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(kInitAckSemaphore));
    noc_semaphore_wait_min(init_ack, kProducerNodes / kActivationNodes);
    noc_semaphore_set(init_ack, 0);

    const uint32_t gate_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t up_addr = get_common_arg_val<uint32_t>(2);
    const auto gate = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<6, 53>(), 1>(), gate_addr, kTileBytes);
    const auto up = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<7, 53>(), 2>(), up_addr, kTileBytes);

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
                {.page_id = (job / 48) * 116 + (job % 48)},
                {});
            noc0.async_read(
                up,
                CoreLocalMem<uint32_t>(up_cb.get_write_ptr()),
                kTileBytes,
                {.page_id = (job / 48) * 116 + 48 + (job % 48)},
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
}  // namespace resident_layer_phase_mlp_3

void kernel_main() {
    resident_layer_phase_mlp_3::run();
}
