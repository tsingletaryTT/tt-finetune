// Consume resident BF16 dLogits from the probability island and stream the
// frozen LM-head weights needed by this row-local dHidden shard.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint8_t kProducerPhysicalX = 2;
constexpr uint32_t kBf16TileBytes = 2048;

FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t vocabulary_tiles = @VOCAB_TILES@;
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@;
    constexpr uint32_t head_workers = @HEAD_WORKERS@;
    constexpr uint32_t head_start_x = @HEAD_START_X@;
    constexpr uint32_t reduction_tiles_per_phase =
        @HEAD_REDUCTION_TILES_PER_PHASE@;
    constexpr uint32_t reduction_phases =
        vocabulary_tiles / reduction_tiles_per_phase;

    const uint32_t row = get_absolute_logical_y();
    const uint32_t worker = get_absolute_logical_x() - head_start_x;
    const uint32_t local_jobs =
        (hidden_tiles + head_workers - 1) / head_workers;

    auto* start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(10));
    auto* data_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(15));
    // The producer owns initialization of both remote semaphores. If this L1
    // address contains a stale one from an earlier program, first observe the
    // producer's explicit zero epoch and only then accept the start signal.
    while (*start != 0) {}
    noc_semaphore_wait(start, 1);
    noc_semaphore_set(start, 0);

    const auto weight = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                2, 32>(),
            0>(),
        get_common_arg_val<uint32_t>(0), kBf16TileBytes);
    Noc noc;
    DataflowBuffer dlogits(0);
    DataflowBuffer weights(7);

    for (uint32_t phase = 0; phase < reduction_phases; ++phase) {
        // Receive a complete numerical reduction phase before streaming its
        // weights. The compute kernel reuses these resident dLogits tiles for
        // every local hidden job without asking the producer to resend them.
        for (
            uint32_t inner = 0;
            inner < reduction_tiles_per_phase;
            ++inner
        ) {
            dlogits.reserve_back(1);
            // Tell the producer this receive slot is reserved before it sends.
            noc_semaphore_inc(
                get_noc_addr(
                    kProducerPhysicalX,
                    physical_y(row),
                    get_semaphore(14)),
                1);
            noc_async_atomic_barrier();
            noc_semaphore_wait(data_ready, 1);
            noc_semaphore_set(data_ready, 0);
            dlogits.push_back(1);
        }

        for (uint32_t local_job = 0; local_job < local_jobs; ++local_job) {
            const uint32_t k_tile = worker + local_job * head_workers;
            if (k_tile >= hidden_tiles) continue;
            for (
                uint32_t inner = 0;
                inner < reduction_tiles_per_phase;
                ++inner
            ) {
                const uint32_t n_tile =
                    phase * reduction_tiles_per_phase + inner;
                weights.reserve_back(1);
                noc.async_read(
                    weight, weights, kBf16TileBytes,
                    {.page_id = n_tile * hidden_tiles + k_tile},
                    {.offset_bytes = 0});
                noc.async_read_barrier();
                weights.push_back(1);
            }
        }
    }
}
