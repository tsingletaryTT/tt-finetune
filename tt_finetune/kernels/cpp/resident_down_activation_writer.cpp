// Write exact dGate/dUp tiles; all down/activation intermediates stay local.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t kProducerNodes = 100;
constexpr uint32_t kActivationNodes = 10;
constexpr uint32_t kTileBytes = 2048;
}  // namespace

void kernel_main() {
    Noc noc1(1);
    constexpr uint32_t cb_dgate = get_compile_time_arg_val(35);
    constexpr uint32_t cb_dup = get_compile_time_arg_val(36);
    constexpr uint32_t total_jobs = @TOTAL_JOBS@;
    constexpr uint32_t producer_jobs = @JOBS_PER_NODE@;
    constexpr uint32_t arena_cols = @K_TILES@;
    constexpr uint32_t m_tiles = @M_TILES@;

    CircularBuffer dgate(cb_dgate);
    CircularBuffer dup(cb_dup);
    const uint32_t activation_node = get_absolute_logical_y();
    const uint32_t arena_addr = get_common_arg_val<uint32_t>(0);
    const auto arena = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                8, @TOTAL_CBS@>(),
            0>(), arena_addr, kTileBytes);

    for (uint32_t producer = activation_node;
         producer < kProducerNodes;
         producer += kActivationNodes) {
        for (uint32_t local = 0; local < producer_jobs; ++local) {
            const uint32_t job = producer * producer_jobs + local;
            if (job >= total_jobs) continue;
            dgate.wait_front(1);
            noc1.async_write(
                CoreLocalMem<uint32_t>(dgate.get_read_ptr()), arena,
                kTileBytes, {}, {.page_id = job});
            noc1.async_write_barrier();
            dgate.pop_front(1);

            dup.wait_front(1);
            noc1.async_write(
                CoreLocalMem<uint32_t>(dup.get_read_ptr()), arena,
                kTileBytes, {}, {.page_id = m_tiles * arena_cols + job});
            noc1.async_write_barrier();
            dup.pop_front(1);
        }
    }
}
