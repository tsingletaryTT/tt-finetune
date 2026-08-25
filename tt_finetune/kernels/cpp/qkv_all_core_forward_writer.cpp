// Publish all-worker Q/K/V outputs and rank checkpoints into the private arena.

#include <cstdint>
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t kTileBytes = 2048, kWorkers = 110, kColumns = 11;
    constexpr uint32_t cb_saved_h = 4, cb_output = 12;
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@;
    constexpr uint32_t kv_tiles = @KV_TILES@;
    const uint32_t node =
        get_absolute_logical_y() * kColumns + get_absolute_logical_x();
    const auto arena = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<13, 15>(), 0>(),
        get_common_arg_val<uint32_t>(0), kTileBytes);
    CircularBuffer output(cb_output), saved_h(cb_saved_h);
    Noc noc1(1);
    for (uint32_t phase = 0; phase < @PHASE_COUNT@; ++phase) {
        const uint32_t output_tiles = phase == 0 ? hidden_tiles : kv_tiles;
        const uint32_t output_base =
            phase == 0 ? @Q_BASE@ : (phase == 1 ? @K_BASE@ : @V_BASE@);
        const uint32_t h_base =
            phase == 0 ? @HQ_BASE@ : (phase == 1 ? @HK_BASE@ : @HV_BASE@);
        const uint32_t total_jobs = seq_tiles * output_tiles;
        const uint32_t jobs_per_node = (total_jobs + kWorkers - 1) / kWorkers;
        for (uint32_t local = 0; local < jobs_per_node; ++local) {
            const uint32_t job = node * jobs_per_node + local;
            if (job >= total_jobs) continue;
            const uint32_t token = job / output_tiles;
            const uint32_t output_index = job - token * output_tiles;
            output.wait_front(1);
            noc1.async_write(CoreLocalMem<uint32_t>(output.get_read_ptr()),
                             arena, kTileBytes, {},
                             {.page_id = output_base + job});
            noc1.async_write_barrier();
            output.pop_front(1);
            if (output_index == 0) {
                saved_h.wait_front(1);
                noc1.async_write(CoreLocalMem<uint32_t>(saved_h.get_read_ptr()),
                                 arena, kTileBytes, {},
                                 {.page_id = h_base + token});
                noc1.async_write_barrier();
                saved_h.pop_front(1);
            }
        }
    }
}
