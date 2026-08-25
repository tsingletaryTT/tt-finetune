// Write the qualified inverse-RoPE Q/K gradients to one packed arena.

#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc1(1);
    constexpr uint32_t cb_output = get_compile_time_arg_val(6);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t dim_tiles = @DIM_TILES@;
    constexpr uint32_t query_heads = @QUERY_HEADS@;
    constexpr uint32_t kv_heads = @KV_HEADS@;
    constexpr uint32_t jobs = (query_heads + kv_heads) * seq_tiles * dim_tiles;
    constexpr uint32_t jobs_per_node = @JOBS_PER_NODE@;
    constexpr uint32_t tile_bytes = 2048;
    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    const uint32_t arena_addr = get_common_arg_val<uint32_t>(0);
    const auto arena = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<4, 7>(), 0>(),
        arena_addr, tile_bytes);
    CircularBuffer output(cb_output);
    for (uint32_t local = 0; local < jobs_per_node; ++local) {
        const uint32_t job = node * jobs_per_node + local;
        if (job >= jobs) continue;
        output.wait_front(1);
        noc1.async_write(
            CoreLocalMem<uint32_t>(output.get_read_ptr()), arena,
            tile_bytes, {}, {.page_id = job});
        noc1.async_write_barrier();
        output.pop_front(1);
    }
}

