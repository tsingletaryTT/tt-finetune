// Write only the final reduced dK or dV tile row.

#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc1(1);
    constexpr uint32_t cb_output = get_compile_time_arg_val(2);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t dim_tiles = @DIM_TILES@;
    constexpr uint32_t kv_heads = @KV_HEADS@;
    constexpr uint32_t jobs = kv_heads * seq_tiles;
    constexpr uint32_t tile_bytes = 2048;
    const uint32_t node = get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= jobs) return;
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const auto output_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 3>(), 0>(),
        output_addr, tile_bytes);
    CircularBuffer output(cb_output);
    output.wait_front(dim_tiles);
    for (uint32_t tile = 0; tile < dim_tiles; ++tile) {
        noc1.async_write(
            CoreLocalMem<uint32_t>(output.get_read_ptr() + tile * tile_bytes),
            output_tensor, tile_bytes, {}, {.page_id = node * dim_tiles + tile});
    }
    noc1.async_write_barrier();
    output.pop_front(dim_tiles);
}
