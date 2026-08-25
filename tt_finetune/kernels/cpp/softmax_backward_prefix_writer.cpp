// Final-only BF16 diagnostic writer.

#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc1(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(8);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t heads = @HEADS@;
    constexpr uint32_t jobs = heads * seq_tiles;
    constexpr uint32_t tile_bytes = 2048;
    const uint32_t node = get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= jobs) return;
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const auto output = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 9>(), 0>(),
        output_addr, tile_bytes);
    CircularBuffer out(cb_out);
    out.wait_front(seq_tiles);
    for (uint32_t tile = 0; tile < seq_tiles; ++tile) {
        noc1.async_write(
            CoreLocalMem<uint32_t>(out.get_read_ptr() + tile * tile_bytes),
            output, tile_bytes, {}, {.page_id = node * seq_tiles + tile});
    }
    noc1.async_write_barrier();
    out.pop_front(seq_tiles);
}
