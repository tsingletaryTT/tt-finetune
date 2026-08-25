// Write transposed dK-head tiles in [head, sequence, dimension] order.

#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc1(1);
    constexpr uint32_t cb_dk = get_compile_time_arg_val(4);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t dim_tiles = @DIM_TILES@;
    constexpr uint32_t query_heads = @QUERY_HEADS@;
    constexpr uint32_t jobs = query_heads * dim_tiles;
    constexpr uint32_t vector_tiles = seq_tiles * dim_tiles;
    constexpr uint32_t tile_bytes = 4096;
    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= jobs) return;
    const uint32_t head = node / dim_tiles;
    const uint32_t dim = node - head * dim_tiles;
    CircularBuffer dk(cb_dk);
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const auto output = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 6>(), 0>(),
        output_addr, tile_bytes);
    dk.wait_front(seq_tiles);
    for (uint32_t seq = 0; seq < seq_tiles; ++seq) {
        noc1.async_write(
            CoreLocalMem<uint32_t>(dk.get_read_ptr() + seq * tile_bytes),
            output, tile_bytes, {},
            {.page_id = head * vector_tiles + seq * dim_tiles + dim});
    }
    noc1.async_write_barrier();
    dk.pop_front(seq_tiles);
}
