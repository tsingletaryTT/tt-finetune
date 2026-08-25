// Publish head output plus opaque raw-FP32 probability pages.

#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc1(1);
    constexpr uint32_t cb_probability_saved = get_compile_time_arg_val(8);
    constexpr uint32_t cb_head = get_compile_time_arg_val(12);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t dim_tiles = @DIM_TILES@;
    constexpr uint32_t query_heads = @QUERY_HEADS@;
    constexpr uint32_t jobs = query_heads * seq_tiles;
    constexpr uint32_t bf16_bytes = 2048;
    const uint32_t node = get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= jobs) return;
    const uint32_t arena_addr = get_common_arg_val<uint32_t>(0);
    const auto arena = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 13>(), 0>(),
        arena_addr, bf16_bytes);
    CircularBuffer probability(cb_probability_saved), head(cb_head);
    head.wait_front(dim_tiles);
    for (uint32_t tile = 0; tile < dim_tiles; ++tile)
        noc1.async_write(
            CoreLocalMem<uint32_t>(head.get_read_ptr() + tile * bf16_bytes),
            arena, bf16_bytes, {}, {.page_id = node * dim_tiles + tile});
    noc1.async_write_barrier();
    head.pop_front(dim_tiles);
    probability.wait_front(seq_tiles);
    constexpr uint32_t probability_base = jobs * dim_tiles;
    for (uint32_t tile = 0; tile < seq_tiles; ++tile) {
        const uint32_t source = probability.get_read_ptr() + tile * 2 * bf16_bytes;
        const uint32_t page = probability_base + 2 * (node * seq_tiles + tile);
        noc1.async_write(CoreLocalMem<uint32_t>(source), arena, bf16_bytes, {}, {.page_id = page});
        noc1.async_write(CoreLocalMem<uint32_t>(source + bf16_bytes), arena, bf16_bytes, {}, {.page_id = page + 1});
    }
    noc1.async_write_barrier();
    probability.pop_front(seq_tiles);
}
