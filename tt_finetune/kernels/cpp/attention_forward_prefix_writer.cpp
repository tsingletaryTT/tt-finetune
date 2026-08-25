// Diagnostic writer for the resident RoPE/QK prefix qualification boundary.

#include <cstdint>
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_q_saved = 9, cb_k_saved = 13, cb_masked = 18;
    constexpr uint32_t seq_tiles = @SEQ_TILES@, dim_tiles = @DIM_TILES@;
    constexpr uint32_t query_heads = @QUERY_HEADS@, kv_heads = @KV_HEADS@;
    constexpr uint32_t group = query_heads / kv_heads;
    constexpr uint32_t jobs = query_heads * seq_tiles;
    constexpr uint32_t tile_bytes = 2048;
    const uint32_t node = get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= jobs) return;
    const uint32_t head = node / seq_tiles;
    const uint32_t row = node - head * seq_tiles;
    const uint32_t kv_head = head / group;
    const uint32_t q_pages = query_heads * seq_tiles * dim_tiles;
    const uint32_t k_pages = kv_heads * seq_tiles * dim_tiles;
    const auto arena = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<5, 20>(), 0>(),
        get_common_arg_val<uint32_t>(0), tile_bytes);
    Noc noc1(1);
    CircularBuffer q(cb_q_saved), k(cb_k_saved), score(cb_masked);
    q.wait_front(dim_tiles);
    for (uint32_t dim = 0; dim < dim_tiles; ++dim)
        noc1.async_write(CoreLocalMem<uint32_t>(q.get_read_ptr() + dim * tile_bytes),
                         arena, tile_bytes, {}, {.page_id = node * dim_tiles + dim});
    noc1.async_write_barrier(); q.pop_front(dim_tiles);

    k.wait_front(seq_tiles * dim_tiles);
    if (head % group == 0 && row == 0) {
        for (uint32_t tile = 0; tile < seq_tiles * dim_tiles; ++tile)
            noc1.async_write(CoreLocalMem<uint32_t>(k.get_read_ptr() + tile * tile_bytes),
                             arena, tile_bytes, {},
                             {.page_id = q_pages + kv_head * seq_tiles * dim_tiles + tile});
        noc1.async_write_barrier();
    }
    k.pop_front(seq_tiles * dim_tiles);

    const uint32_t score_base = q_pages + k_pages;
    score.wait_front(seq_tiles);
    for (uint32_t tile = 0; tile < seq_tiles; ++tile)
        noc1.async_write(CoreLocalMem<uint32_t>(score.get_read_ptr() + tile * tile_bytes),
                         arena, tile_bytes, {},
                         {.page_id = score_base + node * seq_tiles + tile});
    noc1.async_write_barrier(); score.pop_front(seq_tiles);
}
