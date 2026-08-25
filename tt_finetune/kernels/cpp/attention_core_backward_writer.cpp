// Drain only final dQ/dK/dV from the resident attention core.

#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc1(1);
    constexpr uint32_t cb_dq = get_compile_time_arg_val(13);
    constexpr uint32_t cb_dk_final = get_compile_time_arg_val(20);
    constexpr uint32_t cb_dv_final = get_compile_time_arg_val(21);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t dim_tiles = @DIM_TILES@;
    constexpr uint32_t vector_tiles = seq_tiles * dim_tiles;
    constexpr uint32_t query_heads = @QUERY_HEADS@;
    constexpr uint32_t kv_heads = @KV_HEADS@;
    constexpr uint32_t group = query_heads / kv_heads;
    constexpr uint32_t tile_bytes = 4096;
    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= kv_heads) return;

    CircularBuffer dq(cb_dq), dk(cb_dk_final), dv(cb_dv_final);
    const uint32_t arena_addr = get_common_arg_val<uint32_t>(0);
    const auto arena = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<5, 23>(), 0>(),
        arena_addr, tile_bytes);
    for (uint32_t member = 0; member < group; ++member) {
        const uint32_t q_head = node * group + member;
        dq.wait_front(vector_tiles);
        for (uint32_t tile = 0; tile < vector_tiles; ++tile) {
            noc1.async_write(
                CoreLocalMem<uint32_t>(dq.get_read_ptr() + tile * tile_bytes),
                arena, tile_bytes, {},
                {.page_id = q_head * vector_tiles + tile});
        }
        noc1.async_write_barrier();
        dq.pop_front(vector_tiles);
    }
    dk.wait_front(vector_tiles);
    dv.wait_front(vector_tiles);
    const uint32_t dk_base = query_heads * vector_tiles;
    const uint32_t dv_base = (query_heads + kv_heads) * vector_tiles;
    for (uint32_t tile = 0; tile < vector_tiles; ++tile) {
        noc1.async_write(
            CoreLocalMem<uint32_t>(dk.get_read_ptr() + tile * tile_bytes),
            arena, tile_bytes, {},
            {.page_id = dk_base + node * vector_tiles + tile});
        noc1.async_write(
            CoreLocalMem<uint32_t>(dv.get_read_ptr() + tile * tile_bytes),
            arena, tile_bytes, {},
            {.page_id = dv_base + node * vector_tiles + tile});
    }
    noc1.async_write_barrier();
    dk.pop_front(vector_tiles);
    dv.pop_front(vector_tiles);
}
