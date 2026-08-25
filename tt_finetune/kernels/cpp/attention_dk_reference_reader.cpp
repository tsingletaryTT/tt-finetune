// Reference dK reader: one worker per (query head, head-dimension tile).

#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc0(0);
    constexpr uint32_t cb_q_column = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ds = get_compile_time_arg_val(1);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t dim_tiles = @DIM_TILES@;
    constexpr uint32_t query_heads = @QUERY_HEADS@;
    constexpr uint32_t score_tiles = seq_tiles * seq_tiles;
    constexpr uint32_t vector_tiles = seq_tiles * dim_tiles;
    constexpr uint32_t jobs = query_heads * dim_tiles;
    constexpr uint32_t tile_bytes = 4096;
    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= jobs) return;
    const uint32_t head = node / dim_tiles;
    const uint32_t dim = node - head * dim_tiles;

    CircularBuffer q_column(cb_q_column), ds(cb_ds);
    const uint32_t q_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t ds_addr = get_common_arg_val<uint32_t>(1);
    const auto q_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 6>(), 0>(),
        q_addr, tile_bytes);
    const auto ds_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 6>(), 1>(),
        ds_addr, tile_bytes);
    q_column.reserve_back(seq_tiles);
    for (uint32_t seq = 0; seq < seq_tiles; ++seq) {
        noc0.async_read(
            q_tensor,
            CoreLocalMem<uint32_t>(q_column.get_write_ptr() + seq * tile_bytes),
            tile_bytes,
            {.page_id = head * vector_tiles + seq * dim_tiles + dim}, {});
    }
    ds.reserve_back(score_tiles);
    for (uint32_t tile = 0; tile < score_tiles; ++tile) {
        noc0.async_read(
            ds_tensor,
            CoreLocalMem<uint32_t>(ds.get_write_ptr() + tile * tile_bytes),
            tile_bytes, {.page_id = head * score_tiles + tile}, {});
    }
    noc0.async_read_barrier();
    q_column.push_back(seq_tiles);
    ds.push_back(score_tiles);
}
