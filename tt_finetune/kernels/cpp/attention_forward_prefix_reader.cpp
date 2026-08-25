// Token-major Q/K -> query-row-local RoPE/QK inputs.

#include <cstdint>
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_q = 0;
    constexpr uint32_t cb_q_cos = 1;
    constexpr uint32_t cb_q_sin = 2;
    constexpr uint32_t cb_k = 3;
    constexpr uint32_t cb_k_cos = 4;
    constexpr uint32_t cb_k_sin = 5;
    constexpr uint32_t cb_mask = 17;
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t dim_tiles = @DIM_TILES@;
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@;
    constexpr uint32_t kv_tiles = @KV_TILES@;
    constexpr uint32_t query_heads = @QUERY_HEADS@;
    constexpr uint32_t kv_heads = @KV_HEADS@;
    constexpr uint32_t group = query_heads / kv_heads;
    constexpr uint32_t jobs = query_heads * seq_tiles;
    constexpr uint32_t tile_bytes = 2048;
    const uint32_t node = get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= jobs) return;
    const uint32_t head = node / seq_tiles;
    const uint32_t row = node - head * seq_tiles;
    const uint32_t kv_head = head / group;

    const auto query = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 20>(), 0>(),
        get_common_arg_val<uint32_t>(0), tile_bytes);
    const auto key = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 20>(), 1>(),
        get_common_arg_val<uint32_t>(1), tile_bytes);
    const auto cosine = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 20>(), 2>(),
        get_common_arg_val<uint32_t>(2), tile_bytes);
    const auto sine = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<3, 20>(), 3>(),
        get_common_arg_val<uint32_t>(3), tile_bytes);
    const auto mask = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<4, 20>(), 4>(),
        get_common_arg_val<uint32_t>(4), tile_bytes);

    Noc noc0(0);
    CircularBuffer q(cb_q), q_cos(cb_q_cos), q_sin(cb_q_sin);
    CircularBuffer k(cb_k), k_cos(cb_k_cos), k_sin(cb_k_sin), mask_cb(cb_mask);
    q.reserve_back(dim_tiles);
    q_cos.reserve_back(dim_tiles);
    q_sin.reserve_back(dim_tiles);
    for (uint32_t dim = 0; dim < dim_tiles; ++dim) {
        const uint32_t q_page = row * hidden_tiles + head * dim_tiles + dim;
        const uint32_t rope_page = row * dim_tiles + dim;
        noc0.async_read(query, CoreLocalMem<uint32_t>(q.get_write_ptr() + dim * tile_bytes),
                        tile_bytes, {.page_id = q_page}, {});
        noc0.async_read(cosine, CoreLocalMem<uint32_t>(q_cos.get_write_ptr() + dim * tile_bytes),
                        tile_bytes, {.page_id = rope_page}, {});
        noc0.async_read(sine, CoreLocalMem<uint32_t>(q_sin.get_write_ptr() + dim * tile_bytes),
                        tile_bytes, {.page_id = rope_page}, {});
    }
    const uint32_t vector_tiles = seq_tiles * dim_tiles;
    k.reserve_back(vector_tiles);
    k_cos.reserve_back(vector_tiles);
    k_sin.reserve_back(vector_tiles);
    for (uint32_t token = 0; token < seq_tiles; ++token) {
        for (uint32_t dim = 0; dim < dim_tiles; ++dim) {
            const uint32_t local = token * dim_tiles + dim;
            const uint32_t k_page = token * kv_tiles + kv_head * dim_tiles + dim;
            const uint32_t rope_page = token * dim_tiles + dim;
            noc0.async_read(key, CoreLocalMem<uint32_t>(k.get_write_ptr() + local * tile_bytes),
                            tile_bytes, {.page_id = k_page}, {});
            noc0.async_read(cosine, CoreLocalMem<uint32_t>(k_cos.get_write_ptr() + local * tile_bytes),
                            tile_bytes, {.page_id = rope_page}, {});
            noc0.async_read(sine, CoreLocalMem<uint32_t>(k_sin.get_write_ptr() + local * tile_bytes),
                            tile_bytes, {.page_id = rope_page}, {});
        }
    }
    mask_cb.reserve_back(seq_tiles);
    for (uint32_t column = 0; column < seq_tiles; ++column)
        noc0.async_read(mask,
            CoreLocalMem<uint32_t>(mask_cb.get_write_ptr() + column * tile_bytes),
            tile_bytes, {.page_id = row * seq_tiles + column}, {});
    noc0.async_read_barrier();
    q.push_back(dim_tiles); q_cos.push_back(dim_tiles); q_sin.push_back(dim_tiles);
    k.push_back(vector_tiles); k_cos.push_back(vector_tiles); k_sin.push_back(vector_tiles);
    mask_cb.push_back(seq_tiles);
}
