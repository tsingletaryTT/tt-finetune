// FP32 resident GQA-attention backward reader.  One worker owns one KV head
// and streams its query-head siblings through the same local K/V tiles.

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc0(0);
    constexpr uint32_t cb_p = get_compile_time_arg_val(0);
    constexpr uint32_t cb_do = get_compile_time_arg_val(1);
    constexpr uint32_t cb_q = get_compile_time_arg_val(2);
    constexpr uint32_t cb_k = get_compile_time_arg_val(3);
    constexpr uint32_t cb_v = get_compile_time_arg_val(4);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(5);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t dim_tiles = @DIM_TILES@;
    constexpr uint32_t score_tiles = seq_tiles * seq_tiles;
    constexpr uint32_t vector_tiles = seq_tiles * dim_tiles;
    constexpr uint32_t query_heads = @QUERY_HEADS@;
    constexpr uint32_t kv_heads = @KV_HEADS@;
    constexpr uint32_t group = query_heads / kv_heads;
    constexpr uint32_t tile_bytes = 4096;

    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= kv_heads) return;

    CircularBuffer p(cb_p), dout(cb_do), q(cb_q), k(cb_k), v(cb_v);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    const uint32_t p_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t do_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t q_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t k_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t v_addr = get_common_arg_val<uint32_t>(4);
    const auto p_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 23>(), 0>(),
        p_addr, tile_bytes);
    const auto do_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 23>(), 1>(),
        do_addr, tile_bytes);
    const auto q_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 23>(), 2>(),
        q_addr, tile_bytes);
    const auto k_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<3, 23>(), 3>(),
        k_addr, tile_bytes);
    const auto v_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<4, 23>(), 4>(),
        v_addr, tile_bytes);

    k.reserve_back(vector_tiles);
    v.reserve_back(vector_tiles);
    for (uint32_t tile = 0; tile < vector_tiles; ++tile) {
        noc0.async_read(k_tensor,
            CoreLocalMem<uint32_t>(k.get_write_ptr() + tile * tile_bytes),
            tile_bytes, {.page_id = node * vector_tiles + tile}, {});
        noc0.async_read(v_tensor,
            CoreLocalMem<uint32_t>(v.get_write_ptr() + tile * tile_bytes),
            tile_bytes, {.page_id = node * vector_tiles + tile}, {});
    }
    noc0.async_read_barrier();
    k.push_back(vector_tiles);
    v.push_back(vector_tiles);

    for (uint32_t member = 0; member < group; ++member) {
        const uint32_t q_head = node * group + member;
        p.reserve_back(score_tiles);
        dout.reserve_back(vector_tiles);
        q.reserve_back(vector_tiles);
        for (uint32_t tile = 0; tile < score_tiles; ++tile) {
            noc0.async_read(p_tensor,
                CoreLocalMem<uint32_t>(p.get_write_ptr() + tile * tile_bytes),
                tile_bytes, {.page_id = q_head * score_tiles + tile}, {});
        }
        for (uint32_t tile = 0; tile < vector_tiles; ++tile) {
            noc0.async_read(do_tensor,
                CoreLocalMem<uint32_t>(dout.get_write_ptr() + tile * tile_bytes),
                tile_bytes, {.page_id = q_head * vector_tiles + tile}, {});
            noc0.async_read(q_tensor,
                CoreLocalMem<uint32_t>(q.get_write_ptr() + tile * tile_bytes),
                tile_bytes, {.page_id = q_head * vector_tiles + tile}, {});
        }
        noc0.async_read_barrier();
        p.push_back(score_tiles);
        dout.push_back(vector_tiles);
        q.push_back(vector_tiles);
    }
}
