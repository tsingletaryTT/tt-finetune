// Query-row-sharded FP32 attention backward reader.

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc0(0);
    constexpr uint32_t cb_p_row = get_compile_time_arg_val(0);
    constexpr uint32_t cb_p_column = get_compile_time_arg_val(1);
    constexpr uint32_t cb_dout_row = get_compile_time_arg_val(2);
    constexpr uint32_t cb_dout_full = get_compile_time_arg_val(3);
    constexpr uint32_t cb_k = get_compile_time_arg_val(4);
    constexpr uint32_t cb_v = get_compile_time_arg_val(5);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(6);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t dim_tiles = @DIM_TILES@;
    constexpr uint32_t query_heads = @QUERY_HEADS@;
    constexpr uint32_t score_tiles = seq_tiles * seq_tiles;
    constexpr uint32_t vector_tiles = seq_tiles * dim_tiles;
    constexpr uint32_t jobs = query_heads * seq_tiles;
    constexpr uint32_t tile_bytes = 4096;

    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= jobs) return;
    const uint32_t head = node / seq_tiles;
    const uint32_t row = node - head * seq_tiles;

    CircularBuffer p_row(cb_p_row), p_column(cb_p_column);
    CircularBuffer dout_row(cb_dout_row), dout_full(cb_dout_full);
    CircularBuffer k(cb_k), v(cb_v);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    const uint32_t p_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t dout_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t k_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t v_addr = get_common_arg_val<uint32_t>(3);
    const auto p_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 19>(), 0>(),
        p_addr, tile_bytes);
    const auto dout_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 19>(), 1>(),
        dout_addr, tile_bytes);
    const auto k_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 19>(), 2>(),
        k_addr, tile_bytes);
    const auto v_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<3, 19>(), 3>(),
        v_addr, tile_bytes);

    p_row.reserve_back(seq_tiles);
    p_column.reserve_back(seq_tiles);
    for (uint32_t tile = 0; tile < seq_tiles; ++tile) {
        noc0.async_read(
            p_tensor,
            CoreLocalMem<uint32_t>(p_row.get_write_ptr() + tile * tile_bytes),
            tile_bytes,
            {.page_id = head * score_tiles + row * seq_tiles + tile}, {});
        noc0.async_read(
            p_tensor,
            CoreLocalMem<uint32_t>(p_column.get_write_ptr() + tile * tile_bytes),
            tile_bytes,
            {.page_id = head * score_tiles + tile * seq_tiles + row}, {});
    }
    dout_row.reserve_back(dim_tiles);
    dout_full.reserve_back(vector_tiles);
    for (uint32_t tile = 0; tile < vector_tiles; ++tile) {
        noc0.async_read(
            dout_tensor,
            CoreLocalMem<uint32_t>(dout_full.get_write_ptr() + tile * tile_bytes),
            tile_bytes, {.page_id = head * vector_tiles + tile}, {});
    }
    for (uint32_t tile = 0; tile < dim_tiles; ++tile) {
        noc0.async_read(
            dout_tensor,
            CoreLocalMem<uint32_t>(dout_row.get_write_ptr() + tile * tile_bytes),
            tile_bytes,
            {.page_id = head * vector_tiles + row * dim_tiles + tile}, {});
    }
    k.reserve_back(vector_tiles);
    v.reserve_back(vector_tiles);
    for (uint32_t tile = 0; tile < vector_tiles; ++tile) {
        noc0.async_read(
            k_tensor,
            CoreLocalMem<uint32_t>(k.get_write_ptr() + tile * tile_bytes),
            tile_bytes, {.page_id = head * vector_tiles + tile}, {});
    }
    // V is staged directly in V.T matmul order: dimension-major across
    // sequence tiles.  Sequential packing after within-tile transpose is
    // numerically observable and matches TTNN's permute reader.
    for (uint32_t dim = 0; dim < dim_tiles; ++dim) {
        for (uint32_t seq = 0; seq < seq_tiles; ++seq) {
            const uint32_t target = dim * seq_tiles + seq;
            const uint32_t source = @VALUE_PRETRANSPOSED@
                ? target : (seq * dim_tiles + dim);
        noc0.async_read(
            v_tensor,
                CoreLocalMem<uint32_t>(v.get_write_ptr() + target * tile_bytes),
                tile_bytes, {.page_id = head * vector_tiles + source}, {});
        }
    }
    noc0.async_read_barrier();
    p_row.push_back(seq_tiles);
    p_column.push_back(seq_tiles);
    dout_row.push_back(dim_tiles);
    dout_full.push_back(vector_tiles);
    k.push_back(vector_tiles);
    v.push_back(vector_tiles);
}
