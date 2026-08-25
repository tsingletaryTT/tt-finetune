// Minimal row-sharded reader for exact FP32 softmax-backward qualification.

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
    constexpr uint32_t cb_dp = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t heads = @HEADS@;
    constexpr uint32_t jobs = heads * seq_tiles;
    constexpr uint32_t score_tiles = seq_tiles * seq_tiles;
    constexpr uint32_t p_tile_bytes = 4096;
    constexpr uint32_t bf16_tile_bytes = 2048;

    const uint32_t node = get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= jobs) return;
    const uint32_t head = node / seq_tiles;
    const uint32_t row = node - head * seq_tiles;

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    const uint32_t p_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t dp_addr = get_common_arg_val<uint32_t>(1);
    const auto p_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 9>(), 0>(),
        p_addr, p_tile_bytes);
    const auto dp_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 9>(), 1>(),
        dp_addr, bf16_tile_bytes);

    CircularBuffer p(cb_p), dp(cb_dp);
    p.reserve_back(seq_tiles);
    dp.reserve_back(seq_tiles);
    for (uint32_t tile = 0; tile < seq_tiles; ++tile) {
        const uint32_t page = head * score_tiles + row * seq_tiles + tile;
        noc0.async_read(
            p_tensor,
            CoreLocalMem<uint32_t>(p.get_write_ptr() + tile * p_tile_bytes),
            p_tile_bytes, {.page_id = page}, {});
        noc0.async_read(
            dp_tensor,
            CoreLocalMem<uint32_t>(dp.get_write_ptr() + tile * bf16_tile_bytes),
            bf16_tile_bytes, {.page_id = page}, {});
    }
    noc0.async_read_barrier();
    p.push_back(seq_tiles);
    dp.push_back(seq_tiles);
}
