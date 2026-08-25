// Query-row reader for exact causal softmax + value forward.

#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    Noc noc0(0);
    constexpr uint32_t cb_score = get_compile_time_arg_val(0);
    constexpr uint32_t cb_max_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sum_scaler = get_compile_time_arg_val(5);
    constexpr uint32_t cb_probability_fp32 = get_compile_time_arg_val(7);
    constexpr uint32_t cb_probability_bf16 = get_compile_time_arg_val(9);
    constexpr uint32_t cb_value = get_compile_time_arg_val(10);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t dim_tiles = @DIM_TILES@;
    constexpr uint32_t query_heads = @QUERY_HEADS@;
    constexpr uint32_t kv_heads = @KV_HEADS@;
    constexpr uint32_t group = query_heads / kv_heads;
    constexpr uint32_t jobs = query_heads * seq_tiles;
    constexpr uint32_t bf16_tile_bytes = 2048;
    const uint32_t node = get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= jobs) return;
    const uint32_t head = node / seq_tiles;
    const uint32_t row = node - head * seq_tiles;
    const uint32_t kv_head = head / group;

    const uint32_t score_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t value_addr = get_common_arg_val<uint32_t>(1);
    const auto score_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 13>(), 0>(),
        score_addr, bf16_tile_bytes);
    const auto value_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 13>(), 1>(),
        value_addr, bf16_tile_bytes);
    CircularBuffer score(cb_score), value(cb_value);
    score.reserve_back(seq_tiles);
    for (uint32_t tile = 0; tile < seq_tiles; ++tile) {
        const uint32_t page = head * seq_tiles * seq_tiles + row * seq_tiles + tile;
        noc0.async_read(
            score_tensor,
            CoreLocalMem<uint32_t>(score.get_write_ptr() + tile * bf16_tile_bytes),
            bf16_tile_bytes, {.page_id = page}, {});
    }
    value.reserve_back(seq_tiles * dim_tiles);
    for (uint32_t tile = 0; tile < seq_tiles * dim_tiles; ++tile) {
        noc0.async_read(
            value_tensor,
            CoreLocalMem<uint32_t>(value.get_write_ptr() + tile * bf16_tile_bytes),
            bf16_tile_bytes,
            {.page_id = kv_head * seq_tiles * dim_tiles + tile}, {});
    }
    noc0.async_read_barrier();
    score.push_back(seq_tiles);
    value.push_back(seq_tiles * dim_tiles);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_max_scaler, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_sum_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    // TTNN's standalone FP32 -> BF16 edge is IEEE round-to-nearest-even.
    // Keeping this conversion on the data-movement RISC-V has two useful
    // properties for the resident owner: it does not reset the Tensix math
    // state between softmax and P@V, and it gives an explicit bit contract
    // for the exact-mode halfway cases.  The physical FP32/BF16 tile element
    // order is identical; only the datum width changes.
    CircularBuffer probability_fp32(cb_probability_fp32);
    CircularBuffer probability_bf16(cb_probability_bf16);
    probability_fp32.wait_front(seq_tiles);
    probability_bf16.reserve_back(seq_tiles);
    for (uint32_t tile = 0; tile < seq_tiles; ++tile) {
        auto* source = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            probability_fp32.get_read_ptr() + tile * 4096);
        auto* destination = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
            probability_bf16.get_write_ptr() + tile * 2048);
        for (uint32_t element = 0; element < 1024; ++element) {
            const uint32_t bits = source[element];
            const uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
            const uint16_t value = static_cast<uint16_t>(rounded >> 16);
            destination[element] = value;
        }
    }
    probability_bf16.push_back(seq_tiles);
    probability_fp32.pop_front(seq_tiles);
}
