// One-call RoPE/QK/softmax/PV reader. All score/probability edges stay local.

#include <cstdint>
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/moreh_common.hpp"

void kernel_main() {
    WAYPOINT("AFR0");
    constexpr uint32_t cb_q = 0, cb_q_cos = 1, cb_q_sin = 2;
    constexpr uint32_t cb_k = 3, cb_k_cos = 4, cb_k_sin = 5;
    constexpr uint32_t cb_mask = 17;
    constexpr uint32_t cb_max_scaler = 21, cb_sum_scaler = 24;
    constexpr uint32_t cb_probability_fp32 = 26, cb_score_scale_fp32 = 28;
    constexpr uint32_t cb_value = 29;
    constexpr uint32_t seq_tiles = @SEQ_TILES@, dim_tiles = @DIM_TILES@;
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@, kv_tiles = @KV_TILES@;
    constexpr uint32_t query_heads = @QUERY_HEADS@, kv_heads = @KV_HEADS@;
    constexpr uint32_t group = query_heads / kv_heads;
    constexpr uint32_t jobs = query_heads * seq_tiles;
    constexpr uint32_t tile_bytes = 2048;
    const uint32_t node = get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= jobs) return;
    const uint32_t head = node / seq_tiles;
    const uint32_t row = node - head * seq_tiles;
    const uint32_t kv_head = head / group;

    const auto query = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 32>(), 0>(),
        get_common_arg_val<uint32_t>(0), tile_bytes);
    const auto key = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 32>(), 1>(),
        get_common_arg_val<uint32_t>(1), tile_bytes);
    const auto value = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 32>(), 2>(),
        get_common_arg_val<uint32_t>(2), tile_bytes);
    const auto cosine = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<3, 32>(), 3>(),
        get_common_arg_val<uint32_t>(3), tile_bytes);
    const auto sine = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<4, 32>(), 4>(),
        get_common_arg_val<uint32_t>(4), tile_bytes);
    const auto mask = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<5, 32>(), 5>(),
        get_common_arg_val<uint32_t>(5), tile_bytes);

    Noc noc0(0);
    CircularBuffer q(cb_q), q_cos(cb_q_cos), q_sin(cb_q_sin);
    CircularBuffer k(cb_k), k_cos(cb_k_cos), k_sin(cb_k_sin);
    CircularBuffer v(cb_value), mask_cb(cb_mask);
    q.reserve_back(dim_tiles); q_cos.reserve_back(dim_tiles); q_sin.reserve_back(dim_tiles);
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
    k.reserve_back(vector_tiles); k_cos.reserve_back(vector_tiles);
    k_sin.reserve_back(vector_tiles); v.reserve_back(vector_tiles);
    for (uint32_t token = 0; token < seq_tiles; ++token) {
        for (uint32_t dim = 0; dim < dim_tiles; ++dim) {
            const uint32_t local = token * dim_tiles + dim;
            const uint32_t page = token * kv_tiles + kv_head * dim_tiles + dim;
            const uint32_t rope_page = token * dim_tiles + dim;
            noc0.async_read(key, CoreLocalMem<uint32_t>(k.get_write_ptr() + local * tile_bytes),
                            tile_bytes, {.page_id = page}, {});
            noc0.async_read(value, CoreLocalMem<uint32_t>(v.get_write_ptr() + local * tile_bytes),
                            tile_bytes, {.page_id = page}, {});
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
    k.push_back(vector_tiles); k_cos.push_back(vector_tiles);
    k_sin.push_back(vector_tiles); v.push_back(vector_tiles);
    mask_cb.push_back(seq_tiles);
    WAYPOINT("AFR1");

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_max_scaler, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_sum_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    WAYPOINT("AFR2");

#if @FP32_CANONICAL_ATTENTION@ && !@CANONICAL_POW2_SCALE@
    CircularBuffer score_scale(cb_score_scale_fp32);
    score_scale.reserve_back(1);
    auto* scale_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        score_scale.get_write_ptr());
    for (uint32_t element = 0; element < 1024; ++element)
        scale_words[element] = @SCALE_BITS@;
    score_scale.push_back(1);
#endif

#if @CANONICAL_MOREH_SOFTMAX@
    WAYPOINT("AFR3");
    // Mirror SoftmaxProgramFactoryGeneralWSmall's dedicated FP32 tail mask.
    // S=128 has no partial column tile, but the materialized all-ones mask is
    // still part of the kernel's observable initialization schedule.
    DataflowBuffer softmax_tail_mask(@SOFTMAX_TAIL_MASK_CB@);
    generate_mask_w<uint32_t>(softmax_tail_mask, 32);
#endif

#if @PUBLISH_RAW_SCORE_CHECKPOINT@ && !@CANONICAL_POW2_SCALE@
    // Snapshot the post-BF16-boundary FP32 softmax input without DRAM. The
    // compute RISC waits for the CB16 credit before consuming CB20.
    CircularBuffer softmax_input(20), saved_softmax_input(27), checkpoint_ack(16);
    softmax_input.wait_front(seq_tiles);
    saved_softmax_input.reserve_back(seq_tiles);
    for (uint32_t tile = 0; tile < seq_tiles; ++tile) {
        auto* source = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            softmax_input.get_read_ptr() + tile * 4096);
        auto* destination = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            saved_softmax_input.get_write_ptr() + tile * 4096);
        for (uint32_t word = 0; word < 1024; ++word)
            destination[word] = source[word];
    }
    saved_softmax_input.push_back(seq_tiles);
    checkpoint_ack.reserve_back(1);
    checkpoint_ack.push_back(1);
#endif

#if !@PUBLISH_RAW_SCORE_CHECKPOINT@
    // The canonical compute kernel packs only one output FIFO. Preserve that
    // schedule exactly, then copy the finished FP32 bytes locally for the
    // backward checkpoint instead of issuing a second compute-side pack.
    CircularBuffer probability(26), saved_probability(27);
    probability.wait_front(seq_tiles);
    WAYPOINT("AFR4");
    saved_probability.reserve_back(seq_tiles);
    for (uint32_t tile = 0; tile < seq_tiles; ++tile) {
        auto* source = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            probability.get_read_ptr() + tile * 4096);
        auto* destination = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            saved_probability.get_write_ptr() + tile * 4096);
        for (uint32_t word = 0; word < 1024; ++word)
            destination[word] = source[word];
    }
    saved_probability.push_back(seq_tiles);
    WAYPOINT("AFR5");
#endif

#if !@FP32_CANONICAL_ATTENTION@
    // Legacy exact device-local RNE handoff from FP32 softmax to BF16 P@V.
    CircularBuffer probability_fp32(cb_probability_fp32);
    CircularBuffer probability_bf16(@PROBABILITY_BF16_CB@);
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
            destination[element] = static_cast<uint16_t>(rounded >> 16);
        }
    }
    probability_bf16.push_back(seq_tiles);
    probability_fp32.pop_front(seq_tiles);
#endif
    WAYPOINT("AFR9");
}
