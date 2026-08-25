// Receive Q/K/V directly from the 50-core producer island and read only
// RoPE/mask data from DRAM. Payload completion always precedes credits.

#include <cstdint>
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {
constexpr uint8_t kPhysicalX[11] = {1,2,3,4,5,6,7,10,11,12,13};
FORCE_INLINE uint8_t physical_y(uint32_t y) { return static_cast<uint8_t>(y + 2); }
FORCE_INLINE uint64_t projection_zero_l1(uint32_t address) {
    return get_noc_addr(kPhysicalX[0], physical_y(0), address);
}
}

void kernel_main() {
    constexpr uint32_t cb_q = 0, cb_q_cos = 1, cb_q_sin = 2;
    constexpr uint32_t cb_k = 3, cb_k_cos = 4, cb_k_sin = 5;
    constexpr uint32_t cb_mask = 17;
    constexpr uint32_t cb_max_scaler = 21, cb_sum_scaler = 24;
    constexpr uint32_t cb_probability_fp32 = 26, cb_score_scale_fp32 = 28;
    constexpr uint32_t cb_value = 29;
    constexpr uint32_t seq_tiles = @SEQ_TILES@, dim_tiles = @DIM_TILES@;
    constexpr uint32_t query_heads = @QUERY_HEADS@, kv_heads = @KV_HEADS@;
    constexpr uint32_t jobs = query_heads * seq_tiles, tile_bytes = 2048;
    constexpr uint32_t q_ready_sem = 0, k_ready_sem = 1, v_ready_sem = 2;
    constexpr uint32_t consumer_ready_sem = 3;
    constexpr uint32_t invocation_done_sem = 15;
    const uint32_t x = get_absolute_logical_x(), y = get_absolute_logical_y();
    if (x < 5) return;
    const uint32_t node = y * 6 + (x - 5);
    if (node >= jobs) return;
    volatile tt_l1_ptr uint32_t* invocation_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(invocation_done_sem));
    noc_semaphore_set(invocation_done, 0);
    const uint32_t head = node / seq_tiles;
    const uint32_t row = node - head * seq_tiles;

    CircularBuffer q(cb_q), k(cb_k), v(cb_value);
    q.reserve_back(dim_tiles);
    k.reserve_back(seq_tiles * dim_tiles);
    v.reserve_back(seq_tiles * dim_tiles);
    volatile tt_l1_ptr uint32_t* q_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(q_ready_sem));
    volatile tt_l1_ptr uint32_t* k_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(k_ready_sem));
    volatile tt_l1_ptr uint32_t* v_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(v_ready_sem));
    noc_semaphore_set(q_ready, 0); noc_semaphore_set(k_ready, 0);
    noc_semaphore_set(v_ready, 0);
    @RESET_O_GO@
    @PUBLISH_TARGET_MAILBOX@
    noc_semaphore_inc(projection_zero_l1(get_semaphore(consumer_ready_sem)), 1);
    noc_async_atomic_barrier();

    const auto cosine = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<3, 32>(), 0>(),
        get_common_arg_val<uint32_t>(0), tile_bytes);
    const auto sine = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<4, 32>(), 1>(),
        get_common_arg_val<uint32_t>(1), tile_bytes);
    const auto mask = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<5, 32>(), 2>(),
        get_common_arg_val<uint32_t>(2), tile_bytes);
    Noc noc0(0);
    CircularBuffer q_cos(cb_q_cos), q_sin(cb_q_sin);
    CircularBuffer k_cos(cb_k_cos), k_sin(cb_k_sin), mask_cb(cb_mask);
    q_cos.reserve_back(dim_tiles); q_sin.reserve_back(dim_tiles);
    for (uint32_t dim = 0; dim < dim_tiles; ++dim) {
        const uint32_t page = row * dim_tiles + dim;
        noc0.async_read(cosine, CoreLocalMem<uint32_t>(q_cos.get_write_ptr() + dim * tile_bytes),
                        tile_bytes, {.page_id = page}, {});
        noc0.async_read(sine, CoreLocalMem<uint32_t>(q_sin.get_write_ptr() + dim * tile_bytes),
                        tile_bytes, {.page_id = page}, {});
    }
    const uint32_t vector_tiles = seq_tiles * dim_tiles;
    k_cos.reserve_back(vector_tiles); k_sin.reserve_back(vector_tiles);
    for (uint32_t token = 0; token < seq_tiles; ++token)
        for (uint32_t dim = 0; dim < dim_tiles; ++dim) {
            const uint32_t local = token * dim_tiles + dim;
            const uint32_t page = local;
            noc0.async_read(cosine, CoreLocalMem<uint32_t>(k_cos.get_write_ptr() + local * tile_bytes),
                            tile_bytes, {.page_id = page}, {});
            noc0.async_read(sine, CoreLocalMem<uint32_t>(k_sin.get_write_ptr() + local * tile_bytes),
                            tile_bytes, {.page_id = page}, {});
        }
    mask_cb.reserve_back(seq_tiles);
    for (uint32_t column = 0; column < seq_tiles; ++column)
        noc0.async_read(mask,
            CoreLocalMem<uint32_t>(mask_cb.get_write_ptr() + column * tile_bytes),
            tile_bytes, {.page_id = row * seq_tiles + column}, {});
    noc0.async_read_barrier();
    noc_semaphore_wait_min(q_ready, dim_tiles);
    noc_semaphore_wait_min(k_ready, vector_tiles);
    noc_semaphore_wait_min(v_ready, vector_tiles);
    noc_semaphore_set(q_ready, 0); noc_semaphore_set(k_ready, 0);
    noc_semaphore_set(v_ready, 0);
    q.push_back(dim_tiles); k.push_back(vector_tiles); v.push_back(vector_tiles);
    q_cos.push_back(dim_tiles); q_sin.push_back(dim_tiles);
    k_cos.push_back(vector_tiles); k_sin.push_back(vector_tiles);
    mask_cb.push_back(seq_tiles);

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_max_scaler, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_sum_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
#if @FP32_CANONICAL_ATTENTION@
    CircularBuffer score_scale(cb_score_scale_fp32);
    score_scale.reserve_back(1);
    auto* scale_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        score_scale.get_write_ptr());
    for (uint32_t element = 0; element < 1024; ++element)
        scale_words[element] = @SCALE_BITS@;
    score_scale.push_back(1);
#endif
#if !@FP32_CANONICAL_ATTENTION@
    CircularBuffer probability_fp32(cb_probability_fp32);
    CircularBuffer probability_bf16(28);
    probability_fp32.wait_front(seq_tiles); probability_bf16.reserve_back(seq_tiles);
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
    probability_bf16.push_back(seq_tiles); probability_fp32.pop_front(seq_tiles);
#endif
    noc_semaphore_wait_min(invocation_done, 2);
}
