// Receive Q/K/V directly from the 50-core producer island and read only
// RoPE/mask data from DRAM. Payload completion always precedes credits.

#include <cstdint>
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/moreh_common.hpp"

namespace {
constexpr uint8_t kPhysicalX[11] = {1,2,3,4,5,6,7,10,11,12,13};
FORCE_INLINE uint8_t physical_y(uint32_t y) { return static_cast<uint8_t>(y + 2); }
FORCE_INLINE uint64_t projection_zero_l1(uint32_t address) {
    return get_noc_addr(kPhysicalX[0], physical_y(0), address);
}
FORCE_INLINE uint64_t attention_l1(uint32_t node, uint32_t address) {
    const uint32_t logical_x = 5 + node % 6;
    const uint32_t logical_y = node / 6;
    return get_noc_addr(
        kPhysicalX[logical_x], physical_y(logical_y), address);
}
}

void kernel_main() {
    constexpr bool prefix_only = @PROJECTION_ROPE_PREFIX_ONLY@;
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
    constexpr uint32_t grouped_probability_ready_sem = 13;
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

#if @GROUPED_PV_OWNER@
    // This is initialized before the global consumer-ready credit below.
    // Q/K/V production cannot start until every attention reader has issued
    // that credit, so no peer can publish a probability row before the head
    // leader has cleared its ready counter.
    const bool grouped_pv_leader = row == 0;
    if (grouped_pv_leader) {
        volatile tt_l1_ptr uint32_t* grouped_ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(grouped_probability_ready_sem));
        noc_semaphore_set(grouped_ready, 0);
    }
#endif

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
    if constexpr (!prefix_only) {
        mask_cb.reserve_back(seq_tiles);
        for (uint32_t column = 0; column < seq_tiles; ++column)
            noc0.async_read(mask,
                CoreLocalMem<uint32_t>(mask_cb.get_write_ptr() + column * tile_bytes),
                tile_bytes, {.page_id = row * seq_tiles + column}, {});
    }
    noc0.async_read_barrier();
    noc_semaphore_wait_min(q_ready, dim_tiles);
    noc_semaphore_wait_min(k_ready, vector_tiles);
    noc_semaphore_wait_min(v_ready, vector_tiles);
    noc_semaphore_set(q_ready, 0); noc_semaphore_set(k_ready, 0);
    noc_semaphore_set(v_ready, 0);
    q.push_back(dim_tiles); k.push_back(vector_tiles); v.push_back(vector_tiles);
    q_cos.push_back(dim_tiles); q_sin.push_back(dim_tiles);
    k_cos.push_back(vector_tiles); k_sin.push_back(vector_tiles);
    if constexpr (!prefix_only) mask_cb.push_back(seq_tiles);

    if constexpr (!prefix_only) {
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_max_scaler, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_sum_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
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
    DataflowBuffer softmax_tail_mask(28);
    generate_mask_w<uint32_t>(softmax_tail_mask, 32);
#endif
#if @PUBLISH_RAW_SCORE_CHECKPOINT@ && !@CANONICAL_POW2_SCALE@
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
    checkpoint_ack.reserve_back(1); checkpoint_ack.push_back(1);
#endif
#if !@PUBLISH_RAW_SCORE_CHECKPOINT@
    CircularBuffer probability(26), saved_probability(27);
    probability.wait_front(seq_tiles);
#if @GROUPED_PV_OWNER@
    // Recreate TTNN's accepted per-head 4x2 reuse block without a DRAM or
    // PJRT boundary.  Each row publishes its four FP32 probability tiles
    // directly into the row-0 leader's CB16, ordered K-major exactly as the
    // 2-D matmul reader would have produced them.
    // CB27 has no earlier compute-consumer epoch.  Use it as the canonical
    // K-major grouped-matmul input, avoiding the one-tile unpack offset left
    // by CB16's scale-completion credit.
    constexpr uint32_t grouped_rows = 4;
    constexpr uint32_t group_row_base = 0;
    const uint32_t row_in_group = row;
    constexpr uint32_t leader_row = 0;
    if (grouped_pv_leader)
        saved_probability.reserve_back(seq_tiles * grouped_rows);
    const uint32_t leader = head * seq_tiles + leader_row;
    const uint32_t grouped_base = saved_probability.get_write_ptr();
    for (uint32_t k = 0; k < seq_tiles; ++k) {
        const uint32_t source = probability.get_read_ptr() + k * 4096;
        const uint32_t destination =
            grouped_base + (k * grouped_rows + row_in_group) * 4096;
        noc_async_write(
            source, attention_l1(leader, destination), 4096);
    }
    noc_async_write_barrier();
    noc_semaphore_inc(
        attention_l1(leader, get_semaphore(grouped_probability_ready_sem)),
        1);
    noc_async_atomic_barrier();
    if (grouped_pv_leader) {
        volatile tt_l1_ptr uint32_t* grouped_ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(grouped_probability_ready_sem));
        noc_semaphore_wait_min(grouped_ready, grouped_rows);
        noc_semaphore_set(grouped_ready, 0);
        saved_probability.push_back(seq_tiles * grouped_rows);
        // BF16 -> FP32 is an exact exponent-field extension. Perform it on
        // the data-movement RISC into a fresh reader-produced CB28 instead of
        // running a typecast in the same compute kernel immediately before
        // P@V. The accepted TTNN path also enters P@V through a new program
        // boundary, and that clean compute state is numerically observable in
        // the fourth K block.
        CircularBuffer value_fp32(28);
        value_fp32.reserve_back(seq_tiles * dim_tiles);
        for (uint32_t tile = 0; tile < seq_tiles * dim_tiles; ++tile) {
            auto* source = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                v.get_read_ptr() + tile * 2048);
            auto* destination = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                value_fp32.get_write_ptr() + tile * 4096);
            for (uint32_t element = 0; element < 1024; ++element)
                destination[element] =
                    static_cast<uint32_t>(source[element]) << 16;
        }
        value_fp32.push_back(seq_tiles * dim_tiles);
    } else {
        // The compute RISC must not retire its local probability generation
        // until this reader has finished the NoC publication above.
        CircularBuffer publication_ack(28);
        publication_ack.reserve_back(1);
        publication_ack.push_back(1);
    }
#else
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
#endif
#endif
#if !@FP32_CANONICAL_ATTENTION@
    CircularBuffer probability_fp32(cb_probability_fp32);
    CircularBuffer probability_bf16(@PROBABILITY_BF16_CB@);
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
    }
    noc_semaphore_wait_min(invocation_done, 2);
}
