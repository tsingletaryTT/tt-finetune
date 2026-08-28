// Publish only backward checkpoints and final head output from resident attention.

#include <cstdint>
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t kProjectionColumns = 5;
constexpr uint8_t kPhysicalX[11] = {1,2,3,4,5,6,7,10,11,12,13};
FORCE_INLINE uint8_t physical_y(uint32_t y) { return static_cast<uint8_t>(y + 2); }
FORCE_INLINE uint64_t projection_l1(uint32_t node, uint32_t address) {
    return get_noc_addr(kPhysicalX[node % kProjectionColumns],
                        physical_y(node / kProjectionColumns), address);
}
}

void kernel_main() {
    constexpr bool prefix_only = @PROJECTION_ROPE_PREFIX_ONLY@;
    constexpr uint32_t cb_q_saved = 9, cb_k_saved = 13;
    constexpr uint32_t cb_probability_saved = @ATTENTION_CHECKPOINT_CB@;
    constexpr uint32_t cb_head = 31;
    constexpr uint32_t seq_tiles = @SEQ_TILES@, dim_tiles = @DIM_TILES@;
    constexpr uint32_t query_heads = @QUERY_HEADS@, kv_heads = @KV_HEADS@;
    constexpr uint32_t group = query_heads / kv_heads;
    constexpr uint32_t jobs = query_heads * seq_tiles;
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t invocation_writer_count_sem = 14;
    constexpr uint32_t invocation_done_sem = 15;
    const uint32_t node = get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= jobs) return;
    const uint32_t head_index = node / seq_tiles;
    const uint32_t row = node - head_index * seq_tiles;
    const uint32_t kv_head = head_index / group;
    constexpr uint32_t q_pages = query_heads * seq_tiles * dim_tiles;
    constexpr uint32_t k_pages = kv_heads * seq_tiles * dim_tiles;
    constexpr uint32_t head_pages = q_pages;
    constexpr uint32_t head_base = q_pages + k_pages;
    constexpr uint32_t probability_base = head_base + head_pages;
    const auto arena = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<6, 32>(), 0>(),
        get_common_arg_val<uint32_t>(0), tile_bytes);
    Noc noc1(1);
    CircularBuffer q(cb_q_saved), k(cb_k_saved), probability(cb_probability_saved);
#if @PUBLISH_RAW_SCORE_CHECKPOINT@
    // CB16 carries only ownership credit.  The compute RISC must not consume
    // CB23 until this writer has copied its exact FP32 bytes to the arena.
    CircularBuffer raw_checkpoint_ack(16);
#endif
    CircularBuffer head(cb_head), value(29);
    q.wait_front(dim_tiles);
    for (uint32_t dim = 0; dim < dim_tiles; ++dim)
        noc1.async_write(CoreLocalMem<uint32_t>(q.get_read_ptr() + dim * tile_bytes),
                         arena, tile_bytes, {}, {.page_id = node * dim_tiles + dim});
    noc1.async_write_barrier(); q.pop_front(dim_tiles);
    k.wait_front(seq_tiles * dim_tiles);
    if (head_index % group == 0 && row == 0) {
        for (uint32_t tile = 0; tile < seq_tiles * dim_tiles; ++tile)
            noc1.async_write(CoreLocalMem<uint32_t>(k.get_read_ptr() + tile * tile_bytes),
                             arena, tile_bytes, {},
                             {.page_id = q_pages + kv_head * seq_tiles * dim_tiles + tile});
        noc1.async_write_barrier();
    }
    k.pop_front(seq_tiles * dim_tiles);
    if constexpr (!prefix_only) {
#if @PUBLISH_RAW_SCORE_CHECKPOINT@
    // Publish the raw FP32 score before waiting for the downstream attention
    // head: producing that head consumes CB23.  The writer deliberately does
    // not pop CB23; ownership returns to compute through the CB16 credit.
    probability.wait_front(seq_tiles);
    for (uint32_t tile = 0; tile < seq_tiles; ++tile) {
        const uint32_t source = probability.get_read_ptr() + tile * 4096;
        const uint32_t page = probability_base + 2 * (node * seq_tiles + tile);
        noc1.async_write(CoreLocalMem<uint32_t>(source), arena, tile_bytes, {}, {.page_id = page});
        noc1.async_write(CoreLocalMem<uint32_t>(source + tile_bytes), arena, tile_bytes, {},
                         {.page_id = page + 1});
    }
    noc1.async_write_barrier();
    raw_checkpoint_ack.reserve_back(1);
    raw_checkpoint_ack.push_back(1);
#endif
    head.wait_front(dim_tiles);
    @HEAD_TO_O_HANDOFF@
    for (uint32_t dim = 0; dim < dim_tiles; ++dim)
        noc1.async_write(CoreLocalMem<uint32_t>(head.get_read_ptr() + dim * tile_bytes),
                         arena, tile_bytes, {},
                         {.page_id = head_base + node * dim_tiles + dim});
    noc1.async_write_barrier(); head.pop_front(dim_tiles);
#if !@PUBLISH_RAW_SCORE_CHECKPOINT@
    probability.wait_front(seq_tiles);
    for (uint32_t tile = 0; tile < seq_tiles; ++tile) {
        const uint32_t source = probability.get_read_ptr() + tile * 4096;
        const uint32_t page = probability_base + 2 * (node * seq_tiles + tile);
        noc1.async_write(CoreLocalMem<uint32_t>(source), arena, tile_bytes, {}, {.page_id = page});
        noc1.async_write(CoreLocalMem<uint32_t>(source + tile_bytes), arena, tile_bytes, {},
                         {.page_id = page + 1});
    }
    noc1.async_write_barrier(); probability.pop_front(seq_tiles);
#endif
    } else {
        // V is already checkpointed by the projection writer. Retire the
        // replicated attention-island stream so the prefix program returns
        // every CB epoch empty without running QK/softmax/P@V.
        value.wait_front(seq_tiles * dim_tiles);
        value.pop_front(seq_tiles * dim_tiles);
    }

    noc_semaphore_inc(
        projection_l1(0, get_semaphore(invocation_writer_count_sem)), 1);
    noc_async_atomic_barrier();
    volatile tt_l1_ptr uint32_t* invocation_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(invocation_done_sem));
    // In the composed full-layer owner, workers outside `jobs` executed no
    // attention writer epoch and own no attention CB state.  They can join
    // the following all-core MLP phase immediately.  Active workers retain
    // the global completion fence before rebasing their consumed streams.
    if (node < jobs) {
        noc_semaphore_wait(invocation_done, 1);
    }
    CircularBuffer invocation_release(0), invocation_acknowledgement(1);
    invocation_release.reserve_back(1);
    invocation_release.push_back(1);
    invocation_acknowledgement.wait_front(1);
    invocation_acknowledgement.pop_front(1);
    noc_semaphore_set(invocation_done, 2);
}
