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

FORCE_INLINE uint32_t round_shift_right(uint32_t value, uint32_t shift) {
    if (shift == 0) return value;
    if (shift >= 32) return 0;
    const uint32_t truncated = value >> shift;
    const uint32_t remainder_mask = (1u << shift) - 1u;
    const uint32_t remainder = value & remainder_mask;
    const uint32_t halfway = 1u << (shift - 1);
    return truncated +
        (remainder > halfway || (remainder == halfway && (truncated & 1u)));
}

FORCE_INLINE uint32_t canonical_scaled_score(uint32_t bits) {
    constexpr uint32_t shift = @SCALE_SHIFT@;
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t exponent = (bits & 0x7f800000u) >> 23;
    const uint32_t fraction = bits & 0x007fffffu;
    uint32_t scaled = bits;
    if (exponent != 0xffu && shift != 0) {
        if (exponent > shift) {
            scaled = bits - (shift << 23);
        } else if (exponent != 0 || fraction != 0) {
            const uint32_t significand =
                exponent == 0 ? fraction : (fraction | 0x00800000u);
            const uint32_t subnormal_shift =
                exponent == 0 ? shift : (shift + 1u - exponent);
            scaled = sign | round_shift_right(significand, subnormal_shift);
        }
    }
    // Match the observable Blackhole TTNN scalar-multiply contract used by
    // the accepted outer attention tail: every finite non-zero result is one
    // FP32 representation bit closer to zero than exact IEEE power-of-two
    // scaling.
    const uint32_t magnitude = scaled & 0x7fffffffu;
    const uint32_t scaled_exponent = scaled & 0x7f800000u;
    if (magnitude != 0 && scaled_exponent != 0x7f800000u) return scaled - 1u;
    return scaled;
}
}

void kernel_main() {
    WAYPOINT("AFW0");
    constexpr bool prefix_only = @PROJECTION_ROPE_PREFIX_ONLY@;
    constexpr bool grouped_pv_owner = @GROUPED_PV_OWNER@;
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
    const bool grouped_pv_leader = row == 0;
    constexpr uint32_t grouped_row_base = 0;
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
#if @CANONICAL_POW2_SCALE@
    CircularBuffer scale_complete_ack(16);
#endif
    CircularBuffer head(cb_head), value(29);
    q.wait_front(dim_tiles);
    for (uint32_t dim = 0; dim < dim_tiles; ++dim)
        noc1.async_write(CoreLocalMem<uint32_t>(q.get_read_ptr() + dim * tile_bytes),
                         arena, tile_bytes, {}, {.page_id = node * dim_tiles + dim});
    noc1.async_write_barrier(); q.pop_front(dim_tiles);
    WAYPOINT("AFW1");
    k.wait_front(seq_tiles * dim_tiles);
    if (head_index % group == 0 && row == 0) {
        for (uint32_t tile = 0; tile < seq_tiles * dim_tiles; ++tile)
            noc1.async_write(CoreLocalMem<uint32_t>(k.get_read_ptr() + tile * tile_bytes),
                             arena, tile_bytes, {},
                             {.page_id = q_pages + kv_head * seq_tiles * dim_tiles + tile});
        noc1.async_write_barrier();
    }
    k.pop_front(seq_tiles * dim_tiles);
    WAYPOINT("AFW2");
    if constexpr (!prefix_only) {
#if @CANONICAL_POW2_SCALE@
    // The accepted graph scales raw FP32 QK through TTNN's scalar multiply.
    // Reproduce its exact raw-bit contract on the data-movement RISC, keeping
    // the source in L1. CB16 carries only a completion credit; the compute
    // RISC remains CB23's consumer so its unpack counter stays aligned when
    // CB23 is reused for the subsequent exponential FIFO.
    CircularBuffer raw_score(23);
    raw_score.wait_front(seq_tiles);
#endif
#if @PUBLISH_RAW_SCORE_CHECKPOINT@ && !@CANONICAL_POW2_SCALE@
    // Publish the raw FP32 score before the exact resident scale consumes it.
    probability.wait_front(seq_tiles);
    for (uint32_t tile = 0; tile < seq_tiles; ++tile) {
        const uint32_t source = probability.get_read_ptr() + tile * 4096;
        const uint32_t page = probability_base + 2 * (node * seq_tiles + tile);
        noc1.async_write(CoreLocalMem<uint32_t>(source), arena, tile_bytes, {}, {.page_id = page});
        noc1.async_write(CoreLocalMem<uint32_t>(source + tile_bytes), arena, tile_bytes, {},
                         {.page_id = page + 1});
    }
    noc1.async_write_barrier();
    CircularBuffer raw_checkpoint_ack(16);
    raw_checkpoint_ack.reserve_back(1);
    raw_checkpoint_ack.push_back(1);
#endif
#if @CANONICAL_POW2_SCALE@
    for (uint32_t tile = 0; tile < seq_tiles; ++tile) {
        auto* source = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            raw_score.get_read_ptr() + tile * 4096);
        auto* destination = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            raw_score.get_read_ptr() + tile * 4096);
        for (uint32_t word = 0; word < 1024; ++word)
            destination[word] = canonical_scaled_score(source[word]);
    }
#if @PUBLISH_RAW_SCORE_CHECKPOINT@
    // Diagnostic mode: publish the exact resident scale handoff rather than
    // the raw QK source so the boundary can be compared independently from
    // the following mask/softmax schedule.
    for (uint32_t tile = 0; tile < seq_tiles; ++tile) {
        const uint32_t source = raw_score.get_read_ptr() + tile * 4096;
        const uint32_t page = probability_base + 2 * (node * seq_tiles + tile);
        noc1.async_write(CoreLocalMem<uint32_t>(source), arena, tile_bytes, {},
                         {.page_id = page});
        noc1.async_write(CoreLocalMem<uint32_t>(source + tile_bytes), arena,
                         tile_bytes, {}, {.page_id = page + 1});
    }
    noc1.async_write_barrier();
#endif
    scale_complete_ack.reserve_back(1);
    scale_complete_ack.push_back(1);
#endif
#if !@PUBLISH_RAW_SCORE_CHECKPOINT@
    if (grouped_pv_owner && grouped_pv_leader) {
        // CB27 is K-major for the exact 4x2 P@V program. Publish its
        // backward checkpoint in row-major arena order without consuming the
        // FIFO; the compute RISC is its sole pop owner.
        constexpr uint32_t grouped_rows = 4;
        probability.wait_front(seq_tiles * grouped_rows);
        for (uint32_t owned_row = 0; owned_row < grouped_rows;
             ++owned_row) {
            for (uint32_t k = 0; k < seq_tiles; ++k) {
                const uint32_t source = probability.get_read_ptr() +
                    (k * grouped_rows + owned_row) * 4096;
                const uint32_t page = probability_base + 2 * (
                    (head_index * seq_tiles + grouped_row_base + owned_row) *
                    seq_tiles + k);
                noc1.async_write(
                    CoreLocalMem<uint32_t>(source), arena, tile_bytes, {},
                    {.page_id = page});
                noc1.async_write(
                    CoreLocalMem<uint32_t>(source + tile_bytes), arena,
                    tile_bytes, {}, {.page_id = page + 1});
            }
        }
        noc1.async_write_barrier();
        // CB16 already carried the scale-complete writer->compute credit, so
        // both RISC views share the same advanced epoch. Reuse that direction
        // for the probability-publication credit and leave CB28 exclusively
        // reader-produced for the clean FP32 V input.
        CircularBuffer checkpoint_ack(16);
        checkpoint_ack.reserve_back(1);
        checkpoint_ack.push_back(1);
    }
#endif
    if (grouped_pv_owner && grouped_pv_leader) {
        CircularBuffer fp32_head(19);
        constexpr uint32_t fp32_head_tiles = seq_tiles * dim_tiles;
        fp32_head.wait_front(fp32_head_tiles);
        constexpr uint32_t fp32_head_physical_offset =
            dim_tiles + seq_tiles * dim_tiles + seq_tiles;
#if @PUBLISH_FP32_HEAD_CHECKPOINT@
        for (uint32_t tile = 0; tile < fp32_head_tiles; ++tile) {
            const uint32_t source = fp32_head.get_read_ptr() +
                (fp32_head_physical_offset + tile) * 4096;
            const uint32_t page = probability_base + 2 * (
                head_index * fp32_head_tiles + tile);
            noc1.async_write(
                CoreLocalMem<uint32_t>(source), arena, tile_bytes, {},
                {.page_id = page});
            noc1.async_write(
                CoreLocalMem<uint32_t>(source + tile_bytes), arena,
                tile_bytes, {}, {.page_id = page + 1});
        }
        noc1.async_write_barrier();
#endif
        // A separate TTNN typecast program starts from a clean packer state.
        // Reusing the softmax/matmul compute packer changed 1,184 otherwise
        // exact BF16 words. Perform the IEEE round-to-nearest-even conversion
        // here from the bit-exact FP32 tiles and publish CB31 directly.
        head.reserve_back(fp32_head_tiles);
        for (uint32_t tile = 0; tile < fp32_head_tiles; ++tile) {
            auto* source = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                fp32_head.get_read_ptr() +
                (fp32_head_physical_offset + tile) * 4096);
            auto* destination = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                head.get_write_ptr() + tile * tile_bytes);
            for (uint32_t element = 0; element < 1024; ++element) {
                const uint32_t bits = source[element];
                const uint32_t rounded =
                    bits + 0x7fffu + ((bits >> 16) & 1u);
                destination[element] = static_cast<uint16_t>(rounded >> 16);
            }
        }
        head.push_back(fp32_head_tiles);
        CircularBuffer fp32_head_conversion_ack(16);
        fp32_head_conversion_ack.reserve_back(1);
        fp32_head_conversion_ack.push_back(1);
    }
    const bool owns_head = !grouped_pv_owner || grouped_pv_leader;
    const uint32_t owned_rows = grouped_pv_owner ? 4 : 1;
    if (owns_head)
        head.wait_front(owned_rows * dim_tiles);
    WAYPOINT("AFW3");
    @HEAD_TO_O_HANDOFF@
    if (owns_head) {
        for (uint32_t owned_row = 0; owned_row < owned_rows; ++owned_row) {
            const uint32_t output_row =
                grouped_pv_owner ? grouped_row_base + owned_row : row;
            for (uint32_t dim = 0; dim < dim_tiles; ++dim) {
                const uint32_t source_tile = owned_row * dim_tiles + dim;
                noc1.async_write(
                    CoreLocalMem<uint32_t>(
                        head.get_read_ptr() + source_tile * tile_bytes),
                    arena, tile_bytes, {},
                    {.page_id = head_base +
                        (head_index * seq_tiles + output_row) * dim_tiles + dim});
            }
        }
        noc1.async_write_barrier();
        head.pop_front(owned_rows * dim_tiles);
    }
    WAYPOINT("AFW4");
#if !@PUBLISH_RAW_SCORE_CHECKPOINT@
    if (!grouped_pv_owner) {
        const uint32_t probability_rows = 1;
        probability.wait_front(probability_rows * seq_tiles);
        for (uint32_t output_row = 0; output_row < probability_rows;
             ++output_row) {
            const uint32_t logical_row = row;
            for (uint32_t tile = 0; tile < seq_tiles; ++tile) {
                const uint32_t source = probability.get_read_ptr() +
                    (output_row * seq_tiles + tile) * 4096;
                const uint32_t page = probability_base + 2 * (
                    (head_index * seq_tiles + logical_row) * seq_tiles + tile);
                noc1.async_write(
                    CoreLocalMem<uint32_t>(source), arena, tile_bytes, {},
                    {.page_id = page});
                noc1.async_write(
                    CoreLocalMem<uint32_t>(source + tile_bytes), arena,
                    tile_bytes, {}, {.page_id = page + 1});
            }
        }
        noc1.async_write_barrier();
        probability.pop_front(probability_rows * seq_tiles);
    }
    WAYPOINT("AFW5");
#endif
    } else {
        // V is already checkpointed by the projection writer. Retire the
        // replicated attention-island stream so the prefix program returns
        // every CB epoch empty without running QK/softmax/P@V.
        value.wait_front(seq_tiles * dim_tiles);
        value.pop_front(seq_tiles * dim_tiles);
    }

#if @COMPOSED_INVOCATION_BARRIER@
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
#endif
    WAYPOINT("AFW9");
}
