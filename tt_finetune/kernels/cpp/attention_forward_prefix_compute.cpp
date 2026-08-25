// Exact BF16 RoPE boundaries followed by the TTNN-style causal QK schedule.

#define FP32_DEST_ACC_EN 1

#include <cstdint>
#include <type_traits>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/dataflow/circular_buffer.h"

ALWI void add_tiles_bf16(
    uint32_t left, uint32_t right, uint32_t left_tile,
    uint32_t right_tile, uint32_t destination) {
    UNPACK((llk_unpack_AB(left, right, left_tile, right_tile)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWADD, BroadcastType::NONE, false,
          MathFidelity::LoFi, EltwiseBinaryReuseDestType::NONE>(
        left, right, destination, false)));
}
ALWI void tile_regs_commit_bf16() {
    MATH((llk_math_dest_section_done<false>()));
}
ALWI void pack_reconfig_data_format_bf16(uint32_t cb) {
    PACK((llk_pack_reconfig_data_format<false>(cb)));
}
template <bool OutOfOrder = false>
ALWI void pack_tile_bf16(
    uint32_t source, uint32_t cb, uint32_t output_tile = 0) {
    PACK((llk_pack<false, OutOfOrder, PackMode::Default>(
        source, cb, output_tile)));
}
ALWI void tile_regs_release_bf16() {
    PACK((llk_pack_dest_section_done<false>()));
}

namespace {

template <uint32_t Op, bool Negate = false>
FORCE_INLINE void bf16_binary_one(
    uint32_t left_id, uint32_t right_id, uint32_t output_id,
    uint32_t left_tile, uint32_t right_tile) {
    CircularBuffer left(left_id), right(right_id), output(output_id);
    left.wait_front(left_tile + 1); right.wait_front(right_tile + 1);
    output.reserve_back(1);
    unary_op_init_common(left_id, output_id);
    if constexpr (Op == 0) mul_binary_tile_init();
    else if constexpr (Op == 1) sub_binary_tile_init();
    else add_binary_tile_init();
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(right_id, left_id);
    copy_tile(left_id, left_tile, 0);
    copy_tile_to_dst_init_short_with_dt(left_id, right_id);
    copy_tile(right_id, right_tile, 1);
    if constexpr (Op == 0) mul_binary_tile(0, 1, 0);
    else if constexpr (Op == 1) sub_binary_tile(0, 1, 0);
    else add_binary_tile(0, 1, 0);
    if constexpr (Negate) {
        binop_with_scalar_tile_init();
        { volatile int32_t minus_one = 0xbf800000u; mul_unary_tile(0, minus_one); }
    }
    typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
    typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(0);
    tile_regs_commit(); tile_regs_wait();
    PACK((pack_reconfig_data_format(output_id)));
    pack_tile(0, output_id);
    tile_regs_release(); output.push_back(1);
}

template <uint32_t Op, bool ExplicitTypecast = true>
FORCE_INLINE void bf16_binary_one_indexed(
    uint32_t left_id, uint32_t right_id, uint32_t output_id,
    uint32_t tile) {
    CircularBuffer left(left_id), right(right_id), output(output_id);
    left.wait_front(tile + 1); right.wait_front(tile + 1);
    output.reserve_back(1);
    unary_op_init_common(left_id, output_id);
    if constexpr (Op == 1) sub_binary_tile_init(); else add_binary_tile_init();
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(right_id, left_id); copy_tile(left_id, tile, 0);
    copy_tile_to_dst_init_short_with_dt(left_id, right_id); copy_tile(right_id, tile, 1);
    if constexpr (Op == 1) sub_binary_tile(0, 1, 0); else add_binary_tile(0, 1, 0);
    if constexpr (ExplicitTypecast) {
        typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(0);
    }
    tile_regs_commit(); tile_regs_wait();
    PACK((pack_reconfig_data_format(output_id)));
    pack_tile(0, output_id);
    tile_regs_release(); output.push_back(1);
}

FORCE_INLINE void copy_tiles_preserve(
    uint32_t input_id, uint32_t output_id, uint32_t tiles) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(tiles); output.reserve_back(tiles);
    compute_kernel_hw_startup(input_id, output_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire(); copy_tile_init(input_id); copy_tile(input_id, tile, 0);
        tile_regs_commit(); tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, tile); tile_regs_release();
    }
    output.push_back(tiles);
}

FORCE_INLINE void add_bf16_destination_mode(
    uint32_t left_id, uint32_t right_id, uint32_t output_id,
    uint32_t tiles) {
    CircularBuffer left(left_id), right(right_id), output(output_id);
    left.wait_front(tiles); right.wait_front(tiles);
    add_tiles_init(left_id, right_id);
    disable_fp32_dest_acc();
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        output.reserve_back(1); tile_regs_acquire();
        add_tiles_bf16(left_id, right_id, tile, tile, 0);
        tile_regs_commit_bf16(); tile_regs_wait();
        pack_reconfig_data_format_bf16(output_id);
        pack_tile_bf16(0, output_id);
        tile_regs_release_bf16(); output.push_back(1);
    }
    enable_fp32_dest_acc();
}

FORCE_INLINE void rope_tile_fused(
    uint32_t value_id, uint32_t cosine_id, uint32_t sine_id,
    uint32_t output_id, uint32_t tile, uint32_t peer, bool subtract) {
    CircularBuffer value(value_id), cosine(cosine_id), sine(sine_id), output(output_id);
    value.wait_front((tile > peer ? tile : peer) + 1);
    cosine.wait_front(tile + 1); sine.wait_front(tile + 1);
    output.reserve_back(1);
    unary_op_init_common(value_id, output_id);
    mul_binary_tile_init();
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(cosine_id, value_id);
    copy_tile(value_id, tile, 0);
    copy_tile_to_dst_init_short_with_dt(value_id, cosine_id);
    copy_tile(cosine_id, tile, 1);
    mul_binary_tile(0, 1, 0);
    copy_tile_to_dst_init_short_with_dt(sine_id, value_id);
    copy_tile(value_id, peer, 1);
    copy_tile_to_dst_init_short_with_dt(value_id, sine_id);
    copy_tile(sine_id, tile, 2);
    mul_binary_tile(1, 2, 1);
    if (subtract) {
        sub_binary_tile_init(); sub_binary_tile(0, 1, 0);
    } else {
        add_binary_tile_init(); add_binary_tile(0, 1, 0);
    }
    typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
    typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(0);
    tile_regs_commit(); tile_regs_wait();
    PACK((pack_reconfig_data_format(output_id))); pack_tile(0, output_id);
    tile_regs_release(); output.push_back(1);
}

FORCE_INLINE void scale_bf16(
    uint32_t input_id, uint32_t output_id, uint32_t tiles, uint32_t scale_bits) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(tiles); output.reserve_back(tiles);
    init_sfpu(input_id, output_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire(); copy_tile_init(input_id); copy_tile(input_id, tile, 0);
        binop_with_scalar_tile_init();
        { volatile int32_t scalar = scale_bits; mul_unary_tile(0, scalar); }
        typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(0);
        tile_regs_commit(); tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id))); pack_tile<true>(0, output_id, tile);
        tile_regs_release();
    }
    output.push_back(tiles); input.pop_front(tiles);
}

FORCE_INLINE void bf16_tiles_to_fp32(
    uint32_t input_id, uint32_t output_id, uint32_t tiles) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(tiles); output.reserve_back(tiles);
    compute_kernel_hw_startup(input_id, output_id);
    copy_tile_to_dst_init_short(input_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        copy_tile(input_id, tile, 0);
        typecast_tile_init<
            static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float16_b),
            static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float32)>();
        typecast_tile<
            static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float16_b),
            static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float32)>(0);
        tile_regs_commit(); tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, tile);
        tile_regs_release();
    }
    output.push_back(tiles);
}

FORCE_INLINE void transpose_bf16_tiles_to_fp32(
    uint32_t input_id, uint32_t output_id,
    uint32_t row_tiles, uint32_t column_tiles) {
    const uint32_t tiles = row_tiles * column_tiles;
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(tiles); output.reserve_back(tiles);
    transpose_init(input_id);
    for (uint32_t column = 0; column < column_tiles; ++column) {
        for (uint32_t row = 0; row < row_tiles; ++row) {
            const uint32_t source = row * column_tiles + column;
            const uint32_t target = column * row_tiles + row;
            tile_regs_acquire();
            transpose_tile(input_id, source, 0);
            typecast_tile_init<
                static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float16_b),
                static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float32)>();
            typecast_tile<
                static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float16_b),
                static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float32)>(0);
            tile_regs_commit(); tile_regs_wait();
            PACK((pack_reconfig_data_format(output_id)));
            pack_tile<true>(0, output_id, target);
            tile_regs_release();
        }
    }
    output.push_back(tiles);
}

template <bool Multiply>
FORCE_INLINE void fp32_binary_tiles(
    uint32_t left_id, uint32_t right_id, uint32_t output_id,
    uint32_t tiles, bool broadcast_right) {
    CircularBuffer left(left_id), right(right_id), output(output_id);
    left.wait_front(tiles);
    right.wait_front(broadcast_right ? 1 : tiles);
    output.reserve_back(tiles);
    binary_op_init_common(left_id, right_id, output_id);
    if constexpr (Multiply) mul_tiles_init(left_id, right_id);
    else add_tiles_init(left_id, right_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        if constexpr (Multiply)
            mul_tiles(left_id, right_id, tile, broadcast_right ? 0 : tile, 0);
        else
            add_tiles(left_id, right_id, tile, broadcast_right ? 0 : tile, 0);
        tile_regs_commit(); tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, tile);
        tile_regs_release();
    }
    output.push_back(tiles);
    left.pop_front(tiles);
    if (!broadcast_right) right.pop_front(tiles);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_q = 0, cb_q_cos = 1, cb_q_sin = 2;
    constexpr uint32_t cb_k = 3, cb_k_cos = 4, cb_k_sin = 5;
    constexpr uint32_t cb_q_cos_product = 6, cb_q_sin_product = 7;
    constexpr uint32_t cb_q_rotated = 8, cb_q_saved = 9;
    constexpr uint32_t cb_k_cos_product = 10, cb_k_sin_product = 11;
    constexpr uint32_t cb_k_rotated = 12, cb_k_saved = 13;
    constexpr uint32_t cb_kt = 14, cb_score = 15, cb_scaled = 16;
    constexpr uint32_t cb_mask = 17, cb_masked = 18;
    constexpr uint32_t cb_qk_partial = 19;
    constexpr uint32_t seq_tiles = @SEQ_TILES@, dim_tiles = @DIM_TILES@;
    constexpr uint32_t jobs = @QUERY_HEADS@ * seq_tiles;
    constexpr uint32_t scale_bits = @SCALE_BITS@;
    const uint32_t node = get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= jobs) return;

    for (uint32_t dim = 0; dim < dim_tiles; ++dim) {
        bf16_binary_one<0>(cb_q, cb_q_cos, cb_q_cos_product, dim, dim);
        const uint32_t peer = dim == 0 ? 1 : 0;
        if (dim == 0)
            bf16_binary_one<0, true>(cb_q, cb_q_sin, cb_q_sin_product, peer, dim);
        else
            bf16_binary_one<0>(cb_q, cb_q_sin, cb_q_sin_product, peer, dim);
    }
    add_bf16_destination_mode(
        cb_q_cos_product, cb_q_sin_product, cb_q_rotated, dim_tiles);
    copy_tiles_preserve(cb_q_rotated, cb_q_saved, dim_tiles);
    CircularBuffer q(cb_q), qcos(cb_q_cos), qsin(cb_q_sin);
    CircularBuffer qcosp(cb_q_cos_product), qsinp(cb_q_sin_product);
    q.pop_front(dim_tiles); qcos.pop_front(dim_tiles); qsin.pop_front(dim_tiles);
    qcosp.pop_front(dim_tiles); qsinp.pop_front(dim_tiles);

    const uint32_t vector_tiles = seq_tiles * dim_tiles;
    for (uint32_t tile = 0; tile < vector_tiles; ++tile) {
        const uint32_t dim = tile % dim_tiles;
        const uint32_t peer = tile - dim + (dim == 0 ? 1 : 0);
        bf16_binary_one<0>(cb_k, cb_k_cos, cb_k_cos_product, tile, tile);
        if (dim == 0)
            bf16_binary_one<0, true>(cb_k, cb_k_sin, cb_k_sin_product, peer, tile);
        else
            bf16_binary_one<0>(cb_k, cb_k_sin, cb_k_sin_product, peer, tile);
    }
    add_bf16_destination_mode(
        cb_k_cos_product, cb_k_sin_product, cb_k_rotated, vector_tiles);
    copy_tiles_preserve(cb_k_rotated, cb_k_saved, vector_tiles);
    CircularBuffer k(cb_k), kcos(cb_k_cos), ksin(cb_k_sin);
    CircularBuffer kcosp(cb_k_cos_product), ksinp(cb_k_sin_product);
    k.pop_front(vector_tiles); kcos.pop_front(vector_tiles); ksin.pop_front(vector_tiles);
    kcosp.pop_front(vector_tiles); ksinp.pop_front(vector_tiles);

#if @FP32_CANONICAL_ATTENTION@
    // Canonical model path: RoPE is packed to BF16, then Q/K are promoted to
    // FP32.  QK remains FP32 and receives the 1/sqrt(D) scale and causal mask
    // before softmax.  These rounding boundaries are observable in the
    // untouched lowered training graph.
    constexpr uint32_t cb_q_fp32 = 19, cb_kt_fp32 = 20;
    constexpr uint32_t cb_raw_score_fp32 = 23, cb_qk_partial_fp32 = 30;
    constexpr uint32_t cb_score_scale_fp32 = 28;
    bf16_tiles_to_fp32(cb_q_rotated, cb_q_fp32, dim_tiles);
    transpose_bf16_tiles_to_fp32(
        cb_k_rotated, cb_kt_fp32, seq_tiles, dim_tiles);
    CircularBuffer qrot(cb_q_rotated), krot(cb_k_rotated);
    qrot.pop_front(dim_tiles); krot.pop_front(vector_tiles);

    CircularBuffer q32(cb_q_fp32), kt32(cb_kt_fp32);
    CircularBuffer raw_score(cb_raw_score_fp32);
    q32.wait_front(dim_tiles); kt32.wait_front(vector_tiles);

    // TTNN's simple-BMM reader does not expose a resident matrix to the
    // compute RISC.  It streams one Q tile and one K^T tile for every output
    // tile, and bmm.cpp always consumes tile zero before popping both FIFOs.
    // Reproduce that queue-visible contract explicitly.  This is intentionally
    // a correctness-first schedule; once qualified, the duplicate stream can
    // be produced by the data-movement RISC without changing the arithmetic
    // program.
    for (uint32_t n = 0; n < seq_tiles; ++n) {
        for (uint32_t phase = 0; phase < dim_tiles; ++phase) {
            q32.reserve_back(1);
            compute_kernel_hw_startup(cb_q_fp32, cb_q_fp32);
            tile_regs_acquire();
            copy_tile_init(cb_q_fp32);
            copy_tile(cb_q_fp32, phase, 0);
            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(cb_q_fp32)));
            pack_tile(0, cb_q_fp32);
            tile_regs_release();
            q32.push_back(1);

            kt32.reserve_back(1);
            compute_kernel_hw_startup(cb_kt_fp32, cb_kt_fp32);
            tile_regs_acquire();
            copy_tile_init(cb_kt_fp32);
            copy_tile(cb_kt_fp32, phase * seq_tiles + n, 0);
            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(cb_kt_fp32)));
            pack_tile(0, cb_kt_fp32);
            tile_regs_release();
            kt32.push_back(1);
        }
    }
    q32.pop_front(dim_tiles);
    kt32.pop_front(vector_tiles);

    compute_kernel_hw_startup<SrcOrder::Reverse>(
        cb_q_fp32, cb_kt_fp32, cb_raw_score_fp32);
    // The canonical no-program-config TTNN path selects the simple ``bmm``
    // factory.  Its compute kernel initializes ``matmul_tiles`` once, keeps
    // the complete K reduction in one destination register and packs only the
    // final FP32 tile.  ``matmul_block`` is mathematically equivalent here but
    // uses a different LLK schedule and is observably non-exact on Blackhole.
    // Mirror bmm.cpp literally: matmul_init followed by one matmul_tiles call
    // for each K tile.
    matmul_init(cb_q_fp32, cb_kt_fp32);
    for (uint32_t n = 0; n < seq_tiles; ++n) {
        tile_regs_acquire();
        for (uint32_t phase = 0; phase < dim_tiles; ++phase) {
            q32.wait_front(1);
            kt32.wait_front(1);
            matmul_tiles(cb_q_fp32, cb_kt_fp32, 0, 0, 0);
            q32.pop_front(1);
            kt32.pop_front(1);
        }
        tile_regs_commit();
        raw_score.reserve_back(1);
        tile_regs_wait();
        // ``compute_kernel_hw_startup`` already configured the packer for
        // the FP32 output.  Keep the canonical bmm.cpp pack boundary too:
        // one ordinary pack_tile, without a per-output reconfiguration or
        // block-pack helper.
        pack_tile(0, cb_raw_score_fp32);
        tile_regs_release();
        raw_score.push_back(1);
    }
#if @PUBLISH_RAW_SCORE_CHECKPOINT@
    // The writer copies CB23's exact bytes directly to the private arena and
    // publishes one CB16 credit when its NoC write barrier completes.  Wait
    // for that credit before the multiply below consumes and pops raw_score.
    CircularBuffer raw_checkpoint_ack(cb_scaled);
    raw_checkpoint_ack.wait_front(1);
    raw_checkpoint_ack.pop_front(1);
#endif
    // Match TTNN's explicit FP32 multiply, BF16->FP32 mask conversion and
    // FP32 add operations. A scalar SFPU shortcut rounds differently.
    fp32_binary_tiles<true>(
        cb_raw_score_fp32, cb_score_scale_fp32, cb_kt_fp32,
        seq_tiles, true);
    bf16_tiles_to_fp32(cb_mask, cb_q_fp32, seq_tiles);
    CircularBuffer mask_bf16(cb_mask); mask_bf16.pop_front(seq_tiles);
    fp32_binary_tiles<false>(
        cb_kt_fp32, cb_q_fp32, cb_masked, seq_tiles, false);
    CircularBuffer score_scale(cb_score_scale_fp32);
    score_scale.pop_front(1);
#else
    CircularBuffer krot(cb_k_rotated), kt(cb_kt);
    krot.wait_front(vector_tiles); kt.reserve_back(vector_tiles);
    transpose_init(cb_k_rotated);
    for (uint32_t dim = 0; dim < dim_tiles; ++dim) {
        for (uint32_t token = 0; token < seq_tiles; ++token) {
            const uint32_t source = token * dim_tiles + dim;
            const uint32_t target = dim * seq_tiles + token;
            tile_regs_acquire(); transpose_tile(cb_k_rotated, source, 0);
            tile_regs_commit(); tile_regs_wait();
            PACK((pack_reconfig_data_format(cb_kt)));
            pack_tile<true>(0, cb_kt, target); tile_regs_release();
        }
    }
    kt.push_back(vector_tiles); krot.pop_front(vector_tiles);

    CircularBuffer qrot(cb_q_rotated), score(cb_score), partial(cb_qk_partial);
    qrot.wait_front(dim_tiles); kt.wait_front(vector_tiles);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q_rotated, cb_kt, cb_score);
    for (uint32_t n = 0; n < seq_tiles; ++n) {
        for (uint32_t phase = 0; phase < dim_tiles; ++phase) {
            tile_regs_acquire();
            if (phase > 0) {
                copy_tile_to_dst_init_short_with_dt(cb_kt, cb_qk_partial);
                partial.wait_front(1);
                copy_block_matmul_partials(cb_qk_partial, 0, 0, 1);
                partial.pop_front(1);
                reconfig_data_format_srca(cb_qk_partial, cb_kt);
            }
            matmul_block_init(cb_q_rotated, cb_kt, false, 1, 1, 1);
            matmul_block(
                cb_q_rotated, cb_kt, phase, phase * seq_tiles + n,
                0, false, 1, 1, 1);
            tile_regs_commit();
            const bool last = phase + 1 == dim_tiles;
            if (last) score.reserve_back(1); else partial.reserve_back(1);
            tile_regs_wait();
            const uint32_t target = last ? cb_score : cb_qk_partial;
            PACK((pack_reconfig_data_format(target)));
            pack_tile_block(0, target, 1);
            tile_regs_release();
            if (last) score.push_back(1); else partial.push_back(1);
        }
    }
    qrot.pop_front(dim_tiles); kt.pop_front(vector_tiles);

    scale_bf16(cb_score, cb_scaled, seq_tiles, scale_bits);
    for (uint32_t tile = 0; tile < seq_tiles; ++tile)
        bf16_binary_one<2>(cb_scaled, cb_mask, cb_masked, tile, tile);
    CircularBuffer scaled(cb_scaled), mask(cb_mask);
    scaled.pop_front(seq_tiles); mask.pop_front(seq_tiles);
#endif
}
