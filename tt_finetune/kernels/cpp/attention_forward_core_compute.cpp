// Exact TTNN-style FP32 softmax followed by the BF16 probability @ V edge.

#include <cstdint>
#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/reg_api.h"
#include "api/compute/softmax.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"

namespace {
template <uint32_t cb_in, uint32_t cb_scaler, uint32_t cb_max, uint32_t cb_exps>
FORCE_INLINE void numeric_stable(uint32_t width) {
    CircularBuffer input(cb_in), maximum(cb_max), exps(cb_exps);
    compute_kernel_lib::reduce<
        PoolType::MAX, ReduceDim::REDUCE_ROW, cb_in, cb_scaler, cb_max,
        compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
            compute_kernel_lib::ReduceInputBlockShape::row(width));
    // Mirror attention/compute/softmax.cpp's small numeric-stable schedule.
    // At S=128 width and ndst are both four tiles.
    const uint32_t ndst = width < 4 ? width : 4;
    exp_tile_init<true>();
    reconfig_data_format_srcb(cb_max);
    maximum.wait_front(1);
    sub_bcast_cols_init_short(cb_in, cb_max);
    for (uint32_t start = 0; start < width; start += ndst) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < ndst; ++i)
            sub_tiles_bcast_cols(cb_in, cb_max, start + i, 0, i);
        exps.reserve_back(ndst);
        for (uint32_t i = 0; i < ndst; ++i) exp_tile<true>(i);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < ndst; ++i)
            pack_tile(i, cb_exps);
        tile_regs_release();
        exps.push_back(ndst);
    }
    input.pop_front(width);
    maximum.pop_front(1);
    exps.wait_front(width);
}

// Reproduce SoftmaxProgramFactoryGeneralWSmall's FP32 arithmetic schedule.
// The generic Moreh W kernel deliberately separates max reduction, x-max,
// exp and normalization with an FP32 pack boundary after every tile.  The
// earlier resident implementation fused x-max and exp in four-tile DEST
// chunks; that is mathematically equivalent, but it changes Blackhole's
// observable SFPU/reduction rounding and therefore cannot replace the exact
// training graph.
template <
    uint32_t input_id,
    uint32_t max_scaler_id,
    uint32_t sum_scaler_id,
    uint32_t maximum_id,
    uint32_t temporary_id,
    uint32_t x_minus_max_id,
    uint32_t exps_id,
    uint32_t reciprocal_id,
    uint32_t probability_id,
    uint32_t tail_mask_id>
FORCE_INLINE void canonical_moreh_softmax_fp32(uint32_t width) {
    WAYPOINT("MH00");
    DataflowBuffer input(input_id);
    DataflowBuffer max_scaler(max_scaler_id);
    DataflowBuffer sum_scaler(sum_scaler_id);
    DataflowBuffer maximum(maximum_id);
    DataflowBuffer temporary(temporary_id);
    DataflowBuffer x_minus_max(x_minus_max_id);
    DataflowBuffer exps(exps_id);
    DataflowBuffer reciprocal(reciprocal_id);
    DataflowBuffer probability(probability_id);
    DataflowBuffer tail_mask(tail_mask_id);

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t one_tile = 1;
    binary_op_init_common(input_id, max_scaler_id, probability_id);
    tail_mask.wait_front(one_tile);
    // Moreh's reduction helpers consume the scaler tiles through the unpack
    // interface but do not establish their producer/consumer epoch.  The
    // canonical kernel explicitly waits for both persistent scaler tiles
    // before entering either reduction; omitting these waits leaves the
    // resident program racing reader initialization and can deadlock every
    // active Tensix core.
    max_scaler.wait_front(one_tile);
    sum_scaler.wait_front(one_tile);
    WAYPOINT("MH01");

    if (width == 1) {
        mask_tile_to_cb(
            input, tail_mask, temporary, 0, 0, /*pop0=*/0, /*popm=*/0);
        compute_kernel_lib::reduce<
            PoolType::MAX, ReduceDim::REDUCE_ROW,
            temporary_id, max_scaler_id, maximum_id>(
                compute_kernel_lib::ReduceInputBlockShape::single());
    } else {
        compute_kernel_lib::reduce<
            PoolType::MAX, ReduceDim::REDUCE_ROW,
            input_id, max_scaler_id, maximum_id,
            compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                compute_kernel_lib::ReduceInputBlockShape::row(width - 1));
        mask_tile_to_cb(
            input, tail_mask, temporary, width - 1, 0,
            /*pop0=*/0, /*popm=*/0);
        compute_kernel_lib::reduce<
            PoolType::MAX, ReduceDim::REDUCE_ROW,
            temporary_id, max_scaler_id, maximum_id>(
                compute_kernel_lib::ReduceInputBlockShape::row(1),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::Accumulate::at(maximum_id, 1));
    }
    WAYPOINT("MH02");

    x_minus_max.reserve_back(width);
    maximum.wait_front(1);
    for (uint32_t tile = 0; tile < width; ++tile) {
        tile_regs_acquire();
        sub_bcast_cols_init_short_with_dt(input, maximum);
        sub_tiles_bcast<BroadcastType::COL>(
            input_id, maximum_id, tile, 0, dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile_with_dt(dst0, x_minus_max);
        tile_regs_release();
    }
    maximum.pop_front(1);
    input.pop_front(width);
    x_minus_max.push_back(width);
    WAYPOINT("MH03");

    exps.reserve_back(width);
    x_minus_max.wait_front(width);
    for (uint32_t tile = 0; tile < width; ++tile) {
        tile_regs_acquire();
        copy_tile_init_with_dt(x_minus_max);
        copy_tile(x_minus_max_id, tile, dst0);
        exp_tile_init();
        exp_tile(dst0);
        if (tile == width - 1) {
            copy_tile_init_with_dt(tail_mask);
            copy_tile(tail_mask_id, 0, dst1);
            mask_tile_init();
            mask_tile(dst0, dst1);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile_with_dt(dst0, exps);
        tile_regs_release();
    }
    // The mask has a dedicated FIFO in the resident schedule. Retire its one
    // generation once the final exponential tile has consumed it; the slot is
    // never reused for a different producer/consumer epoch in this invocation.
    tail_mask.pop_front(one_tile);
    exps.push_back(width);
    WAYPOINT("MH04");

    compute_kernel_lib::reduce<
        PoolType::SUM, ReduceDim::REDUCE_ROW,
        exps_id, sum_scaler_id, reciprocal_id,
        compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
            compute_kernel_lib::ReduceInputBlockShape::row(width),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            [](uint32_t destination) {
                recip_tile_init();
                recip_tile(destination);
            });
    WAYPOINT("MH05");

    WAYPOINT("MHA0");
    probability.reserve_back(width);
    WAYPOINT("MHA1");
    reciprocal.wait_front(1);
    WAYPOINT("MHA2");
    exps.wait_front(width);
    WAYPOINT("MHA3");
    for (uint32_t tile = 0; tile < width; ++tile) {
        WAYPOINT("MHA4");
        tile_regs_acquire();
        mul_bcast_cols_init_short_with_dt(exps, reciprocal);
        mul_tiles_bcast_cols(
            exps_id, reciprocal_id, tile, 0, dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile_with_dt(dst0, probability);
        tile_regs_release();
    }
    WAYPOINT("MHA5");
    reciprocal.pop_front(1);
    x_minus_max.pop_front(width);
    exps.pop_front(width);
    probability.push_back(width);
    WAYPOINT("MH06");
}

FORCE_INLINE void matmul_probability_value(
    uint32_t lhs_id, uint32_t rhs_id, uint32_t partial_id,
    uint32_t output_id,
    uint32_t seq_tiles, uint32_t dim_tiles, uint32_t phase_tiles,
    uint32_t destination_row = 0) {
    CircularBuffer lhs(lhs_id), rhs(rhs_id);
    lhs.wait_front(seq_tiles);
    rhs.wait_front(seq_tiles * dim_tiles);
    CircularBuffer partial(partial_id), output(output_id);
    compute_kernel_hw_startup<SrcOrder::Reverse>(
        lhs_id, rhs_id, partial_id);
    matmul_init(lhs_id, rhs_id);
    const uint32_t phases = seq_tiles / phase_tiles;
    // The accepted reuse program issues both N tiles of a 2x2 output
    // subblock in one DEST epoch and spills/reloads those tiles together.
    // A row-resident worker owns only one M row, but preserving the paired-N
    // issue order is still numerically visible in the fourth causal K phase.
    const uint32_t destination_base = destination_row * dim_tiles;
    for (uint32_t phase = 0; phase < phases; ++phase) {
        tile_regs_acquire();
        if (phase > 0) {
            copy_tile_to_dst_init_short_with_dt(rhs_id, partial_id);
            partial.wait_front(dim_tiles);
            for (uint32_t n = 0; n < dim_tiles; ++n)
                copy_tile(partial_id, n, destination_base + n);
            partial.pop_front(dim_tiles);
            reconfig_data_format_srca(partial_id, rhs_id);
            matmul_init(lhs_id, rhs_id);
        }
        for (uint32_t n = 0; n < dim_tiles; ++n) {
            for (uint32_t inner = 0; inner < phase_tiles; ++inner) {
                const uint32_t k = phase * phase_tiles + inner;
                matmul_tiles(
                    lhs_id, rhs_id, k, k * dim_tiles + n,
                    destination_base + n);
            }
        }
        tile_regs_commit();
        const bool last = phase + 1 == phases;
        if (last) output.reserve_back(dim_tiles);
        else partial.reserve_back(dim_tiles);
        tile_regs_wait();
        const uint32_t target = last ? output_id : partial_id;
        PACK((pack_reconfig_data_format(target)));
        for (uint32_t n = 0; n < dim_tiles; ++n)
            pack_tile(destination_base + n, target);
        tile_regs_release();
        if (last) output.push_back(dim_tiles);
        else partial.push_back(dim_tiles);
    }
    lhs.pop_front(seq_tiles);
    rhs.pop_front(seq_tiles * dim_tiles);
}

FORCE_INLINE void matmul_probability_value_grouped(
    uint32_t lhs_id, uint32_t rhs_id, uint32_t partial_id,
    uint32_t output_id, uint32_t seq_tiles, uint32_t dim_tiles,
    uint32_t grouped_rows) {
    CircularBuffer lhs(lhs_id), rhs(rhs_id);
    CircularBuffer partial(partial_id), output(output_id);
    compute_kernel_hw_startup<SrcOrder::Reverse>(
        lhs_id, rhs_id, partial_id);
    matmul_init(lhs_id, rhs_id);

    // Exact specialization of TTNN's accepted bmm_large_block_zm schedule:
    //   in0_block_w=1, out_block=4x2, subblock=2x2, K blocks=seq_tiles.
    // CB16 is K-major (four M rows per K block) and V is KxN.  Keeping all
    // four output rows in the same 2x2 DEST epoch is numerically observable
    // on Blackhole and cannot be replaced by four independent row kernels.
    constexpr uint32_t subblock_h = 2;
    constexpr uint32_t subblock_w = 2;
    constexpr uint32_t subblock_tiles = subblock_h * subblock_w;
    const uint32_t row_subblocks = grouped_rows / subblock_h;
    for (uint32_t block = 0; block < seq_tiles; ++block) {
        // Match bmm_large_block_zm's input epoch exactly: one 4x1 A block
        // and one 1x2 B block become the CB fronts for each K phase.  Merely
        // indexing the complete 16/8-tile generations absolutely changes the
        // unpack schedule for the second M subblock on Blackhole.
        lhs.wait_front(grouped_rows);
        rhs.wait_front(dim_tiles);
        const bool last = block + 1 == seq_tiles;
        for (uint32_t row_subblock = 0;
             row_subblock < row_subblocks; ++row_subblock) {
            tile_regs_acquire();
            if (block > 0) {
                copy_tile_to_dst_init_short_with_dt(rhs_id, partial_id);
                partial.wait_front(subblock_tiles);
                for (uint32_t tile = 0; tile < subblock_tiles; ++tile)
                    copy_tile(partial_id, tile, tile);
                partial.pop_front(subblock_tiles);
                reconfig_data_format_srca(partial_id, rhs_id);
                matmul_init(lhs_id, rhs_id);
            }
            uint32_t destination = 0;
            for (uint32_t h = 0; h < subblock_h; ++h) {
                const uint32_t lhs_tile =
                    row_subblock * subblock_h + h;
                for (uint32_t w = 0; w < subblock_w; ++w) {
                    const uint32_t rhs_tile = w;
                    matmul_tiles(
                        lhs_id, rhs_id, lhs_tile, rhs_tile,
                        destination++);
                }
            }
            tile_regs_commit();
            if (last) output.reserve_back(subblock_tiles);
            else partial.reserve_back(subblock_tiles);
            tile_regs_wait();
            const uint32_t target = last ? output_id : partial_id;
            PACK((pack_reconfig_data_format(target)));
            for (uint32_t tile = 0; tile < subblock_tiles; ++tile)
                pack_tile(tile, target);
            tile_regs_release();
            if (last) output.push_back(subblock_tiles);
            else partial.push_back(subblock_tiles);
        }
        lhs.pop_front(grouped_rows);
        rhs.pop_front(dim_tiles);
    }
}

FORCE_INLINE void bf16_value_to_fp32(
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
    output.push_back(tiles); input.pop_front(tiles);
}

FORCE_INLINE void fp32_head_to_bf16(
    uint32_t input_id, uint32_t output_id, uint32_t tiles) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(tiles); output.reserve_back(tiles);
    compute_kernel_hw_startup(input_id, output_id);
    copy_tile_to_dst_init_short(input_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        copy_tile(input_id, tile, 0);
        typecast_tile_init<
            static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float32),
            static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float16_b)>();
        typecast_tile<
            static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float32),
            static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float16_b)>(0);
        tile_regs_commit(); tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, tile);
        tile_regs_release();
    }
    output.push_back(tiles); input.pop_front(tiles);
}

}  // namespace

void kernel_main() {
    WAYPOINT("AFC3");
    constexpr uint32_t cb_score = get_compile_time_arg_val(0);
    constexpr uint32_t cb_score_fp32 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_max_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_max = get_compile_time_arg_val(3);
    constexpr uint32_t cb_exps = get_compile_time_arg_val(4);
    constexpr uint32_t cb_sum_scaler = get_compile_time_arg_val(5);
    constexpr uint32_t cb_recip = get_compile_time_arg_val(6);
    constexpr uint32_t cb_probability = get_compile_time_arg_val(7);
    constexpr uint32_t cb_probability_saved = get_compile_time_arg_val(8);
    constexpr uint32_t cb_probability_bf16 = get_compile_time_arg_val(9);
    constexpr uint32_t cb_value = get_compile_time_arg_val(10);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(11);
    constexpr uint32_t cb_head = get_compile_time_arg_val(12);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t dim_tiles = @DIM_TILES@;
    constexpr uint32_t jobs = @QUERY_HEADS@ * seq_tiles;
    const uint32_t node = get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= jobs) return;

#if @FP32_CANONICAL_ATTENTION@
    // The prefix already produced canonical FP32 masked logits.  Feeding
    // softmax directly avoids the old BF16 score spill and re-promotion.
    CircularBuffer score_fp32(cb_score);
    compute_kernel_hw_startup(cb_score, cb_exps);
#else
    // The framework performs an explicit BF16 -> FP32 conversion before the
    // attention-optimized softmax. Preserve that visible boundary exactly.
    CircularBuffer score(cb_score), score_fp32(cb_score_fp32);
    score.wait_front(seq_tiles);
    score_fp32.reserve_back(seq_tiles);
    compute_kernel_hw_startup(cb_score, cb_score_fp32);
    copy_tile_to_dst_init_short(cb_score);
    for (uint32_t start = 0; start < seq_tiles; start += 4) {
        const uint32_t chunk = seq_tiles - start < 4 ? seq_tiles - start : 4;
        tile_regs_acquire();
        for (uint32_t i = 0; i < chunk; ++i) {
            copy_tile(cb_score, start + i, i);
            typecast_tile_init<
                static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float16_b),
                static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float32)>();
            typecast_tile<
                static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float16_b),
                static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float32)>(i);
        }
        tile_regs_commit(); tile_regs_wait();
        PACK((pack_reconfig_data_format(cb_score_fp32)));
        for (uint32_t i = 0; i < chunk; ++i)
            pack_tile<true>(i, cb_score_fp32, start + i);
        tile_regs_release();
    }
    score_fp32.push_back(seq_tiles);
    score.pop_front(seq_tiles);
#if @PUBLISH_RAW_SCORE_CHECKPOINT@ && !@CANONICAL_POW2_SCALE@
    // Diagnostic only: the data-movement RISC snapshots the exact FP32 input
    // generation before softmax consumes it, then returns a local CB credit.
    CircularBuffer softmax_input_checkpoint_ack(16);
    softmax_input_checkpoint_ack.wait_front(1);
    softmax_input_checkpoint_ack.pop_front(1);
#endif
#endif

#if @CANONICAL_MOREH_SOFTMAX@
    WAYPOINT("AFC7");
    // CB20 is empty after the scale/mask prefix and is reused for the
    // canonical last-tile temporary before becoming the FP32 V buffer.
    // CB30 is compute-owned throughout the resident tail.  It first owns
    // x-minus-max and, after that FIFO is retired, becomes the P@V partial
    // buffer.  Do not use CB28 here: its earlier score-scale generation was
    // produced by the data-movement RISC, so its compute packer and unpacker
    // intentionally have different counter histories after role reversal.
#if @FP32_CANONICAL_ATTENTION@
    canonical_moreh_softmax_fp32<
        cb_score, cb_max_scaler, cb_sum_scaler, cb_max,
        cb_score_fp32, cb_partial, cb_exps, cb_recip,
        cb_probability, cb_probability_bf16>(seq_tiles);
#else
    // The framework-visible score has a BF16 pack boundary. CB20 contains its
    // exact FP32 promotion. Reuse CB27 for Moreh's one-tile temporary before
    // the reader publishes the final probability checkpoint there. Keep the
    // tail mask (CB28) and reciprocal (CB25) in distinct FIFOs: sharing their
    // address left the unpack/pack RISC views in different epochs and hung the
    // normalization phase on repeated physical-CB use.
    canonical_moreh_softmax_fp32<
        cb_score_fp32, cb_max_scaler, cb_sum_scaler, cb_max,
        cb_probability_saved, cb_partial, cb_exps, cb_recip,
        cb_probability, cb_probability_bf16>(seq_tiles);
#endif
    WAYPOINT("AFC4");
#else
    CircularBuffer max_scaler(cb_max_scaler), sum_scaler(cb_sum_scaler);
    max_scaler.wait_front(1);
    sum_scaler.wait_front(1);
    binary_op_init_common(
#if @FP32_CANONICAL_ATTENTION@
        cb_score,
#else
        cb_score_fp32,
#endif
        cb_max_scaler, cb_exps);
    numeric_stable<
#if @FP32_CANONICAL_ATTENTION@
        cb_score,
#else
        cb_score_fp32,
#endif
        cb_max_scaler, cb_max, cb_exps>(seq_tiles);
    CircularBuffer exps(cb_exps), recip(cb_recip);
    compute_kernel_lib::reduce<
        PoolType::SUM, ReduceDim::REDUCE_ROW,
        cb_exps, cb_sum_scaler, cb_recip,
        compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
            compute_kernel_lib::ReduceInputBlockShape::row(seq_tiles),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            [](uint32_t) { recip_tile_init(); recip_tile(0); });

    CircularBuffer probability(cb_probability);
    recip.wait_front(1);
    exps.wait_front(seq_tiles);
    reconfig_data_format(cb_exps, cb_recip);
    pack_reconfig_data_format(cb_probability);
    mul_bcast_cols_init_short(cb_exps, cb_recip);
    const uint32_t ndst = seq_tiles < 4 ? seq_tiles : 4;
    for (uint32_t start = 0; start < seq_tiles; start += ndst) {
        tile_regs_acquire();
        probability.reserve_back(ndst);
        for (uint32_t i = 0; i < ndst; ++i)
            mul_tiles_bcast<BroadcastType::COL>(cb_exps, cb_recip, start + i, 0, i);
        tile_regs_commit(); tile_regs_wait();
        for (uint32_t i = 0; i < ndst; ++i)
            pack_tile(i, cb_probability);
        tile_regs_release();
        probability.push_back(ndst);
    }
    recip.pop_front(1); exps.pop_front(seq_tiles);
    max_scaler.pop_front(1); sum_scaler.pop_front(1);
    // This fused program continues from softmax reductions directly into a
    // matmul. On Blackhole the reduction API leaves a packer edge mask and
    // math-reduction state active until explicitly cleared; a separate TTNN
    // softmax program gets that reset at its kernel boundary. Without this
    // reset P@V's fourth causal K block is packed under stale reduction state,
    // producing the only remaining row-3 divergence.
    reduce_uninit(cb_exps);
    WAYPOINT("AFC4");
#endif

#if @FP32_CANONICAL_ATTENTION@
    // Canonical model path keeps both probability and V in FP32 through P@V,
    // then performs one explicit BF16 conversion at the final head boundary.
    constexpr uint32_t cb_value_fp32 = 20;
    constexpr uint32_t cb_head_fp32 = 19;
#if @GROUPED_PV_OWNER@
    constexpr uint32_t cb_grouped_probability = cb_probability_saved;
    const uint32_t row = node % seq_tiles;
    constexpr uint32_t grouped_rows = 4;
    const bool grouped_pv_leader = row == 0;
    if (!grouped_pv_leader) {
        // Reader-side NoC publication must finish before these local source
        // generations can be retired. CB28 has no FP32-canonical tail payload
        // and carries this one-credit acknowledgement.
        CircularBuffer publication_ack(28);
        publication_ack.wait_front(1);
        publication_ack.pop_front(1);
        CircularBuffer local_probability(cb_probability);
        CircularBuffer local_value(cb_value);
        local_probability.wait_front(seq_tiles);
        local_value.wait_front(seq_tiles * dim_tiles);
        local_probability.pop_front(seq_tiles);
        local_value.pop_front(seq_tiles * dim_tiles);
        return;
    }
    // The writer publishes the K-major CB27 checkpoint to the row-major
    // backward arena first, then returns a CB16 credit.  Only compute pops
    // CB27, so the grouped matmul begins after that read-only publication.
    CircularBuffer checkpoint_ack(16);
    checkpoint_ack.wait_front(1);
    checkpoint_ack.pop_front(1);
    WAYPOINT("AFC5");
    matmul_probability_value_grouped(
        cb_grouped_probability, cb_probability_bf16, cb_partial,
        cb_head_fp32, seq_tiles, dim_tiles, grouped_rows);
    CircularBuffer local_value(cb_value);
    local_value.wait_front(seq_tiles * dim_tiles);
    local_value.pop_front(seq_tiles * dim_tiles);
    CircularBuffer fp32_head_conversion_ack(16);
    fp32_head_conversion_ack.wait_front(1);
    fp32_head_conversion_ack.pop_front(1);
    CircularBuffer grouped_fp32_head(cb_head_fp32);
    grouped_fp32_head.wait_front(grouped_rows * dim_tiles);
    grouped_fp32_head.pop_front(grouped_rows * dim_tiles);
    CircularBuffer local_probability(cb_probability);
    local_probability.wait_front(seq_tiles);
    local_probability.pop_front(seq_tiles);
    WAYPOINT("AFC6");
#else
    bf16_value_to_fp32(
        cb_value, cb_value_fp32, seq_tiles * dim_tiles);
    WAYPOINT("AFC5");
    matmul_probability_value(
        cb_probability, cb_value_fp32, cb_partial, cb_head_fp32,
        seq_tiles, dim_tiles, @PV_PHASE_TILES@,
        (node % seq_tiles) % 2);
    WAYPOINT("AFC6");
    fp32_head_to_bf16(cb_head_fp32, cb_head, dim_tiles);
#endif
    WAYPOINT("AFC9");
#else
    // The reader RISC-V consumes the FP32 probability CB and publishes the
    // exact-RNE BF16 CB without a PJRT or DRAM boundary. Waiting here is a
    // device-local producer/consumer edge.
    matmul_probability_value(
        @PROBABILITY_BF16_CB@, cb_value, cb_partial, cb_head,
        seq_tiles, dim_tiles, @PV_PHASE_TILES@);
#endif
}
