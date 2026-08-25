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
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace {
template <uint32_t cb_in, uint32_t cb_scaler, uint32_t cb_max, uint32_t cb_exps>
FORCE_INLINE void numeric_stable(uint32_t width) {
    CircularBuffer input(cb_in), maximum(cb_max), exps(cb_exps);
    compute_kernel_lib::reduce<
        PoolType::MAX, ReduceDim::REDUCE_ROW, cb_in, cb_scaler, cb_max,
        compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
            compute_kernel_lib::ReduceInputBlockShape::row(width));
    maximum.wait_front(1);
    exp_tile_init<true>();
    reconfig_data_format_srcb(cb_max);
    sub_bcast_cols_init_short(cb_in, cb_max);
    input.wait_front(width);
    exps.reserve_back(width);
    for (uint32_t start = 0; start < width; start += 4) {
        const uint32_t chunk = width - start < 4 ? width - start : 4;
        tile_regs_acquire();
        for (uint32_t i = 0; i < chunk; ++i)
            sub_tiles_bcast_cols(cb_in, cb_max, start + i, 0, i);
        for (uint32_t i = 0; i < chunk; ++i) exp_tile<true>(i);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(cb_exps)));
        for (uint32_t i = 0; i < chunk; ++i)
            pack_tile<true>(i, cb_exps, start + i);
        tile_regs_release();
    }
    exps.push_back(width);
    input.pop_front(width);
    maximum.pop_front(1);
}

FORCE_INLINE void matmul_probability_value(
    uint32_t lhs_id, uint32_t rhs_id, uint32_t partial_id,
    uint32_t output_id,
    uint32_t seq_tiles, uint32_t dim_tiles, uint32_t phase_tiles) {
    CircularBuffer lhs(lhs_id), rhs(rhs_id);
    lhs.wait_front(seq_tiles);
    rhs.wait_front(seq_tiles * dim_tiles);
    CircularBuffer partial(partial_id), output(output_id);
    compute_kernel_hw_startup<SrcOrder::Reverse>(
        lhs_id, rhs_id, partial_id);
    const uint32_t phases = seq_tiles / phase_tiles;
    // TTNN's selected 2-D multicast config has per_core_N=1 and a 1x1
    // output subblock. Reproduce that destination-register issue order even
    // though this resident worker owns all D tiles for its query row.
    for (uint32_t n = 0; n < dim_tiles; ++n) {
        for (uint32_t phase = 0; phase < phases; ++phase) {
            tile_regs_acquire();
            if (phase > 0) {
                copy_tile_to_dst_init_short_with_dt(rhs_id, partial_id);
                partial.wait_front(1);
                copy_block_matmul_partials(partial_id, 0, 0, 1);
                partial.pop_front(1);
                reconfig_data_format_srca(partial_id, rhs_id);
            }
            matmul_block_init(
                lhs_id, rhs_id, false, 1, 1, phase_tiles);
            for (uint32_t inner = 0; inner < phase_tiles; ++inner) {
                const uint32_t k = phase * phase_tiles + inner;
                matmul_block(
                    lhs_id, rhs_id, k, k * dim_tiles + n, 0,
                    false, 1, 1, phase_tiles);
            }
            tile_regs_commit();
            const bool last = phase + 1 == phases;
            if (last) output.reserve_back(1);
            else partial.reserve_back(1);
            tile_regs_wait();
            const uint32_t target = last ? output_id : partial_id;
            PACK((pack_reconfig_data_format(target)));
            pack_tile_block(0, target, 1);
            tile_regs_release();
            if (last) output.push_back(1);
            else partial.push_back(1);
        }
    }
    lhs.pop_front(seq_tiles);
    rhs.pop_front(seq_tiles * dim_tiles);
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
#endif

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

    CircularBuffer probability(cb_probability), saved(cb_probability_saved);
    recip.wait_front(1);
    exps.wait_front(seq_tiles);
    probability.reserve_back(seq_tiles);
    saved.reserve_back(seq_tiles);
    reconfig_data_format(cb_exps, cb_recip);
    mul_bcast_cols_init_short(cb_exps, cb_recip);
    for (uint32_t start = 0; start < seq_tiles; start += 4) {
        const uint32_t chunk = seq_tiles - start < 4 ? seq_tiles - start : 4;
        tile_regs_acquire();
        for (uint32_t i = 0; i < chunk; ++i)
            mul_tiles_bcast<BroadcastType::COL>(cb_exps, cb_recip, start + i, 0, i);
        tile_regs_commit(); tile_regs_wait();
        PACK((pack_reconfig_data_format(cb_probability)));
        for (uint32_t i = 0; i < chunk; ++i) {
            pack_tile<true>(i, cb_probability, start + i);
            pack_tile<true>(i, cb_probability_saved, start + i);
        }
        tile_regs_release();
    }
    probability.push_back(seq_tiles);
    saved.push_back(seq_tiles);
    recip.pop_front(1); exps.pop_front(seq_tiles);

#if @FP32_CANONICAL_ATTENTION@
    // Canonical model path keeps both probability and V in FP32 through P@V,
    // then performs one explicit BF16 conversion at the final head boundary.
    constexpr uint32_t cb_value_fp32 = 20;
    constexpr uint32_t cb_head_fp32 = 19;
    bf16_value_to_fp32(
        cb_value, cb_value_fp32, seq_tiles * dim_tiles);
    matmul_probability_value(
        cb_probability, cb_value_fp32, cb_partial, cb_head_fp32,
        seq_tiles, dim_tiles, @PV_PHASE_TILES@);
    fp32_head_to_bf16(cb_head_fp32, cb_head, dim_tiles);
#else
    // The reader RISC-V consumes the FP32 probability CB and publishes the
    // exact-RNE BF16 CB without a PJRT or DRAM boundary. Waiting here is a
    // device-local producer/consumer edge.
    matmul_probability_value(
        cb_probability_bf16, cb_value, cb_partial, cb_head,
        seq_tiles, dim_tiles, @PV_PHASE_TILES@);
#endif
}
