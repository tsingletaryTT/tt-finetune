// Reproduce StableHLO cross-entropy's exact dLogits schedule after the
// resident max/sum/log replay.  No probability or FP32 dLogits tile leaves L1.

#include <cstdint>
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"

namespace {

ALWI void sub_bcast_cols(
    DataflowBuffer lhs, DataflowBuffer reduced_rhs,
    DataflowBuffer expanded_rhs, DataflowBuffer output) {
    expanded_rhs.reserve_back(1);
    reduced_rhs.wait_front(1);
    pack_reconfig_data_format(output.get_id(), expanded_rhs.get_id());
    unary_bcast_init<BroadcastType::COL>(
        reduced_rhs.get_id(), expanded_rhs.get_id());
    tile_regs_acquire();
    unary_bcast<BroadcastType::COL>(reduced_rhs.get_id(), 0, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(1, expanded_rhs.get_id());
    expanded_rhs.push_back(1);
    tile_regs_release();

    lhs.wait_front(1);
    expanded_rhs.wait_front(1);
    output.reserve_back(1);
    pack_reconfig_data_format(expanded_rhs.get_id(), output.get_id());
#ifdef ARCH_BLACKHOLE
    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(output.get_id())));
#endif
    sub_binary_tile_init();
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(expanded_rhs.get_id(), lhs.get_id());
    copy_tile(lhs.get_id(), 0, 0);
    copy_tile_to_dst_init_short_with_dt(lhs.get_id(), expanded_rhs.get_id());
    copy_tile(expanded_rhs.get_id(), 0, 1);
    sub_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, output.get_id());
    tile_regs_release();
    output.push_back(1);
    lhs.pop_front(1);
    expanded_rhs.pop_front(1);
}

template <bool Multiply, bool PopRhs = true, bool PopLhs = true>
ALWI void binary_to_cb(
    DataflowBuffer lhs, DataflowBuffer rhs, DataflowBuffer output) {
    lhs.wait_front(1);
    rhs.wait_front(1);
    output.reserve_back(1);
    // Reinitialize both unpack paths after the preceding SFPU exp.  A unary
    // init leaves SrcB carrying the exp/broadcast format state; on Blackhole
    // that made the correct -1/count tile decode as a much larger value.
    binary_op_init_common(lhs.get_id(), rhs.get_id(), output.get_id());
    if constexpr (Multiply) mul_binary_tile_init();
    else sub_binary_tile_init();
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(rhs.get_id(), lhs.get_id());
    copy_tile(lhs.get_id(), 0, 0);
    copy_tile_to_dst_init_short_with_dt(lhs.get_id(), rhs.get_id());
    copy_tile(rhs.get_id(), 0, 1);
    if constexpr (Multiply) mul_binary_tile(0, 1, 0);
    else sub_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(output.get_id())));
    pack_tile(0, output.get_id());
    tile_regs_release();
    output.push_back(1);
    if constexpr (PopLhs) lhs.pop_front(1);
    if constexpr (PopRhs) rhs.pop_front(1);
}

ALWI void copy_keep_to_cb(DataflowBuffer input, DataflowBuffer output) {
    input.wait_front(1);
    output.reserve_back(1);
    unary_op_init_common(input.get_id(), output.get_id());
    tile_regs_acquire();
    copy_tile_init(input.get_id());
    copy_tile(input.get_id(), 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(output.get_id())));
    pack_tile(0, output.get_id());
    tile_regs_release();
    output.push_back(1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t wt = @VOCAB_TILES@;
    constexpr auto cb_logits = tt::CBIndex::c_7;
    constexpr auto cb_bf16_startup = tt::CBIndex::c_1;
    constexpr auto cb_probability = tt::CBIndex::c_16;
    constexpr auto cb_shifted = tt::CBIndex::c_22;
    constexpr auto cb_expanded = tt::CBIndex::c_23;
    constexpr auto cb_cast = tt::CBIndex::c_24;
    constexpr auto cb_logsum = tt::CBIndex::c_25;
    constexpr auto cb_log_probability = tt::CBIndex::c_26;
    constexpr auto cb_maximum = tt::CBIndex::c_27;
    constexpr auto cb_seed = tt::CBIndex::c_28;
    constexpr auto cb_product = tt::CBIndex::c_30;
    constexpr auto cb_tf32_row_seed = tt::CBIndex::c_18;
    // Dedicated final-output CB. Reusing c26 would expose log_probability to
    // the writer before exp consumes it; a CB can only have one concurrent
    // consumer epoch.
    constexpr auto cb_dlogits = tt::CBIndex::c_20;

    binary_op_init_common(cb_logits, cb_bf16_startup, cb_probability);
    DataflowBuffer logits(cb_logits), maximum(cb_maximum), cast(cb_cast);
    DataflowBuffer expanded(cb_expanded), shifted(cb_shifted);
    DataflowBuffer probability(cb_probability), logsum(cb_logsum);
    DataflowBuffer log_probability(cb_log_probability);
    DataflowBuffer seed(cb_seed), tf32_row_seed(cb_tf32_row_seed);
    DataflowBuffer negative_target(tt::CBIndex::c_17);
    DataflowBuffer loss_contribution(tt::CBIndex::c_21);
    DataflowBuffer product(cb_product), dlogits(cb_dlogits);

    maximum.wait_front(1);
    for (uint32_t tile = 0; tile < wt; ++tile) {
        binary_op_init_common(cb_logits, cb_bf16_startup, cb_probability);
        copy_tile_to_cb(logits, cast);
        sub_bcast_cols(cast, maximum, expanded, shifted);
        exp_tile_to_cb(shifted, probability);
    }

    logsum.wait_front(1);

    for (uint32_t tile = 0; tile < wt; ++tile) {
        binary_op_init_common(cb_logits, cb_bf16_startup, cb_probability);
        copy_tile_to_cb(logits, cast);
        sub_bcast_cols(cast, maximum, expanded, shifted);
        sub_bcast_cols(shifted, logsum, expanded, log_probability);
        // StableHLO loss numerator: (-one_hot * log_probability) * valid.
        // negative_target is already zero for ignored rows.  Retain
        // log_probability in L1 for the immediately following exponential.
        binary_to_cb<true, true, false>(
            log_probability, negative_target, loss_contribution);
        exp_tile_to_cb(log_probability, probability);

#if @DIAGNOSTIC_ROW_SEED@
        probability.pop_front(1);
        seed.pop_front(1);
        copy_keep_to_cb(tf32_row_seed, dlogits);
#elif @DIAGNOSTIC_SEED@
        probability.pop_front(1);
        copy_tile_to_cb(seed, dlogits);
#else
        // StableHLO order: probability * row_seed_sum, then
        // seed - product.  Both operations remain FP32; writer performs the
        // one final BF16 conversion.
        binary_to_cb<true, false>(
            probability, tf32_row_seed, product);
#if @DIAGNOSTIC_PRODUCT@
        seed.pop_front(1);
        copy_tile_to_cb(product, dlogits);
#else
        binary_to_cb<false>(seed, product, dlogits);
#endif
#endif
    }
    tf32_row_seed.pop_front(1);
    logsum.pop_front(1);
    maximum.pop_front(1);
}
