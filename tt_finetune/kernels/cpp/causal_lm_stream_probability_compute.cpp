// Exact replay: log_probability = (FP32(logits)-max)-logsum, probability=exp.

#include <cstdint>
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/debug/waypoint.h"

ALWI void probability_sub_bcast_cols(
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

void kernel_main() {
    WAYPOINT("PC00");
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
    binary_op_init_common(cb_logits, cb_bf16_startup, cb_probability);
    WAYPOINT("PC01");
    DataflowBuffer logits(cb_logits), maximum(cb_maximum), cast(cb_cast);
    DataflowBuffer expanded(cb_expanded), shifted(cb_shifted);
    DataflowBuffer probability(cb_probability), logsum(cb_logsum);
    DataflowBuffer log_probability(cb_log_probability);

    maximum.wait_front(1);
    WAYPOINT("PC02");
    for (uint32_t tile = 0; tile < wt; ++tile) {
        binary_op_init_common(cb_logits, cb_bf16_startup, cb_probability);
        WAYPOINT("PC03");
        copy_tile_to_cb(logits, cast);
        WAYPOINT("PC04");
        probability_sub_bcast_cols(cast, maximum, expanded, shifted);
        WAYPOINT("PC05");
        exp_tile_to_cb(shifted, probability);
        WAYPOINT("PC06");
    }

    WAYPOINT("PC20");
    logsum.wait_front(1);
    WAYPOINT("PC21");
    for (uint32_t tile = 0; tile < wt; ++tile) {
        binary_op_init_common(cb_logits, cb_bf16_startup, cb_probability);
        copy_tile_to_cb(logits, cast);
        probability_sub_bcast_cols(cast, maximum, expanded, shifted);
        probability_sub_bcast_cols(
            shifted, logsum, expanded, log_probability);
        exp_tile_to_cb(log_probability, probability);
    }
    logsum.pop_front(1);
    maximum.pop_front(1);
}
