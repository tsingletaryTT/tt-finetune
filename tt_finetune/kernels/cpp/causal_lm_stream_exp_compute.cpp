// Exact FP32 cast/subtract/exp producer for the streamed reduction.

#include <cstdint>
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"

ALWI void sfpu_sub_bcast_cols(
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
    constexpr uint32_t wt = @VOCAB_TILES@;
    constexpr auto cb_logits = tt::CBIndex::c_7;
    constexpr auto cb_bf16_startup = tt::CBIndex::c_1;
    // Keep every pack-produced temporary in the hardware output-CB bank,
    // exactly as the accepted Moreh large-softmax kernel does.
    constexpr auto cb_exponential = tt::CBIndex::c_16;
    constexpr auto cb_shifted = tt::CBIndex::c_22;
    constexpr auto cb_expanded = tt::CBIndex::c_23;
    constexpr auto cb_cast = tt::CBIndex::c_24;
    constexpr auto cb_maximum = tt::CBIndex::c_27;
    // The accepted Moreh kernel starts the compute thread with two BF16 input
    // formats and an FP32 output format, then explicitly reconfigures the pack
    // path inside each helper.  The maximum itself is FP32, so using it as the
    // startup RHS changes persistent unpack state after the first SFPU tile.
    binary_op_init_common(cb_logits, cb_bf16_startup, cb_exponential);
    DataflowBuffer logits(cb_logits), maximum(cb_maximum), cast(cb_cast);
    DataflowBuffer expanded(cb_expanded), shifted(cb_shifted);
    DataflowBuffer exponential(cb_exponential);
    maximum.wait_front(1);
    for (uint32_t tile = 0; tile < wt; ++tile) {
        // This island does not run the preceding Moreh MAX reduction locally;
        // restore the same common unpack/pack baseline before every streamed
        // tile so the previous SFPU subtract cannot leak hardware state into
        // the next BF16->FP32 copy.
        binary_op_init_common(cb_logits, cb_bf16_startup, cb_exponential);
        copy_tile_to_cb(logits, cast);
        sfpu_sub_bcast_cols(cast, maximum, expanded, shifted);
        exp_tile_to_cb(shifted, exponential);
    }
    maximum.pop_front(1);
}
