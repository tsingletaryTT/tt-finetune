// Dedicated StableHLO-order sparse loss-numerator reduction.

#include <cstdint>
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void kernel_main() {
    constexpr uint32_t wt = @VOCAB_TILES@;
    constexpr auto cb_loss_terms = tt::CBIndex::c_13;
    constexpr auto cb_loss_scaler = tt::CBIndex::c_9;
    constexpr auto cb_token_scaler = tt::CBIndex::c_10;
    constexpr auto cb_row_loss = tt::CBIndex::c_14;
    constexpr auto cb_loss_numerator = tt::CBIndex::c_15;

    compute_kernel_hw_startup(
        cb_loss_terms, cb_loss_scaler, cb_loss_numerator);
#if @DIAGNOSTIC_LOSS_TERMS@
    copy_tile_to_cb(
        DataflowBuffer(cb_loss_terms), DataflowBuffer(cb_loss_numerator));
    for (uint32_t tile = 1; tile < wt; ++tile) {
        DataflowBuffer(cb_loss_terms).wait_front(1);
        DataflowBuffer(cb_loss_terms).pop_front(1);
    }
    DataflowBuffer(cb_loss_scaler).pop_front(1);
    DataflowBuffer(cb_token_scaler).pop_front(1);
#else
    compute_kernel_lib::reduce<
        PoolType::SUM, ReduceDim::REDUCE_ROW,
        cb_loss_terms, cb_loss_scaler, cb_row_loss,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT,
        ReduceFp32Mode::Fast>(
            compute_kernel_lib::ReduceInputBlockShape::row(wt));
    DataflowBuffer(cb_loss_scaler).pop_front(1);
    compute_kernel_lib::reduce<
        PoolType::SUM, ReduceDim::REDUCE_COL,
        cb_row_loss, cb_token_scaler, cb_loss_numerator,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT,
        ReduceFp32Mode::Fast>(
            compute_kernel_lib::ReduceInputBlockShape::col(1));
    DataflowBuffer(cb_token_scaler).pop_front(1);
#endif
}
