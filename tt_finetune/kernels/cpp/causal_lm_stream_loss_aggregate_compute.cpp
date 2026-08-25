// Deterministically join token-row loss partials in row order.

#include <cstdint>
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/compute/eltwise_binary_sfpu.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void kernel_main() {
    constexpr uint32_t row_tiles = @ROW_TILES@;
    constexpr auto cb_partials = tt::CBIndex::c_16;
    constexpr auto cb_scaler = tt::CBIndex::c_9;
    constexpr auto cb_global_numerator = tt::CBIndex::c_17;
    constexpr auto cb_count = tt::CBIndex::c_18;
    constexpr auto cb_loss = tt::CBIndex::c_20;
    compute_kernel_hw_startup(cb_partials, cb_scaler, cb_global_numerator);
    if constexpr (row_tiles == 1) {
        copy_tile_to_cb(
            DataflowBuffer(cb_partials),
            DataflowBuffer(cb_global_numerator));
        DataflowBuffer(cb_scaler).pop_front(1);
    } else {
        compute_kernel_lib::reduce<
            PoolType::SUM, ReduceDim::REDUCE_COL,
            cb_partials, cb_scaler, cb_global_numerator,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT,
            ReduceFp32Mode::Accurate>(
                compute_kernel_lib::ReduceInputBlockShape::col(row_tiles));
        DataflowBuffer(cb_scaler).pop_front(1);
    }

    DataflowBuffer numerator(cb_global_numerator), count(cb_count), loss(cb_loss);
    numerator.wait_front(1);
    count.wait_front(1);
    loss.reserve_back(1);
    binary_op_init_common(cb_global_numerator, cb_count, cb_loss);
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(cb_count, cb_global_numerator);
    copy_tile(cb_global_numerator, 0, 0);
    copy_tile_to_dst_init_short_with_dt(cb_global_numerator, cb_count);
    copy_tile(cb_count, 0, 1);
#if @OUTPUT_LOSS@
    div_binary_tile_init();
    div_binary_tile(0, 1, 0);
#endif
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(cb_loss)));
    pack_tile<true>(0, cb_loss, 0);
    tile_regs_release();
    loss.push_back(1);
    numerator.pop_front(1);
    count.pop_front(1);
}
