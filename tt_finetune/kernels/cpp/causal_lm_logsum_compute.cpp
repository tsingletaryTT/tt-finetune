// Match StableHLO's FP32 SUM reduction boundary followed by accurate LOG.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t row_tiles = @ROW_TILES@;
    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= row_tiles) {
        return;
    }

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_scaler = tt::CBIndex::c_2;
    constexpr auto cb_sum = tt::CBIndex::c_3;
    constexpr auto cb_logsum = tt::CBIndex::c_4;
    constexpr uint32_t vocabulary_tiles = @VOCAB_TILES@;

    compute_kernel_hw_startup(cb_input, cb_scaler, cb_sum);
    compute_kernel_lib::reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_input,
        cb_scaler,
        cb_sum,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT,
        ReduceFp32Mode::Fast>(
        compute_kernel_lib::ReduceInputBlockShape::row(vocabulary_tiles),
        compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
        compute_kernel_lib::NoAccumulation{});

    @FINAL_LOG@
    DataflowBuffer scaler(cb_scaler);
    scaler.pop_front(1);
}
