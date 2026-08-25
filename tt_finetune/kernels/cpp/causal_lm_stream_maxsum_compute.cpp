// Exact BF16 max followed by StableHLO's full-width FP32 SUM schedule.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t wt = @VOCAB_TILES@;
    compute_kernel_hw_startup(
        tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_4);
    compute_kernel_lib::reduce<
        PoolType::MAX, ReduceDim::REDUCE_ROW,
        tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_4,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast>(
        compute_kernel_lib::ReduceInputBlockShape::row(wt));
    DataflowBuffer(tt::CBIndex::c_1).pop_front(1);

    compute_kernel_lib::reduce<
        PoolType::SUM, ReduceDim::REDUCE_ROW,
        tt::CBIndex::c_3, tt::CBIndex::c_2, tt::CBIndex::c_5,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast>(
        compute_kernel_lib::ReduceInputBlockShape::row(wt));
    DataflowBuffer(tt::CBIndex::c_2).pop_front(1);
}
