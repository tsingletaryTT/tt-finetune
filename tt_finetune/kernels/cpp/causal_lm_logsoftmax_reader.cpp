// Exact large-W log-softmax reader adapted from TTNN MorehSoftmaxWLarge.

#include <cstdint>
#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_mask = 1;
    constexpr uint32_t cb_max_scaler = 2;
    constexpr uint32_t cb_sum_scaler = 3;
    constexpr uint32_t wt = @VOCAB_TILES@;
    constexpr uint32_t row_tiles = @ROW_TILES@;
    constexpr uint32_t tile_bytes = 2048;
    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= row_tiles) return;

    const uint32_t source_address = get_common_arg_val<uint32_t>(0);
    const auto source = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 29>(),
            0>(),
        source_address,
        tile_bytes);

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_max_scaler, ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_sum_scaler, ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_ROW>();
    DataflowBuffer mask(cb_mask);
    generate_mask_w<uint16_t>(mask, 32);

    Noc noc;
    DataflowBuffer input(cb_in);
    const uint32_t first_page = node * wt;
    for (uint32_t pass = 0; pass < @PASSES@; ++pass) {
        for (uint32_t tile = 0; tile < wt; ++tile) {
            input.reserve_back(1);
            noc.async_read(
                source,
                input,
                tile_bytes,
                {.page_id = first_page + tile},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            input.push_back(1);
        }
    }
}
