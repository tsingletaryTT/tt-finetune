// Read one BF16 tile for the FP32-destination producer island.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    if (get_absolute_logical_x() != 0 || get_absolute_logical_y() != 9) {
        return;
    }

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
#ifdef TWO_ISLAND_PJRT_ARTIFACT
    constexpr auto input_args = TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 7>(),
        0>();
#else
    constexpr auto input_args = TensorAccessorArgs<0>();
#endif
    const auto input = TensorAccessor(input_args, input_addr);

#ifdef TWO_ISLAND_REAL_MATMUL
    constexpr uint32_t cb_lhs = 3;
    constexpr uint32_t cb_weight = 4;
    const uint32_t weight_addr = get_common_arg_val<uint32_t>(1);
#ifdef TWO_ISLAND_PJRT_ARTIFACT
    constexpr auto weight_args = TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 7>(),
        1>();
#else
    constexpr auto weight_args =
        TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
#endif
    const auto weight = TensorAccessor(weight_args, weight_addr);

    Noc noc;
    CircularBuffer lhs(cb_lhs);
    CircularBuffer rhs(cb_weight);
    lhs.reserve_back(1);
    rhs.reserve_back(1);
    noc.async_read(input, lhs, 2048, {.page_id = 0}, {});
    noc.async_read(weight, rhs, 2048, {.page_id = 0}, {});
    noc.async_read_barrier();
    lhs.push_back(1);
    rhs.push_back(1);
#else
    constexpr uint32_t cb_handoff = 0;
    constexpr uint32_t cb_input = 1;
    Noc noc;
    CircularBuffer handoff(cb_handoff);
    CircularBuffer local_input(cb_input);
    handoff.reserve_back(1);
    local_input.reserve_back(1);
    noc.async_read(input, handoff, 2048, {.page_id = 0}, {});
    noc.async_read(input, local_input, 2048, {.page_id = 0}, {});
    noc.async_read_barrier();
    handoff.push_back(1);
    local_input.push_back(1);
#endif
}
