// Read one packed logical-gradient region on each Blackhole worker core.

#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t jobs = @JOBS@;
    constexpr uint32_t jobs_per_layer = @JOBS_PER_LAYER@;
    constexpr uint32_t page_offsets[jobs_per_layer] = {
        @PAGE_OFFSET_PATTERN@
    };
    constexpr uint32_t page_counts[jobs_per_layer] = {
        @PAGE_COUNT_PATTERN@
    };
    // TT-Lang's current `grid="full"` lowering exposes an 8x8 worker grid.
    // Use the serialized program geometry, not Blackhole's physical 11x10
    // worker count, when mapping a logical core to a packed-gradient job.
    const uint32_t node =
        get_absolute_logical_y() * @GRID_WIDTH@ + get_absolute_logical_x();
    Noc noc0(0);
    CircularBuffer gradient(get_compile_time_arg_val(0));
    CircularBuffer scaler(get_compile_time_arg_val(1));
    CircularBuffer exponent(get_compile_time_arg_val(5));

    // Runtime tensor addresses follow operation-tensor order, independently
    // of which operand the generated shell touches first: gradient is common
    // argument zero and scaler is common argument one.  Six CB IDs precede
    // the tensor-accessor compile-time payload.
    const uint32_t scaler_address = get_common_arg_val<uint32_t>(1);
    auto scaler_args = TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 6>(),
        1>();
    TensorAccessor scaler_accessor(scaler_args, scaler_address, 2048);

    const uint32_t gradient_address = get_common_arg_val<uint32_t>(0);
    auto gradient_args = TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 6>(),
        0>();
    TensorAccessor gradient_accessor(gradient_args, gradient_address, 2048);
    for (uint32_t job = node; job < jobs; job += @CORE_COUNT@) {
        const uint32_t layer = job / jobs_per_layer;
        const uint32_t local_job = job % jobs_per_layer;
        const uint32_t page_offset =
            page_offsets[local_job] + layer * @LAYER_PAGE_STRIDE@;
        scaler.reserve_back(1);
        noc0.async_read(
            scaler_accessor,
            CoreLocalMem<uint32_t>(scaler.get_write_ptr()),
            scaler_accessor.get_aligned_page_size(),
            {.page_id = 0}, {});
        noc0.async_read_barrier();
        scaler.push_back(1);

        exponent.reserve_back(1);
        noc0.async_read(
            scaler_accessor,
            CoreLocalMem<uint32_t>(exponent.get_write_ptr()),
            scaler_accessor.get_aligned_page_size(),
            {.page_id = 1}, {});
        noc0.async_read_barrier();
        exponent.push_back(1);

        for (uint32_t page = 0; page < page_counts[local_job]; ++page) {
            gradient.reserve_back(1);
            noc0.async_read(
                gradient_accessor,
                CoreLocalMem<uint32_t>(gradient.get_write_ptr()),
                gradient_accessor.get_aligned_page_size(),
                {.page_id = page_offset + page}, {});
            noc0.async_read_barrier();
            gradient.push_back(1);
        }
    }
}
