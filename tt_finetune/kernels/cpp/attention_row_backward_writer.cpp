// Diagnostic writer: final dQ/dV plus scaled dS for the future L1 handoff.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc1(1);
    constexpr uint32_t cb_dv = get_compile_time_arg_val(8);
    constexpr uint32_t cb_ds_scaled = get_compile_time_arg_val(15);
    constexpr uint32_t cb_dq = get_compile_time_arg_val(16);
    constexpr uint32_t cb_dp_snapshot = get_compile_time_arg_val(18);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t dim_tiles = @DIM_TILES@;
    constexpr uint32_t query_heads = @QUERY_HEADS@;
    constexpr uint32_t jobs = query_heads * seq_tiles;
    constexpr uint32_t arena_row_tiles = seq_tiles;
    constexpr uint32_t section_tiles = jobs * arena_row_tiles;
    constexpr uint32_t diagnostic_stage = @DIAGNOSTIC_STAGE@;
    constexpr uint32_t tile_bytes = 4096;
    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= jobs) return;

    constexpr uint32_t diagnostic_cb = diagnostic_stage == 0 ? cb_ds_scaled :
        (diagnostic_stage == 1 ? get_compile_time_arg_val(10) :
        (diagnostic_stage == 2 ? get_compile_time_arg_val(14) :
        (diagnostic_stage == 3 ? get_compile_time_arg_val(11) :
        (diagnostic_stage == 4 ? get_compile_time_arg_val(13) :
                                 get_compile_time_arg_val(12)))));
    CircularBuffer dv(cb_dv), diagnostic(diagnostic_cb), dq(cb_dq);
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const auto output = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<4, 19>(), 0>(),
        output_addr, tile_bytes);
    if (diagnostic_stage < 3) dq.wait_front(dim_tiles);
    constexpr uint32_t diagnostic_tiles = diagnostic_stage == 5 ? 1 : seq_tiles;
    diagnostic.wait_front(diagnostic_tiles);
    dv.wait_front(dim_tiles);
    for (uint32_t tile = 0; tile < dim_tiles; ++tile) {
        if (diagnostic_stage < 3)
            noc1.async_write(
                CoreLocalMem<uint32_t>(dq.get_read_ptr() + tile * tile_bytes),
                output, tile_bytes, {},
                {.page_id = node * arena_row_tiles + tile});
        noc1.async_write(
            CoreLocalMem<uint32_t>(dv.get_read_ptr() + tile * tile_bytes),
            output, tile_bytes, {},
            {.page_id = 2 * section_tiles + node * arena_row_tiles + tile});
    }
    for (uint32_t tile = 0; tile < diagnostic_tiles; ++tile) {
        noc1.async_write(
            CoreLocalMem<uint32_t>(diagnostic.get_read_ptr() + tile * tile_bytes),
            output, tile_bytes, {},
            {.page_id = section_tiles + node * arena_row_tiles + tile});
    }
    noc1.async_write_barrier();
    if (diagnostic_stage < 3) dq.pop_front(dim_tiles);
    if (diagnostic_stage != 1) diagnostic.pop_front(diagnostic_tiles);
    // In dP diagnostic mode the compute kernel deliberately owns the CB for
    // the invocation lifetime so the writer can observe the unmodified FP32
    // matmul result without a TF32 copy-tile round trip.
    dv.pop_front(dim_tiles);
}
