// Write only the final per-query-head dV tile row.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc1(1);
    constexpr uint32_t cb_dv = get_compile_time_arg_val(3);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t dim_tiles = @DIM_TILES@;
    constexpr uint32_t query_heads = @QUERY_HEADS@;
    constexpr uint32_t rows_per_worker = @ROWS_PER_WORKER@;
    constexpr uint32_t workers_per_head = seq_tiles / rows_per_worker;
    constexpr uint32_t jobs = query_heads * workers_per_head;
    constexpr uint32_t tile_bytes = @TILE_BYTES@;
    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= jobs) return;

    CircularBuffer dv(cb_dv);
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const auto output = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 5>(), 0>(),
        output_addr, tile_bytes);
    const uint32_t head = node / workers_per_head;
    const uint32_t output_row =
        (node - head * workers_per_head) * rows_per_worker;
    dv.wait_front(rows_per_worker * dim_tiles);
    for (uint32_t row = 0; row < rows_per_worker; ++row) {
        for (uint32_t tile = 0; tile < dim_tiles; ++tile) {
            noc1.async_write(
                CoreLocalMem<uint32_t>(dv.get_read_ptr() +
                    (row * dim_tiles + tile) * tile_bytes),
                output, tile_bytes, {},
                {.page_id = (head * seq_tiles + output_row + row) *
                    dim_tiles + tile});
        }
    }
    noc1.async_write_barrier();
    dv.pop_front(rows_per_worker * dim_tiles);
}
