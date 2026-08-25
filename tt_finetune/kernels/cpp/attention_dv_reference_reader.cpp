// Reference-compatible FP32 attention dV reader.
//
// One worker owns one (query head, output sequence tile-row).  It reads the
// corresponding probability *column* tiles so the compute kernel can form
// P.T without materializing the complete transpose, plus the complete dO
// head.  This mirrors the query-head row sharding selected by TT-MLIR.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc0(0);
    constexpr uint32_t cb_p_column = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dout = get_compile_time_arg_val(1);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t dim_tiles = @DIM_TILES@;
    constexpr uint32_t query_heads = @QUERY_HEADS@;
    constexpr uint32_t score_tiles = seq_tiles * seq_tiles;
    constexpr uint32_t vector_tiles = seq_tiles * dim_tiles;
    constexpr uint32_t rows_per_worker = @ROWS_PER_WORKER@;
    constexpr uint32_t workers_per_head = seq_tiles / rows_per_worker;
    constexpr uint32_t jobs = query_heads * workers_per_head;
    constexpr uint32_t tile_bytes = @TILE_BYTES@;

    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= jobs) return;
    const uint32_t head = node / workers_per_head;
    const uint32_t output_row =
        (node - head * workers_per_head) * rows_per_worker;

    CircularBuffer p_column(cb_p_column), dout(cb_dout);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t dout_addr = get_common_arg_val<uint32_t>(1);
    const auto p_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 5>(), 0>(),
        p_addr, tile_bytes);
    const auto dout_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 5>(), 1>(),
        dout_addr, tile_bytes);

    p_column.reserve_back(rows_per_worker * seq_tiles);
    for (uint32_t row = 0; row < rows_per_worker; ++row) {
        for (uint32_t k = 0; k < seq_tiles; ++k) {
            const uint32_t page =
                head * score_tiles + k * seq_tiles + output_row + row;
            noc0.async_read(
                p_tensor,
                CoreLocalMem<uint32_t>(p_column.get_write_ptr() +
                    (row * seq_tiles + k) * tile_bytes),
                tile_bytes, {.page_id = page}, {});
        }
    }
    dout.reserve_back(vector_tiles);
    for (uint32_t tile = 0; tile < vector_tiles; ++tile) {
        noc0.async_read(
            dout_tensor,
            CoreLocalMem<uint32_t>(dout.get_write_ptr() + tile * tile_bytes),
            tile_bytes, {.page_id = head * vector_tiles + tile}, {});
    }
    noc0.async_read_barrier();
    p_column.push_back(rows_per_worker * seq_tiles);
    dout.push_back(vector_tiles);
}
