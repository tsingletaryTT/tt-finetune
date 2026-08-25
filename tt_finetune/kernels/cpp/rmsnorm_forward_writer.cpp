// Final BF16 writer for the exact resident RMSNorm forward qualifier.

#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_output = @WRITER_CB_ID@;
    constexpr uint32_t output_tiles = @WRITER_TILES@;
    constexpr uint32_t published_block_tiles = @WRITER_BLOCK_TILES@;
    constexpr bool replicate_scalar = @REPLICATE_SCALAR@;
    constexpr uint32_t token_tiles = @TOKEN_TILES@;
    constexpr uint32_t total_cbs = 23;
    constexpr uint32_t tile_bytes = @OUTPUT_TILE_BYTES@;
    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= token_tiles) return;

    const uint32_t output_addr =
        get_common_arg_val<uint32_t>(@OUTPUT_POSITION@);
    const auto output_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                @OUTPUT_TENSOR_INDEX@, total_cbs>(), @OUTPUT_POSITION@>(),
        output_addr, tile_bytes);
    Noc noc1(1);
    CircularBuffer output(cb_output);
    uint32_t page = node * output_tiles;
    if constexpr (replicate_scalar) {
        // RMSNorm backward consumes one full-width FP32 rstd page row. The
        // exact scalar statistic is produced once; replicate its raw tile
        // through NoC without introducing another compute/rounding boundary.
        output.wait_front(1);
        for (uint32_t tile = 0; tile < output_tiles; ++tile) {
            noc1.async_write(
                CoreLocalMem<uint32_t>(output.get_read_ptr()),
                output_tensor, tile_bytes, {}, {.page_id = page++});
        }
        noc1.async_write_barrier();
        output.pop_front(1);
        return;
    }
    for (uint32_t start = 0; start < output_tiles;
         start += published_block_tiles) {
        const uint32_t valid =
            output_tiles - start < published_block_tiles
                ? output_tiles - start
                : published_block_tiles;
        // Compute always publishes a complete padded block.  Consume that
        // complete block as well: leaving the tail padding in the FIFO makes
        // the next invocation observe stale CB state.
        output.wait_front(published_block_tiles);
        for (uint32_t tile = 0; tile < valid; ++tile) {
            noc1.async_write(
                CoreLocalMem<uint32_t>(
                    output.get_read_ptr() + tile * tile_bytes),
                output_tensor, tile_bytes, {}, {.page_id = page++});
        }
        noc1.async_write_barrier();
        output.pop_front(published_block_tiles);
    }
}
