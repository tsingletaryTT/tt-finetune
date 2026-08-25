// Final-only BF16 writer for resident RMSNorm backward.

#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_output = get_compile_time_arg_val(18);
    constexpr uint32_t cb_reader_release = get_compile_time_arg_val(13);
    constexpr uint32_t cb_compute_release = get_compile_time_arg_val(14);
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@;
    constexpr uint32_t token_tiles = @TOKEN_TILES@;
    constexpr uint32_t total_cbs = 21;
    constexpr uint32_t tile_bytes = @OUTPUT_TILE_BYTES@;
    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= token_tiles) return;

    const uint32_t output_addr =
        get_common_arg_val<uint32_t>(@OUTPUT_POSITION@);
    const auto output_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                @OUTPUT_TENSOR@, total_cbs>(), @OUTPUT_POSITION@>(),
        output_addr, tile_bytes);
    Noc noc1(1);
    CircularBuffer output(cb_output);
    output.wait_front(hidden_tiles);
    for (uint32_t tile = 0; tile < hidden_tiles; ++tile) {
        noc1.async_write(
            CoreLocalMem<uint32_t>(
                output.get_read_ptr() + tile * tile_bytes),
            output_tensor, tile_bytes, {},
            {.page_id = node * hidden_tiles + tile});
    }
    noc1.async_write_barrier();
    output.pop_front(hidden_tiles);

    // Close the three-RISC epoch only after every output page owned by this
    // core is visible. This prevents a subsequent program's reader from
    // overtaking the current writer on the same Tensix core.
    CircularBuffer reader_release(cb_reader_release);
    CircularBuffer compute_release(cb_compute_release);
    reader_release.reserve_back(1);
    reader_release.push_back(1);
    compute_release.reserve_back(1);
    compute_release.push_back(1);
}
