// Read one GQA member group for one logical sequence tile.

#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc0(0);
    constexpr uint32_t cb_input = get_compile_time_arg_val(0);
    constexpr uint32_t cb_zero = get_compile_time_arg_val(1);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t dim_tiles = @DIM_TILES@;
    constexpr uint32_t kv_heads = @KV_HEADS@;
    constexpr uint32_t group = @GROUP@;
    constexpr uint32_t jobs = kv_heads * seq_tiles;
    constexpr uint32_t vector_tiles = seq_tiles * dim_tiles;
    constexpr uint32_t tile_bytes = 2048;
    const uint32_t node = get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= jobs) return;
    const uint32_t kv_head = node / seq_tiles;
    const uint32_t row = node - kv_head * seq_tiles;
    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const auto input_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 3>(), 0>(),
        input_addr, tile_bytes);
    CircularBuffer input(cb_input), zero(cb_zero);
    input.reserve_back(dim_tiles * group);
    uint32_t linear = 0;
    for (uint32_t tile = 0; tile < dim_tiles; ++tile) {
        for (uint32_t member = 0; member < group; ++member) {
            const uint32_t head = kv_head * group + member;
            const uint32_t page = head * vector_tiles + row * dim_tiles + tile;
            noc0.async_read(
                input_tensor,
                CoreLocalMem<uint32_t>(input.get_write_ptr() + linear * tile_bytes),
                tile_bytes, {.page_id = page}, {});
            ++linear;
        }
    }
    zero.reserve_back(1);
    noc0.async_write_zeros(zero, tile_bytes);
    noc0.write_zeros_l1_barrier();
    noc0.async_read_barrier();
    input.push_back(dim_tiles * group);
    zero.push_back(1);
}
