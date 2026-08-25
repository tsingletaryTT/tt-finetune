// Exact resident RMSNorm-backward reader for one 32-token row per core.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dy = get_compile_time_arg_val(1);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(2);
    constexpr uint32_t cb_rstd = get_compile_time_arg_val(3);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(8);
    constexpr uint32_t cb_reader_release = get_compile_time_arg_val(13);
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@;
    constexpr uint32_t token_tiles = @TOKEN_TILES@;
    constexpr uint32_t total_cbs = 21;
    constexpr uint32_t bf16_tile_bytes = 2048;
    constexpr uint32_t fp32_tile_bytes = 4096;

    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= token_tiles) return;

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scaler, ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_ROW>();

    const uint32_t x_addr = get_common_arg_val<uint32_t>(@X_POSITION@);
    const uint32_t dy_addr = get_common_arg_val<uint32_t>(@DY_POSITION@);
    const uint32_t weight_addr =
        get_common_arg_val<uint32_t>(@WEIGHT_POSITION@);
    const uint32_t rstd_addr =
        get_common_arg_val<uint32_t>(@RSTD_POSITION@);
    const auto x_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                0, total_cbs>(), @X_POSITION@>(),
        x_addr, bf16_tile_bytes);
    const auto dy_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                1, total_cbs>(), @DY_POSITION@>(),
        dy_addr, bf16_tile_bytes);
    const auto weight_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                2, total_cbs>(), @WEIGHT_POSITION@>(),
        weight_addr, bf16_tile_bytes);
    const auto rstd_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                3, total_cbs>(), @RSTD_POSITION@>(),
        rstd_addr, fp32_tile_bytes);

    Noc noc0(0);
    CircularBuffer x(cb_x), dy(cb_dy), weight(cb_weight), rstd(cb_rstd);
    x.reserve_back(hidden_tiles);
    dy.reserve_back(hidden_tiles);
    weight.reserve_back(hidden_tiles);
    for (uint32_t tile = 0; tile < hidden_tiles; ++tile) {
        const uint32_t page = node * hidden_tiles + tile;
        noc0.async_read(
            x_tensor,
            CoreLocalMem<uint32_t>(x.get_write_ptr() + tile * bf16_tile_bytes),
            bf16_tile_bytes, {.page_id = page}, {});
        noc0.async_read(
            dy_tensor,
            CoreLocalMem<uint32_t>(dy.get_write_ptr() + tile * bf16_tile_bytes),
            bf16_tile_bytes, {.page_id = page}, {});
        noc0.async_read(
            weight_tensor,
            CoreLocalMem<uint32_t>(
                weight.get_write_ptr() + tile * bf16_tile_bytes),
            // Gamma is one 32-row tile repeated over every token shard.  It
            // is deliberately represented as [32, hidden], not
            // [sequence, hidden].  Applying the token-shard offset here
            // made every worker except node 0 read beyond the tensor.  The
            // bad address happened to be stable in isolated runs but
            // changed when this primitive was embedded in a larger resident
            // allocation.
            bf16_tile_bytes, {.page_id = tile}, {});
    }
    rstd.reserve_back(hidden_tiles);
    for (uint32_t tile = 0; tile < hidden_tiles; ++tile) {
        noc0.async_read(
            rstd_tensor,
            CoreLocalMem<uint32_t>(
                rstd.get_write_ptr() + tile * fp32_tile_bytes),
            fp32_tile_bytes,
            {.page_id = node * hidden_tiles + tile}, {});
    }
    noc0.async_read_barrier();
    x.push_back(hidden_tiles);
    dy.push_back(hidden_tiles);
    weight.push_back(hidden_tiles);
    rstd.push_back(hidden_tiles);

    // A following compact program may start its NCRISC as soon as this
    // reader exits, while this program's BRISC is still draining output NoC
    // writes. Keep the reader epoch alive until the writer publishes its
    // local completion token.
    CircularBuffer reader_release(cb_reader_release);
    reader_release.wait_front(1);
    reader_release.pop_front(1);
}
