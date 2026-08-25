// Exact TTNN-compatible RMSNorm forward reader: one 32-token row per core.

#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/layernorm_dataflow_utils.h"

namespace generic = norm::kernel_util::generic;
namespace layernorm_dataflow_utils =
    norm::layernorm::device::kernels::dataflow;

namespace {

// binary_ng's scalar-broadcast reader materializes a full physical FP32 tile
// by repeating the first scalar element 1024 times before the SFPU consumes
// it.  A reduce-scaler tile has a different sparse face layout and is only
// valid for the FPU reduction primitive; using it as a generic binary operand
// changes the native RMSNorm numerical schedule.
FORCE_INLINE void prepare_full_fp32_scalar(
    CircularBuffer& cb, uint32_t scalar_bits) {
    cb.reserve_back(1);
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        cb.get_write_ptr());
    for (uint32_t i = 0; i < 1024; ++i) {
        ptr[i] = scalar_bits;
    }
    cb.push_back(1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(5);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_eps = get_compile_time_arg_val(3);
    constexpr uint32_t cb_inverse_hidden = 4;
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@;
    constexpr uint32_t padded_tiles = @PADDED_TILES@;
    constexpr uint32_t token_tiles = @TOKEN_TILES@;
    constexpr uint32_t total_cbs = 23;
    constexpr uint32_t tile_bytes = 2048;
    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= token_tiles) return;

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scaler, ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    CircularBuffer inverse_hidden(cb_inverse_hidden), eps(cb_eps);
#if @OUTPUT_STAGE@ != 11
    constexpr float inverse_hidden_value = 1.0f / @HIDDEN_SIZE@;
    prepare_full_fp32_scalar(
        inverse_hidden,
        __builtin_bit_cast(uint32_t, inverse_hidden_value));
#endif
#if @OUTPUT_STAGE@ == 0 || @OUTPUT_STAGE@ == 9 || @OUTPUT_STAGE@ == 11
    // The production graph fuses the complete expression to ttnn.rms_norm.
    // Its reader publishes epsilon as a BF16 column-broadcast tile.
    generate_bcast_col_scalar(eps, uint32_t{@EPSILON_BITS@});
#else
    // Explicit diagnostic stages use binary_ng's full FP32 scalar tile.
    prepare_full_fp32_scalar(eps, uint32_t{@EPSILON_BITS@});
#endif

    const uint32_t x_addr = get_common_arg_val<uint32_t>(@X_POSITION@);
    const uint32_t weight_addr =
        get_common_arg_val<uint32_t>(@WEIGHT_POSITION@);
    const auto x_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                0, total_cbs>(), @X_POSITION@>(),
        x_addr, tile_bytes);
    const auto weight_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                1, total_cbs>(), @WEIGHT_POSITION@>(),
        weight_addr, tile_bytes);

    Noc noc0;
    CircularBuffer x(cb_x), weight(cb_weight);
    // Match reader_unary_interleaved_ln.cpp exactly: X and gamma are
    // interleaved and each padded block is published as its own CB epoch.
    for (auto block : generic::blocks(hidden_tiles, uint32_t{4})) {
        layernorm_dataflow_utils::read_block_to_cb(
            noc0, x, x_tensor, tile_bytes,
            node * hidden_tiles + block.start(), block);
        layernorm_dataflow_utils::read_block_to_cb(
            noc0, weight, weight_tensor, tile_bytes,
            block.start(), block);
    }

}
