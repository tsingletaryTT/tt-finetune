// Exact inverse RoPE with the framework's two BF16 product boundaries.

#define FP32_DEST_ACC_EN 1

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"

namespace {

template <uint32_t Op>
FORCE_INLINE void bf16_binary(
    uint32_t left_id, uint32_t right_id, uint32_t output_id) {
    CircularBuffer left(left_id), right(right_id), output(output_id);
    left.wait_front(1);
    right.wait_front(1);
    output.reserve_back(1);
    unary_op_init_common(left_id, output_id);
    if constexpr (Op == 0) mul_binary_tile_init();
    else if constexpr (Op == 1) sub_binary_tile_init();
    else add_binary_tile_init();
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(right_id, left_id);
    copy_tile(left_id, 0, 0);
    copy_tile_to_dst_init_short_with_dt(left_id, right_id);
    copy_tile(right_id, 0, 1);
    if constexpr (Op == 0) mul_binary_tile(0, 1, 0);
    else if constexpr (Op == 1) sub_binary_tile(0, 1, 0);
    else add_binary_tile(0, 1, 0);
    // Standalone TTNN BF16 binary operations round at this boundary even
    // though the surrounding resident program keeps FP32 destination mode
    // enabled for matmuls.
    typecast_tile_init<
        (uint32_t)DataFormat::Float32,
        (uint32_t)DataFormat::Float16_b>();
    typecast_tile<
        (uint32_t)DataFormat::Float32,
        (uint32_t)DataFormat::Float16_b>(0);
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(output_id)));
    pack_tile(0, output_id);
    tile_regs_release();
    output.push_back(1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_gradient = get_compile_time_arg_val(0);
    constexpr uint32_t cb_cosine = get_compile_time_arg_val(1);
    constexpr uint32_t cb_peer_gradient = get_compile_time_arg_val(2);
    constexpr uint32_t cb_peer_sine = get_compile_time_arg_val(3);
    constexpr uint32_t cb_cosine_product = get_compile_time_arg_val(4);
    constexpr uint32_t cb_sine_product = get_compile_time_arg_val(5);
    constexpr uint32_t cb_output = get_compile_time_arg_val(6);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t dim_tiles = @DIM_TILES@;
    constexpr uint32_t query_heads = @QUERY_HEADS@;
    constexpr uint32_t kv_heads = @KV_HEADS@;
    constexpr uint32_t jobs = (query_heads + kv_heads) * seq_tiles * dim_tiles;
    constexpr uint32_t jobs_per_node = @JOBS_PER_NODE@;
    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();

    for (uint32_t local = 0; local < jobs_per_node; ++local) {
        const uint32_t job = node * jobs_per_node + local;
        if (job >= jobs) continue;
        const uint32_t dim = job % dim_tiles;
        bf16_binary<0>(cb_gradient, cb_cosine, cb_cosine_product);
        bf16_binary<0>(cb_peer_gradient, cb_peer_sine, cb_sine_product);
        if (dim == 0)
            bf16_binary<2>(cb_cosine_product, cb_sine_product, cb_output);
        else
            bf16_binary<1>(cb_cosine_product, cb_sine_product, cb_output);
        CircularBuffer gradient(cb_gradient), cosine(cb_cosine);
        CircularBuffer peer_gradient(cb_peer_gradient), peer_sine(cb_peer_sine);
        CircularBuffer cosine_product(cb_cosine_product);
        CircularBuffer sine_product(cb_sine_product);
        gradient.pop_front(1);
        cosine.pop_front(1);
        peer_gradient.pop_front(1);
        peer_sine.pop_front(1);
        cosine_product.pop_front(1);
        sine_product.pop_front(1);
    }
}
