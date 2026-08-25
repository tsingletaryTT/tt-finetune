// Shape-identical, row-sharded FP32 softmax-backward numerical probe.

#define FP32_DEST_ACC_EN 1

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

namespace {

FORCE_INLINE void copy_tiles_preserve(
    uint32_t input_id, uint32_t output_id, uint32_t tiles) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(tiles);
    output.reserve_back(tiles);
    compute_kernel_hw_startup(input_id, output_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        copy_tile_init(input_id);
        copy_tile(input_id, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, tile);
        tile_regs_release();
    }
    output.push_back(tiles);
}

FORCE_INLINE void fp32_to_bf16_preserve(
    uint32_t input_id, uint32_t output_id, uint32_t tiles) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(tiles);
    output.reserve_back(tiles);
    init_sfpu(input_id, output_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        copy_tile_init(input_id);
        copy_tile(input_id, tile, 0);
        typecast_tile_init<
            (uint32_t)DataFormat::Float32,
            (uint32_t)DataFormat::Float16_b>();
        typecast_tile<
            (uint32_t)DataFormat::Float32,
            (uint32_t)DataFormat::Float16_b>(0);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, tile);
        tile_regs_release();
    }
    output.push_back(tiles);
}

FORCE_INLINE void scale_bf16_tiles(
    uint32_t input_id, uint32_t output_id, uint32_t tiles,
    uint32_t scale_bits) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(tiles);
    output.reserve_back(tiles);
    init_sfpu(input_id, output_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        copy_tile_init(input_id);
        copy_tile(input_id, tile, 0);
        binop_with_scalar_tile_init();
        { volatile int32_t scalar = scale_bits; mul_unary_tile(0, scalar); }
        typecast_tile_init<
            (uint32_t)DataFormat::Float32,
            (uint32_t)DataFormat::Float16_b>();
        typecast_tile<
            (uint32_t)DataFormat::Float32,
            (uint32_t)DataFormat::Float16_b>(0);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, tile);
        tile_regs_release();
    }
    output.push_back(tiles);
    input.pop_front(tiles);
}

template <uint32_t Op>
FORCE_INLINE void sfpu_binary_tile_to_cb(
    uint32_t left_id, uint32_t right_id, uint32_t output_id,
    uint32_t left_tile, uint32_t right_tile) {
    CircularBuffer left(left_id), right(right_id), output(output_id);
    left.wait_front(left_tile + 1);
    right.wait_front(right_tile + 1);
    output.reserve_back(1);
    unary_op_init_common(left_id, output_id);
    if constexpr (Op == 0) mul_binary_tile_init();
    else sub_binary_tile_init();
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(right_id, left_id);
    copy_tile(left_id, left_tile, 0);
    copy_tile_to_dst_init_short_with_dt(left_id, right_id);
    copy_tile(right_id, right_tile, 1);
    if constexpr (Op == 0) mul_binary_tile(0, 1, 0);
    else sub_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(output_id)));
    pack_tile(0, output_id);
    tile_regs_release();
    output.push_back(1);
}

FORCE_INLINE void materialize_col_broadcast(
    uint32_t row_sum_id, uint32_t broadcast_id) {
    CircularBuffer row_sum(row_sum_id), broadcast(broadcast_id);
    row_sum.wait_front(1);
    broadcast.reserve_back(1);
    unary_op_init_common(row_sum_id, broadcast_id);
    unary_bcast_init<BroadcastType::COL>(row_sum_id, broadcast_id);
    tile_regs_acquire();
    unary_bcast<BroadcastType::COL>(row_sum_id, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(broadcast_id)));
    pack_tile<true>(0, broadcast_id, 0);
    tile_regs_release();
    broadcast.push_back(1);
}

FORCE_INLINE void fpu_sub_tile_to_cb(
    uint32_t matrix_id, uint32_t broadcast_id, uint32_t output_id,
    uint32_t matrix_tile) {
    CircularBuffer matrix(matrix_id), broadcast(broadcast_id), output(output_id);
    matrix.wait_front(matrix_tile + 1);
    broadcast.wait_front(1);
    output.reserve_back(1);
    binary_op_init_common(matrix_id, broadcast_id, output_id);
    sub_tiles_init(matrix_id, broadcast_id);
    tile_regs_acquire();
    sub_tiles(matrix_id, broadcast_id, matrix_tile, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(output_id)));
    pack_tile<true>(0, output_id, 0);
    tile_regs_release();
    output.push_back(1);
}

FORCE_INLINE void sfpu_sub_row_sum_to_cb(
    uint32_t matrix_id, uint32_t row_sum_id, uint32_t output_id,
    uint32_t matrix_tile) {
    CircularBuffer matrix(matrix_id), row_sum(row_sum_id), output(output_id);
    matrix.wait_front(matrix_tile + 1);
    row_sum.wait_front(1);
    output.reserve_back(1);
    unary_op_init_common(matrix_id, output_id);
    unary_bcast_init<BroadcastType::COL>(row_sum_id, output_id);
    sub_binary_tile_init();
    tile_regs_acquire();
    unary_bcast<BroadcastType::COL>(row_sum_id, 0, 1);
    copy_tile_to_dst_init_short_with_dt(row_sum_id, matrix_id);
    copy_tile(matrix_id, matrix_tile, 0);
    sub_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(output_id)));
    pack_tile<true>(0, output_id, 0);
    tile_regs_release();
    output.push_back(1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_p = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dp = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_dp_fp32 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_product = get_compile_time_arg_val(4);
    constexpr uint32_t cb_rowsum = get_compile_time_arg_val(5);
    constexpr uint32_t cb_centered = get_compile_time_arg_val(6);
    constexpr uint32_t cb_ds = get_compile_time_arg_val(7);
    constexpr uint32_t cb_out = get_compile_time_arg_val(8);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t heads = @HEADS@;
    constexpr uint32_t jobs = heads * seq_tiles;
    constexpr uint32_t stage = @STAGE@;
    constexpr uint32_t scale_bits = @SCALE_BITS@;
    const uint32_t node = get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= jobs) return;

    copy_tiles_preserve(cb_dp, cb_dp_fp32, seq_tiles);
    CircularBuffer dp_input(cb_dp);
    dp_input.pop_front(seq_tiles);
    for (uint32_t tile = 0; tile < seq_tiles; ++tile)
        sfpu_binary_tile_to_cb<0>(cb_p, cb_dp_fp32, cb_product, tile, tile);
    if constexpr (stage == 0) {
        fp32_to_bf16_preserve(cb_product, cb_out, seq_tiles);
        return;
    }

    compute_kernel_lib::reduce<
        PoolType::SUM, ReduceDim::REDUCE_ROW,
        cb_product, cb_scaler, cb_rowsum,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT,
        ReduceFp32Mode::Fast>(
            compute_kernel_lib::ReduceInputBlockShape::row(seq_tiles));
    CircularBuffer rowsum(cb_rowsum);
    rowsum.wait_front(1);
    if constexpr (stage == 3) {
        materialize_col_broadcast(cb_rowsum, cb_product);
        for (uint32_t tile = 0; tile < seq_tiles; ++tile)
            copy_tiles_preserve(cb_product, cb_out, 1);
        return;
    }
    if constexpr (!@SUB_SFPU@)
        materialize_col_broadcast(cb_rowsum, cb_product);
    for (uint32_t tile = 0; tile < seq_tiles; ++tile) {
        if constexpr (@SUB_SFPU@)
            sfpu_sub_row_sum_to_cb(
                cb_dp_fp32, cb_rowsum, cb_centered, tile);
        else
            fpu_sub_tile_to_cb(
                cb_dp_fp32, cb_product, cb_centered, tile);
    }
    if constexpr (stage == 1) {
        fp32_to_bf16_preserve(cb_centered, cb_out, seq_tiles);
        return;
    }

    for (uint32_t tile = 0; tile < seq_tiles; ++tile)
        sfpu_binary_tile_to_cb<0>(cb_p, cb_centered, cb_ds, tile, tile);
    if constexpr (stage == 4) {
        fp32_to_bf16_preserve(cb_ds, cb_dp, seq_tiles);
        scale_bf16_tiles(cb_dp, cb_out, seq_tiles, scale_bits);
    } else {
        fp32_to_bf16_preserve(cb_ds, cb_out, seq_tiles);
    }
}
