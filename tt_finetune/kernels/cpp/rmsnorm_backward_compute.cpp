// Exact StableHLO RMSNorm input derivative with explicit FP32/BF16 edges.

#define FP32_DEST_ACC_EN 1

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

namespace {

template <bool ToFp32>
FORCE_INLINE void typecast_tiles(
    uint32_t input_id, uint32_t output_id, uint32_t tiles) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(tiles);
    output.reserve_back(tiles);
    init_sfpu(input_id, output_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        copy_tile_init(input_id);
        copy_tile(input_id, tile, 0);
        if constexpr (ToFp32) {
            typecast_tile_init<
                (uint32_t)DataFormat::Float16_b,
                (uint32_t)DataFormat::Float32>();
            typecast_tile<
                (uint32_t)DataFormat::Float16_b,
                (uint32_t)DataFormat::Float32>(0);
        } else {
            typecast_tile_init<
                (uint32_t)DataFormat::Float32,
                (uint32_t)DataFormat::Float16_b>();
            typecast_tile<
                (uint32_t)DataFormat::Float32,
                (uint32_t)DataFormat::Float16_b>(0);
        }
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, tile);
        tile_regs_release();
    }
    output.push_back(tiles);
}

FORCE_INLINE void copy_tiles_to_cb(
    uint32_t input_id, uint32_t output_id, uint32_t tiles) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(tiles);
    output.reserve_back(tiles);
    unary_op_init_common(input_id, output_id);
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

template <uint32_t Op, bool NarrowToBf16 = false>
FORCE_INLINE void binary_tiles(
    uint32_t left_id, uint32_t right_id, uint32_t output_id,
    uint32_t tiles, bool pop_inputs = false) {
    CircularBuffer left(left_id), right(right_id), output(output_id);
    left.wait_front(tiles);
    right.wait_front(tiles);
    output.reserve_back(tiles);
    unary_op_init_common(left_id, output_id);
    if constexpr (Op == 0) mul_binary_tile_init();
    else add_binary_tile_init();
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(right_id, left_id);
        copy_tile(left_id, tile, 0);
        copy_tile_to_dst_init_short_with_dt(left_id, right_id);
        copy_tile(right_id, tile, 1);
        if constexpr (Op == 0) mul_binary_tile(0, 1, 0);
        else add_binary_tile(0, 1, 0);
        if constexpr (NarrowToBf16) {
            typecast_tile_init<
                (uint32_t)DataFormat::Float32,
                (uint32_t)DataFormat::Float16_b>();
            typecast_tile<
                (uint32_t)DataFormat::Float32,
                (uint32_t)DataFormat::Float16_b>(0);
        }
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, tile);
        tile_regs_release();
    }
    output.push_back(tiles);
    if (pop_inputs) {
        left.pop_front(tiles);
        right.pop_front(tiles);
    }
}

FORCE_INLINE void scalar_multiply(
    uint32_t input_id, uint32_t output_id, uint32_t tiles,
    uint32_t scalar_bits, bool pop_input = false) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(tiles);
    output.reserve_back(tiles);
    init_sfpu(input_id, output_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        copy_tile_init(input_id);
        copy_tile(input_id, tile, 0);
        binop_with_scalar_tile_init();
        { volatile int32_t scalar = scalar_bits; mul_unary_tile(0, scalar); }
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, tile);
        tile_regs_release();
    }
    output.push_back(tiles);
    if (pop_input) input.pop_front(tiles);
}

FORCE_INLINE void divide_by_scalar(
    uint32_t input_id, uint32_t output_id, uint32_t tiles,
    float scalar, bool pop_input = false) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(tiles);
    output.reserve_back(tiles);
    init_sfpu(input_id, output_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        copy_tile_init(input_id);
        copy_tile(input_id, tile, 0);
        fill_tile_init();
        fill_tile(1, scalar);
        div_binary_tile_init();
        div_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, tile);
        tile_regs_release();
    }
    output.push_back(tiles);
    if (pop_input) input.pop_front(tiles);
}

FORCE_INLINE void derivative_times_scaled(
    uint32_t x_id, uint32_t scaled_id, uint32_t output_id,
    uint32_t tiles, uint32_t two_bits) {
    CircularBuffer x(x_id), scaled(scaled_id), output(output_id);
    x.wait_front(tiles);
    scaled.wait_front(tiles);
    output.reserve_back(tiles);
    unary_op_init_common(x_id, output_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        copy_tile_init(x_id);
        copy_tile(x_id, tile, 0);
        binop_with_scalar_tile_init();
        { volatile int32_t scalar = two_bits; mul_unary_tile(0, scalar); }
        copy_tile_to_dst_init_short_with_dt(x_id, scaled_id);
        copy_tile(scaled_id, tile, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, tile);
        tile_regs_release();
    }
    output.push_back(tiles);
}

FORCE_INLINE void cube_tiles(
    uint32_t input_id, uint32_t output_id, uint32_t tiles) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(tiles);
    output.reserve_back(tiles);
    init_sfpu(input_id, output_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        copy_tile_init(input_id);
        copy_tile(input_id, tile, 0);
        power_iterative_tile_init();
        power_iterative_tile(0, 3);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, tile);
        tile_regs_release();
    }
    output.push_back(tiles);
}

FORCE_INLINE void multiply_col_broadcast(
    uint32_t left_id, uint32_t column_id, uint32_t output_id,
    uint32_t tiles) {
    CircularBuffer left(left_id), column(column_id), output(output_id);
    left.wait_front(tiles);
    column.wait_front(1);
    output.reserve_back(tiles);
    reconfig_data_format(left_id, column_id);
    pack_reconfig_data_format(output_id);
    mul_bcast_cols_init_short(left_id, column_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(left_id, column_id, tile, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, output_id, tile);
        tile_regs_release();
    }
    output.push_back(tiles);
}

FORCE_INLINE void sfpu_multiply_col_broadcast(
    uint32_t left_id, uint32_t column_id, uint32_t temporary_id,
    uint32_t output_id, uint32_t tiles) {
    CircularBuffer left(left_id), column(column_id), temporary(temporary_id);
    column.wait_front(1);
    temporary.reserve_back(1);
    unary_op_init_common(column_id, temporary_id);
    unary_bcast_init<BroadcastType::COL>(column_id, temporary_id);
    tile_regs_acquire();
    unary_bcast<BroadcastType::COL>(column_id, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(temporary_id)));
    pack_tile<true>(0, temporary_id, 0);
    tile_regs_release();
    temporary.push_back(1);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        binary_tiles<0>(left_id, temporary_id, output_id, 1);
        left.pop_front(1);
    }
    temporary.pop_front(1);
    column.pop_front(1);
}

FORCE_INLINE void multiply_row_broadcast(
    uint32_t left_id, uint32_t row_id, uint32_t temporary_id,
    uint32_t output_id,
    uint32_t tiles) {
    CircularBuffer row(row_id), temporary(temporary_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        row.wait_front(1);
        temporary.reserve_back(1);
        unary_op_init_common(row_id, temporary_id);
        unary_bcast_init<BroadcastType::ROW>(row_id, temporary_id);
        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(row_id, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(temporary_id)));
        pack_tile<true>(0, temporary_id, 0);
        tile_regs_release();
        temporary.push_back(1);
        binary_tiles<0, true>(
            left_id, temporary_id, output_id, 1, true);
        row.pop_front(1);
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dy = get_compile_time_arg_val(1);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(2);
    constexpr uint32_t cb_rstd = get_compile_time_arg_val(3);
    constexpr uint32_t cb_weighted = get_compile_time_arg_val(4);
    constexpr uint32_t cb_x_fp32 = get_compile_time_arg_val(5);
    constexpr uint32_t cb_weighted_fp32 = get_compile_time_arg_val(6);
    constexpr uint32_t cb_product = get_compile_time_arg_val(7);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(8);
    constexpr uint32_t cb_correlation = get_compile_time_arg_val(9);
    constexpr uint32_t cb_direct = get_compile_time_arg_val(10);
    constexpr uint32_t cb_coefficient = get_compile_time_arg_val(11);
    constexpr uint32_t cb_rstd_cube = get_compile_time_arg_val(12);
    constexpr uint32_t cb_scaled = get_compile_time_arg_val(13);
    constexpr uint32_t cb_derivative = get_compile_time_arg_val(14);
    constexpr uint32_t cb_correction = get_compile_time_arg_val(15);
    constexpr uint32_t cb_sum = get_compile_time_arg_val(16);
    constexpr uint32_t cb_broadcast = get_compile_time_arg_val(17);
    constexpr uint32_t cb_output = get_compile_time_arg_val(18);
    constexpr uint32_t cb_row_broadcast = get_compile_time_arg_val(19);
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@;
    constexpr uint32_t token_tiles = @TOKEN_TILES@;
    constexpr uint32_t neg_half_bits = @NEG_HALF_BITS@;
    constexpr uint32_t inverse_hidden_bits = @INVERSE_HIDDEN_BITS@;
    constexpr float hidden_scalar = @HIDDEN_FLOAT@;
    constexpr uint32_t two_bits = @TWO_BITS@;
    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= token_tiles) return;

    compute_kernel_hw_startup(cb_x, cb_scaler, cb_output);

    // weighted = BF16(dY * gamma), then both operands cross the explicit
    // StableHLO BF16->FP32 edge before the correlation product.
    multiply_row_broadcast(
        cb_dy, cb_weight, cb_row_broadcast, cb_weighted, hidden_tiles);
#if @OUTPUT_STAGE@ == 0
    copy_tiles_to_cb(cb_weighted, cb_output, hidden_tiles);
    return;
#endif
    typecast_tiles<true>(cb_x, cb_x_fp32, hidden_tiles);
    typecast_tiles<true>(cb_weighted, cb_weighted_fp32, hidden_tiles);
    binary_tiles<0>(
        cb_weighted_fp32, cb_x_fp32, cb_product, hidden_tiles);

    compute_kernel_lib::reduce<
        PoolType::SUM, ReduceDim::REDUCE_ROW,
        cb_product, cb_scaler, cb_correlation,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT,
        ReduceFp32Mode::Fast>(
            compute_kernel_lib::ReduceInputBlockShape::row(hidden_tiles));

    // Preserve every FP32 operation boundary from AOTAutograd.
    scalar_multiply(
        cb_correlation, cb_coefficient, 1, neg_half_bits);
    CircularBuffer correlation(cb_correlation);
    correlation.pop_front(1);
    cube_tiles(cb_rstd, cb_rstd_cube, hidden_tiles);
    sfpu_multiply_col_broadcast(
        cb_rstd_cube, cb_coefficient, cb_broadcast, cb_scaled,
        hidden_tiles);
    divide_by_scalar(
        cb_scaled, cb_rstd_cube, hidden_tiles, hidden_scalar, true);
    CircularBuffer scaled(cb_scaled);
    // rstd is passed in an explicitly expanded tiled form. Narrow [T, 1]
    // tensors use a compact row-vector tile representation at the PJRT ABI;
    // treating that representation as a column broadcast silently retains
    // only token row zero. The eventual layer owner computes and consumes
    // this expanded field entirely in L1, so this is not a production DRAM
    // expansion.
    binary_tiles<0>(
        cb_weighted_fp32, cb_rstd, cb_direct, hidden_tiles, true);
    derivative_times_scaled(
        cb_x_fp32, cb_rstd_cube, cb_correction, hidden_tiles, two_bits);
#if @OUTPUT_STAGE@ == 1
    typecast_tiles<false>(cb_direct, cb_output, hidden_tiles);
#elif @OUTPUT_STAGE@ == 2
    typecast_tiles<false>(cb_correction, cb_output, hidden_tiles);
#elif @OUTPUT_STAGE@ == 3
    copy_tiles_to_cb(cb_rstd_cube, cb_output, hidden_tiles);
#elif @OUTPUT_STAGE@ == 4
    copy_tiles_to_cb(cb_correction, cb_output, hidden_tiles);
#else
    binary_tiles<1, true>(
        cb_direct, cb_correction, cb_output, hidden_tiles, true);
#endif
}
