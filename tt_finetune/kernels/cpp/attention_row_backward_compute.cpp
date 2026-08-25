// Exact-schedule query-row attention backward: dV, dP, scaled dS and dQ.

#if @FP32_DEST_ACC_EN@
#define FP32_DEST_ACC_EN 1
#endif

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"

namespace {

FORCE_INLINE void transpose_tiles(
    uint32_t input_id, uint32_t output_id,
    uint32_t rows, uint32_t cols) {
    CircularBuffer input(input_id), output(output_id);
    const uint32_t tiles = rows * cols;
    input.wait_front(tiles);
    output.reserve_back(tiles);
    compute_kernel_hw_startup(input_id, output_id);
    transpose_init(input_id);
    for (uint32_t start = 0; start < tiles; start += 4) {
        const uint32_t chunk = (tiles - start) < 4 ? (tiles - start) : 4;
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < chunk; ++tile)
            transpose_tile(input_id, start + tile, tile);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        for (uint32_t tile = 0; tile < chunk; ++tile) {
            const uint32_t source = start + tile;
            const uint32_t row = source / cols;
            const uint32_t col = source - row * cols;
            pack_tile<true>(tile, output_id, col * rows + row);
        }
        tile_regs_release();
    }
    output.push_back(tiles);
}

FORCE_INLINE void transpose_linear(
    uint32_t input_id, uint32_t output_id, uint32_t tiles) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(tiles);
    output.reserve_back(tiles);
    compute_kernel_hw_startup(input_id, output_id);
    transpose_init(input_id);
    for (uint32_t start = 0; start < tiles; start += 4) {
        const uint32_t chunk = (tiles - start) < 4 ? (tiles - start) : 4;
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < chunk; ++tile)
            transpose_tile(input_id, start + tile, tile);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        for (uint32_t tile = 0; tile < chunk; ++tile)
            pack_tile<true>(tile, output_id, start + tile);
        tile_regs_release();
    }
    output.push_back(tiles);
}

FORCE_INLINE void matmul_row(
    uint32_t lhs_id, uint32_t rhs_id, uint32_t output_id,
    uint32_t partial_id, uint32_t n_tiles, uint32_t k_tiles,
    uint32_t phase_tiles, uint32_t n_subblock_tiles) {
    CircularBuffer lhs(lhs_id), rhs(rhs_id), output(output_id), partial(partial_id);
    lhs.wait_front(k_tiles);
    rhs.wait_front(k_tiles * n_tiles);
    compute_kernel_hw_startup<SrcOrder::Reverse>(lhs_id, rhs_id, output_id);
    const uint32_t phases = k_tiles / phase_tiles;
    for (uint32_t n_start = 0; n_start < n_tiles;
         n_start += n_subblock_tiles) {
        matmul_block_init(
            lhs_id, rhs_id, false, n_subblock_tiles, 1, phase_tiles);
        for (uint32_t phase = 0; phase < phases; ++phase) {
            tile_regs_acquire();
            if (phase > 0) {
                copy_tile_to_dst_init_short_with_dt(rhs_id, partial_id);
                partial.wait_front(n_subblock_tiles);
                copy_block_matmul_partials(
                    partial_id, 0, 0, n_subblock_tiles);
                partial.pop_front(n_subblock_tiles);
                reconfig_data_format_srca(partial_id, rhs_id);
            }
            matmul_block_init(
                lhs_id, rhs_id, false,
                n_subblock_tiles, 1, phase_tiles);
            for (uint32_t inner = 0; inner < phase_tiles; ++inner) {
                const uint32_t k = phase * phase_tiles + inner;
                matmul_block(
                    lhs_id, rhs_id, k,
                    k * n_tiles + n_start, 0,
                    false, n_subblock_tiles, 1, phase_tiles);
            }
            tile_regs_commit();
            const bool last = phase + 1 == phases;
            if (last) output.reserve_back(n_subblock_tiles);
            else partial.reserve_back(n_subblock_tiles);
            tile_regs_wait();
            const uint32_t target = last ? output_id : partial_id;
            PACK((pack_reconfig_data_format(target)));
            pack_tile_block(0, target, n_subblock_tiles);
            tile_regs_release();
            if (last) output.push_back(n_subblock_tiles);
            else partial.push_back(n_subblock_tiles);
        }
    }
}

// MatmulMultiCoreReuseProgramConfig uses bmm_large_block_zm for the sharded
// dP operand.  Its observable inner traversal is output-tile-major and calls
// matmul_tiles, unlike matmul_block's K-major subblock traversal.
FORCE_INLINE void matmul_row_ttnn_reuse(
    uint32_t lhs_id, uint32_t rhs_id, uint32_t output_id,
    uint32_t n_tiles, uint32_t k_tiles) {
    CircularBuffer lhs(lhs_id), rhs(rhs_id), output(output_id);
    lhs.wait_front(k_tiles);
    rhs.wait_front(k_tiles * n_tiles);
    compute_kernel_hw_startup<SrcOrder::Reverse>(lhs_id, rhs_id, output_id);
    matmul_init(lhs_id, rhs_id);
    tile_regs_acquire();
    for (uint32_t n = 0; n < n_tiles; ++n) {
        for (uint32_t k = 0; k < k_tiles; ++k)
            matmul_tiles(lhs_id, rhs_id, k, k * n_tiles + n, n);
    }
    tile_regs_commit();
    output.reserve_back(n_tiles);
    tile_regs_wait();
    PACK((pack_reconfig_data_format(output_id)));
    for (uint32_t n = 0; n < n_tiles; ++n)
        pack_tile(n, output_id);
    tile_regs_release();
    output.push_back(n_tiles);
}

FORCE_INLINE void scale_tiles(
    uint32_t input_id, uint32_t output_id,
    uint32_t tiles, uint32_t scale_bits, bool preserve_input) {
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
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, tile);
        tile_regs_release();
    }
    output.push_back(tiles);
    if (!preserve_input) input.pop_front(tiles);
}

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

template <bool Multiply>
FORCE_INLINE void sfpu_binary_tile_to_cb(
    uint32_t left_id, uint32_t right_id, uint32_t output_id,
    uint32_t left_tile, uint32_t right_tile) {
    CircularBuffer left(left_id), right(right_id), output(output_id);
    left.wait_front(left_tile + 1);
    right.wait_front(right_tile + 1);
    output.reserve_back(1);
    unary_op_init_common(left_id, output_id);
    tile_regs_acquire();
    tile_regs_wait();
    copy_tile_to_dst_init_short_with_dt(right_id, left_id);
    copy_tile(left_id, left_tile, 0);
    copy_tile_to_dst_init_short_with_dt(left_id, right_id);
    copy_tile(right_id, right_tile, 1);
    if constexpr (Multiply) {
        mul_binary_tile(0, 1, 0);
    } else {
        sub_binary_tile(0, 1, 0);
    }
    pack_tile(0, output_id);
    tile_regs_commit();
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
    constexpr uint32_t cb_p_row = get_compile_time_arg_val(0);
    constexpr uint32_t cb_p_column = get_compile_time_arg_val(1);
    constexpr uint32_t cb_dout_row = get_compile_time_arg_val(2);
    constexpr uint32_t cb_dout_full = get_compile_time_arg_val(3);
    constexpr uint32_t cb_k = get_compile_time_arg_val(4);
    constexpr uint32_t cb_v = get_compile_time_arg_val(5);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(6);
    constexpr uint32_t cb_pt = get_compile_time_arg_val(7);
    constexpr uint32_t cb_dv = get_compile_time_arg_val(8);
    constexpr uint32_t cb_vt = get_compile_time_arg_val(9);
    constexpr uint32_t cb_dp = get_compile_time_arg_val(10);
    constexpr uint32_t cb_product = get_compile_time_arg_val(11);
    constexpr uint32_t cb_rowsum = get_compile_time_arg_val(12);
    constexpr uint32_t cb_centered = get_compile_time_arg_val(13);
    constexpr uint32_t cb_ds = get_compile_time_arg_val(14);
    constexpr uint32_t cb_ds_scaled = get_compile_time_arg_val(15);
    constexpr uint32_t cb_dq = get_compile_time_arg_val(16);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(17);
    constexpr uint32_t cb_dp_snapshot = get_compile_time_arg_val(18);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t dim_tiles = @DIM_TILES@;
    constexpr uint32_t query_heads = @QUERY_HEADS@;
    constexpr uint32_t jobs = query_heads * seq_tiles;
    constexpr uint32_t scale_bits = @SCALE_BITS@;
    constexpr uint32_t dp_phase_tiles = @DP_PHASE_TILES@;
    constexpr uint32_t dp_n_subblock_tiles = @DP_N_SUBBLOCK_TILES@;
    constexpr uint32_t diagnostic_stage = @DIAGNOSTIC_STAGE@;

    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= jobs) return;
    CircularBuffer p_row(cb_p_row), p_column(cb_p_column);
    CircularBuffer dout_row(cb_dout_row), dout_full(cb_dout_full);
    CircularBuffer k(cb_k), v(cb_v), pt(cb_pt), vt(cb_vt);
    CircularBuffer dp(cb_dp), product(cb_product), rowsum(cb_rowsum);
    CircularBuffer centered(cb_centered), ds(cb_ds);

    // dP row = dO row @ V.T.
    if constexpr (@VALUE_PRETRANSPOSED@)
        copy_tiles_preserve(cb_v, cb_vt, seq_tiles * dim_tiles);
    else
        transpose_linear(cb_v, cb_vt, seq_tiles * dim_tiles);
    matmul_row(cb_dout_row, cb_vt, cb_dp, cb_partial,
               seq_tiles, dim_tiles, dp_phase_tiles, dp_n_subblock_tiles);
    vt.pop_front(seq_tiles * dim_tiles);

    // dV row = P.T row @ dO.  The one-tile spill boundary is required by the
    // exact TT-XLA FP32 contract on both qualified model geometries.
    transpose_tiles(cb_p_column, cb_pt, 1, seq_tiles);
    matmul_row(cb_pt, cb_dout_full, cb_dv, cb_partial,
               dim_tiles, seq_tiles, 1, dim_tiles);
    pt.pop_front(seq_tiles);

    // dS row = P * (dP - sum(P*dP)); then apply scale before dQ/dK,
    // matching the optimization-level-2 TTNN lowering exactly.
    p_row.wait_front(seq_tiles);
    dp.wait_front(seq_tiles);
    for (uint32_t tile = 0; tile < seq_tiles; ++tile)
        sfpu_binary_tile_to_cb<true>(
            cb_p_row, cb_dp, cb_product, tile, tile);
    if (diagnostic_stage == 3) return;
    compute_kernel_lib::reduce<
        PoolType::SUM, ReduceDim::REDUCE_ROW,
        cb_product, cb_scaler, cb_rowsum,
        compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
            compute_kernel_lib::ReduceInputBlockShape::row(seq_tiles));
    if (diagnostic_stage == 5) return;
    rowsum.wait_front(1);
    for (uint32_t tile = 0; tile < seq_tiles; ++tile) {
        sfpu_sub_row_sum_to_cb(
            cb_dp, cb_rowsum, cb_centered, tile);
        if (diagnostic_stage != 4) {
            sfpu_binary_tile_to_cb<true>(
                cb_p_row, cb_centered, cb_ds, tile, 0);
            centered.pop_front(1);
        }
    }
    rowsum.pop_front(1);
    if (diagnostic_stage == 4) return;
    if (diagnostic_stage != 1) dp.pop_front(seq_tiles);
    p_row.pop_front(seq_tiles);
    scale_tiles(
        cb_ds, cb_ds_scaled, seq_tiles, scale_bits,
        diagnostic_stage == 2);

    // dQ row = scaled-dS row @ K, with the same one-K-tile FP32 boundary.
    matmul_row(cb_ds_scaled, cb_k, cb_dq, cb_partial,
               dim_tiles, seq_tiles, 1, dim_tiles);

    p_column.pop_front(seq_tiles);
    dout_row.pop_front(dim_tiles);
    dout_full.pop_front(seq_tiles * dim_tiles);
    k.pop_front(seq_tiles * dim_tiles);
    v.pop_front(seq_tiles * dim_tiles);
}
