// Exact-contract FP32 attention-core backward for causal GQA.
// Python substitutes the qualified static geometry.  The worker retains all
// dP/dS and transpose intermediates in L1 and publishes only dQ/dK/dV.

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
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

FORCE_INLINE void transpose_matrix(
    uint32_t input_id, uint32_t output_id,
    uint32_t rows, uint32_t cols) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(rows * cols);
    output.reserve_back(rows * cols);
    compute_kernel_hw_startup(input_id, output_id);
    transpose_init(input_id);
    const uint32_t tiles = rows * cols;
    for (uint32_t tile_start = 0; tile_start < tiles; tile_start += 4) {
        constexpr uint32_t max_chunk = 4;
        const uint32_t remaining = tiles - tile_start;
        const uint32_t chunk = remaining < max_chunk ? remaining : max_chunk;
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < chunk; ++tile) {
            transpose_tile(input_id, tile_start + tile, tile);
        }
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        for (uint32_t tile = 0; tile < chunk; ++tile) {
            const uint32_t source = tile_start + tile;
            const uint32_t row = source / cols;
            const uint32_t col = source % cols;
            pack_tile<true>(tile, output_id, col * rows + row);
        }
        tile_regs_release();
    }
    output.push_back(rows * cols);
}

FORCE_INLINE void matmul_matrix(
    uint32_t lhs_id, uint32_t rhs_id, uint32_t output_id,
    uint32_t partial_id,
    uint32_t m_tiles, uint32_t n_tiles, uint32_t k_tiles,
    uint32_t phase_tiles) {
    CircularBuffer lhs(lhs_id), rhs(rhs_id), output(output_id), partial(partial_id);
    lhs.wait_front(m_tiles * k_tiles);
    rhs.wait_front(k_tiles * n_tiles);
    compute_kernel_hw_startup<SrcOrder::Reverse>(lhs_id, rhs_id, output_id);
    const uint32_t phases = k_tiles / phase_tiles;
    // TT-NN's qualified FP32 program holds the complete N-wide output
    // subblock in destination registers. Computing one N tile at a time is
    // algebraically equivalent but changes the observable accumulation
    // schedule.
    matmul_block_init(lhs_id, rhs_id, false, n_tiles, 1, phase_tiles);
    for (uint32_t m = 0; m < m_tiles; ++m) {
        for (uint32_t phase = 0; phase < phases; ++phase) {
            tile_regs_acquire();
            if (phase > 0) {
                copy_tile_to_dst_init_short_with_dt(rhs_id, partial_id);
                partial.wait_front(n_tiles);
                copy_block_matmul_partials(partial_id, 0, 0, n_tiles);
                partial.pop_front(n_tiles);
                reconfig_data_format_srca(partial_id, rhs_id);
            }
            matmul_block_init(
                lhs_id, rhs_id, false, n_tiles, 1, phase_tiles);
            for (uint32_t inner = 0; inner < phase_tiles; ++inner) {
                const uint32_t k = phase * phase_tiles + inner;
                matmul_block(
                    lhs_id, rhs_id,
                    m * k_tiles + k,
                    k * n_tiles,
                    0, false, n_tiles, 1, phase_tiles);
            }
            tile_regs_commit();
            const bool last = phase + 1 == phases;
            if (last) output.reserve_back(n_tiles);
            else partial.reserve_back(n_tiles);
            tile_regs_wait();
            const uint32_t target = last ? output_id : partial_id;
            PACK((pack_reconfig_data_format(target)));
            pack_tile_block(0, target, n_tiles);
            tile_regs_release();
            if (last) output.push_back(n_tiles);
            else partial.push_back(n_tiles);
        }
    }
}

FORCE_INLINE void scale_tiles(
    uint32_t input_id, uint32_t output_id,
    uint32_t tiles, uint32_t scale_bits) {
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
    input.pop_front(tiles);
}

FORCE_INLINE void copy_tiles(
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
    input.pop_front(tiles);
}

FORCE_INLINE void add_tiles_block(
    uint32_t left_id, uint32_t right_id,
    uint32_t output_id, uint32_t tiles) {
    CircularBuffer left(left_id), right(right_id), output(output_id);
    left.wait_front(tiles);
    right.wait_front(tiles);
    output.reserve_back(tiles);
    binary_op_init_common(left_id, right_id, output_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        add_tiles_init(left_id, right_id);
        add_tiles(left_id, right_id, tile, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, tile);
        tile_regs_release();
    }
    output.push_back(tiles);
    left.pop_front(tiles);
    right.pop_front(tiles);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_p = get_compile_time_arg_val(0);
    constexpr uint32_t cb_do = get_compile_time_arg_val(1);
    constexpr uint32_t cb_q = get_compile_time_arg_val(2);
    constexpr uint32_t cb_k = get_compile_time_arg_val(3);
    constexpr uint32_t cb_v = get_compile_time_arg_val(4);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(5);
    constexpr uint32_t cb_pt = get_compile_time_arg_val(6);
    constexpr uint32_t cb_vt = get_compile_time_arg_val(7);
    constexpr uint32_t cb_dv = get_compile_time_arg_val(8);
    constexpr uint32_t cb_dp = get_compile_time_arg_val(9);
    constexpr uint32_t cb_product = get_compile_time_arg_val(10);
    constexpr uint32_t cb_rowsum = get_compile_time_arg_val(11);
    constexpr uint32_t cb_ds = get_compile_time_arg_val(12);
    constexpr uint32_t cb_dq = get_compile_time_arg_val(13);
    constexpr uint32_t cb_dst = get_compile_time_arg_val(14);
    constexpr uint32_t cb_dk = get_compile_time_arg_val(15);
    constexpr uint32_t cb_dk_a = get_compile_time_arg_val(16);
    constexpr uint32_t cb_dk_b = get_compile_time_arg_val(17);
    constexpr uint32_t cb_dv_a = get_compile_time_arg_val(18);
    constexpr uint32_t cb_dv_b = get_compile_time_arg_val(19);
    constexpr uint32_t cb_dk_final = get_compile_time_arg_val(20);
    constexpr uint32_t cb_dv_final = get_compile_time_arg_val(21);
    constexpr uint32_t cb_vector_scratch = get_compile_time_arg_val(22);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t dim_tiles = @DIM_TILES@;
    constexpr uint32_t query_heads = @QUERY_HEADS@;
    constexpr uint32_t kv_heads = @KV_HEADS@;
    constexpr uint32_t group = query_heads / kv_heads;
    constexpr uint32_t score_tiles = seq_tiles * seq_tiles;
    constexpr uint32_t vector_tiles = seq_tiles * dim_tiles;
    constexpr uint32_t scale_bits = @SCALE_BITS@;
    constexpr uint32_t long_k_phase_tiles = @LONG_K_PHASE_TILES@;
    static_assert(group % 2 == 1, "qualified GQA groups are odd");

    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= kv_heads) return;
    CircularBuffer p(cb_p), dout(cb_do), q(cb_q), k(cb_k), v(cb_v);
    CircularBuffer dp(cb_dp), product(cb_product), rowsum(cb_rowsum);
    CircularBuffer ds(cb_ds), dst(cb_dst), pt(cb_pt), vt(cb_vt);
    DataflowBuffer dfb_p(cb_p), dfb_dp(cb_dp), dfb_product(cb_product);
    DataflowBuffer dfb_rowsum(cb_rowsum), dfb_ds(cb_ds), dfb_dst(cb_dst);

    for (uint32_t member = 0; member < group; ++member) {
        // dV_partial = P.T @ dO.
        transpose_matrix(cb_p, cb_pt, seq_tiles, seq_tiles);
        matmul_matrix(cb_pt, cb_do, cb_dv, cb_vector_scratch,
                      seq_tiles, dim_tiles, seq_tiles,
                      long_k_phase_tiles);
        pt.pop_front(score_tiles);
        if (member == 0) copy_tiles(cb_dv, cb_dv_a, vector_tiles);
        else if (member % 2 == 1)
            add_tiles_block(cb_dv_a, cb_dv, cb_dv_b, vector_tiles);
        else
            add_tiles_block(cb_dv_b, cb_dv, cb_dv_a, vector_tiles);

        // dP = dO @ V.T.
        transpose_matrix(cb_v, cb_vt, seq_tiles, dim_tiles);
        matmul_matrix(cb_do, cb_vt, cb_dp, cb_vector_scratch,
                      seq_tiles, seq_tiles, dim_tiles, 2);
        vt.pop_front(vector_tiles);

        // dS = P * (dP - reduce_sum(P*dP, rows)).  This is the exact FP32
        // AOTAutograd order visible in StableHLO, not a recomputed softmax.
        p.wait_front(score_tiles);
        dp.wait_front(score_tiles);
        for (uint32_t tile = 0; tile < score_tiles; ++tile)
            mul_tiles_to_cb(dfb_p, dfb_dp, dfb_product, tile, tile, 0, 0);
        for (uint32_t row = 0; row < seq_tiles; ++row) {
            compute_kernel_lib::reduce<
                PoolType::SUM, ReduceDim::REDUCE_ROW,
                cb_product, cb_scaler, cb_rowsum,
                compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
                    compute_kernel_lib::ReduceInputBlockShape::row(seq_tiles));
        }
        for (uint32_t row = 0; row < seq_tiles; ++row) {
            dfb_rowsum.wait_front(1);
            for (uint32_t col = 0; col < seq_tiles; ++col) {
                const uint32_t tile = row * seq_tiles + col;
                sub_tiles_bcast_cols_to_cb(
                    dfb_dp, dfb_rowsum, dfb_dst,
                    tile, 0, 0, 0);
                mul_tiles_to_cb(
                    dfb_p, dfb_dst, dfb_ds,
                    tile, 0, 0, 1);
            }
            dfb_rowsum.pop_front(1);
        }
        dp.pop_front(score_tiles);

        // dQ = (dS @ K) * scale.
        matmul_matrix(cb_ds, cb_k, cb_vector_scratch, cb_dv,
                      seq_tiles, dim_tiles, seq_tiles,
                      long_k_phase_tiles);
        scale_tiles(cb_vector_scratch, cb_dq, vector_tiles, scale_bits);

        // dK_partial = (dS.T @ Q) * scale.
        transpose_matrix(cb_ds, cb_dst, seq_tiles, seq_tiles);
        matmul_matrix(cb_dst, cb_q, cb_vector_scratch, cb_dv,
                      seq_tiles, dim_tiles, seq_tiles,
                      long_k_phase_tiles);
        dst.pop_front(score_tiles);
        scale_tiles(cb_vector_scratch, cb_dk, vector_tiles, scale_bits);
        if (member == 0) copy_tiles(cb_dk, cb_dk_a, vector_tiles);
        else if (member % 2 == 1)
            add_tiles_block(cb_dk_a, cb_dk, cb_dk_b, vector_tiles);
        else
            add_tiles_block(cb_dk_b, cb_dk, cb_dk_a, vector_tiles);

        ds.pop_front(score_tiles);
        p.pop_front(score_tiles);
        dout.pop_front(vector_tiles);
        q.pop_front(vector_tiles);
    }
    k.pop_front(vector_tiles);
    v.pop_front(vector_tiles);
    copy_tiles(cb_dk_a, cb_dk_final, vector_tiles);
    copy_tiles(cb_dv_a, cb_dv_final, vector_tiles);
}
