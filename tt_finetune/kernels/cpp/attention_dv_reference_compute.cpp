// Reference-compatible FP32 attention dV tile program.

#define FP32_DEST_ACC_EN @FP32_DEST@

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/dataflow/circular_buffer.h"

namespace {

FORCE_INLINE void transpose_probability_column(
    uint32_t input_id, uint32_t output_id, uint32_t tiles) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(tiles);
    output.reserve_back(tiles);
    compute_kernel_hw_startup(input_id, output_id);
    transpose_init(input_id);
    tile_regs_acquire();
    for (uint32_t tile = 0; tile < tiles; ++tile)
        transpose_tile(input_id, tile, tile);
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(output_id)));
    for (uint32_t tile = 0; tile < tiles; ++tile)
        pack_tile<true>(tile, output_id, tile);
    tile_regs_release();
    output.push_back(tiles);
    input.pop_front(tiles);
}

FORCE_INLINE void matmul_row(
    uint32_t lhs_id, uint32_t rhs_id, uint32_t output_id,
    uint32_t partial_id, uint32_t n_tiles, uint32_t k_tiles,
    uint32_t phase_tiles, uint32_t out_subblock_tiles,
    uint32_t m_tiles) {
    CircularBuffer lhs(lhs_id), rhs(rhs_id), output(output_id), partial(partial_id);
    lhs.wait_front(m_tiles * k_tiles);
    rhs.wait_front(k_tiles * n_tiles);
    // TTNN initializes BMM against its FP32 mm-partials CB and only
    // reconfigures the packer to the final BF16 output at last_out.
    compute_kernel_hw_startup<SrcOrder::Reverse>(lhs_id, rhs_id, partial_id);
    for (uint32_t n_start = 0; n_start < n_tiles; n_start += out_subblock_tiles) {
      const uint32_t phases = k_tiles / phase_tiles;
      matmul_block_init(
          lhs_id, rhs_id, false, out_subblock_tiles, m_tiles, phase_tiles);
      for (uint32_t phase = 0; phase < phases; ++phase) {
        tile_regs_acquire();
        if (phase > 0) {
            copy_tile_to_dst_init_short_with_dt(rhs_id, partial_id);
            partial.wait_front(m_tiles * out_subblock_tiles);
            copy_block_matmul_partials(
                partial_id, 0, 0, m_tiles * out_subblock_tiles);
            partial.pop_front(m_tiles * out_subblock_tiles);
            reconfig_data_format_srca(partial_id, rhs_id);
        }
        matmul_block_init(
            lhs_id, rhs_id, false, out_subblock_tiles, m_tiles, phase_tiles);
        for (uint32_t inner = 0; inner < phase_tiles; ++inner) {
            const uint32_t k = phase * phase_tiles + inner;
            matmul_block(
                lhs_id, rhs_id, k, k * n_tiles + n_start, 0,
                false, out_subblock_tiles, m_tiles, phase_tiles);
        }
        tile_regs_commit();
        const bool last = phase + 1 == phases;
        if (last) output.reserve_back(m_tiles * out_subblock_tiles);
        else partial.reserve_back(m_tiles * out_subblock_tiles);
        tile_regs_wait();
        const uint32_t target = last ? output_id : partial_id;
        PACK((pack_reconfig_data_format(target)));
        pack_tile_block(0, target, m_tiles * out_subblock_tiles);
        tile_regs_release();
        if (last) output.push_back(m_tiles * out_subblock_tiles);
        else partial.push_back(m_tiles * out_subblock_tiles);
      }
    }
    lhs.pop_front(m_tiles * k_tiles);
    rhs.pop_front(k_tiles * n_tiles);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_p_column = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dout = get_compile_time_arg_val(1);
    constexpr uint32_t cb_pt_row = get_compile_time_arg_val(2);
    constexpr uint32_t cb_dv = get_compile_time_arg_val(3);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(4);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t dim_tiles = @DIM_TILES@;
    constexpr uint32_t query_heads = @QUERY_HEADS@;
    constexpr uint32_t rows_per_worker = @ROWS_PER_WORKER@;
    constexpr uint32_t jobs = query_heads * (seq_tiles / rows_per_worker);
    constexpr uint32_t phase_tiles = @PHASE_TILES@;
    constexpr uint32_t out_subblock_tiles = @OUT_SUBBLOCK_TILES@;
    static_assert(seq_tiles % phase_tiles == 0);

    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= jobs) return;
    transpose_probability_column(
        cb_p_column, cb_pt_row, rows_per_worker * seq_tiles);
    matmul_row(
        cb_pt_row, cb_dout, cb_dv, cb_partial,
        dim_tiles, seq_tiles, phase_tiles, out_subblock_tiles,
        rows_per_worker);
    CircularBuffer pt(cb_pt_row);
    pt.pop_front(rows_per_worker * seq_tiles);
}
