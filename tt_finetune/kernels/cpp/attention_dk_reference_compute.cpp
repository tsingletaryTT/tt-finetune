// Exact-schedule Q.T @ scaled-dS followed by tile-local output transpose.

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
FORCE_INLINE void transpose_linear(
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
}

FORCE_INLINE void matmul_one_k_phases(
    uint32_t lhs_id, uint32_t rhs_id, uint32_t output_id,
    uint32_t partial_id, uint32_t n_tiles, uint32_t k_tiles) {
    CircularBuffer lhs(lhs_id), rhs(rhs_id), output(output_id), partial(partial_id);
    lhs.wait_front(k_tiles);
    rhs.wait_front(k_tiles * n_tiles);
    compute_kernel_hw_startup<SrcOrder::Reverse>(lhs_id, rhs_id, output_id);
    matmul_block_init(lhs_id, rhs_id, false, n_tiles, 1, 1);
    for (uint32_t phase = 0; phase < k_tiles; ++phase) {
        tile_regs_acquire();
        if (phase > 0) {
            copy_tile_to_dst_init_short_with_dt(rhs_id, partial_id);
            partial.wait_front(n_tiles);
            copy_block_matmul_partials(partial_id, 0, 0, n_tiles);
            partial.pop_front(n_tiles);
            reconfig_data_format_srca(partial_id, rhs_id);
        }
        matmul_block_init(lhs_id, rhs_id, false, n_tiles, 1, 1);
        matmul_block(
            lhs_id, rhs_id, phase, phase * n_tiles, 0,
            false, n_tiles, 1, 1);
        tile_regs_commit();
        const bool last = phase + 1 == k_tiles;
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
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_q_column = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ds = get_compile_time_arg_val(1);
    constexpr uint32_t cb_qt = get_compile_time_arg_val(2);
    constexpr uint32_t cb_dk_raw = get_compile_time_arg_val(3);
    constexpr uint32_t cb_dk = get_compile_time_arg_val(4);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(5);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t dim_tiles = @DIM_TILES@;
    constexpr uint32_t query_heads = @QUERY_HEADS@;
    constexpr uint32_t jobs = query_heads * dim_tiles;
    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= jobs) return;
    transpose_linear(cb_q_column, cb_qt, seq_tiles);
    matmul_one_k_phases(
        cb_qt, cb_ds, cb_dk_raw, cb_partial, seq_tiles, seq_tiles);
    CircularBuffer qt(cb_qt), raw(cb_dk_raw), ds(cb_ds);
    qt.pop_front(seq_tiles);
    ds.pop_front(seq_tiles * seq_tiles);
    transpose_linear(cb_dk_raw, cb_dk, seq_tiles);
    raw.pop_front(seq_tiles);
}
