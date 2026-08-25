// Bit-exact LoRA backward arena:
//   rows [0:M]   = dH = BF16(dY * scale) @ B
//   rows [M:M+N] = dB = transpose(H.T @ BF16(dY * scale))
//
// Each phase preserves the reduction order and BF16 boundaries of the
// separately validated native kernels. Python substitutes @...@ constants.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_dh_dy = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dh_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_dh_scaled = get_compile_time_arg_val(2);
    constexpr uint32_t cb_dh_partial = get_compile_time_arg_val(3);
    constexpr uint32_t cb_dh_output = get_compile_time_arg_val(4);

    constexpr uint32_t cb_db_dy = get_compile_time_arg_val(5);
    constexpr uint32_t cb_db_hidden = get_compile_time_arg_val(6);
    constexpr uint32_t cb_db_scaled = get_compile_time_arg_val(7);
    constexpr uint32_t cb_db_transposed_hidden =
        get_compile_time_arg_val(8);
    constexpr uint32_t cb_db_partial = get_compile_time_arg_val(9);
    constexpr uint32_t cb_db_pretranspose =
        get_compile_time_arg_val(10);
    constexpr uint32_t cb_db_output = get_compile_time_arg_val(11);

    constexpr uint32_t reduction_block_tiles = 2;
    constexpr uint32_t num_n_blocks = @NUM_N_BLOCKS@;
    constexpr uint32_t n_tiles =
        num_n_blocks * reduction_block_tiles;
    constexpr uint32_t m_tiles_per_node = @M_TILES_PER_NODE@;
    constexpr uint32_t total_m_tiles = @TOTAL_M_TILES@;
    constexpr uint32_t n_tiles_per_node = @N_TILES_PER_NODE@;
    constexpr uint32_t total_n_tiles = @TOTAL_N_TILES@;
    constexpr uint32_t reduction_tiles = @REDUCTION_TILES@;
    constexpr uint32_t reduction_blocks =
        reduction_tiles / reduction_block_tiles;
    constexpr uint32_t grid_cols = @GRID_COLS@;
    constexpr uint32_t scale_bits = @SCALE_BITS@;

    static_assert(
        reduction_tiles % reduction_block_tiles == 0,
        "exact dB requires an even token tile count");

    CircularBuffer dh_dy(cb_dh_dy);
    CircularBuffer dh_b(cb_dh_b);
    CircularBuffer dh_scaled(cb_dh_scaled);
    CircularBuffer dh_partial(cb_dh_partial);
    CircularBuffer dh_output(cb_dh_output);

    CircularBuffer db_dy(cb_db_dy);
    CircularBuffer db_hidden(cb_db_hidden);
    CircularBuffer db_scaled(cb_db_scaled);
    CircularBuffer db_transposed_hidden(cb_db_transposed_hidden);
    CircularBuffer db_partial(cb_db_partial);
    CircularBuffer db_pretranspose(cb_db_pretranspose);
    CircularBuffer db_output(cb_db_output);

    const uint32_t node_index =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();

    // Phase 1: one complete output-feature reduction per token tile.
    for (uint32_t local_m = 0; local_m < m_tiles_per_node; ++local_m) {
        const uint32_t m_tile =
            node_index * m_tiles_per_node + local_m;
        if (m_tile >= total_m_tiles) {
            continue;
        }

        dh_dy.wait_front(n_tiles);
        dh_scaled.reserve_back(n_tiles);
        compute_kernel_hw_startup(cb_dh_dy, cb_dh_scaled);
        for (uint32_t tile = 0; tile < n_tiles; tile += 2) {
            init_sfpu(cb_dh_dy, cb_dh_scaled);
            tile_regs_acquire();
            copy_tile_init(cb_dh_dy);
            copy_tile(cb_dh_dy, tile, 0);
            copy_tile(cb_dh_dy, tile + 1, 2);
            binop_with_scalar_tile_init();
            {
                volatile int32_t scalar = scale_bits;
                mul_unary_tile(0, scalar);
                mul_unary_tile(2, scalar);
            }
            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(cb_dh_scaled)));
            pack_tile<true>(0, cb_dh_scaled, tile);
            pack_tile<true>(2, cb_dh_scaled, tile + 1);
            tile_regs_release();
        }
        dh_dy.pop_front(n_tiles);
        dh_scaled.push_back(n_tiles);

        compute_kernel_hw_startup<SrcOrder::Reverse>(
            cb_dh_scaled, cb_dh_b, cb_dh_partial);
        matmul_block_init(
            cb_dh_scaled,
            cb_dh_b,
            false,
            1,
            1,
            reduction_block_tiles);

        for (uint32_t block = 0; block < num_n_blocks; ++block) {
            dh_scaled.wait_front(reduction_block_tiles);
            dh_b.wait_front(reduction_block_tiles);
            tile_regs_acquire();
            if (block > 0) {
                copy_tile_to_dst_init_short_with_dt(
                    cb_dh_b, cb_dh_partial);
                dh_partial.wait_front(1);
                copy_block_matmul_partials(
                    cb_dh_partial, 0, 0, 1);
                dh_partial.pop_front(1);
                reconfig_data_format_srca(cb_dh_partial, cb_dh_b);
                matmul_block_init(
                    cb_dh_scaled,
                    cb_dh_b,
                    false,
                    1,
                    1,
                    reduction_block_tiles);
            }
            for (
                uint32_t inner = 0;
                inner < reduction_block_tiles;
                ++inner
            ) {
                matmul_block(
                    cb_dh_scaled,
                    cb_dh_b,
                    inner,
                    inner,
                    0,
                    false,
                    1,
                    1,
                    reduction_block_tiles);
            }

            tile_regs_commit();
            const bool last = block == num_n_blocks - 1;
            const uint32_t target_cb =
                last ? cb_dh_output : cb_dh_partial;
            if (last) {
                dh_output.reserve_back(1);
            } else {
                dh_partial.reserve_back(1);
            }
            tile_regs_wait();
            PACK((pack_reconfig_data_format(target_cb)));
            pack_tile_block(0, target_cb, 1);
            tile_regs_release();
            if (last) {
                dh_output.push_back(1);
            } else {
                dh_partial.push_back(1);
            }
            dh_b.pop_front(reduction_block_tiles);
            dh_scaled.pop_front(reduction_block_tiles);
        }
    }

    // Phase 2: one complete token reduction per output-feature tile.
    for (uint32_t local_n = 0; local_n < n_tiles_per_node; ++local_n) {
        const uint32_t n_tile =
            node_index * n_tiles_per_node + local_n;
        if (n_tile >= total_n_tiles) {
            continue;
        }

        db_dy.wait_front(reduction_tiles);
        db_scaled.reserve_back(reduction_tiles);
        compute_kernel_hw_startup(cb_db_dy, cb_db_scaled);
        for (
            uint32_t tile = 0;
            tile < reduction_tiles;
            tile += 2
        ) {
            init_sfpu(cb_db_dy, cb_db_scaled);
            tile_regs_acquire();
            copy_tile_init(cb_db_dy);
            copy_tile(cb_db_dy, tile, 0);
            copy_tile(cb_db_dy, tile + 1, 2);
            binop_with_scalar_tile_init();
            {
                volatile int32_t scalar = scale_bits;
                mul_unary_tile(0, scalar);
                mul_unary_tile(2, scalar);
            }
            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(cb_db_scaled)));
            pack_tile<true>(0, cb_db_scaled, tile);
            pack_tile<true>(2, cb_db_scaled, tile + 1);
            tile_regs_release();
        }
        db_dy.pop_front(reduction_tiles);
        db_scaled.push_back(reduction_tiles);

        db_hidden.wait_front(reduction_tiles);
        db_transposed_hidden.reserve_back(reduction_tiles);
        compute_kernel_hw_startup(
            cb_db_hidden, cb_db_transposed_hidden);
        transpose_init(cb_db_hidden);
        for (
            uint32_t tile_start = 0;
            tile_start < reduction_tiles;
            tile_start += 4
        ) {
            constexpr uint32_t max_chunk = 4;
            const uint32_t remaining = reduction_tiles - tile_start;
            const uint32_t chunk =
                remaining < max_chunk ? remaining : max_chunk;
            tile_regs_acquire();
            for (uint32_t tile = 0; tile < chunk; ++tile) {
                transpose_tile(
                    cb_db_hidden, tile_start + tile, tile);
            }
            tile_regs_commit();
            tile_regs_wait();
            PACK((
                pack_reconfig_data_format(cb_db_transposed_hidden)
            ));
            for (uint32_t tile = 0; tile < chunk; ++tile) {
                pack_tile<true>(
                    tile,
                    cb_db_transposed_hidden,
                    tile_start + tile);
            }
            tile_regs_release();
        }
        db_hidden.pop_front(reduction_tiles);
        db_transposed_hidden.push_back(reduction_tiles);

        compute_kernel_hw_startup<SrcOrder::Reverse>(
            cb_db_transposed_hidden,
            cb_db_scaled,
            cb_db_partial);
        matmul_block_init(
            cb_db_transposed_hidden,
            cb_db_scaled,
            false,
            1,
            1,
            reduction_block_tiles);

        for (
            uint32_t block = 0;
            block < reduction_blocks;
            ++block
        ) {
            db_transposed_hidden.wait_front(reduction_block_tiles);
            db_scaled.wait_front(reduction_block_tiles);
            tile_regs_acquire();
            if (block > 0) {
                copy_tile_to_dst_init_short_with_dt(
                    cb_db_scaled, cb_db_partial);
                db_partial.wait_front(1);
                copy_block_matmul_partials(
                    cb_db_partial, 0, 0, 1);
                db_partial.pop_front(1);
                reconfig_data_format_srca(
                    cb_db_partial, cb_db_scaled);
                matmul_block_init(
                    cb_db_transposed_hidden,
                    cb_db_scaled,
                    false,
                    1,
                    1,
                    reduction_block_tiles);
            }
            for (
                uint32_t inner = 0;
                inner < reduction_block_tiles;
                ++inner
            ) {
                matmul_block(
                    cb_db_transposed_hidden,
                    cb_db_scaled,
                    inner,
                    inner,
                    0,
                    false,
                    1,
                    1,
                    reduction_block_tiles);
            }

            tile_regs_commit();
            const bool last = block == reduction_blocks - 1;
            const uint32_t target_cb =
                last ? cb_db_pretranspose : cb_db_partial;
            if (last) {
                db_pretranspose.reserve_back(1);
            } else {
                db_partial.reserve_back(1);
            }
            tile_regs_wait();
            PACK((pack_reconfig_data_format(target_cb)));
            pack_tile_block(0, target_cb, 1);
            tile_regs_release();
            if (last) {
                db_pretranspose.push_back(1);
            } else {
                db_partial.push_back(1);
            }
            db_scaled.pop_front(reduction_block_tiles);
            db_transposed_hidden.pop_front(reduction_block_tiles);
        }

        db_pretranspose.wait_front(1);
        db_output.reserve_back(1);
        compute_kernel_hw_startup(
            cb_db_pretranspose, cb_db_output);
        transpose_init(cb_db_pretranspose);
        tile_regs_acquire();
        transpose_tile(cb_db_pretranspose, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(cb_db_output)));
        pack_tile<true>(0, cb_db_output, 0);
        tile_regs_release();
        db_output.push_back(1);
        db_pretranspose.pop_front(1);
    }
}
