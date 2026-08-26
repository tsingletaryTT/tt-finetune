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
#include "api/compute/eltwise_binary.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "api/debug/waypoint.h"

namespace resident_layer_phase_attention {
// Bit-exact LoRA backward arena:
//   rows [0:M]   = dH = BF16(dY * scale) @ B
//   rows [M:M+N] = dB = transpose(H.T @ BF16(dY * scale))
//
// Each phase preserves the reduction order and BF16 boundaries of the
// separately validated native kernels. Python substitutes @...@ constants.



__attribute__((optimize("Os"), noinline)) void lora_backward_phase(
    uint32_t projection_n_tiles, uint32_t projection_workers) {
    constexpr uint32_t cb_dh_dy = get_compile_time_arg_val(4);
    constexpr uint32_t cb_dh_b = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dh_scaled = get_compile_time_arg_val(2);
    constexpr uint32_t cb_dh_partial = get_compile_time_arg_val(54);
    constexpr uint32_t cb_dh_output = get_compile_time_arg_val(17);

    constexpr uint32_t cb_db_dy = get_compile_time_arg_val(11);
    constexpr uint32_t cb_db_hidden = get_compile_time_arg_val(12);
    constexpr uint32_t cb_db_scaled = get_compile_time_arg_val(10);
    constexpr uint32_t cb_db_transposed_hidden =
        get_compile_time_arg_val(18);
    constexpr uint32_t cb_db_partial = get_compile_time_arg_val(49);
    constexpr uint32_t cb_db_pretranspose =
        get_compile_time_arg_val(55);
    constexpr uint32_t cb_db_output = get_compile_time_arg_val(42);

    constexpr uint32_t reduction_block_tiles = 2;
    const uint32_t num_n_blocks =
        projection_n_tiles / reduction_block_tiles;
    const uint32_t n_tiles = projection_n_tiles;
    constexpr uint32_t m_tiles_per_node = 1;
    constexpr uint32_t total_m_tiles = 4;
    const uint32_t n_tiles_per_node =
        (projection_n_tiles + projection_workers - 1) /
        projection_workers;
    const uint32_t total_n_tiles = projection_n_tiles;
    constexpr uint32_t reduction_tiles = 4;
    constexpr uint32_t reduction_blocks =
        reduction_tiles / reduction_block_tiles;
    constexpr uint32_t grid_cols = 5;
    constexpr uint32_t scale_bits = 1073741824;

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

__attribute__((optimize("Os"), noinline)) void q_lora_backward_phase(
    uint32_t projection_n_tiles, uint32_t projection_workers) {
    constexpr uint32_t cb_dh_dy = 32;
    constexpr uint32_t cb_dh_b = 36;
    constexpr uint32_t cb_dh_scaled = 38;
    constexpr uint32_t cb_dh_partial = 39;
    constexpr uint32_t cb_dh_output = 40;

    constexpr uint32_t cb_db_dy = 33;
    constexpr uint32_t cb_db_hidden = 37;
    constexpr uint32_t cb_db_scaled = 41;
    constexpr uint32_t cb_db_transposed_hidden = 42;
    constexpr uint32_t cb_db_partial = 43;
    constexpr uint32_t cb_db_pretranspose = 44;
    constexpr uint32_t cb_db_output = 34;

    constexpr uint32_t reduction_block_tiles = 2;
    const uint32_t num_n_blocks =
        projection_n_tiles / reduction_block_tiles;
    const uint32_t n_tiles = projection_n_tiles;
    constexpr uint32_t m_tiles_per_node = 1;
    constexpr uint32_t total_m_tiles = 4;
    const uint32_t n_tiles_per_node =
        (projection_n_tiles + projection_workers - 1) /
        projection_workers;
    const uint32_t total_n_tiles = projection_n_tiles;
    constexpr uint32_t reduction_tiles = 4;
    constexpr uint32_t reduction_blocks =
        reduction_tiles / reduction_block_tiles;
    constexpr uint32_t grid_cols = 5;
    constexpr uint32_t scale_bits = 1073741824;

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

// Appended after lora_backward_arena_exact_compute.cpp has been rewritten
// from kernel_main() into lora_backward_phase(). CBs 0..3 belong to this
// exact frozen-base dX phase; the shifted LoRA phase uses CBs 4..15.

__attribute__((optimize("Os"), noinline)) void resident_down_phase_one(
    uint32_t projection_n_tiles, uint32_t projection_workers) {
    constexpr uint32_t cb_dy = get_compile_time_arg_val(40);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(41);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(37);
    constexpr uint32_t cb_result = get_compile_time_arg_val(25);

    constexpr uint32_t jobs_per_node = 4;
    constexpr uint32_t total_jobs = 112;
    const uint32_t n_tiles = projection_n_tiles;
    constexpr uint32_t reduction_tiles_per_phase =
        1;
    const uint32_t reduction_blocks =
        n_tiles / reduction_tiles_per_phase;
    constexpr uint32_t grid_cols = 5;

    CircularBuffer dy(cb_dy);
    CircularBuffer weight(cb_weight);
    CircularBuffer partial(cb_partial);
    CircularBuffer result(cb_result);

    const uint32_t node_index =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();

    compute_kernel_hw_startup<SrcOrder::Reverse>(
        cb_dy, cb_weight, cb_partial);
    matmul_block_init(
        cb_dy,
        cb_weight,
        false,
        1,
        1,
        reduction_tiles_per_phase);
    for (uint32_t local_job = 0; local_job < jobs_per_node; ++local_job) {
        const uint32_t job = node_index * jobs_per_node + local_job;
        if (job >= total_jobs) {
            continue;
        }
        for (uint32_t block = 0; block < reduction_blocks; ++block) {
            dy.wait_front(reduction_tiles_per_phase);
            weight.wait_front(reduction_tiles_per_phase);
            tile_regs_acquire();
            if (block > 0) {
                copy_tile_to_dst_init_short_with_dt(
                    cb_weight, cb_partial);
                partial.wait_front(1);
                copy_block_matmul_partials(cb_partial, 0, 0, 1);
                partial.pop_front(1);
                reconfig_data_format_srca(cb_partial, cb_weight);
                matmul_block_init(
                    cb_dy,
                    cb_weight,
                    false,
                    1,
                    1,
                    reduction_tiles_per_phase);
            }
            for (
                uint32_t inner = 0;
                inner < reduction_tiles_per_phase;
                ++inner
            ) {
                matmul_block(
                    cb_dy,
                    cb_weight,
                    inner,
                    inner,
                    0,
                    false,
                    1,
                    1,
                    reduction_tiles_per_phase);
            }
            tile_regs_commit();
            const bool last = block == reduction_blocks - 1;
            const uint32_t target_cb = last ? cb_result : cb_partial;
            if (last) {
                result.reserve_back(1);
            } else {
                partial.reserve_back(1);
            }
            tile_regs_wait();
            PACK((pack_reconfig_data_format(target_cb)));
            pack_tile_block(0, target_cb, 1);
            tile_regs_release();
            if (last) {
                result.push_back(1);
            } else {
                partial.push_back(1);
            }
            weight.pop_front(reduction_tiles_per_phase);
            dy.pop_front(reduction_tiles_per_phase);
        }
    }

    lora_backward_phase(projection_n_tiles, projection_workers);
}

__attribute__((optimize("Os"), noinline)) void q_resident_down_phase_one(
    uint32_t projection_n_tiles, uint32_t projection_workers) {
    constexpr uint32_t cb_dy = 45;
    constexpr uint32_t cb_weight = 46;
    constexpr uint32_t cb_partial = 48;
    constexpr uint32_t cb_result = 47;

    constexpr uint32_t jobs_per_node = 4;
    constexpr uint32_t total_jobs = 112;
    const uint32_t n_tiles = projection_n_tiles;
    constexpr uint32_t reduction_tiles_per_phase =
        1;
    const uint32_t reduction_blocks =
        n_tiles / reduction_tiles_per_phase;
    constexpr uint32_t grid_cols = 5;

    CircularBuffer dy(cb_dy);
    CircularBuffer weight(cb_weight);
    CircularBuffer partial(cb_partial);
    CircularBuffer result(cb_result);

    const uint32_t node_index =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();

    compute_kernel_hw_startup<SrcOrder::Reverse>(
        cb_dy, cb_weight, cb_partial);
    matmul_block_init(
        cb_dy,
        cb_weight,
        false,
        1,
        1,
        reduction_tiles_per_phase);
    for (uint32_t local_job = 0; local_job < jobs_per_node; ++local_job) {
        const uint32_t job = node_index * jobs_per_node + local_job;
        if (job >= total_jobs) {
            continue;
        }
        for (uint32_t block = 0; block < reduction_blocks; ++block) {
            dy.wait_front(reduction_tiles_per_phase);
            weight.wait_front(reduction_tiles_per_phase);
            tile_regs_acquire();
            if (block > 0) {
                copy_tile_to_dst_init_short_with_dt(
                    cb_weight, cb_partial);
                partial.wait_front(1);
                copy_block_matmul_partials(cb_partial, 0, 0, 1);
                partial.pop_front(1);
                reconfig_data_format_srca(cb_partial, cb_weight);
                matmul_block_init(
                    cb_dy,
                    cb_weight,
                    false,
                    1,
                    1,
                    reduction_tiles_per_phase);
            }
            for (
                uint32_t inner = 0;
                inner < reduction_tiles_per_phase;
                ++inner
            ) {
                matmul_block(
                    cb_dy,
                    cb_weight,
                    inner,
                    inner,
                    0,
                    false,
                    1,
                    1,
                    reduction_tiles_per_phase);
            }
            tile_regs_commit();
            const bool last = block == reduction_blocks - 1;
            const uint32_t target_cb = last ? cb_result : cb_partial;
            if (last) {
                result.reserve_back(1);
            } else {
                partial.reserve_back(1);
            }
            tile_regs_wait();
            PACK((pack_reconfig_data_format(target_cb)));
            pack_tile_block(0, target_cb, 1);
            tile_regs_release();
            if (last) {
                result.push_back(1);
            } else {
                partial.push_back(1);
            }
            weight.pop_front(reduction_tiles_per_phase);
            dy.pop_front(reduction_tiles_per_phase);
        }
    }

    q_lora_backward_phase(projection_n_tiles, projection_workers);
}

// Exact two-member second phase of frozen Linear + LoRA backward.
//
// All intermediate member sums are deliberately packed to BF16 circular
// buffers in L1. This preserves the accepted PEFT rounding schedule without
// exposing any intermediate tensor to DRAM or the outer autograd graph.



namespace {

ALWI void add_tiles_bf16(
    uint32_t cb0,
    uint32_t cb1,
    uint32_t dst
) {
    UNPACK((llk_unpack_AB(cb0, cb1, 0, 0)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWADD,
          BroadcastType::NONE,
          false,
          MathFidelity::LoFi,
          EltwiseBinaryReuseDestType::NONE>(
        cb0, cb1, dst, false)));
}

ALWI void commit_bf16() {
    MATH((llk_math_dest_section_done<false>()));
}

ALWI void pack_reconfig_bf16(uint32_t cb) {
    PACK((llk_pack_reconfig_data_format<false>(cb)));
}

ALWI void pack_bf16(uint32_t dst, uint32_t cb) {
    PACK((llk_pack<false, false, PackMode::Default>(dst, cb, 0)));
}

ALWI void release_bf16() {
    PACK((llk_pack_dest_section_done<false>()));
}

__attribute__((optimize("Os"), noinline)) void run_lora_dx(
    uint32_t cb_dh,
    uint32_t cb_a,
    uint32_t cb_output,
    uint32_t jobs_per_node,
    uint32_t total_jobs,
    uint32_t node_index
) {
    CircularBuffer dh(cb_dh);
    CircularBuffer a(cb_a);
    CircularBuffer output(cb_output);
    compute_kernel_hw_startup<SrcOrder::Reverse>(
        cb_dh, cb_a, cb_output);
    matmul_block_init(cb_dh, cb_a, false, 1, 1, 1);
    for (uint32_t local_job = 0; local_job < jobs_per_node; ++local_job) {
        const uint32_t job = node_index * jobs_per_node + local_job;
        if (job >= total_jobs) {
            continue;
        }
        dh.wait_front(1);
        a.wait_front(1);
        output.reserve_back(1);
        matmul_block_init(cb_dh, cb_a, false, 1, 1, 1);
        tile_regs_acquire();
        matmul_block(cb_dh, cb_a, 0, 0, 0, false, 1, 1, 1);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(cb_output)));
        pack_tile<true>(0, cb_output, 0);
        tile_regs_release();
        output.push_back(1);
        a.pop_front(1);
        dh.pop_front(1);
    }
}

__attribute__((optimize("Os"), noinline)) void run_bf16_add(
    uint32_t cb0,
    uint32_t cb1,
    uint32_t cb_output,
    uint32_t jobs_per_node,
    uint32_t total_jobs,
    uint32_t node_index
) {
    CircularBuffer first(cb0);
    CircularBuffer second(cb1);
    CircularBuffer output(cb_output);
    add_tiles_init(cb0, cb1);
    for (uint32_t local_job = 0; local_job < jobs_per_node; ++local_job) {
        const uint32_t job = node_index * jobs_per_node + local_job;
        if (job >= total_jobs) {
            continue;
        }
        first.wait_front(1);
        second.wait_front(1);
        output.reserve_back(1);
        tile_regs_acquire();
        add_tiles_bf16(cb0, cb1, 0);
        commit_bf16();
        tile_regs_wait();
        pack_reconfig_bf16(cb_output);
        pack_bf16(0, cb_output);
        release_bf16();
        output.push_back(1);
        second.pop_front(1);
        first.pop_front(1);
    }
}

__attribute__((optimize("Os"), noinline)) void run_da(
    uint32_t cb_dh,
    uint32_t cb_x,
    uint32_t cb_transposed_x,
    uint32_t cb_partial,
    uint32_t cb_pretranspose,
    uint32_t cb_output,
    uint32_t reduction_tiles,
    uint32_t k_tiles_per_node,
    uint32_t total_k_tiles,
    uint32_t node_index
) {
    constexpr uint32_t reduction_block_tiles = 2;
    const uint32_t reduction_blocks =
        reduction_tiles / reduction_block_tiles;
    CircularBuffer dh(cb_dh);
    CircularBuffer x(cb_x);
    CircularBuffer transposed_x(cb_transposed_x);
    CircularBuffer partial(cb_partial);
    CircularBuffer pretranspose(cb_pretranspose);
    CircularBuffer output(cb_output);

    for (uint32_t local_k = 0; local_k < k_tiles_per_node; ++local_k) {
        const uint32_t k_tile =
            node_index * k_tiles_per_node + local_k;
        if (k_tile >= total_k_tiles) {
            continue;
        }

        x.wait_front(reduction_tiles);
        transposed_x.reserve_back(reduction_tiles);
        compute_kernel_hw_startup(cb_x, cb_transposed_x);
        transpose_init(cb_x);
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
                transpose_tile(cb_x, tile_start + tile, tile);
            }
            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(cb_transposed_x)));
            for (uint32_t tile = 0; tile < chunk; ++tile) {
                pack_tile<true>(
                    tile,
                    cb_transposed_x,
                    tile_start + tile);
            }
            tile_regs_release();
        }
        x.pop_front(reduction_tiles);
        transposed_x.push_back(reduction_tiles);

        compute_kernel_hw_startup<SrcOrder::Reverse>(
            cb_transposed_x, cb_dh, cb_partial);
        matmul_block_init(
            cb_transposed_x,
            cb_dh,
            false,
            1,
            1,
            reduction_block_tiles);
        for (uint32_t block = 0; block < reduction_blocks; ++block) {
            transposed_x.wait_front(reduction_block_tiles);
            // Complete dH window is already resident.
            tile_regs_acquire();
            if (block > 0) {
                copy_tile_to_dst_init_short_with_dt(cb_dh, cb_partial);
                partial.wait_front(1);
                copy_block_matmul_partials(cb_partial, 0, 0, 1);
                partial.pop_front(1);
                reconfig_data_format_srca(cb_partial, cb_dh);
                matmul_block_init(
                    cb_transposed_x,
                    cb_dh,
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
                    cb_transposed_x,
                    cb_dh,
                    inner,
                    block * reduction_block_tiles + inner,
                    0,
                    false,
                    1,
                    1,
                    reduction_block_tiles);
            }
            tile_regs_commit();
            const bool last = block == reduction_blocks - 1;
            const uint32_t target =
                last ? cb_pretranspose : cb_partial;
            if (last) {
                pretranspose.reserve_back(1);
            } else {
                partial.reserve_back(1);
            }
            tile_regs_wait();
            PACK((pack_reconfig_data_format(target)));
            pack_tile_block(0, target, 1);
            tile_regs_release();
            if (last) {
                pretranspose.push_back(1);
            } else {
                partial.push_back(1);
            }
            // Keep dH resident for LoRA dO.
            transposed_x.pop_front(reduction_block_tiles);
        }

        pretranspose.wait_front(1);
        output.reserve_back(1);
        compute_kernel_hw_startup(cb_pretranspose, cb_output);
        transpose_init(cb_pretranspose);
        tile_regs_acquire();
        transpose_tile(cb_pretranspose, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(cb_output)));
        pack_tile<true>(0, cb_output, 0);
        tile_regs_release();
        output.push_back(1);
        pretranspose.pop_front(1);
    }
}

}  // namespace

// Compute tail appended after the exact phase-one helpers.


__attribute__((optimize("Os"), noinline)) void run_resident_down_lora_dx(
    uint32_t cb_dh,
    uint32_t cb_a,
    uint32_t cb_output,
    uint32_t jobs_per_node,
    uint32_t total_jobs,
    uint32_t k_tiles,
    uint32_t node_index
) {
    CircularBuffer dh(cb_dh);
    CircularBuffer a(cb_a);
    CircularBuffer output(cb_output);
    const uint32_t first_job = node_index * jobs_per_node;
    if (first_job >= total_jobs) return;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_dh, cb_a, cb_output);
    matmul_block_init(cb_dh, cb_a, false, 1, 1, 1);
    for (uint32_t local = 0; local < jobs_per_node; ++local) {
        const uint32_t job = first_job + local;
        if (job >= total_jobs) continue;
        const uint32_t m = job / k_tiles;
        a.wait_front(1);
        output.reserve_back(1);
        matmul_block_init(cb_dh, cb_a, false, 1, 1, 1);
        tile_regs_acquire();
        matmul_block(cb_dh, cb_a, m, 0, 0, false, 1, 1, 1);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(cb_output)));
        pack_tile<true>(0, cb_output, 0);
        tile_regs_release();
        output.push_back(1);
        a.pop_front(1);
    }
}

__attribute__((optimize("Os"), noinline)) void resident_down_tail() {
    constexpr uint32_t cb_base = get_compile_time_arg_val(25);
    constexpr uint32_t cb_dh = get_compile_time_arg_val(19);
    constexpr uint32_t cb_a = get_compile_time_arg_val(20);
    constexpr uint32_t cb_x = get_compile_time_arg_val(13);
    constexpr uint32_t cb_transposed_x = get_compile_time_arg_val(14);
    constexpr uint32_t cb_da_partial = get_compile_time_arg_val(50);
    constexpr uint32_t cb_da_pretranspose = get_compile_time_arg_val(56);
    constexpr uint32_t cb_da = get_compile_time_arg_val(43);
    constexpr uint32_t cb_lora_dx = get_compile_time_arg_val(21);
    constexpr uint32_t cb_final_dx = get_compile_time_arg_val(22);
    constexpr uint32_t cb_k_accum_dx = 30;

    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t k_tiles = 28;
    constexpr uint32_t jobs_per_node = 4;
    constexpr uint32_t da_k_tiles_per_node = 1;
    constexpr uint32_t total_jobs = m_tiles * k_tiles;
    constexpr uint32_t grid_cols = 5;
    const uint32_t node =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();

    CircularBuffer resident_dh(cb_dh);
    resident_dh.wait_front(m_tiles);

    {
        DeviceZoneScopedN("resident_down_da_compute");
        run_da(
            cb_dh, cb_x, cb_transposed_x, cb_da_partial,
            cb_da_pretranspose, cb_da, m_tiles,
            da_k_tiles_per_node, k_tiles, node);
    }
    {
        DeviceZoneScopedN("resident_down_lora_dx_compute");
        run_resident_down_lora_dx(
            cb_dh, cb_a, cb_lora_dx,
            jobs_per_node, total_jobs, k_tiles, node);
    }

    // Match the framework-visible BF16 add boundary exactly.
    disable_fp32_dest_acc();
    {
        DeviceZoneScopedN("resident_down_final_add");
        run_bf16_add(
            cb_base, cb_lora_dx, cb_final_dx,
            jobs_per_node, total_jobs, node);
    }
    enable_fp32_dest_acc();
}

// Bit-exact LoRA backward arena:
//   rows [0:M]   = dH = BF16(dY * scale) @ B
//   rows [M:M+N] = dB = transpose(H.T @ BF16(dY * scale))
//
// Each phase preserves the reduction order and BF16 boundaries of the
// separately validated native kernels. Python substitutes @...@ constants.




// Appended after lora_backward_arena_exact_compute.cpp has been rewritten
// from kernel_main() into kv_lora_backward_phase(). CBs 0..3 belong to this
// exact frozen-base dX phase; the shifted LoRA phase uses CBs 4..15.


// Exact two-member second phase of frozen Linear + LoRA backward.
//
// All intermediate member sums are deliberately packed to BF16 circular
// buffers in L1. This preserves the accepted PEFT rounding schedule without
// exposing any intermediate tensor to DRAM or the outer autograd graph.



namespace {









}  // namespace

// Compute tail appended after the exact phase-one helpers.


__attribute__((optimize("Os"), noinline)) void kv_run_resident_down_lora_dx(
    uint32_t cb_dh,
    uint32_t cb_a,
    uint32_t cb_output,
    uint32_t jobs_per_node,
    uint32_t total_jobs,
    uint32_t k_tiles,
    uint32_t node_index
) {
    CircularBuffer dh(cb_dh);
    CircularBuffer a(cb_a);
    CircularBuffer output(cb_output);
    const uint32_t first_job = node_index * jobs_per_node;
    if (first_job >= total_jobs) return;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_dh, cb_a, cb_output);
    matmul_block_init(cb_dh, cb_a, false, 1, 1, 1);
    for (uint32_t local = 0; local < jobs_per_node; ++local) {
        const uint32_t job = first_job + local;
        if (job >= total_jobs) continue;
        const uint32_t m = job / k_tiles;
        a.wait_front(1);
        output.reserve_back(1);
        matmul_block_init(cb_dh, cb_a, false, 1, 1, 1);
        tile_regs_acquire();
        matmul_block(cb_dh, cb_a, m, 0, 0, false, 1, 1, 1);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(cb_output)));
        pack_tile<true>(0, cb_output, 0);
        tile_regs_release();
        output.push_back(1);
        a.pop_front(1);
    }
}

__attribute__((optimize("Os"), noinline)) void kv_resident_down_tail(bool accumulate) {
    constexpr uint32_t cb_base = get_compile_time_arg_val(25);
    constexpr uint32_t cb_dh = get_compile_time_arg_val(19);
    constexpr uint32_t cb_a = get_compile_time_arg_val(20);
    constexpr uint32_t cb_x = get_compile_time_arg_val(13);
    constexpr uint32_t cb_transposed_x = get_compile_time_arg_val(14);
    constexpr uint32_t cb_da_partial = get_compile_time_arg_val(50);
    constexpr uint32_t cb_da_pretranspose = get_compile_time_arg_val(56);
    constexpr uint32_t cb_da = get_compile_time_arg_val(43);
    constexpr uint32_t cb_lora_dx = get_compile_time_arg_val(21);
    constexpr uint32_t cb_final_dx = get_compile_time_arg_val(22);
    constexpr uint32_t cb_k_accum_dx = 30;
    constexpr uint32_t cb_accum_scratch = 26;

    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t k_tiles = 28;
    constexpr uint32_t jobs_per_node = 4;
    constexpr uint32_t da_k_tiles_per_node = 1;
    constexpr uint32_t total_jobs = m_tiles * k_tiles;
    constexpr uint32_t grid_cols = 5;
    const uint32_t node =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();

    CircularBuffer resident_dh(cb_dh);
    resident_dh.wait_front(m_tiles);

    {
        DeviceZoneScopedN("resident_down_da_compute");
        run_da(
            cb_dh, cb_x, cb_transposed_x, cb_da_partial,
            cb_da_pretranspose, cb_da, m_tiles,
            da_k_tiles_per_node, k_tiles, node);
    }
    {
        DeviceZoneScopedN("resident_down_lora_dx_compute");
        kv_run_resident_down_lora_dx(
            cb_dh, cb_a, cb_lora_dx,
            jobs_per_node, total_jobs, k_tiles, node);
    }

    // Match the framework-visible BF16 add boundary exactly.
    disable_fp32_dest_acc();
    {
        DeviceZoneScopedN("resident_down_final_add");
        if (accumulate) {
        // Preserve the framework's two new BF16 edges:
        // accumulator + LoRA, followed by + frozen-base.
        run_bf16_add(
            cb_final_dx, cb_lora_dx, cb_accum_scratch,
            jobs_per_node, total_jobs, node);
        run_bf16_add(
            cb_accum_scratch, cb_base, cb_k_accum_dx,
            jobs_per_node, total_jobs, node);
        } else {
            run_bf16_add(
                cb_base, cb_lora_dx, cb_final_dx,
                jobs_per_node, total_jobs, node);
        }
    }
    enable_fp32_dest_acc();
}

static __attribute__((noinline, noclone)) void kv_compute_phase(bool accumulate) {
    {
        DeviceZoneScopedN("resident_down_phase_one_compute");
        resident_down_phase_one(4, 50);
    }
    kv_resident_down_tail(accumulate);
    CircularBuffer resident_dh(get_compile_time_arg_val(19));
    resident_dh.pop_front(4);
    // CB7 is empty after dH and becomes a count-only compute->writer fence.
    if (accumulate) {
        CircularBuffer phase_done(28);
        PACK((*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                phase_done.get_write_ptr() <<
                LOG_BASE_2_OF_L1_ALIGNMENT) = 1));
    } else {
        CircularBuffer phase_done(25);
        PACK((*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                phase_done.get_write_ptr() <<
                LOG_BASE_2_OF_L1_ALIGNMENT) = 1));
    }
}

// Bit-exact LoRA backward arena:
//   rows [0:M]   = dH = BF16(dY * scale) @ B
//   rows [M:M+N] = dB = transpose(H.T @ BF16(dY * scale))
//
// Each phase preserves the reduction order and BF16 boundaries of the
// separately validated native kernels. Python substitutes @...@ constants.




// Appended after lora_backward_arena_exact_compute.cpp has been rewritten
// from kernel_main() into lora_backward_phase(). CBs 0..3 belong to this
// exact frozen-base dX phase; the shifted LoRA phase uses CBs 4..15.


// Compute tail appended after the exact phase-one helpers.



__attribute__((optimize("Os"), noinline)) void q_resident_down_tail() {
    constexpr uint32_t cb_base = 47;
    constexpr uint32_t cb_dh = 49;
    constexpr uint32_t cb_a = 50;
    constexpr uint32_t cb_x = 51;
    constexpr uint32_t cb_transposed_x = 52;
    constexpr uint32_t cb_da_partial = 53;
    constexpr uint32_t cb_da_pretranspose = 54;
    constexpr uint32_t cb_da = 35;
    constexpr uint32_t cb_lora_dx = 55;
    constexpr uint32_t cb_final_dx = 56;
    constexpr uint32_t cb_k_accum_dx = 30;
    constexpr uint32_t cb_accum_scratch = 31;

    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t k_tiles = 28;
    constexpr uint32_t jobs_per_node = 4;
    constexpr uint32_t da_k_tiles_per_node = 1;
    constexpr uint32_t total_jobs = m_tiles * k_tiles;
    constexpr uint32_t grid_cols = 5;
    const uint32_t node =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();

    CircularBuffer resident_dh(cb_dh);
    resident_dh.wait_front(m_tiles);

    {
        DeviceZoneScopedN("resident_down_da_compute");
        run_da(
            cb_dh, cb_x, cb_transposed_x, cb_da_partial,
            cb_da_pretranspose, cb_da, m_tiles,
            da_k_tiles_per_node, k_tiles, node);
    }
    {
        DeviceZoneScopedN("resident_down_lora_dx_compute");
        run_resident_down_lora_dx(
            cb_dh, cb_a, cb_lora_dx,
            jobs_per_node, total_jobs, k_tiles, node);
    }

    // Match the framework-visible BF16 add boundary exactly.
    disable_fp32_dest_acc();
    {
        DeviceZoneScopedN("resident_down_final_add");
        // Preserve the framework's two new BF16 edges:
        // accumulator + LoRA, followed by + frozen-base.
        run_bf16_add(
            cb_k_accum_dx, cb_lora_dx, cb_accum_scratch,
            jobs_per_node, total_jobs, node);
        run_bf16_add(
            cb_accum_scratch, cb_base, cb_final_dx,
            jobs_per_node, total_jobs, node);
    }
    enable_fp32_dest_acc();
}

static __attribute__((noinline, noclone)) void q_compute_phase() {
    {
        DeviceZoneScopedN("resident_down_phase_one_compute");
        q_resident_down_phase_one(28, 50);
    }
    q_resident_down_tail();
    CircularBuffer resident_dh(49);
    resident_dh.pop_front(4);
    // CB7 is empty after dH and becomes a count-only compute->writer fence.
    CircularBuffer phase_done(29);
PACK((*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        phase_done.get_write_ptr() <<
        LOG_BASE_2_OF_L1_ALIGNMENT) = 1));
}

void run() {
    {
    {
        DeviceZoneScopedN("resident_down_phase_one_compute");
        WAYPOINT("PB0");
        resident_down_phase_one(28, 50);
        WAYPOINT("PB1");
    }
    WAYPOINT("PT0");
    resident_down_tail();
    WAYPOINT("PT1");

    // Output dH has reached both terminal consumers (dA and LoRA dO). The
    // split path reclaimed CB16 at program teardown; the one-call owner must
    // end that lifetime explicitly before V reuses the same local buffer.
    CircularBuffer output_resident_dh(get_compile_time_arg_val(19));
    output_resident_dh.pop_front(4);
    }
    {
    kv_compute_phase(false);
    kv_compute_phase(true);
    q_compute_phase();
    }
}
}  // namespace resident_layer_phase_attention

void kernel_main() {
    resident_layer_phase_attention::run();
}
