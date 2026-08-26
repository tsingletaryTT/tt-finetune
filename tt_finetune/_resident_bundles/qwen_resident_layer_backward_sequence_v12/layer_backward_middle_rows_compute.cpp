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
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/debug/waypoint.h"

namespace resident_layer_phase_mlp {
// Bit-exact LoRA backward arena:
//   rows [0:M]   = dH = BF16(dY * scale) @ B
//   rows [M:M+N] = dB = transpose(H.T @ BF16(dY * scale))
//
// Each phase preserves the reduction order and BF16 boundaries of the
// separately validated native kernels. Python substitutes @...@ constants.



FORCE_INLINE void lora_backward_phase() {
    constexpr uint32_t cb_dh_dy = get_compile_time_arg_val(64);
    constexpr uint32_t cb_dh_b = get_compile_time_arg_val(65);
    constexpr uint32_t cb_dh_scaled = get_compile_time_arg_val(66);
    constexpr uint32_t cb_dh_partial = get_compile_time_arg_val(99);
    constexpr uint32_t cb_dh_output = get_compile_time_arg_val(73);

    constexpr uint32_t cb_db_dy = get_compile_time_arg_val(87);
    constexpr uint32_t cb_db_hidden = get_compile_time_arg_val(81);
    constexpr uint32_t cb_db_scaled = get_compile_time_arg_val(82);
    constexpr uint32_t cb_db_transposed_hidden =
        get_compile_time_arg_val(83);
    constexpr uint32_t cb_db_partial = get_compile_time_arg_val(77);
    constexpr uint32_t cb_db_pretranspose =
        get_compile_time_arg_val(90);
    constexpr uint32_t cb_db_output = get_compile_time_arg_val(84);

    constexpr uint32_t reduction_block_tiles = 2;
    constexpr uint32_t num_n_blocks = 14;
    constexpr uint32_t n_tiles =
        num_n_blocks * reduction_block_tiles;
    constexpr uint32_t m_tiles_per_node = 1;
    constexpr uint32_t total_m_tiles = 4;
    constexpr uint32_t n_tiles_per_node = 1;
    constexpr uint32_t total_n_tiles = 28;
    constexpr uint32_t reduction_tiles = 4;
    constexpr uint32_t reduction_blocks =
        reduction_tiles / reduction_block_tiles;
    constexpr uint32_t grid_cols = 10;
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

FORCE_INLINE void resident_down_phase_one() {
    constexpr uint32_t cb_dy = get_compile_time_arg_val(79);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(80);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(78);
    constexpr uint32_t cb_result = get_compile_time_arg_val(71);

    constexpr uint32_t jobs_per_node = 7;
    constexpr uint32_t total_jobs = 608;
    constexpr uint32_t n_tiles = 28;
    constexpr uint32_t reduction_tiles_per_phase =
        2;
    constexpr uint32_t reduction_blocks =
        n_tiles / reduction_tiles_per_phase;
    constexpr uint32_t grid_cols = 10;

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

    lora_backward_phase();
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

ALWI void run_lora_dx(
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

ALWI void run_bf16_add(
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

ALWI void run_da(
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
            // Keep dH resident for LoRA dProduct.
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


ALWI void run_resident_down_lora_dx(
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

FORCE_INLINE void resident_down_tail() {
    constexpr uint32_t cb_base = get_compile_time_arg_val(71);
    constexpr uint32_t cb_dh = get_compile_time_arg_val(85);
    constexpr uint32_t cb_a = get_compile_time_arg_val(63);
    constexpr uint32_t cb_x = get_compile_time_arg_val(67);
    constexpr uint32_t cb_transposed_x = get_compile_time_arg_val(68);
    constexpr uint32_t cb_da_partial = get_compile_time_arg_val(75);
    constexpr uint32_t cb_da_pretranspose = get_compile_time_arg_val(94);
    constexpr uint32_t cb_da = get_compile_time_arg_val(91);
    constexpr uint32_t cb_lora_dx = get_compile_time_arg_val(69);
    constexpr uint32_t cb_final_dx = get_compile_time_arg_val(70);

    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t k_tiles = 152;
    constexpr uint32_t jobs_per_node = 7;
    constexpr uint32_t da_k_tiles_per_node = 2;
    constexpr uint32_t total_jobs = m_tiles * k_tiles;
    constexpr uint32_t grid_cols = 10;
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


ALWI void full_mlp_bf16_unary_init(uint32_t input_id, uint32_t output_id) {
#ifndef ARCH_QUASAR
    state_configure<Operand::SRCA, Operand::PACK>(
        input_id, output_id, __builtin_LINE());
    UNPACK((llk_unpack_hw_configure<false>(input_id)));
    UNPACK((llk_unpack_A_init<
            BroadcastType::NONE, false,
            EltwiseBinaryReuseDestType::NONE,
            UnpackToDestEn>(false, false, input_id)));
    PACK((llk_pack_hw_configure<false>(output_id)));
    PACK((llk_pack_init(output_id)));
    PACK((llk_pack_dest_init<false, PackMode::Default>()));
    MATH((llk_math_eltwise_unary_datacopy_init<
          DataCopyType::A2D, false, BroadcastType::NONE>(input_id)));
    MATH((llk_math_pack_sync_init<false>()));
    MATH((llk_math_hw_configure<false>(input_id, input_id)));
    MATH((ckernel::math::_configure_unary_preserve_zero_flag_state_()));
#endif
}

ALWI void full_mlp_bf16_copy_init(uint32_t input_id) {
    UNPACK((llk_unpack_A_init<
            BroadcastType::NONE, false,
            EltwiseBinaryReuseDestType::NONE,
            UnpackToDestEn>(false, false, input_id)));
    MATH((llk_math_eltwise_unary_datacopy_init<
          DataCopyType::A2D, false, BroadcastType::NONE>(input_id)));
}

ALWI void full_mlp_bf16_copy_tile(
    uint32_t input_id, uint32_t input_tile, uint32_t destination_tile) {
    UNPACK((llk_unpack_A<
            BroadcastType::NONE, false,
            EltwiseBinaryReuseDestType::NONE,
            UnpackToDestEn>(input_id, input_tile)));
    MATH((llk_math_eltwise_unary_datacopy<
          DataCopyType::A2D, false, BroadcastType::NONE,
          UnpackToDestEn>(destination_tile, input_id)));
}

static __attribute__((noinline, noclone)) void full_mlp_exact_sigmoid_one(
    uint32_t input_id, uint32_t output_id) {
    CircularBuffer input(input_id);
    CircularBuffer output(output_id);
    full_mlp_bf16_unary_init(input_id, output_id);
    input.wait_front(1);
    output.reserve_back(1);
    tile_regs_acquire();
    full_mlp_bf16_copy_init(input_id);
    full_mlp_bf16_copy_tile(input_id, 0, 0);
    MATH((SFPU_UNARY_INIT_FN(sigmoid, sfpu::sigmoid_init, (false))));
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE, false, calculate_sigmoid,
        (false, false, 8), 0, VectorMode::RC)));
    MATH((llk_math_dest_section_done<false>()));
    tile_regs_wait();
    PACK((llk_pack_reconfig_data_format<false>(output_id)));
    PACK((llk_pack<false, false, PackMode::Default>(0, output_id, 0)));
    PACK((llk_pack_dest_section_done<false>()));
    output.push_back(1);
    input.pop_front(1);
}

static __attribute__((noinline, noclone)) void full_mlp_exact_multiply_one(
    uint32_t lhs_id, uint32_t rhs_id, uint32_t output_id) {
    CircularBuffer lhs(lhs_id);
    CircularBuffer rhs(rhs_id);
    CircularBuffer output(output_id);
    full_mlp_bf16_unary_init(lhs_id, output_id);
    MATH((SFPU_BINARY_INIT_FN(
        unused, sfpu::sfpu_binary_init,
        (false, ckernel::BinaryOp::MUL))));
    lhs.wait_front(1);
    rhs.wait_front(1);
    output.reserve_back(1);
    tile_regs_acquire();
    full_mlp_bf16_copy_init(lhs_id);
    full_mlp_bf16_copy_tile(lhs_id, 0, 0);
    full_mlp_bf16_copy_init(rhs_id);
    full_mlp_bf16_copy_tile(rhs_id, 0, 1);
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE, false, calculate_sfpu_binary_mul,
        (false, ckernel::BinaryOp::MUL, 8, false),
        0, 1, 0, VectorMode::RC)));
    MATH((llk_math_dest_section_done<false>()));
    tile_regs_wait();
    PACK((llk_pack_reconfig_data_format<false>(output_id)));
    PACK((llk_pack<false, false, PackMode::Default>(0, output_id, 0)));
    PACK((llk_pack_dest_section_done<false>()));
    output.push_back(1);
}

static __attribute__((noinline, noclone)) void full_mlp_exact_product_one(
    uint32_t gate_id, uint32_t up_id, uint32_t sigmoid_id,
    uint32_t silu_id, uint32_t output_id) {
    CircularBuffer gate(gate_id);
    CircularBuffer up(up_id);
    CircularBuffer sigmoid(sigmoid_id);
    CircularBuffer silu(silu_id);
    full_mlp_exact_sigmoid_one(gate_id, sigmoid_id);
    full_mlp_exact_multiply_one(gate_id, sigmoid_id, silu_id);
    sigmoid.pop_front(1);
    full_mlp_exact_multiply_one(silu_id, up_id, output_id);
    silu.pop_front(1);
    gate.pop_front(1);
    up.pop_front(1);
}

FORCE_INLINE void full_recompute_swiglu_products_phase() {
    constexpr uint32_t cb_gate = get_compile_time_arg_val(62);
    constexpr uint32_t cb_up = get_compile_time_arg_val(92);
    constexpr uint32_t cb_sigmoid = get_compile_time_arg_val(93);
    constexpr uint32_t cb_silu = get_compile_time_arg_val(76);
    constexpr uint32_t cb_gate_saved = get_compile_time_arg_val(89);
    constexpr uint32_t cb_h_product = get_compile_time_arg_val(62);
    constexpr uint32_t cb_da_product = get_compile_time_arg_val(67);
    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t k_tiles = 152;
    constexpr uint32_t da_k_tiles_per_node = 2;
    const uint32_t node =
        get_absolute_logical_y() * 10 + get_absolute_logical_x();
    // Prevent the SFPI/LTO pipeline from cloning the complete exact SFPU
    // sequence once per compile-time tile. These values are invariant at
    // runtime, but volatile keeps the generated program as compact loops.
    volatile uint32_t runtime_k_tiles = k_tiles;
    volatile uint32_t runtime_m_tiles = m_tiles;
    volatile uint32_t runtime_da_tiles = da_k_tiles_per_node;
    if (node < m_tiles) {
        for (uint32_t k = 0; k < runtime_k_tiles; ++k) {
            full_mlp_exact_product_one(
                cb_gate, cb_up, cb_sigmoid, cb_silu, cb_h_product);
        }
    }
    for (uint32_t local_k = 0; local_k < runtime_da_tiles; ++local_k) {
        const uint32_t k = node * da_k_tiles_per_node + local_k;
        if (k >= k_tiles) continue;
        for (uint32_t m = 0; m < runtime_m_tiles; ++m) {
            full_mlp_exact_product_one(
                cb_gate, cb_up, cb_sigmoid, cb_silu, cb_da_product);
        }
    }
    // Restore the FP32 destination mode used by all following reductions.
    enable_fp32_dest_acc();
}

FORCE_INLINE void full_recompute_down_h_phase() {
    constexpr uint32_t cb_product = get_compile_time_arg_val(62);
    constexpr uint32_t cb_a = get_compile_time_arg_val(92);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(76);
    constexpr uint32_t cb_h = get_compile_time_arg_val(89);
    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t k_tiles = 152;
    const uint32_t node =
        get_absolute_logical_y() * 10 + get_absolute_logical_x();
    // Every dA owner (not only the token-row H owners) must consume its
    // gate/up stream and populate CB18 before the common down tail starts.
    if (node >= m_tiles) return;
    CircularBuffer product(cb_product);
    CircularBuffer a(cb_a);
    CircularBuffer partial(cb_partial);
    CircularBuffer h(cb_h);
    compute_kernel_hw_startup<SrcOrder::Reverse>(
        cb_product, cb_a, cb_partial);
    matmul_block_init(cb_product, cb_a, true, 1, 1, 2);
    for (uint32_t block = 0; block < k_tiles / 2; ++block) {
        product.wait_front(2);
        a.wait_front(2);
        tile_regs_acquire();
        if (block > 0) {
            copy_tile_to_dst_init_short_with_dt(cb_a, cb_partial);
            partial.wait_front(1);
            copy_tile(cb_partial, 0, 0);
            partial.pop_front(1);
            reconfig_data_format_srca(cb_partial, cb_a);
        }
        matmul_block_init(cb_product, cb_a, true, 1, 1, 2);
        // Preserve the accepted per-rank dynamic-throttle reset even for
        // the single physical rank tile.
        matmul_block_init(cb_product, cb_a, true, 1, 1, 2);
        for (uint32_t inner = 0; inner < 2; ++inner) {
            matmul_block(
                cb_product, cb_a, inner, inner, 0,
                true, 1, 1, 2);
        }
        tile_regs_commit();
        const bool last = block == k_tiles / 2 - 1;
        if (last) {
            h.reserve_back(1);
        } else {
            partial.reserve_back(1);
        }
        tile_regs_wait();
        const uint32_t target = last ? cb_h : cb_partial;
        PACK((pack_reconfig_data_format(target)));
        pack_tile<true>(0, target, 0);
        tile_regs_release();
        if (last) {
            h.push_back(1);
        } else {
            partial.push_back(1);
        }
        a.pop_front(2);
        product.pop_front(2);
    }
}
FORCE_INLINE void full_down_compute_phase() {
    full_recompute_down_h_phase();
    {
        DeviceZoneScopedN("resident_down_phase_one_compute");
        resident_down_phase_one();
    }
    resident_down_tail();
    CircularBuffer full_down_resident_dh(get_compile_time_arg_val(85));
    full_down_resident_dh.pop_front(4);
}

// Bit-exact LoRA backward arena:
//   rows [0:M]   = dH = BF16(dY * scale) @ B
//   rows [M:M+N] = dB = transpose(H.T @ BF16(dY * scale))
//
// Each phase preserves the reduction order and BF16 boundaries of the
// separately validated native kernels. Python substitutes @...@ constants.



FORCE_INLINE void full_grouped_lora_backward_phase() {
    constexpr uint32_t cb_dh_dy = get_compile_time_arg_val(62);
    constexpr uint32_t cb_dh_b = get_compile_time_arg_val(92);
    constexpr uint32_t cb_dh_scaled = get_compile_time_arg_val(93);
    constexpr uint32_t cb_dh_partial = get_compile_time_arg_val(76);
    constexpr uint32_t cb_dh_output = get_compile_time_arg_val(89);

    constexpr uint32_t cb_db_dy = get_compile_time_arg_val(87);
    constexpr uint32_t cb_db_hidden = get_compile_time_arg_val(81);
    constexpr uint32_t cb_db_scaled = get_compile_time_arg_val(82);
    constexpr uint32_t cb_db_transposed_hidden =
        get_compile_time_arg_val(83);
    constexpr uint32_t cb_db_partial = get_compile_time_arg_val(77);
    constexpr uint32_t cb_db_pretranspose =
        get_compile_time_arg_val(90);
    constexpr uint32_t cb_db_output = get_compile_time_arg_val(84);

    constexpr uint32_t reduction_block_tiles = 2;
    constexpr uint32_t num_n_blocks = 76;
    constexpr uint32_t n_tiles =
        num_n_blocks * reduction_block_tiles;
    constexpr uint32_t m_tiles_per_node = 1;
    constexpr uint32_t total_m_tiles = 4;
    constexpr uint32_t n_tiles_per_node = 4;
    constexpr uint32_t total_n_tiles = 152;
    constexpr uint32_t reduction_tiles = 4;
    constexpr uint32_t reduction_blocks =
        reduction_tiles / reduction_block_tiles;
    constexpr uint32_t grid_cols = 10;
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

    const uint32_t global_node_index =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();
    constexpr uint32_t member_nodes = 50;
    const uint32_t node_index = global_node_index % member_nodes;

    // Phase 1: one complete output-feature reduction per token tile.
    // Reader and compute exchange one exact pre-scaled two-tile reduction
    // block at a time; no long-K intermediate row is resident.
    for (uint32_t local_m = 0; local_m < m_tiles_per_node; ++local_m) {
        const uint32_t m_tile = node_index >= 46
            ? (node_index - 46) * m_tiles_per_node + local_m
            : total_m_tiles;
        if (m_tile >= total_m_tiles) {
            continue;
        }

        compute_kernel_hw_startup<SrcOrder::Reverse>(
            cb_dh_scaled, cb_dh_b, cb_dh_partial);
        matmul_block_init(
            cb_dh_scaled,
            cb_dh_b,
            false,
            1,
            1,
            reduction_block_tiles);

#pragma GCC unroll 1
        for (uint32_t block = 0; block < num_n_blocks; ++block) {
            dh_scaled.wait_front(reduction_block_tiles);
            dh_b.wait_front(reduction_block_tiles);
            tile_regs_acquire();
            if (block > 0) {
                copy_tile_to_dst_init_short_with_dt(
                    cb_dh_b, cb_dh_partial);
                dh_partial.wait_front(1);
                copy_block_matmul_partials(cb_dh_partial, 0, 0, 1);
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
            for (uint32_t inner = 0; inner < reduction_block_tiles; ++inner) {
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
// from kernel_main() into full_grouped_lora_backward_phase(). CBs 0..3 belong to this
// exact frozen-base dX phase; the shifted LoRA phase uses CBs 4..15.

FORCE_INLINE void full_grouped_resident_phase_one() {
    constexpr uint32_t cb_dy = get_compile_time_arg_val(79);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(80);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(78);
    constexpr uint32_t cb_result = get_compile_time_arg_val(71);

    constexpr uint32_t jobs_per_node = 3;
    constexpr uint32_t total_jobs = 112;
    constexpr uint32_t n_tiles = 152;
    constexpr uint32_t reduction_tiles_per_phase =
        1;
    constexpr uint32_t reduction_blocks =
        n_tiles / reduction_tiles_per_phase;
    constexpr uint32_t grid_cols = 10;

    CircularBuffer dy(cb_dy);
    CircularBuffer weight(cb_weight);
    CircularBuffer partial(cb_partial);
    CircularBuffer result(cb_result);

    const uint32_t global_node_index =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();
    constexpr uint32_t member_nodes = 50;
    const uint32_t node_index = global_node_index % member_nodes;

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

    full_grouped_lora_backward_phase();
}

// Exact two-member second phase of frozen Linear + LoRA backward.
//
// All intermediate member sums are deliberately packed to BF16 circular
// buffers in L1. This preserves the accepted PEFT rounding schedule without
// exposing any intermediate tensor to DRAM or the outer autograd graph.



namespace {

ALWI void full_grouped_add_tiles_bf16(
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

ALWI void full_grouped_commit_bf16() {
    MATH((llk_math_dest_section_done<false>()));
}

ALWI void full_grouped_pack_reconfig_bf16(uint32_t cb) {
    PACK((llk_pack_reconfig_data_format<false>(cb)));
}

ALWI void full_grouped_pack_bf16(uint32_t dst, uint32_t cb) {
    PACK((llk_pack<false, false, PackMode::Default>(dst, cb, 0)));
}

ALWI void full_grouped_release_bf16() {
    PACK((llk_pack_dest_section_done<false>()));
}

ALWI void full_grouped_run_lora_dx(
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

ALWI void full_grouped_run_bf16_add(
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
        full_grouped_add_tiles_bf16(cb0, cb1, 0);
        full_grouped_commit_bf16();
        tile_regs_wait();
        full_grouped_pack_reconfig_bf16(cb_output);
        full_grouped_pack_bf16(0, cb_output);
        full_grouped_release_bf16();
        output.push_back(1);
        second.pop_front(1);
        first.pop_front(1);
    }
}

ALWI void full_grouped_run_da(
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
            // Keep dH resident for LoRA dX.
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

// Appended after:
//   1. exact phase-one helpers/kernel rewritten as full_grouped_resident_phase_one()
//   2. grouped-tail helpers with its old kernel rewritten as unused_tail()
//
// The helpers retain the accepted HiFi4/FP32 reduction schedule and explicit
// BF16 packing points.  This file only changes residency and core ownership.


ALWI void full_grouped_run_resident_lora_dx(
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
    if (first_job >= total_jobs) {
        return;
    }
    compute_kernel_hw_startup<SrcOrder::Reverse>(
        cb_dh, cb_a, cb_output);
    matmul_block_init(cb_dh, cb_a, false, 1, 1, 1);
    for (uint32_t local = 0; local < jobs_per_node; ++local) {
        const uint32_t job = first_job + local;
        if (job >= total_jobs) {
            continue;
        }
        const uint32_t m_tile = job / k_tiles;
        a.wait_front(1);
        output.reserve_back(1);
        matmul_block_init(cb_dh, cb_a, false, 1, 1, 1);
        tile_regs_acquire();
        matmul_block(
            cb_dh, cb_a, m_tile, 0, 0, false, 1, 1, 1);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(cb_output)));
        pack_tile<true>(0, cb_output, 0);
        tile_regs_release();
        output.push_back(1);
        a.pop_front(1);
    }
}

FORCE_INLINE void full_grouped_resident_tail() {
    constexpr uint32_t cb_base = get_compile_time_arg_val(71);
    constexpr uint32_t cb_dh = get_compile_time_arg_val(85);
    constexpr uint32_t cb_a = get_compile_time_arg_val(63);
    constexpr uint32_t cb_lora = get_compile_time_arg_val(93);
    constexpr uint32_t cb_peer_sum = get_compile_time_arg_val(88);
    constexpr uint32_t cb_x = get_compile_time_arg_val(87);
    constexpr uint32_t cb_transposed_x = get_compile_time_arg_val(86);
    // Keep producer ownership on the compute RISC across both phases.
    constexpr uint32_t cb_intermediate = get_compile_time_arg_val(82);
    constexpr uint32_t cb_final_dx = get_compile_time_arg_val(83);
    constexpr uint32_t cb_da_partial = get_compile_time_arg_val(77);
    constexpr uint32_t cb_da_pretranspose = get_compile_time_arg_val(90);
    constexpr uint32_t cb_da = get_compile_time_arg_val(84);

    constexpr uint32_t grid_cols = 10;
    constexpr uint32_t member_nodes = 50;
    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t k_tiles = 28;
    constexpr uint32_t jobs_per_node = 3;
    constexpr uint32_t total_jobs = m_tiles * k_tiles;

    const uint32_t global_node =
        get_absolute_logical_y() * grid_cols
        + get_absolute_logical_x();
    const uint32_t member = global_node / member_nodes;
    const uint32_t node = global_node - member * member_nodes;

    // One readiness wait covers both indexed consumers. Neither advances the
    // dH CB front, so dA and LoRA dX see the same four resident tiles.
    CircularBuffer resident_dh(cb_dh);
    resident_dh.wait_front(m_tiles);

    {
        DeviceZoneScopedN("resident_tail_da_compute");
        full_grouped_run_da(
            cb_dh,
            cb_x,
            cb_transposed_x,
            cb_da_partial,
            cb_da_pretranspose,
            cb_da,
            m_tiles,
            1,
            k_tiles,
            node);
    }

    {
        DeviceZoneScopedN("resident_tail_lora_compute");
        full_grouped_run_resident_lora_dx(
            cb_dh,
            cb_a,
            cb_lora,
            jobs_per_node,
            total_jobs,
            k_tiles,
            node);
    }

    // Exactly the accepted staged tree:
    //   s1 = BF16(base1 + lora1)   (member-one half)
    //   s2 = BF16(s1 + lora0)      (member-zero half, after L1 NoC)
    //   dX = BF16(s2 + base0)
    disable_fp32_dest_acc();
    if (member == 1) {
        DeviceZoneScopedN("resident_member1_bf16_add");
        full_grouped_run_bf16_add(
            cb_base,
            cb_lora,
            cb_peer_sum,
            jobs_per_node,
            total_jobs,
            node);
    } else {
        DeviceZoneScopedN("resident_member0_bf16_adds");
        full_grouped_run_bf16_add(
            cb_peer_sum,
            cb_lora,
            cb_intermediate,
            jobs_per_node,
            total_jobs,
            node);
        full_grouped_run_bf16_add(
            cb_intermediate,
            cb_base,
            cb_final_dx,
            jobs_per_node,
            total_jobs,
            node);
    }
    enable_fp32_dest_acc();
}

FORCE_INLINE void full_grouped_compute_phase() {
    {
        DeviceZoneScopedN("resident_phase_one_compute");
        full_grouped_resident_phase_one();
    }
    full_grouped_resident_tail();
}

void run() {
    full_down_compute_phase();
    full_grouped_compute_phase();
}
}  // namespace resident_layer_phase_mlp

namespace resident_layer_phase_attention {
// Exact resident dV -> dP -> softmax backward -> dQ -> dK -> GQA reduce.

#define FP32_DEST_ACC_EN 1


#ifdef TRISC_MATH
namespace ckernel::sfpu {
// TTNN's BF16 FPU subtraction rounds its only possible halfway case away
// from zero.  With BF16 operands, subtraction is either already BF16 or has
// exactly one discarded bit, so add 0x8000 to the magnitude field and mask.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_bf16_sub_round() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; ++d) {
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPIADD(
            0, p_sfpu::LREG13, p_sfpu::LREG1,
            sfpi::SFPIADD_MOD1_CC_NONE);  // +0x7fff
        TTI_SFPIADD(
            0, p_sfpu::LREG12, p_sfpu::LREG1,
            sfpi::SFPIADD_MOD1_CC_NONE);  // +1 => +0x8000
        TTI_SFPAND(0, p_sfpu::LREG14, p_sfpu::LREG1, 0);
        TTI_SFPSTORE(
            p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
    }
}
}  // namespace ckernel::sfpu
#endif

namespace {

constexpr bool kBf16 = 1;

// Kept byte-for-byte structurally equivalent to the separately qualified
// attention_dv_reference primitive.  Small-looking loop/signature changes
// can alter Tensix instruction scheduling, so the resident owner calls this
// exact implementation for its BF16 dV edge.
FORCE_INLINE void transpose_probability_column_exact(
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

FORCE_INLINE void matmul_dv_exact(
    uint32_t lhs_id, uint32_t rhs_id, uint32_t output_id,
    uint32_t partial_id, uint32_t n_tiles, uint32_t k_tiles,
    uint32_t phase_tiles, uint32_t out_subblock_tiles,
    uint32_t m_tiles) {
    CircularBuffer lhs(lhs_id), rhs(rhs_id), output(output_id), partial(partial_id);
    lhs.wait_front(m_tiles * k_tiles);
    rhs.wait_front(k_tiles * n_tiles);
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

FORCE_INLINE void transpose_tiles(
    uint32_t input_id, uint32_t output_id, uint32_t rows, uint32_t cols) {
    CircularBuffer input(input_id), output(output_id);
    const uint32_t tiles = rows * cols;
    input.wait_front(tiles);
    output.reserve_back(tiles);
    compute_kernel_hw_startup(input_id, output_id);
    transpose_init(input_id);
    for (uint32_t start = 0; start < tiles; start += 4) {
        const uint32_t chunk = (tiles - start) < 4 ? tiles - start : 4;
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
        const uint32_t chunk = (tiles - start) < 4 ? tiles - start : 4;
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

FORCE_INLINE void transpose_strided_column(
    uint32_t input_id, uint32_t output_id, uint32_t tiles,
    uint32_t stride, uint32_t offset) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(tiles * stride);
    output.reserve_back(tiles);
    compute_kernel_hw_startup(input_id, output_id);
    transpose_init(input_id);
    for (uint32_t start = 0; start < tiles; start += 4) {
        const uint32_t chunk = (tiles - start) < 4 ? tiles - start : 4;
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < chunk; ++tile)
            transpose_tile(
                input_id, (start + tile) * stride + offset, tile);
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
    uint32_t n_subblock_tiles, uint32_t phase_tiles,
    bool startup_on_output = false) {
    CircularBuffer lhs(lhs_id), rhs(rhs_id), output(output_id), partial(partial_id);
    lhs.wait_front(k_tiles);
    rhs.wait_front(k_tiles * n_tiles);
    compute_kernel_hw_startup<SrcOrder::Reverse>(
        lhs_id, rhs_id, startup_on_output ? output_id : partial_id);
    for (uint32_t n_start = 0; n_start < n_tiles; n_start += n_subblock_tiles) {
        matmul_block_init(lhs_id, rhs_id, false, n_subblock_tiles, 1, 1);
        const uint32_t phases = k_tiles / phase_tiles;
        for (uint32_t phase = 0; phase < phases; ++phase) {
            tile_regs_acquire();
            if (phase > 0) {
                copy_tile_to_dst_init_short_with_dt(rhs_id, partial_id);
                partial.wait_front(n_subblock_tiles);
                copy_block_matmul_partials(partial_id, 0, 0, n_subblock_tiles);
                partial.pop_front(n_subblock_tiles);
                reconfig_data_format_srca(partial_id, rhs_id);
            }
            matmul_block_init(
                lhs_id, rhs_id, false, n_subblock_tiles, 1, phase_tiles);
            for (uint32_t inner = 0; inner < phase_tiles; ++inner) {
                const uint32_t k = phase * phase_tiles + inner;
                matmul_block(
                    lhs_id, rhs_id, k, k * n_tiles + n_start, 0,
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

// Exact shape used by the real autograd graph for dK: Q.T @ dS, followed by
// an output transpose.  Keeping this operand order matters at the FP32 bit
// level even though it is algebraically equivalent to dS.T @ Q.
FORCE_INLINE void matmul_column(
    uint32_t lhs_id, uint32_t rhs_id, uint32_t output_id,
    uint32_t partial_id, uint32_t m_tiles, uint32_t k_tiles,
    uint32_t phase_tiles) {
    CircularBuffer lhs(lhs_id), rhs(rhs_id), output(output_id), partial(partial_id);
    lhs.wait_front(m_tiles * k_tiles);
    rhs.wait_front(k_tiles);
    compute_kernel_hw_startup<SrcOrder::Reverse>(lhs_id, rhs_id, output_id);
    for (uint32_t m = 0; m < m_tiles; ++m) {
        const uint32_t phases = k_tiles / phase_tiles;
        for (uint32_t phase = 0; phase < phases; ++phase) {
            tile_regs_acquire();
            if (phase > 0) {
                copy_tile_to_dst_init_short_with_dt(rhs_id, partial_id);
                partial.wait_front(1);
                copy_block_matmul_partials(partial_id, 0, 0, 1);
                partial.pop_front(1);
                reconfig_data_format_srca(partial_id, rhs_id);
            }
            matmul_block_init(lhs_id, rhs_id, false, 1, 1, phase_tiles);
            for (uint32_t inner = 0; inner < phase_tiles; ++inner) {
                const uint32_t k = phase * phase_tiles + inner;
                matmul_block(
                    lhs_id, rhs_id, m * k_tiles + k, k, 0,
                    false, 1, 1, phase_tiles);
            }
            tile_regs_commit();
            const bool last = phase + 1 == phases;
            if (last) output.reserve_back(1);
            else partial.reserve_back(1);
            tile_regs_wait();
            const uint32_t target = last ? output_id : partial_id;
            PACK((pack_reconfig_data_format(target)));
            pack_tile_block(0, target, 1);
            tile_regs_release();
            if (last) output.push_back(1);
            else partial.push_back(1);
        }
    }
}

// Reproduce bmm_large_block_zm's IN0_TRANSPOSE_TILE path for a row worker.
// TTNN transposes one input tile immediately before the matching K block,
// then spills the FP32 partial before advancing to the next tile.  Performing
// all transposes up front leaves identical BF16 tiles in L1 but a different
// observable Tensix configuration/reduction schedule.
FORCE_INLINE void matmul_row_streamed_in0_transpose(
    uint32_t original_lhs_id, uint32_t transposed_lhs_id,
    uint32_t rhs_id, uint32_t output_id, uint32_t partial_id,
    uint32_t n_tiles, uint32_t k_tiles) {
    CircularBuffer original_lhs(original_lhs_id);
    CircularBuffer transposed_lhs(transposed_lhs_id);
    CircularBuffer rhs(rhs_id), output(output_id), partial(partial_id);
    original_lhs.wait_front(k_tiles);
    rhs.wait_front(k_tiles * n_tiles);
    compute_kernel_hw_startup<SrcOrder::Reverse>(
        transposed_lhs_id, rhs_id, partial_id);

    for (uint32_t n = 0; n < n_tiles; ++n) {
        bool reload = false;
        for (uint32_t block = 0; block < k_tiles; ++block) {
            reconfig_data_format_srca(rhs_id, original_lhs_id);
            transpose_init(original_lhs_id);
            PACK((pack_reconfig_data_format(transposed_lhs_id)));
            transposed_lhs.reserve_back(1);
            tile_regs_acquire();
            transpose_tile(original_lhs_id, block, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, transposed_lhs_id);
            tile_regs_release();
            transposed_lhs.push_back(1);

            reconfig_data_format_srca(original_lhs_id, rhs_id);
            matmul_block_init(
                transposed_lhs_id, rhs_id, false, 1, 1, 1);
            PACK((pack_reconfig_data_format(partial_id)));
            transposed_lhs.wait_front(1);
            tile_regs_acquire();
            if (reload) {
                copy_tile_to_dst_init_short_with_dt(rhs_id, partial_id);
                partial.wait_front(1);
                copy_block_matmul_partials(partial_id, 0, 0, 1);
                partial.pop_front(1);
                reconfig_data_format_srca(partial_id, rhs_id);
                matmul_block_init(
                    transposed_lhs_id, rhs_id, false, 1, 1, 1);
            }
            matmul_block(
                transposed_lhs_id, rhs_id, 0,
                block * n_tiles + n, 0, false, 1, 1, 1);
            tile_regs_commit();
            const bool last = block + 1 == k_tiles;
            if (last) output.reserve_back(1);
            else partial.reserve_back(1);
            tile_regs_wait();
            const uint32_t target = last ? output_id : partial_id;
            PACK((pack_reconfig_data_format(target)));
            pack_tile_block(0, target, 1);
            tile_regs_release();
            if (last) output.push_back(1);
            else partial.push_back(1);
            transposed_lhs.pop_front(1);
            reload = !last;
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
        if constexpr (kBf16) {
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
    input.pop_front(tiles);
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

FORCE_INLINE void repeat_first_tile_preserve(
    uint32_t input_id, uint32_t output_id, uint32_t copies) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(1);
    output.reserve_back(copies);
    compute_kernel_hw_startup(input_id, output_id);
    for (uint32_t copy = 0; copy < copies; ++copy) {
        tile_regs_acquire();
        copy_tile_init(input_id);
        copy_tile(input_id, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, copy);
        tile_regs_release();
    }
    output.push_back(copies);
}

template <uint32_t Op, bool PackBf16 = kBf16>
FORCE_INLINE void sfpu_binary_tile_to_cb(
    uint32_t left_id, uint32_t right_id, uint32_t output_id,
    uint32_t left_tile, uint32_t right_tile) {
    CircularBuffer left(left_id), right(right_id), output(output_id);
    left.wait_front(left_tile + 1);
    right.wait_front(right_tile + 1);
    output.reserve_back(1);
    unary_op_init_common(left_id, output_id);
    if constexpr (Op == 0) mul_binary_tile_init();
    else if constexpr (Op == 1) sub_binary_tile_init();
    else add_binary_tile_init();
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(right_id, left_id);
    copy_tile(left_id, left_tile, 0);
    copy_tile_to_dst_init_short_with_dt(left_id, right_id);
    copy_tile(right_id, right_tile, 1);
    if constexpr (Op == 0) mul_binary_tile(0, 1, 0);
    else if constexpr (Op == 1) sub_binary_tile(0, 1, 0);
    else add_binary_tile(0, 1, 0);
    if constexpr (PackBf16) {
        // A combined program must keep FP32 destination accumulation for its
        // matmuls.  TTNN's standalone BF16 binary kernels instead use the
        // 16-bit destination path, whose halfway conversion is RNE.  Apply
        // that conversion explicitly before packing so both contracts can
        // coexist in one compute kernel.
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
    pack_tile(0, output_id);
    tile_regs_release();
    output.push_back(1);
}

FORCE_INLINE void fp32_to_bf16_tiles(
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

#if !0
// Mirror binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp.
// TTNN processes the four score tiles owned by a row worker in one DST
// acquire/commit: lhs tiles occupy even registers and rhs tiles odd registers.
// That register schedule is numerically observable in BF16.
FORCE_INLINE void sfpu_multiply_block_to_cb(
    uint32_t left_id, uint32_t right_id, uint32_t output_id,
    uint32_t tiles) {
    CircularBuffer left(left_id), right(right_id), output(output_id);
    left.wait_front(tiles);
    right.wait_front(tiles);
    output.reserve_back(tiles);
    unary_op_init_common(left_id, output_id);
    mul_binary_tile_init();
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(right_id, left_id);
    for (uint32_t tile = 0; tile < tiles; ++tile)
        copy_tile(left_id, tile, tile * 2);
    copy_tile_to_dst_init_short_with_dt(left_id, right_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        copy_tile(right_id, tile, tile * 2 + 1);
        mul_binary_tile(tile * 2, tile * 2 + 1, tile * 2);
    }
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(output_id)));
    for (uint32_t tile = 0; tile < tiles; ++tile)
        pack_tile(tile * 2, output_id);
    tile_regs_release();
    output.push_back(tiles);
}

FORCE_INLINE void fpu_add_tile_to_cb(
    uint32_t left_id, uint32_t right_id, uint32_t output_id,
    uint32_t left_tile, uint32_t right_tile) {
    CircularBuffer left(left_id), right(right_id), output(output_id);
    left.wait_front(left_tile + 1);
    right.wait_front(right_tile + 1);
    output.reserve_back(1);
    binary_op_init_common(left_id, right_id, output_id);
    tile_regs_acquire();
    add_tiles_init(left_id, right_id);
    add_tiles(left_id, right_id, left_tile, right_tile, 0);
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(output_id)));
    pack_tile<true>(0, output_id, 0);
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

// Match binary_ng's broadcast-aware FP32 subtract without materializing the
// row-sum broadcast.  Keeping the broadcast value in DST is numerically
// observable: an FP32 L1 round trip changes BF16 boundary values.
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

template <uint32_t packed_rows_id, uint32_t scaler_id, uint32_t reduced_rows_id>
FORCE_INLINE void exact_gqa_h_reduce(uint32_t tiles) {
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        compute_kernel_lib::reduce<
            PoolType::SUM, ReduceDim::REDUCE_COL,
            packed_rows_id, scaler_id, reduced_rows_id,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT,
            ReduceFp32Mode::Fast>(
                compute_kernel_lib::ReduceInputBlockShape::single(),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{});
    }
}

FORCE_INLINE void fpu_add_tile_to_fp32_cb(
    uint32_t left_id, uint32_t right_id, uint32_t output_id,
    uint32_t left_tile, uint32_t right_tile) {
    CircularBuffer left(left_id), right(right_id), output(output_id);
    output.reserve_back(1);
    binary_op_init_common(left_id, right_id, output_id);
    add_tiles_init(left_id, right_id, false);
    tile_regs_acquire();
    add_tiles(left_id, right_id, left_tile, right_tile, 0);
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(output_id)));
    pack_tile<true>(0, output_id, 0);
    tile_regs_release();
    output.push_back(1);
}

FORCE_INLINE void copy_tile_to_fp32_cb(
    uint32_t input_id, uint32_t output_id, uint32_t input_tile) {
    CircularBuffer input(input_id), output(output_id);
    output.reserve_back(1);
    compute_kernel_hw_startup(input_id, output_id);
    tile_regs_acquire();
    copy_tile_init(input_id);
    copy_tile(input_id, input_tile, 0);
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(output_id)));
    pack_tile<true>(0, output_id, 0);
    tile_regs_release();
    output.push_back(1);
}

FORCE_INLINE void rne_fp32_tile_to_bf16_cb(
    uint32_t input_id, uint32_t output_id) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(1);
    output.reserve_back(1);
    init_sfpu(input_id, output_id);
    tile_regs_acquire();
    copy_tile_init(input_id);
    copy_tile(input_id, 0, 0);
    typecast_tile_init<
        (uint32_t)DataFormat::Float32,
        (uint32_t)DataFormat::Float16_b>();
    typecast_tile<
        (uint32_t)DataFormat::Float32,
        (uint32_t)DataFormat::Float16_b>(0);
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(output_id)));
    pack_tile<true>(0, output_id, 0);
    tile_regs_release();
    output.push_back(1);
}

template <uint32_t input_id, uint32_t output_id>
FORCE_INLINE void diagnostic_gqa_passthrough(uint32_t outputs, uint32_t group) {
    CircularBuffer input(input_id), output(output_id);
    compute_kernel_hw_startup(input_id, output_id);
    for (uint32_t output_tile = 0; output_tile < outputs; ++output_tile) {
        input.wait_front(group);
        output.reserve_back(1);
        tile_regs_acquire();
        copy_tile_init(input_id);
        copy_tile(input_id, 0, 0);
        input.pop_front(group);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, 0);
        tile_regs_release();
        output.push_back(1);
    }
}

FORCE_INLINE void packed_bf16_gqa_sum(
    uint32_t input_id, uint32_t scratch_a_id, uint32_t scratch_b_id,
    uint32_t output_id, uint32_t outputs, uint32_t group) {
    CircularBuffer input(input_id);
    for (uint32_t output_tile = 0; output_tile < outputs; ++output_tile) {
        input.wait_front(group);
        sfpu_binary_tile_to_cb<2>(
            input_id, input_id, scratch_a_id, 0, 1);
        uint32_t source = scratch_a_id;
        uint32_t target = scratch_b_id;
        for (uint32_t member = 2; member < group; ++member) {
            const bool last = member + 1 == group;
            const uint32_t destination = last ? output_id : target;
            sfpu_binary_tile_to_cb<2>(
                source, input_id, destination, 0, member);
            CircularBuffer source_cb(source);
            source_cb.pop_front(1);
            const uint32_t old_source = source;
            source = target;
            target = old_source;
        }
        input.pop_front(group);
    }
}
#endif

}  // namespace

void run() {
    DeviceZoneScopedN("attention_resident_compute");
    constexpr uint32_t cb_p_row = get_compile_time_arg_val(72);
    constexpr uint32_t cb_p_column = get_compile_time_arg_val(74);
    constexpr uint32_t cb_dout_row = get_compile_time_arg_val(88);
    constexpr uint32_t cb_dout_full = get_compile_time_arg_val(67);
    constexpr uint32_t cb_q_full = get_compile_time_arg_val(68);
    constexpr uint32_t cb_k = get_compile_time_arg_val(69);
    constexpr uint32_t cb_v = get_compile_time_arg_val(70);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(101);
    constexpr uint32_t cb_pt = get_compile_time_arg_val(85);
    constexpr uint32_t cb_dv = get_compile_time_arg_val(98);
    constexpr uint32_t cb_vt = get_compile_time_arg_val(65);
    constexpr uint32_t cb_dp = get_compile_time_arg_val(79);
    constexpr uint32_t cb_product = get_compile_time_arg_val(75);
    constexpr uint32_t cb_rowsum = get_compile_time_arg_val(99);
    constexpr uint32_t cb_centered = get_compile_time_arg_val(76);
    constexpr uint32_t cb_ds = get_compile_time_arg_val(77);
    constexpr uint32_t cb_ds_scaled = get_compile_time_arg_val(80);
    constexpr uint32_t cb_dq = get_compile_time_arg_val(86);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(78);
    constexpr uint32_t cb_ds_handoff = get_compile_time_arg_val(81);
    constexpr uint32_t cb_ds_column = get_compile_time_arg_val(82);
    constexpr uint32_t cb_ds_t = get_compile_time_arg_val(66);
    constexpr uint32_t cb_dk = get_compile_time_arg_val(87);
    constexpr uint32_t cb_peer_dk = get_compile_time_arg_val(63);
    constexpr uint32_t cb_peer_dv = get_compile_time_arg_val(64);
    constexpr uint32_t cb_sum_dk_a = get_compile_time_arg_val(62);
    constexpr uint32_t cb_sum_dk_b = get_compile_time_arg_val(83);
    constexpr uint32_t cb_sum_dv_a = get_compile_time_arg_val(100);
    constexpr uint32_t cb_sum_dv_b = get_compile_time_arg_val(89);
    constexpr uint32_t cb_final_dk = get_compile_time_arg_val(90);
    constexpr uint32_t cb_final_dv = get_compile_time_arg_val(91);
    constexpr uint32_t cb_p_bf16 = get_compile_time_arg_val(84);
    constexpr uint32_t cb_rope_cos = get_compile_time_arg_val(92);
    constexpr uint32_t cb_rope_sin = get_compile_time_arg_val(93);
    constexpr uint32_t cb_rope_cos_product = get_compile_time_arg_val(94);
    constexpr uint32_t cb_rope_sin_product = get_compile_time_arg_val(95);
    constexpr uint32_t cb_rope_q = get_compile_time_arg_val(96);
    constexpr uint32_t cb_rope_k = get_compile_time_arg_val(97);
    constexpr uint32_t seq_tiles = 4;
    constexpr uint32_t dim_tiles = 2;
    constexpr uint32_t query_heads = 14;
    constexpr uint32_t kv_heads = 2;
    constexpr uint32_t group = query_heads / kv_heads;
    constexpr uint32_t jobs = query_heads * seq_tiles;
    constexpr uint32_t reducer_jobs = kv_heads * seq_tiles;
    constexpr uint32_t vector_tiles = seq_tiles * dim_tiles;
    constexpr uint32_t scale_bits = 1040187392;
    constexpr uint32_t matmul_phase_tiles = 1;
    constexpr bool diagnostic_ds = 0;
    constexpr bool diagnostic_dv = 0;
    constexpr bool diagnostic_pt = 0;
    constexpr bool diagnostic_dq = 0;
    constexpr bool diagnostic_dk = 0;
    constexpr bool diagnostic_dp = 0;
    constexpr bool diagnostic_product = 0;
    constexpr bool diagnostic_centered = 0;
    constexpr bool diagnostic_rowsum = 0;
    constexpr bool diagnostic_unscaled_ds = 0;
    constexpr bool diagnostic_dout_handoff = 0;
    constexpr bool probability_bf16_input = 0;

    const uint32_t node =
        get_absolute_logical_y() * 6 + (get_absolute_logical_x() - 5);
    if (node >= jobs) return;
    const uint32_t head = node / seq_tiles;
    const uint32_t member = head % group;
    CircularBuffer p_row(cb_p_row), p_column(cb_p_column);
    CircularBuffer dout_row(cb_dout_row), dout_full(cb_dout_full);
    CircularBuffer q_full(cb_q_full), k(cb_k), v(cb_v), pt(cb_pt), vt(cb_vt);
    CircularBuffer dp(cb_dp), rowsum(cb_rowsum), centered(cb_centered);
    CircularBuffer ds(cb_ds), ds_column(cb_ds_column), ds_t(cb_ds_t);

    if constexpr (diagnostic_dout_handoff) {
        copy_tiles_preserve(cb_dout_row, cb_dv, dim_tiles);
        return;
    }

    // Match eager autograd's execution order as well as its individual
    // matmul schedules: dV is the first backward dot, followed by dP.  The
    // order is numerically observable because dV's qualified TTNN schedule
    // starts from a fresh matmul-engine state.
    if constexpr (kBf16) {
        if constexpr (!probability_bf16_input) {
            fp32_to_bf16_tiles(cb_p_column, cb_p_bf16, seq_tiles);
            // The framework executes the FP32->BF16 cast and BMM as separate
            // programs. Re-arm FP32 destination mode here so the resident
            // BMM starts from the same hardware mode boundary.
            disable_fp32_dest_acc();
            enable_fp32_dest_acc();
        }
        constexpr uint32_t packed_p_id =
            probability_bf16_input ? cb_p_column : cb_p_bf16;
        if constexpr (diagnostic_pt) {
            transpose_tiles(packed_p_id, cb_pt, 1, seq_tiles);
            copy_tiles_preserve(cb_pt, cb_dv, dim_tiles);
            return;
        }
        transpose_probability_column_exact(packed_p_id, cb_pt, seq_tiles);
        matmul_dv_exact(
            cb_pt, cb_dout_full, cb_dv, cb_partial,
            dim_tiles, seq_tiles, 1, dim_tiles, 1);
    } else {
        transpose_tiles(cb_p_column, cb_pt, 1, seq_tiles);
        matmul_row(
            cb_pt, cb_dout_full, cb_dv, cb_partial,
            dim_tiles, seq_tiles, dim_tiles, matmul_phase_tiles);
        pt.pop_front(seq_tiles);
    }
    WAYPOINT("ADV");
    if constexpr (diagnostic_dv) return;

    transpose_linear(cb_v, cb_vt, vector_tiles);
    matmul_row(
        cb_dout_row, cb_vt, cb_dp, cb_partial,
        seq_tiles, dim_tiles, 4,
        matmul_phase_tiles == 1 ? 1 : dim_tiles);
    vt.pop_front(vector_tiles);
    if constexpr (diagnostic_dp) {
        copy_tiles_preserve(cb_dp, cb_ds_handoff, seq_tiles);
        dp.pop_front(seq_tiles);
        return;
    }

    p_row.wait_front(seq_tiles);
    dp.wait_front(seq_tiles);
    // StableHLO has an observable BF16 -> FP32 conversion between dP and the
    // softmax-backward multiply. CB18 must retain TTNN's default TF32 matmul
    // partial-reload contract, so stage full-FP32 dP in CB15. CB15 is empty
    // until dS is produced and can be reused after this staging value is
    // consumed.
    copy_tiles_preserve(cb_dp, cb_ds, seq_tiles);
    for (uint32_t tile = 0; tile < seq_tiles; ++tile)
        sfpu_binary_tile_to_cb<0, false>(
            cb_p_row, cb_ds, cb_product, tile, tile);
    if constexpr (diagnostic_product) {
        fp32_to_bf16_tiles(cb_product, cb_ds_handoff, seq_tiles);
        CircularBuffer product(cb_product);
        CircularBuffer dp_fp32(cb_ds);
        product.pop_front(seq_tiles);
        dp_fp32.pop_front(seq_tiles);
        dp.pop_front(seq_tiles);
        p_row.pop_front(seq_tiles);
        return;
    }
#if !0
    compute_kernel_lib::reduce<
        PoolType::SUM, ReduceDim::REDUCE_ROW,
        cb_product, cb_scaler, cb_rowsum,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT,
        ReduceFp32Mode::Fast>(
            compute_kernel_lib::ReduceInputBlockShape::row(seq_tiles));
    // The scaler is produced once per invocation.  The reduction helper
    // treats it as reusable and therefore does not advance its CB.  This
    // owner has exactly one row reduction, so release it now; otherwise stale
    // scaler pages accumulate and the fifth invocation blocks its reader
    // before Q/K/V are loaded.
    CircularBuffer row_reduce_scaler(cb_scaler);
    row_reduce_scaler.pop_front(1);
    rowsum.wait_front(1);
    if constexpr (diagnostic_rowsum) {
        repeat_first_tile_preserve(cb_rowsum, cb_ds_handoff, seq_tiles);
        CircularBuffer dp_fp32(cb_ds);
        dp_fp32.pop_front(seq_tiles);
        rowsum.pop_front(1);
        dp.pop_front(seq_tiles);
        p_row.pop_front(seq_tiles);
        return;
    }
    for (uint32_t tile = 0; tile < seq_tiles; ++tile)
        sfpu_sub_row_sum_to_cb(
            cb_ds, cb_rowsum, cb_centered, tile);
    CircularBuffer dp_fp32(cb_ds);
    dp_fp32.pop_front(seq_tiles);
    if constexpr (diagnostic_centered) {
        fp32_to_bf16_tiles(cb_centered, cb_ds_handoff, seq_tiles);
        centered.pop_front(seq_tiles);
        rowsum.pop_front(1);
        dp.pop_front(seq_tiles);
        p_row.pop_front(seq_tiles);
        return;
    }
    for (uint32_t tile = 0; tile < seq_tiles; ++tile)
        sfpu_binary_tile_to_cb<0, false>(
            cb_p_row, cb_centered, cb_ds, tile, tile);
    centered.pop_front(seq_tiles);
    if constexpr (diagnostic_unscaled_ds) {
        fp32_to_bf16_tiles(cb_ds, cb_ds_handoff, seq_tiles);
        CircularBuffer diagnostic_ds(cb_ds);
        diagnostic_ds.pop_front(seq_tiles);
        rowsum.pop_front(1);
        dp.pop_front(seq_tiles);
        p_row.pop_front(seq_tiles);
        return;
    }
    rowsum.pop_front(1);
    dp.pop_front(seq_tiles);
    p_row.pop_front(seq_tiles);
    // The real autograd graph applies score scale once to dS before both dQ
    // and dK. This is the observable TTNN boundary qualified by the row op.
    if constexpr (kBf16) {
        fp32_to_bf16_tiles(cb_ds, cb_p_bf16, seq_tiles);
        CircularBuffer ds_fp32(cb_ds);
        ds_fp32.pop_front(seq_tiles);
        scale_tiles(cb_p_bf16, cb_ds_scaled, seq_tiles, scale_bits);
    } else {
        scale_tiles(cb_ds, cb_ds_scaled, seq_tiles, scale_bits);
    }
    copy_tiles_preserve(cb_ds_scaled, cb_ds_handoff, seq_tiles);
    WAYPOINT("ADS");
    if constexpr (diagnostic_ds) return;
    matmul_row(
        cb_ds_scaled, cb_k, cb_dq, cb_partial,
        dim_tiles, seq_tiles, dim_tiles, matmul_phase_tiles);
    CircularBuffer ds_scaled(cb_ds_scaled);
    ds_scaled.pop_front(seq_tiles);
    WAYPOINT("ADQ");
    if constexpr (diagnostic_dq) return;

    // Match eager-autograd StableHLO exactly: Q.T @ scaled_dS followed by an
    // output transpose.  The algebraically equivalent dS.T @ Q schedule has
    // a different BF16 reduction tree and is therefore not interchangeable.
    for (uint32_t dim = 0; dim < dim_tiles; ++dim) {
        transpose_strided_column(
            cb_q_full, cb_ds_t, seq_tiles, dim_tiles, dim);
        matmul_column(
            cb_ds_t, cb_ds_column, cb_sum_dv_b, cb_partial,
            1, seq_tiles, matmul_phase_tiles);
        ds_t.pop_front(seq_tiles);
    }
    transpose_linear(cb_sum_dv_b, cb_dk, dim_tiles);
    CircularBuffer dk_pretranspose(cb_sum_dv_b);
    dk_pretranspose.pop_front(dim_tiles);
    ds_column.pop_front(seq_tiles);
    WAYPOINT("ADK");

    if constexpr (diagnostic_dk) return;

    if (member == 0) {
        // The reader packs 32 logical sequence rows for each dK tile and then
        // 32 rows for each dV tile.  Each input is exactly the single H tile
        // produced by TTNN's transpose(group, H); the writer performs the
        // inverse row scatter entirely in local L1.
        if constexpr (!kBf16) {
            exact_gqa_h_reduce<cb_sum_dk_a, cb_sum_dv_a, cb_sum_dk_b>(
                2 * dim_tiles * 32);
            CircularBuffer local_dk(cb_dk), local_dv(cb_dv);
            CircularBuffer peer_dk(cb_peer_dk), peer_dv(cb_peer_dv);
            local_dk.pop_front(dim_tiles);
            local_dv.pop_front(dim_tiles);
            peer_dk.pop_front((group - 1) * dim_tiles);
            peer_dv.pop_front((group - 1) * dim_tiles);
        } else {
            // TTNN's reduce over the GQA member dimension accumulates all
            // members in one FP32 destination register and rounds once at
            // the final BF16 pack.  Pairwise packed BF16 additions introduce
            // an extra rounding boundary and are observably nonexact.
            CircularBuffer gathered(cb_sum_dk_a);
            CircularBuffer zero(cb_sum_dv_a);
            CircularBuffer reduced(cb_sum_dk_b);
            zero.wait_front(1);
            WAYPOINT("AG0");
            // The standalone exact reducer begins from a fresh binary
            // compute configuration.  This owner reaches the same phase
            // after a transpose/matmul pipeline, so explicitly restart the
            // unpack/math/pack configuration before reusing DST register 0.
            compute_kernel_hw_startup(
                cb_sum_dk_a, cb_sum_dv_a, cb_sum_dk_b);
            binary_op_init_common(cb_sum_dk_a, cb_sum_dv_a, cb_sum_dk_b);
            for (uint32_t output_tile = 0;
                 output_tile < 2 * dim_tiles; ++output_tile) {
                gathered.wait_front(group);
                WAYPOINT("AG1");
                copy_tile_to_dst_init_short_with_dt(
                    cb_sum_dv_a, cb_sum_dk_a);
                tile_regs_acquire();
                copy_tile(cb_sum_dk_a, 0, 0);
                reconfig_data_format_srcb(cb_sum_dk_a, cb_sum_dv_a);
                add_tiles_init(cb_sum_dk_a, cb_sum_dv_a, true);
                for (uint32_t source_member = 1;
                     source_member < group; ++source_member) {
                    add_tiles(
                        cb_sum_dk_a, cb_sum_dv_a,
                        source_member, 0, 0);
                }
                gathered.pop_front(group);
                tile_regs_commit();
                reduced.reserve_back(1);
                tile_regs_wait();
                PACK((pack_reconfig_data_format(cb_sum_dk_b)));
                pack_tile(0, cb_sum_dk_b);
                tile_regs_release();
                reduced.push_back(1);
                WAYPOINT("AG2");
            }
            zero.pop_front(1);
        }
    }
    WAYPOINT("AGD");

    // Inverse RoPE remains inside the attention island. Match the framework's
    // two separately packed BF16 products before the final add/sub boundary.
    CircularBuffer rope_cos(cb_rope_cos), rope_sin(cb_rope_sin);
    CircularBuffer dq_gradient(cb_dq), rope_q(cb_rope_q);
    compute_kernel_hw_startup(cb_dq, cb_rope_cos, cb_rope_cos_product);
    for (uint32_t dim = 0; dim < dim_tiles; ++dim) {
        const uint32_t peer = dim ^ 1u;
        sfpu_binary_tile_to_cb<0, true>(
            cb_dq, cb_rope_cos, cb_rope_cos_product, dim, dim);
        sfpu_binary_tile_to_cb<0, true>(
            cb_dq, cb_rope_sin, cb_rope_sin_product, peer, peer);
        if (dim == 0)
            fpu_add_tile_to_cb(
                cb_rope_cos_product, cb_rope_sin_product, cb_rope_q, 0, 0);
        else
            fpu_sub_tile_to_cb(
                cb_rope_cos_product, cb_rope_sin_product, cb_rope_q, 0);
        CircularBuffer cos_product(cb_rope_cos_product);
        CircularBuffer sin_product(cb_rope_sin_product);
        cos_product.pop_front(1);
        sin_product.pop_front(1);
    }
    dq_gradient.pop_front(dim_tiles);

    if (member == 0) {
        CircularBuffer reduced(cb_sum_dk_b), rope_k(cb_rope_k);
        compute_kernel_hw_startup(
            cb_sum_dk_b, cb_rope_cos, cb_rope_cos_product);
        for (uint32_t dim = 0; dim < dim_tiles; ++dim) {
            const uint32_t peer = dim ^ 1u;
            sfpu_binary_tile_to_cb<0, true>(
                cb_sum_dk_b, cb_rope_cos, cb_rope_cos_product, dim, dim);
            sfpu_binary_tile_to_cb<0, true>(
                cb_sum_dk_b, cb_rope_sin, cb_rope_sin_product, peer, peer);
            if (dim == 0)
                fpu_add_tile_to_cb(
                    cb_rope_cos_product, cb_rope_sin_product, cb_rope_k, 0, 0);
            else
                fpu_sub_tile_to_cb(
                    cb_rope_cos_product, cb_rope_sin_product, cb_rope_k, 0);
            CircularBuffer cos_product(cb_rope_cos_product);
            CircularBuffer sin_product(cb_rope_sin_product);
            cos_product.pop_front(1);
            sin_product.pop_front(1);
        }
    }
    rope_cos.pop_front(dim_tiles);
    rope_sin.pop_front(dim_tiles);

    p_column.pop_front(seq_tiles);
    dout_row.pop_front(dim_tiles);
    dout_full.pop_front(vector_tiles);
    q_full.pop_front(vector_tiles);
    k.pop_front(vector_tiles);
    v.pop_front(vector_tiles);
#endif
}
}  // namespace resident_layer_phase_attention

void kernel_main() {
    resident_layer_phase_mlp::run();
    CircularBuffer layer_compute_release(get_compile_time_arg_val(102));
    layer_compute_release.wait_front(1);
    layer_compute_release.pop_front(1);
    resident_layer_phase_attention::run();
}
