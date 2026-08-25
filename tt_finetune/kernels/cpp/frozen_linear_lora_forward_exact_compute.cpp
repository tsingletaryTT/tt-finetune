// Exact N=1-tile frozen Linear + LoRA forward for Blackhole.
//
// Per (M tile, N tile) job:
//   H      = X @ A.T
//   base   = X @ W.T
//   delta  = H @ B.T
//   output = base + BF16(delta * scale)
//
// Both long-K reductions use two-tile FP32 spill/reload phases. Python
// substitutes @...@ compile-time constants before TT-Metal JIT compilation.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

ALWI void add_tiles_bf16(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t itile0,
    uint32_t itile1,
    uint32_t idst
) {
    UNPACK((llk_unpack_AB(icb0, icb1, itile0, itile1)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWADD,
          BroadcastType::NONE,
          false,
          MathFidelity::LoFi,
          EltwiseBinaryReuseDestType::NONE>(
        icb0, icb1, idst, false)));
}

ALWI void tile_regs_commit_bf16() {
    MATH((llk_math_dest_section_done<false>()));
}

ALWI void pack_reconfig_data_format_bf16(uint32_t cb) {
    PACK((llk_pack_reconfig_data_format<false>(cb)));
}

ALWI void pack_tile_bf16(
    uint32_t ifrom_dst,
    uint32_t cb,
    uint32_t output_tile_index = 0
) {
    PACK((llk_pack<false, false, PackMode::Default>(
        ifrom_dst, cb, output_tile_index)));
}

ALWI void tile_regs_release_bf16() {
    PACK((llk_pack_dest_section_done<false>()));
}

void kernel_main() {
    constexpr uint32_t cb_h_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_a = get_compile_time_arg_val(1);
    constexpr uint32_t cb_h_partial = get_compile_time_arg_val(2);
    constexpr uint32_t cb_h = get_compile_time_arg_val(3);
    constexpr uint32_t cb_saved_h = get_compile_time_arg_val(4);
    constexpr uint32_t cb_base_x = get_compile_time_arg_val(5);
    constexpr uint32_t cb_w = get_compile_time_arg_val(6);
    constexpr uint32_t cb_base_partial = get_compile_time_arg_val(7);
    constexpr uint32_t cb_base = get_compile_time_arg_val(8);
    constexpr uint32_t cb_b = get_compile_time_arg_val(9);
    constexpr uint32_t cb_delta = get_compile_time_arg_val(10);
    constexpr uint32_t cb_scaled_delta = get_compile_time_arg_val(11);
    constexpr uint32_t cb_output = get_compile_time_arg_val(12);
    @BIAS_DECLARATIONS@

    constexpr uint32_t k_block_tiles = @H_K_BLOCK_TILES@;
    constexpr uint32_t num_k_blocks = @NUM_K_BLOCKS@;
    constexpr uint32_t rank_tiles = @RANK_TILES@;
    constexpr uint32_t delta_rank_tiles = @DELTA_RANK_TILES@;
    constexpr uint32_t delta_h_tile_offset = @DELTA_H_TILE_OFFSET@;
    constexpr uint32_t base_k_block_tiles = @BASE_K_BLOCK_TILES@;
    constexpr uint32_t num_base_k_blocks = @NUM_BASE_K_BLOCKS@;
    constexpr bool base_transpose_rhs = @BASE_TRANSPOSE_RHS@;
    constexpr uint32_t num_n_blocks = @NUM_N_BLOCKS@;
    constexpr uint32_t n_tiles_per_block = @N_TILES_PER_BLOCK@;
    constexpr uint32_t jobs_per_node = @JOBS_PER_NODE@;
    constexpr uint32_t total_jobs = @TOTAL_JOBS@;
    constexpr uint32_t grid_cols = @GRID_COLS@;
    constexpr uint32_t scale_bits = @SCALE_BITS@;
    // 0: legacy (both redundant init sites); 1: skip the post-reload
    // duplicate; 2: skip the per-rank duplicate; 3: TTNN-like (skip both).
    // Modes 1/2 are diagnostic stepping stones because the resident CB
    // lifecycle is not identical to the standalone TTNN matmul factory.
    constexpr uint32_t lora_init_schedule = @LORA_INIT_SCHEDULE@;

    CircularBuffer h_x(cb_h_x);
    CircularBuffer a(cb_a);
    CircularBuffer h_partial(cb_h_partial);
    CircularBuffer h(cb_h);
    CircularBuffer saved_h(cb_saved_h);
    CircularBuffer base_x(cb_base_x);
    CircularBuffer w(cb_w);
    CircularBuffer base_partial(cb_base_partial);
    CircularBuffer base(cb_base);
    CircularBuffer b(cb_b);
    CircularBuffer delta(cb_delta);
    CircularBuffer scaled_delta(cb_scaled_delta);
    CircularBuffer output(cb_output);
    @BIAS_BUFFER_OBJECTS@

    const uint32_t node_index =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();

    for (uint32_t local_job = 0; local_job < jobs_per_node; ++local_job) {
        const uint32_t job = node_index * jobs_per_node + local_job;
        if (job >= total_jobs) {
            continue;
        }
        const uint32_t n_index = job % num_n_blocks;

        // Schedule 4 is a literal one-output adaptation of TTNN's
        // bmm_large_block_zm_fused_bias_activation reduction loop.  Keep it
        // separate from the historical resident loop so the order of every
        // init, reserve, pack and pop remains inspectable.
        if constexpr (lora_init_schedule == 4) {
            // This branch is discarded for every schedule except 4, but a
            // non-dependent static_assert is still evaluated by the Tensix
            // compiler while parsing the enclosing function.  Make the
            // schedule-4 restriction explicit without rejecting the general
            // multi-tile-rank schedules 0/1/2.
            static_assert(
                rank_tiles == 1 || lora_init_schedule != 4);
            static_assert(
                k_block_tiles == 1 || lora_init_schedule != 4);
            compute_kernel_hw_startup<SrcOrder::Reverse>(
                cb_h_x, cb_a, cb_h_partial);
            matmul_block_init(cb_h_x, cb_a, true, 1, 1, 1);
            bool enable_reload = false;
            for (uint32_t block = 0; block < num_k_blocks; ++block) {
                const bool last = block == num_k_blocks - 1;
                h_x.wait_front(1);
                a.wait_front(1);
                tile_regs_acquire();
                if (enable_reload) {
                    copy_tile_to_dst_init_short_with_dt(
                        cb_a, cb_h_partial);
                    h_partial.wait_front(1);
                    copy_block_matmul_partials(
                        cb_h_partial, 0, 0, 1);
                    h_partial.pop_front(1);
                    reconfig_data_format_srca(cb_h_partial, cb_a);
                    matmul_block_init(cb_h_x, cb_a, true, 1, 1, 1);
                }
                matmul_block(
                    cb_h_x, cb_a, 0, 0, 0, true, 1, 1, 1);
                tile_regs_commit();
                if (last) {
                    h.reserve_back(1);
                    if (n_index == 0) {
                        saved_h.reserve_back(1);
                    }
                    tile_regs_wait();
                    if (n_index == 0) {
                        PACK((pack_reconfig_data_format(cb_saved_h)));
                        pack_tile_block(0, cb_saved_h, 1);
                    }
                    PACK((pack_reconfig_data_format(cb_h)));
                    pack_tile_block(0, cb_h, 1);
                    tile_regs_release();
                    if (n_index == 0) {
                        saved_h.push_back(1);
                    }
                    h.push_back(1);
                } else {
                    h_partial.reserve_back(1);
                    tile_regs_wait();
                    PACK((pack_reconfig_data_format(cb_h_partial)));
                    pack_tile_block(0, cb_h_partial, 1);
                    tile_regs_release();
                    h_partial.push_back(1);
                    enable_reload = true;
                }
                h_x.pop_front(1);
                a.pop_front(1);
            }
        } else {
        // Exact H = X @ A.T.
        compute_kernel_hw_startup<SrcOrder::Reverse>(
            cb_h_x, cb_a, cb_h_partial);
        matmul_block_init(
            cb_h_x, cb_a, true, rank_tiles, 1, k_block_tiles);
        for (uint32_t block = 0; block < num_k_blocks; ++block) {
            h_x.wait_front(k_block_tiles);
            a.wait_front(rank_tiles * k_block_tiles);
            tile_regs_acquire();
            if (block > 0) {
                copy_tile_to_dst_init_short_with_dt(
                    cb_a, cb_h_partial);
                h_partial.wait_front(rank_tiles);
                copy_block_matmul_partials(
                    cb_h_partial, 0, 0, rank_tiles);
                h_partial.pop_front(rank_tiles);
                reconfig_data_format_srca(cb_h_partial, cb_a);
                matmul_block_init(
                    cb_h_x, cb_a, true,
                    rank_tiles, 1, k_block_tiles);
            }
            if constexpr (rank_tiles > 1) {
                // Match TTNN's native multi-N subblock: all physical rank
                // tiles share one matmul block and one spill/reload boundary.
                // Treating each tile as a separate rank-32 projection is
                // numerically close but changes the observable BF16 contract.
                for (uint32_t inner = 0; inner < k_block_tiles; ++inner) {
                    matmul_block(
                        cb_h_x, cb_a, inner, inner, 0, true,
                        rank_tiles, 1, k_block_tiles);
                }
            } else {
                if constexpr (
                    lora_init_schedule != 1 && lora_init_schedule != 3
                ) {
                    matmul_block_init(
                        cb_h_x,
                        cb_a,
                        true,
                        1,
                        1,
                        k_block_tiles);
                }
                for (uint32_t inner = 0; inner < k_block_tiles; ++inner) {
                    matmul_block(
                        cb_h_x,
                        cb_a,
                        inner,
                        inner,
                        0,
                        true,
                        1,
                        1,
                        k_block_tiles);
                }
            }
            tile_regs_commit();
            const bool last = block == num_k_blocks - 1;
            const uint32_t target_cb =
                last ? cb_h : cb_h_partial;
            if (last) {
                h.reserve_back(rank_tiles);
                if (n_index == 0) {
                    saved_h.reserve_back(rank_tiles);
                }
            } else {
                h_partial.reserve_back(rank_tiles);
            }
            tile_regs_wait();
            PACK((pack_reconfig_data_format(target_cb)));
            pack_tile_block(0, target_cb, rank_tiles);
            if (last && n_index == 0) {
                PACK((pack_reconfig_data_format(cb_saved_h)));
                pack_tile_block(0, cb_saved_h, rank_tiles);
            }
            tile_regs_release();
            if (last) {
                h.push_back(rank_tiles);
                if (n_index == 0) {
                    saved_h.push_back(rank_tiles);
                }
            } else {
                h_partial.push_back(rank_tiles);
            }
            a.pop_front(rank_tiles * k_block_tiles);
            h_x.pop_front(k_block_tiles);
        }
        }

        // Exact base = X @ W.T for one <=4-tile N block.
        compute_kernel_hw_startup<SrcOrder::Reverse>(
            cb_base_x, cb_w, cb_base_partial);
        matmul_block_init(
            cb_base_x,
            cb_w,
            base_transpose_rhs,
            n_tiles_per_block,
            1,
            base_k_block_tiles);
        for (
            uint32_t block = 0;
            block < num_base_k_blocks;
            ++block
        ) {
            const bool last = block == num_base_k_blocks - 1;
            base_x.wait_front(base_k_block_tiles);
            w.wait_front(
                n_tiles_per_block * base_k_block_tiles);
            @BASE_BIAS_WAIT@
            tile_regs_acquire();
            if (block > 0) {
                copy_tile_to_dst_init_short_with_dt(
                    cb_w, cb_base_partial);
                base_partial.wait_front(n_tiles_per_block);
                copy_block_matmul_partials(
                    cb_base_partial,
                    0,
                    0,
                    n_tiles_per_block);
                base_partial.pop_front(n_tiles_per_block);
                reconfig_data_format_srca(cb_base_partial, cb_w);
                matmul_block_init(
                    cb_base_x,
                    cb_w,
                    base_transpose_rhs,
                    n_tiles_per_block,
                    1,
                    base_k_block_tiles);
            }
            for (
                uint32_t inner = 0;
                inner < base_k_block_tiles;
                ++inner
            ) {
                matmul_block(
                    cb_base_x,
                    cb_w,
                    inner,
                    inner,
                    0,
                    base_transpose_rhs,
                    n_tiles_per_block,
                    1,
                    base_k_block_tiles);
            }
            @BASE_BIAS_ACCUMULATE@
            tile_regs_commit();
            const uint32_t target_cb = @BASE_FINAL_TARGET@;
            @BASE_RESERVE@
            tile_regs_wait();
            PACK((pack_reconfig_data_format(target_cb)));
            pack_tile_block(
                0, target_cb, n_tiles_per_block);
            tile_regs_release();
            @BASE_PUSH@
            @BASE_BIAS_FINISH@
            @BASE_BIAS_POP@
            w.pop_front(
                n_tiles_per_block * base_k_block_tiles);
            base_x.pop_front(base_k_block_tiles);
        }

        // delta = H @ B.T.
        h.wait_front(rank_tiles);
        b.wait_front(n_tiles_per_block * delta_rank_tiles);
        delta.reserve_back(n_tiles_per_block);
        compute_kernel_hw_startup<SrcOrder::Reverse>(
            cb_h, cb_b, cb_delta);
        matmul_block_init(
            cb_h,
            cb_b,
            true,
            n_tiles_per_block,
            1,
            delta_rank_tiles);
        tile_regs_acquire();
        for (
            uint32_t inner = 0;
            inner < delta_rank_tiles;
            ++inner
        ) {
            matmul_block(
                cb_h,
                cb_b,
                delta_h_tile_offset + inner,
                inner,
                0,
                true,
                n_tiles_per_block,
                1,
                delta_rank_tiles);
        }
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(cb_delta)));
        for (
            uint32_t tile = 0;
            tile < n_tiles_per_block;
            ++tile
        ) {
            pack_tile<true>(tile, cb_delta, tile);
        }
        tile_regs_release();
        delta.push_back(n_tiles_per_block);
        b.pop_front(n_tiles_per_block * delta_rank_tiles);
        h.pop_front(rank_tiles);

        // Preserve BF16(delta * scale). Base and scaled delta remain queued
        // for the one destination-mode switch after all local jobs.
        delta.wait_front(n_tiles_per_block);
        scaled_delta.reserve_back(n_tiles_per_block);
        init_sfpu(cb_delta, cb_scaled_delta);
        for (
            uint32_t tile = 0;
            tile < n_tiles_per_block;
            ++tile
        ) {
            tile_regs_acquire();
            copy_tile_init(cb_delta);
            copy_tile(cb_delta, tile, 0);
            binop_with_scalar_tile_init();
            {
                volatile int32_t scalar = scale_bits;
                mul_unary_tile(0, scalar);
            }
            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(cb_scaled_delta)));
            pack_tile<true>(0, cb_scaled_delta, tile);
            tile_regs_release();
        }
        scaled_delta.push_back(n_tiles_per_block);
        delta.pop_front(n_tiles_per_block);
    }

    // Repeated FP32/BF16 destination-mode switches across local jobs corrupt
    // SyncHalf destination banks on Blackhole. Stage the complete local batch,
    // switch once, and drain it in output order.
    @FINAL_ADD_BODY@
}
