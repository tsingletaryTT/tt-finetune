// Hand-written TT-Metal compute kernel for bit-exact LoRA forward.
//
// The first projection deliberately mirrors TT-NN's
// bmm_large_block_zm_fused_bias_activation reduction contract:
//   * HiFi4 + FP32 destination accumulation (set by the host descriptor)
//   * two K tiles per reduction block
//   * Float32 partial-result CB between blocks
//   * one final BF16 pack before the second projection
//
// Python substitutes the @...@ constants before TT-Metal JIT compilation.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

// The kernel is compiled with FP32 destination accumulation for the two
// matmuls. The generic graph's final ttnn.add, however, runs with BF16
// destination mode. Blackhole supports switching that hardware mode safely
// in the middle of a compute kernel. These small wrappers also pass `false`
// to the LLK templates so destination addressing, packing, and clearing match
// the active BF16 mode instead of the kernel-wide FP32 compile flag.
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

template <bool out_of_order_output = false>
ALWI void pack_tile_bf16(
    uint32_t ifrom_dst,
    uint32_t cb,
    uint32_t output_tile_index = 0
) {
    PACK((llk_pack<
          false,
          out_of_order_output,
          PackMode::Default>(
        ifrom_dst, cb, output_tile_index)));
}

ALWI void tile_regs_release_bf16() {
    PACK((llk_pack_dest_section_done<false>()));
}

void kernel_main() {
    constexpr uint32_t cb_inputs = get_compile_time_arg_val(0);
    constexpr uint32_t cb_a = get_compile_time_arg_val(1);
    constexpr uint32_t cb_partial_fp32 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_hidden_bf16 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_b = get_compile_time_arg_val(4);
    constexpr uint32_t cb_base = get_compile_time_arg_val(5);
    constexpr uint32_t cb_delta = get_compile_time_arg_val(6);
    constexpr uint32_t cb_output = get_compile_time_arg_val(7);

    constexpr uint32_t k_block_tiles = 2;
    constexpr uint32_t num_k_blocks = @NUM_K_BLOCKS@;
    constexpr uint32_t num_n_blocks = @NUM_N_BLOCKS@;
    constexpr uint32_t n_tiles_per_block = @N_TILES_PER_BLOCK@;
    constexpr uint32_t second_subblock_w = @SECOND_SUBBLOCK_W@;
    constexpr uint32_t m_blocks_per_node = @M_BLOCKS_PER_NODE@;
    constexpr uint32_t total_m_blocks = @TOTAL_M_BLOCKS@;
    constexpr uint32_t grid_cols = @GRID_COLS@;
    constexpr uint32_t scale_bits = @SCALE_BITS@;

    CircularBuffer inputs_cb(cb_inputs);
    CircularBuffer a_cb(cb_a);
    CircularBuffer partial_cb(cb_partial_fp32);
    CircularBuffer hidden_cb(cb_hidden_bf16);
    CircularBuffer b_cb(cb_b);
    CircularBuffer base_cb(cb_base);
    CircularBuffer delta_cb(cb_delta);
    CircularBuffer output_cb(cb_output);

    const uint32_t node_index =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();

    for (uint32_t local_m = 0; local_m < m_blocks_per_node; ++local_m) {
        const uint32_t m_block =
            node_index * m_blocks_per_node + local_m;
        if (m_block >= total_m_blocks) {
            continue;
        }

        compute_kernel_hw_startup<SrcOrder::Reverse>(
            cb_inputs, cb_a, cb_partial_fp32);
        matmul_block_init(
            cb_inputs,
            cb_a,
            true,
            1,
            1,
            k_block_tiles);

        for (uint32_t block = 0; block < num_k_blocks; ++block) {
            inputs_cb.wait_front(k_block_tiles);
            a_cb.wait_front(k_block_tiles);

            tile_regs_acquire();
            if (block > 0) {
                copy_tile_to_dst_init_short_with_dt(
                    cb_a, cb_partial_fp32);
                partial_cb.wait_front(1);
                copy_block_matmul_partials(
                    cb_partial_fp32, 0, 0, 1);
                partial_cb.pop_front(1);
                reconfig_data_format_srca(cb_partial_fp32, cb_a);
                matmul_block_init(
                    cb_inputs,
                    cb_a,
                    true,
                    1,
                    1,
                    k_block_tiles);
            }

            for (uint32_t inner = 0; inner < k_block_tiles; ++inner) {
                matmul_block(
                    cb_inputs,
                    cb_a,
                    inner,
                    inner,
                    0,
                    true,
                    1,
                    1,
                    k_block_tiles);
            }

            tile_regs_commit();
            const bool last = block == num_k_blocks - 1;
            const uint32_t target_cb =
                last ? cb_hidden_bf16 : cb_partial_fp32;
            if (last) {
                hidden_cb.reserve_back(1);
            } else {
                partial_cb.reserve_back(1);
            }
            tile_regs_wait();
            PACK((pack_reconfig_data_format(target_cb)));
            pack_tile_block(0, target_cb, 1);
            tile_regs_release();
            if (last) {
                hidden_cb.push_back(1);
            } else {
                partial_cb.push_back(1);
            }

            a_cb.pop_front(k_block_tiles);
            inputs_cb.pop_front(k_block_tiles);
        }

        hidden_cb.wait_front(1);
        for (uint32_t n_block = 0; n_block < num_n_blocks; ++n_block) {
            b_cb.wait_front(n_tiles_per_block);
            base_cb.wait_front(n_tiles_per_block);
            delta_cb.reserve_back(n_tiles_per_block);

            compute_kernel_hw_startup<SrcOrder::Reverse>(
                cb_hidden_bf16, cb_b, cb_delta);
            // Blackhole exposes only four FP32 destination tiles.  TT-NN's
            // wide matmul therefore walks N in <=4-tile subblocks instead of
            // trying to accumulate the whole output-feature block at once.
            // Mirroring that schedule is required both for correctness and
            // for the same numerical contract as the generic path.
            for (
                uint32_t n_subblock = 0;
                n_subblock < n_tiles_per_block;
                n_subblock += second_subblock_w
            ) {
                matmul_block_init(
                    cb_hidden_bf16,
                    cb_b,
                    true,
                    second_subblock_w,
                    1,
                    1);
                tile_regs_acquire();
                matmul_block(
                    cb_hidden_bf16,
                    cb_b,
                    0,
                    n_subblock,
                    0,
                    true,
                    second_subblock_w,
                    1,
                    1);
                tile_regs_commit();
                tile_regs_wait();
                PACK((pack_reconfig_data_format(cb_delta)));
                for (
                    uint32_t tile = 0;
                    tile < second_subblock_w;
                    ++tile
                ) {
                    pack_tile<true>(
                        tile,
                        cb_delta,
                        n_subblock + tile);
                }
                tile_regs_release();
            }
            delta_cb.push_back(n_tiles_per_block);

            delta_cb.wait_front(n_tiles_per_block);
            // The generic graph materializes `delta * scale` as a BF16
            // tensor before its separate base add. Reuse the now-empty
            // input CB as an L1-only scaled-delta staging buffer so the
            // fused program preserves that exact rounding boundary.
            // Process two output tiles per acquire, matching TT-Lang's
            // generated FP32-destination elementwise schedule.  Destinations
            // (delta, base) use (0,1) and (2,3), respectively.
            // First materialize every scaled BF16 tile. Keeping the whole
            // N block in this L1 staging CB lets us switch destination mode
            // once for the complete add phase. Repeatedly switching between
            // FP32 and BF16 across SyncHalf destination-bank flips corrupts
            // the third and later tiles on Blackhole.
            uint32_t tile = 0;
            for (; tile + 1 < n_tiles_per_block; tile += 2) {
                inputs_cb.reserve_back(2);
                init_sfpu(cb_delta, cb_inputs);
                tile_regs_acquire();
                copy_tile_init(cb_delta);
                copy_tile(cb_delta, tile, 0);
                copy_tile(cb_delta, tile + 1, 2);
                binop_with_scalar_tile_init();
                {
                    volatile int32_t scalar = scale_bits;
                    mul_unary_tile(0, scalar);
                    mul_unary_tile(2, scalar);
                }
                tile_regs_commit();
                tile_regs_wait();
                PACK((pack_reconfig_data_format(cb_inputs)));
                pack_tile<true>(0, cb_inputs, 0);
                pack_tile<true>(2, cb_inputs, 1);
                tile_regs_release();
                inputs_cb.push_back(2);
            }
            if (tile < n_tiles_per_block) {
                inputs_cb.reserve_back(1);
                init_sfpu(cb_delta, cb_inputs);
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
                PACK((pack_reconfig_data_format(cb_inputs)));
                pack_tile<true>(0, cb_inputs, 0);
                tile_regs_release();
                inputs_cb.push_back(1);
            }

            inputs_cb.wait_front(n_tiles_per_block);
            add_tiles_init(cb_base, cb_inputs);
            disable_fp32_dest_acc();
            tile = 0;
            for (; tile + 1 < n_tiles_per_block; tile += 2) {
                output_cb.reserve_back(2);
                tile_regs_acquire();
                add_tiles_bf16(
                    cb_base, cb_inputs, tile, tile, 0);
                add_tiles_bf16(
                    cb_base, cb_inputs, tile + 1, tile + 1, 1);
                tile_regs_commit_bf16();
                tile_regs_wait();
                pack_reconfig_data_format_bf16(cb_output);
                pack_tile_bf16(0, cb_output, 0);
                pack_tile_bf16(1, cb_output, 1);
                tile_regs_release_bf16();
                output_cb.push_back(2);
            }
            if (tile < n_tiles_per_block) {
                output_cb.reserve_back(1);
                tile_regs_acquire();
                add_tiles_bf16(
                    cb_base, cb_inputs, tile, tile, 0);
                tile_regs_commit_bf16();
                tile_regs_wait();
                pack_reconfig_data_format_bf16(cb_output);
                pack_tile_bf16(0, cb_output, 0);
                tile_regs_release_bf16();
                output_cb.push_back(1);
            }
            inputs_cb.pop_front(n_tiles_per_block);
            enable_fp32_dest_acc();

            delta_cb.pop_front(n_tiles_per_block);
            base_cb.pop_front(n_tiles_per_block);
            b_cb.pop_front(n_tiles_per_block);
        }
        hidden_cb.pop_front(1);
    }
}
