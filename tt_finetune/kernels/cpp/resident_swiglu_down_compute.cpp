// Numerically exact resident SwiGLU + frozen down projection + LoRA.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace {

ALWI void bf16_unary_init(uint32_t input_id, uint32_t output_id) {
#ifndef ARCH_QUASAR
    state_configure<Operand::SRCA, Operand::PACK>(
        input_id, output_id, __builtin_LINE());
    UNPACK((llk_unpack_hw_configure<false>(input_id)));
    UNPACK((llk_unpack_A_init<
            BroadcastType::NONE,
            false,
            EltwiseBinaryReuseDestType::NONE,
            UnpackToDestEn>(false, false, input_id)));
    PACK((llk_pack_hw_configure<false>(output_id)));
    PACK((llk_pack_init(output_id)));
    PACK((llk_pack_dest_init<false, PackMode::Default>()));
    MATH((llk_math_eltwise_unary_datacopy_init<
          DataCopyType::A2D,
          false,
          BroadcastType::NONE>(input_id)));
    MATH((llk_math_pack_sync_init<false>()));
    MATH((llk_math_hw_configure<false>(input_id, input_id)));
    MATH((ckernel::math::_configure_unary_preserve_zero_flag_state_()));
#endif
}

ALWI void bf16_copy_init(uint32_t input_id) {
    UNPACK((llk_unpack_A_init<
            BroadcastType::NONE,
            false,
            EltwiseBinaryReuseDestType::NONE,
            UnpackToDestEn>(false, false, input_id)));
    MATH((llk_math_eltwise_unary_datacopy_init<
          DataCopyType::A2D,
          false,
          BroadcastType::NONE>(input_id)));
}

ALWI void bf16_copy_tile(
    uint32_t input_id, uint32_t input_tile, uint32_t destination_tile) {
    UNPACK((llk_unpack_A<
            BroadcastType::NONE,
            false,
            EltwiseBinaryReuseDestType::NONE,
            UnpackToDestEn>(input_id, input_tile)));
    MATH((llk_math_eltwise_unary_datacopy<
          DataCopyType::A2D,
          false,
          BroadcastType::NONE,
          UnpackToDestEn>(destination_tile, input_id)));
}

ALWI void exact_silu_one(uint32_t input_id, uint32_t output_id) {
    CircularBuffer input(input_id);
    CircularBuffer output(output_id);
    bf16_unary_init(input_id, output_id);
    input.wait_front(1);
    output.reserve_back(1);
    tile_regs_acquire();
    bf16_copy_init(input_id);
    bf16_copy_tile(input_id, 0, 0);
    MATH((SFPU_UNARY_INIT_FN(silu, sfpu::silu_init, (false))));
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        false,
        calculate_silu,
        (false, 8),
        0,
        VectorMode::RC)));
    MATH((llk_math_dest_section_done<false>()));
    tile_regs_wait();
    PACK((llk_pack_reconfig_data_format<false>(output_id)));
    PACK((llk_pack<false, false, PackMode::Default>(0, output_id, 0)));
    PACK((llk_pack_dest_section_done<false>()));
    output.push_back(1);
    input.pop_front(1);
}

ALWI void exact_sigmoid_one(uint32_t input_id, uint32_t output_id) {
    CircularBuffer input(input_id);
    CircularBuffer output(output_id);
    bf16_unary_init(input_id, output_id);
    input.wait_front(1);
    output.reserve_back(1);
    tile_regs_acquire();
    bf16_copy_init(input_id);
    bf16_copy_tile(input_id, 0, 0);
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

ALWI void exact_sfpu_multiply_one(
    uint32_t lhs_id, uint32_t rhs_id, uint32_t output_id) {
    CircularBuffer lhs(lhs_id);
    CircularBuffer rhs(rhs_id);
    CircularBuffer output(output_id);
    bf16_unary_init(lhs_id, output_id);
    MATH((SFPU_BINARY_INIT_FN(
        unused, sfpu::sfpu_binary_init,
        (false, ckernel::BinaryOp::MUL))));
    lhs.wait_front(1);
    rhs.wait_front(1);
    output.reserve_back(1);
    tile_regs_acquire();
    bf16_copy_init(lhs_id);
    bf16_copy_tile(lhs_id, 0, 0);
    bf16_copy_init(rhs_id);
    bf16_copy_tile(rhs_id, 0, 1);
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
    lhs.pop_front(1);
    rhs.pop_front(1);
}

ALWI void fp32_silu_one(uint32_t input_id, uint32_t output_id) {
    CircularBuffer input(input_id);
    CircularBuffer output(output_id);
    init_sfpu(input_id, output_id);
    input.wait_front(1);
    output.reserve_back(1);
    tile_regs_acquire();
    copy_tile_init(input_id);
    copy_tile(input_id, 0, 0);
    silu_tile_init();
    silu_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, output_id, 0);
    tile_regs_release();
    output.push_back(1);
    input.pop_front(1);
}

ALWI void fp32_multiply_one(
    uint32_t lhs_id, uint32_t rhs_id, uint32_t output_id) {
    CircularBuffer lhs(lhs_id);
    CircularBuffer rhs(rhs_id);
    CircularBuffer output(output_id);
    unary_op_init_common(lhs_id, output_id);
    mul_binary_tile_init();
    lhs.wait_front(1);
    rhs.wait_front(1);
    output.reserve_back(1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(rhs_id, lhs_id);
    copy_tile(lhs_id, 0, 0);
    copy_tile_to_dst_init_short_with_dt(lhs_id, rhs_id);
    copy_tile(rhs_id, 0, 1);
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, output_id, 0);
    tile_regs_release();
    output.push_back(1);
    lhs.pop_front(1);
    rhs.pop_front(1);
}

ALWI void add_tiles_bf16(
    uint32_t icb0, uint32_t icb1,
    uint32_t itile0, uint32_t itile1, uint32_t idst) {
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
    uint32_t source, uint32_t cb, uint32_t target = 0) {
    PACK((llk_pack<false, false, PackMode::Default>(source, cb, target)));
}
ALWI void tile_regs_release_bf16() {
    PACK((llk_pack_dest_section_done<false>()));
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_gate = get_compile_time_arg_val(0);
    constexpr uint32_t cb_up = get_compile_time_arg_val(1);
    constexpr uint32_t cb_silu = get_compile_time_arg_val(2);
    constexpr uint32_t cb_product_out = get_compile_time_arg_val(3);
    constexpr uint32_t cb_product_resident = get_compile_time_arg_val(4);
    constexpr uint32_t cb_a = get_compile_time_arg_val(5);
    constexpr uint32_t cb_h_partial = get_compile_time_arg_val(6);
    constexpr uint32_t cb_h_mcast = get_compile_time_arg_val(7);
    constexpr uint32_t cb_h_resident = get_compile_time_arg_val(8);
    constexpr uint32_t cb_h_saved = get_compile_time_arg_val(9);
    constexpr uint32_t cb_w = get_compile_time_arg_val(10);
    constexpr uint32_t cb_base_partial = get_compile_time_arg_val(11);
    constexpr uint32_t cb_base = get_compile_time_arg_val(12);
    constexpr uint32_t cb_b = get_compile_time_arg_val(13);
    constexpr uint32_t cb_delta = get_compile_time_arg_val(14);
    constexpr uint32_t cb_scaled = get_compile_time_arg_val(15);
    constexpr uint32_t cb_output = get_compile_time_arg_val(16);
    constexpr uint32_t cb_training_sigmoid = get_compile_time_arg_val(17);

    constexpr uint32_t h_tiles = @H_TILES@;
    constexpr uint32_t i_tiles = @I_TILES@;
    constexpr uint32_t rank_tiles = @RANK_TILES@;
    constexpr uint32_t product_jobs = @PRODUCT_JOBS@;
    constexpr uint32_t product_jobs_per_core = @PRODUCT_JOBS_PER_CORE@;
    constexpr uint32_t output_jobs = @OUTPUT_JOBS@;
    constexpr uint32_t output_jobs_per_core = @OUTPUT_JOBS_PER_CORE@;
    constexpr uint32_t base_k_block_tiles = @BASE_K_BLOCK_TILES@;
    constexpr uint32_t num_base_k_blocks = @NUM_BASE_K_BLOCKS@;
    constexpr uint32_t num_h_k_blocks = @NUM_H_K_BLOCKS@;
    constexpr uint32_t h_k_block_tiles = @H_K_BLOCK_TILES@;
    constexpr uint32_t scale_bits = @SCALE_BITS@;
    constexpr bool fp32_activation = @FP32_ACTIVATION@;
    constexpr bool decomposed_activation = @DECOMPOSED_ACTIVATION@;
    constexpr uint32_t kCoreCount = 110;
    constexpr uint32_t kOutputCoreCount = 72;
    constexpr uint32_t kProductCoreStart = kOutputCoreCount;
    constexpr uint32_t kProductCoreCount = kCoreCount - kOutputCoreCount;

    CircularBuffer product_resident(cb_product_resident);
    // gate/up are consumed completely by SwiGLU, then recycled as the
    // canonical phase-sized X streams for H and frozen-base reductions.
    CircularBuffer h_x(cb_gate);
    CircularBuffer base_x(cb_up);
    CircularBuffer a(cb_a);
    CircularBuffer h_partial(cb_h_partial);
    CircularBuffer h_mcast(cb_h_mcast);
    CircularBuffer h_resident(cb_h_resident);
    CircularBuffer h_saved(cb_h_saved);
    CircularBuffer w(cb_w);
    CircularBuffer base_partial(cb_base_partial);
    CircularBuffer base(cb_base);
    CircularBuffer b(cb_b);
    CircularBuffer delta(cb_delta);
    CircularBuffer scaled(cb_scaled);
    CircularBuffer output(cb_output);

    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    uint32_t valid_products = 0;
    for (uint32_t local = 0; local < product_jobs_per_core; ++local) {
        if (node >= kProductCoreStart &&
            (node - kProductCoreStart) + local * kProductCoreCount
                < product_jobs) {
            ++valid_products;
        }
    }
    uint32_t valid_outputs = 0;
    for (uint32_t local = 0; local < output_jobs_per_core; ++local) {
        if (node < kOutputCoreCount &&
            node + local * kOutputCoreCount < output_jobs) {
            ++valid_outputs;
        }
    }

    for (uint32_t local = 0; local < valid_products; ++local) {
        DeviceZoneScopedN("resident_mlp_swiglu");
        if constexpr (decomposed_activation) {
            exact_sigmoid_one(cb_gate, cb_training_sigmoid);
            exact_sfpu_multiply_one(
                cb_gate, cb_training_sigmoid, cb_silu);
            exact_sfpu_multiply_one(cb_silu, cb_up, cb_product_out);
        } else if constexpr (fp32_activation) {
            fp32_silu_one(cb_gate, cb_silu);
            fp32_multiply_one(cb_silu, cb_up, cb_product_out);
        } else {
            exact_silu_one(cb_gate, cb_silu);
            exact_sfpu_multiply_one(cb_silu, cb_up, cb_product_out);
        }
    }
    if (valid_products != 0) {
        enable_fp32_dest_acc();
    }

    if (valid_outputs != 0) {
        product_resident.wait_front(valid_outputs * i_tiles);
    }

    // One owner per token row computes H = product @ A.T. The exact accepted
    // LoRA reduction spills every two K tiles to FP32. Reader multicasts the
    // final BF16 H tile to all down-output owners on the device.
    for (uint32_t local = 0; local < output_jobs_per_core; ++local) {
        if (node >= kOutputCoreCount) {
            continue;
        }
        const uint32_t job = node + local * kOutputCoreCount;
        if (job >= output_jobs || job % h_tiles != 0) {
            continue;
        }
        compute_kernel_hw_startup<SrcOrder::Reverse>(
            cb_gate, cb_a, cb_h_partial);
        matmul_block_init(
            cb_gate, cb_a, true, 1, 1, h_k_block_tiles);
        for (uint32_t block = 0; block < num_h_k_blocks; ++block) {
            h_x.wait_front(h_k_block_tiles);
            a.wait_front(h_k_block_tiles * rank_tiles);
            tile_regs_acquire();
            if (block > 0) {
                copy_tile_to_dst_init_short_with_dt(cb_a, cb_h_partial);
                h_partial.wait_front(rank_tiles);
                for (uint32_t tile = 0; tile < rank_tiles; ++tile) {
                    copy_tile(cb_h_partial, tile, tile);
                }
                h_partial.pop_front(rank_tiles);
                reconfig_data_format_srca(cb_h_partial, cb_a);
            }
            matmul_block_init(
                cb_gate, cb_a, true, 1, 1, h_k_block_tiles);
            for (uint32_t rank = 0; rank < rank_tiles; ++rank) {
                matmul_block_init(
                    cb_gate, cb_a, true, 1, 1,
                    h_k_block_tiles);
                for (uint32_t inner = 0; inner < h_k_block_tiles; ++inner) {
                    matmul_block(
                        cb_gate,
                        cb_a,
                        inner,
                        rank * h_k_block_tiles + inner,
                        rank,
                        true, 1, 1, h_k_block_tiles);
                }
            }
            tile_regs_commit();
            const bool last = block == num_h_k_blocks - 1;
            const uint32_t target = last ? cb_h_mcast : cb_h_partial;
            if (last) {
                h_mcast.reserve_back(rank_tiles);
                h_saved.reserve_back(rank_tiles);
            } else {
                h_partial.reserve_back(rank_tiles);
            }
            tile_regs_wait();
            if (last) {
                PACK((pack_reconfig_data_format(cb_h_saved)));
                pack_tile_block(0, cb_h_saved, rank_tiles);
            }
            PACK((pack_reconfig_data_format(target)));
            pack_tile_block(0, target, rank_tiles);
            tile_regs_release();
            if (last) {
                h_saved.push_back(rank_tiles);
                h_mcast.push_back(rank_tiles);
            } else {
                h_partial.push_back(rank_tiles);
            }
            h_x.pop_front(h_k_block_tiles);
            a.pop_front(h_k_block_tiles * rank_tiles);
        }
    }

    if (valid_outputs != 0) {
        h_resident.wait_front(valid_outputs * rank_tiles);
    }
    for (uint32_t local = 0; local < output_jobs_per_core; ++local) {
        if (node >= kOutputCoreCount) {
            continue;
        }
        const uint32_t job = node + local * kOutputCoreCount;
        if (job >= output_jobs) {
            continue;
        }

        // Exact base = product @ frozen_weight.T using the registered
        // per-shape K phase. Product tiles remain at fixed L1 offsets.
        compute_kernel_hw_startup<SrcOrder::Reverse>(
            cb_up, cb_w, cb_base_partial);
        matmul_block_init(
            cb_up, cb_w, false, 1, 1,
            base_k_block_tiles);
        for (uint32_t block = 0; block < num_base_k_blocks; ++block) {
            base_x.wait_front(base_k_block_tiles);
            w.wait_front(base_k_block_tiles);
            tile_regs_acquire();
            if (block > 0) {
                copy_tile_to_dst_init_short_with_dt(cb_w, cb_base_partial);
                base_partial.wait_front(1);
                copy_block_matmul_partials(cb_base_partial, 0, 0, 1);
                base_partial.pop_front(1);
                reconfig_data_format_srca(cb_base_partial, cb_w);
                matmul_block_init(
                    cb_up, cb_w, false, 1, 1,
                    base_k_block_tiles);
            }
            for (uint32_t inner = 0; inner < base_k_block_tiles; ++inner) {
                matmul_block(
                    cb_up,
                    cb_w,
                    inner,
                    inner,
                    0,
                    false, 1, 1, base_k_block_tiles);
            }
            tile_regs_commit();
            const bool last = block == num_base_k_blocks - 1;
            const uint32_t target = last ? cb_base : cb_base_partial;
            if (last) {
                base.reserve_back(1);
            } else {
                base_partial.reserve_back(1);
            }
            tile_regs_wait();
            PACK((pack_reconfig_data_format(target)));
            pack_tile_block(0, target, 1);
            tile_regs_release();
            if (last) {
                base.push_back(1);
            } else {
                base_partial.push_back(1);
            }
            base_x.pop_front(base_k_block_tiles);
            w.pop_front(base_k_block_tiles);
        }

        // delta = H @ B.T.
        b.wait_front(rank_tiles);
        delta.reserve_back(1);
        compute_kernel_hw_startup<SrcOrder::Reverse>(
            cb_h_resident, cb_b, cb_delta);
        matmul_block_init(
            cb_h_resident, cb_b, true, 1, 1, rank_tiles);
        tile_regs_acquire();
        for (uint32_t inner = 0; inner < rank_tiles; ++inner) {
            matmul_block(
                cb_h_resident, cb_b,
                local * rank_tiles + inner, inner, 0,
                true, 1, 1, rank_tiles);
        }
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(cb_delta)));
        pack_tile<true>(0, cb_delta, 0);
        tile_regs_release();
        delta.push_back(1);
        b.pop_front(rank_tiles);

        delta.wait_front(1);
        scaled.reserve_back(1);
        init_sfpu(cb_delta, cb_scaled);
        tile_regs_acquire();
        copy_tile_init(cb_delta);
        copy_tile(cb_delta, 0, 0);
        binop_with_scalar_tile_init();
        {
            volatile int32_t scalar = scale_bits;
            mul_unary_tile(0, scalar);
        }
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(cb_scaled)));
        pack_tile<true>(0, cb_scaled, 0);
        tile_regs_release();
        scaled.push_back(1);
        delta.pop_front(1);
    }

    // Match the generic BF16 add boundary once, after all FP32 work on the
    // core has finished. Repeated mode flips corrupt SyncHalf banks on BH.
    add_tiles_init(cb_base, cb_scaled);
    disable_fp32_dest_acc();
    for (uint32_t local = 0; local < valid_outputs; ++local) {
        base.wait_front(1);
        scaled.wait_front(1);
        output.reserve_back(1);
        tile_regs_acquire();
        add_tiles_bf16(cb_base, cb_scaled, 0, 0, 0);
        tile_regs_commit_bf16();
        tile_regs_wait();
        pack_reconfig_data_format_bf16(cb_output);
        pack_tile_bf16(0, cb_output, 0);
        tile_regs_release_bf16();
        output.push_back(1);
        scaled.pop_front(1);
        base.pop_front(1);
    }
    enable_fp32_dest_acc();
    if (valid_outputs != 0) {
        h_resident.pop_front(valid_outputs * rank_tiles);
        product_resident.pop_front(valid_outputs * i_tiles);
    }
}
