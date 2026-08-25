// Exact two-member second phase of frozen Linear + LoRA backward.
//
// All intermediate member sums are deliberately packed to BF16 circular
// buffers in L1. This preserves the accepted PEFT rounding schedule without
// exposing any intermediate tensor to DRAM or the outer autograd graph.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/dataflow/circular_buffer.h"

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
            dh.wait_front(reduction_block_tiles);
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
                    inner,
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
            dh.pop_front(reduction_block_tiles);
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

void kernel_main() {
    constexpr uint32_t cb_base0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_base1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_dh0 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_a0 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_dh1 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_a1 = get_compile_time_arg_val(5);
    constexpr uint32_t cb_lora0 = get_compile_time_arg_val(6);
    constexpr uint32_t cb_lora1 = get_compile_time_arg_val(7);
    constexpr uint32_t cb_member1_sum = get_compile_time_arg_val(8);
    constexpr uint32_t cb_with_member0_lora =
        get_compile_time_arg_val(9);
    constexpr uint32_t cb_combined = get_compile_time_arg_val(10);

    constexpr uint32_t cb_da0_dh = get_compile_time_arg_val(11);
    constexpr uint32_t cb_da0_x = get_compile_time_arg_val(12);
    constexpr uint32_t cb_da0_transposed_x =
        get_compile_time_arg_val(13);
    constexpr uint32_t cb_da0_partial = get_compile_time_arg_val(14);
    constexpr uint32_t cb_da0_pretranspose =
        get_compile_time_arg_val(15);
    constexpr uint32_t cb_da0_output = get_compile_time_arg_val(16);

    constexpr uint32_t cb_da1_dh = get_compile_time_arg_val(17);
    constexpr uint32_t cb_da1_x = get_compile_time_arg_val(18);
    constexpr uint32_t cb_da1_transposed_x =
        get_compile_time_arg_val(19);
    constexpr uint32_t cb_da1_partial = get_compile_time_arg_val(20);
    constexpr uint32_t cb_da1_pretranspose =
        get_compile_time_arg_val(21);
    constexpr uint32_t cb_da1_output = get_compile_time_arg_val(22);

    constexpr uint32_t jobs_per_node = @DX_JOBS_PER_NODE@;
    constexpr uint32_t total_jobs = @TOTAL_DX_JOBS@;
    constexpr uint32_t k_tiles_per_node = @K_TILES_PER_NODE@;
    constexpr uint32_t total_k_tiles = @TOTAL_K_TILES@;
    constexpr uint32_t reduction_tiles = @REDUCTION_TILES@;
    constexpr uint32_t grid_cols = @GRID_COLS@;
    static_assert(
        reduction_tiles % 2 == 0,
        "exact grouped dA requires an even token tile count");

    const uint32_t node_index =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();

    // Run the two FP32 token reductions before entering BF16 destination
    // mode. This keeps each exact dA path independent of the later staged
    // input-gradient accumulation state.
    da0_phase();
    da1_phase();

    run_lora_dx(
        cb_dh0, cb_a0, cb_lora0,
        jobs_per_node, total_jobs, node_index);
    run_lora_dx(
        cb_dh1, cb_a1, cb_lora1,
        jobs_per_node, total_jobs, node_index);

    // Blackhole's SyncHalf destination mode must switch only once across
    // the complete staged BF16 accumulation tree.
    disable_fp32_dest_acc();
    run_bf16_add(
        cb_base1, cb_lora1, cb_member1_sum,
        jobs_per_node, total_jobs, node_index);
    run_bf16_add(
        cb_member1_sum, cb_lora0, cb_with_member0_lora,
        jobs_per_node, total_jobs, node_index);
    run_bf16_add(
        cb_with_member0_lora, cb_base0, cb_combined,
        jobs_per_node, total_jobs, node_index);
    enable_fp32_dest_acc();
}
