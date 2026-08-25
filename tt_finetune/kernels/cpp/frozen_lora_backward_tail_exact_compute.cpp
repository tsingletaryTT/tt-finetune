// Exact second phase of frozen Linear + LoRA backward.
//
// Inputs from phase one:
//   base_dX
//   dH = BF16(BF16(dY * scale) @ B)
//
// Outputs:
//   dX = BF16(base_dX + BF16(dH @ A))
//   dA = transpose(X.T @ dH)
//
// Python appends the accepted exact dA kernel as da_phase().

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/dataflow/circular_buffer.h"

ALWI void tail_add_tiles_bf16(
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

ALWI void tail_tile_regs_commit_bf16() {
    MATH((llk_math_dest_section_done<false>()));
}

ALWI void tail_pack_reconfig_bf16(uint32_t cb) {
    PACK((llk_pack_reconfig_data_format<false>(cb)));
}

ALWI void tail_pack_tile_bf16(
    uint32_t ifrom_dst,
    uint32_t cb,
    uint32_t output_tile_index = 0
) {
    PACK((llk_pack<false, false, PackMode::Default>(
        ifrom_dst, cb, output_tile_index)));
}

ALWI void tail_tile_regs_release_bf16() {
    PACK((llk_pack_dest_section_done<false>()));
}

void kernel_main() {
    constexpr uint32_t cb_base_dx = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dh = get_compile_time_arg_val(1);
    constexpr uint32_t cb_a = get_compile_time_arg_val(2);
    constexpr uint32_t cb_lora_dx = get_compile_time_arg_val(3);
    constexpr uint32_t cb_combined_dx = get_compile_time_arg_val(4);

    constexpr uint32_t k_tiles = @K_TILES@;
    constexpr uint32_t jobs_per_node = @DX_JOBS_PER_NODE@;
    constexpr uint32_t total_jobs = @TOTAL_DX_JOBS@;
    constexpr uint32_t grid_cols = @GRID_COLS@;

    CircularBuffer base_dx(cb_base_dx);
    CircularBuffer dh(cb_dh);
    CircularBuffer a(cb_a);
    CircularBuffer lora_dx(cb_lora_dx);
    CircularBuffer combined_dx(cb_combined_dx);

    const uint32_t node_index =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();

    // Compute and stage every local BF16 LoRA dX tile first.
    compute_kernel_hw_startup<SrcOrder::Reverse>(
        cb_dh, cb_a, cb_lora_dx);
    matmul_block_init(cb_dh, cb_a, false, 1, 1, 1);
    for (uint32_t local_job = 0; local_job < jobs_per_node; ++local_job) {
        const uint32_t job = node_index * jobs_per_node + local_job;
        if (job >= total_jobs) {
            continue;
        }
        dh.wait_front(1);
        a.wait_front(1);
        lora_dx.reserve_back(1);
        matmul_block_init(cb_dh, cb_a, false, 1, 1, 1);
        tile_regs_acquire();
        matmul_block(
            cb_dh, cb_a, 0, 0, 0, false, 1, 1, 1);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(cb_lora_dx)));
        pack_tile<true>(0, cb_lora_dx, 0);
        tile_regs_release();
        lora_dx.push_back(1);
        a.pop_front(1);
        dh.pop_front(1);
    }

    // Match PyTorch's observable BF16 base + LoRA addition. Switching the
    // destination mode once per core avoids Blackhole SyncHalf corruption.
    add_tiles_init(cb_base_dx, cb_lora_dx);
    disable_fp32_dest_acc();
    for (uint32_t local_job = 0; local_job < jobs_per_node; ++local_job) {
        const uint32_t job = node_index * jobs_per_node + local_job;
        if (job >= total_jobs) {
            continue;
        }
        base_dx.wait_front(1);
        lora_dx.wait_front(1);
        combined_dx.reserve_back(1);
        tile_regs_acquire();
        tail_add_tiles_bf16(
            cb_base_dx, cb_lora_dx, 0, 0, 0);
        tail_tile_regs_commit_bf16();
        tile_regs_wait();
        tail_pack_reconfig_bf16(cb_combined_dx);
        tail_pack_tile_bf16(0, cb_combined_dx, 0);
        tail_tile_regs_release_bf16();
        combined_dx.push_back(1);
        lora_dx.pop_front(1);
        base_dx.pop_front(1);
    }
    enable_fp32_dest_acc();

    da_phase();
}
