// Appended after:
//   1. exact phase-one helpers/kernel rewritten as resident_phase_one()
//   2. grouped-tail helpers with its old kernel rewritten as unused_tail()
//
// The helpers retain the accepted HiFi4/FP32 reduction schedule and explicit
// BF16 packing points.  This file only changes residency and core ownership.

#include "tools/profiler/kernel_profiler.hpp"

ALWI void run_resident_lora_dx(
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

FORCE_INLINE void resident_tail() {
    constexpr uint32_t cb_base = get_compile_time_arg_val(3);
    constexpr uint32_t cb_dh = get_compile_time_arg_val(16);
    constexpr uint32_t cb_a = get_compile_time_arg_val(5);
    constexpr uint32_t cb_lora = get_compile_time_arg_val(6);
    constexpr uint32_t cb_peer_sum = get_compile_time_arg_val(18);
    constexpr uint32_t cb_x = get_compile_time_arg_val(9);
    constexpr uint32_t cb_transposed_x = get_compile_time_arg_val(17);
    // Keep producer ownership on the compute RISC across both phases.
    constexpr uint32_t cb_intermediate = get_compile_time_arg_val(11);
    constexpr uint32_t cb_final_dx = get_compile_time_arg_val(12);
    constexpr uint32_t cb_da_partial = get_compile_time_arg_val(13);
    constexpr uint32_t cb_da_pretranspose = get_compile_time_arg_val(14);
    constexpr uint32_t cb_da = get_compile_time_arg_val(15);

    constexpr uint32_t grid_cols = 11;
    constexpr uint32_t member_nodes = 55;
    constexpr uint32_t m_tiles = @M_TILES@;
    constexpr uint32_t k_tiles = @K_TILES@;
    constexpr uint32_t jobs_per_node = @JOBS_PER_NODE@;
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
        run_da(
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
        run_resident_lora_dx(
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
        run_bf16_add(
            cb_base,
            cb_lora,
            cb_peer_sum,
            jobs_per_node,
            total_jobs,
            node);
    } else {
        DeviceZoneScopedN("resident_member0_bf16_adds");
        run_bf16_add(
            cb_peer_sum,
            cb_lora,
            cb_intermediate,
            jobs_per_node,
            total_jobs,
            node);
        run_bf16_add(
            cb_intermediate,
            cb_base,
            cb_final_dx,
            jobs_per_node,
            total_jobs,
            node);
    }
    enable_fp32_dest_acc();
}

void kernel_main() {
    {
        DeviceZoneScopedN("resident_phase_one_compute");
        resident_phase_one();
    }
    resident_tail();
}
