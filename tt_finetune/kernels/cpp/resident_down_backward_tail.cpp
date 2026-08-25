// Compute tail appended after the exact phase-one helpers.

#include "tools/profiler/kernel_profiler.hpp"

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
    constexpr uint32_t cb_base = get_compile_time_arg_val(3);
    constexpr uint32_t cb_dh = get_compile_time_arg_val(16);
    constexpr uint32_t cb_a = get_compile_time_arg_val(17);
    constexpr uint32_t cb_x = get_compile_time_arg_val(18);
    constexpr uint32_t cb_transposed_x = get_compile_time_arg_val(19);
    constexpr uint32_t cb_da_partial = get_compile_time_arg_val(20);
    constexpr uint32_t cb_da_pretranspose = get_compile_time_arg_val(21);
    constexpr uint32_t cb_da = get_compile_time_arg_val(22);
    constexpr uint32_t cb_lora_dx = get_compile_time_arg_val(23);
    constexpr uint32_t cb_final_dx = get_compile_time_arg_val(24);

    constexpr uint32_t m_tiles = @M_TILES@;
    constexpr uint32_t k_tiles = @K_TILES@;
    constexpr uint32_t jobs_per_node = @JOBS_PER_NODE@;
    constexpr uint32_t da_k_tiles_per_node = @DA_K_TILES_PER_NODE@;
    constexpr uint32_t total_jobs = m_tiles * k_tiles;
    constexpr uint32_t grid_cols = 11;
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

void kernel_main() {
    {
        DeviceZoneScopedN("resident_down_phase_one_compute");
        resident_down_phase_one();
    }
    resident_down_tail();
}
