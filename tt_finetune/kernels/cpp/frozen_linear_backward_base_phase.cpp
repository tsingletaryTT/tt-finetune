// Appended after lora_backward_arena_exact_compute.cpp has been rewritten
// from kernel_main() into lora_backward_phase(). CBs 0..3 belong to this
// exact frozen-base dX phase; the shifted LoRA phase uses CBs 4..15.

void kernel_main() {
    constexpr uint32_t cb_dy = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(1);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(2);
    constexpr uint32_t cb_result = get_compile_time_arg_val(3);

    constexpr uint32_t jobs_per_node = @JOBS_PER_NODE@;
    constexpr uint32_t total_jobs = @TOTAL_JOBS@;
    constexpr uint32_t n_tiles = @N_TILES@;
    constexpr uint32_t reduction_tiles_per_phase =
        @REDUCTION_TILES_PER_PHASE@;
    constexpr uint32_t reduction_blocks =
        n_tiles / reduction_tiles_per_phase;
    constexpr uint32_t grid_cols = @GRID_COLS@;

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
