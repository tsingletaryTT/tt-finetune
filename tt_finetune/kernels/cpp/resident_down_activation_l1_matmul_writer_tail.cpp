
    constexpr uint32_t cb_pulled_dgate = get_compile_time_arg_val(@PULLED_DGATE_ARG@);
    constexpr uint32_t cb_pulled_dup = get_compile_time_arg_val(@PULLED_DUP_ARG@);
    CircularBuffer pulled_dgate(cb_pulled_dgate);
    CircularBuffer pulled_dup(cb_pulled_dup);
    const uint32_t first_output_job = node * jobs_per_node;
    for (uint32_t local = 0; local < jobs_per_node; ++local) {
        const uint32_t job = first_output_job + local;
        if (job >= total_jobs) continue;
        pulled_dgate.wait_front(1);
        pulled_dup.wait_front(1);
        noc1.async_write(
            CoreLocalMem<uint32_t>(pulled_dgate.get_read_ptr()), arena,
            kTileBytes, {}, {.page_id = job});
        noc1.async_write(
            CoreLocalMem<uint32_t>(pulled_dup.get_read_ptr()), arena,
            kTileBytes, {}, {.page_id = m_tiles * arena_cols + job});
        noc1.async_write_barrier();
        pulled_dup.pop_front(1);
        pulled_dgate.pop_front(1);
    }
