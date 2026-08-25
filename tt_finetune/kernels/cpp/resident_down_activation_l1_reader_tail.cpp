
    // The activation island has completed every exact dGate/dUp tile and
    // published the bank base addresses to this core's semaphore mailboxes.
    volatile tt_l1_ptr uint32_t* output_ready_flag =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(@OUTPUT_READY_SEMAPHORE@));
    noc_semaphore_wait(output_ready_flag, 1);
    noc_semaphore_set(output_ready_flag, 0);

    constexpr uint32_t cb_pulled_dgate = get_compile_time_arg_val(@PULLED_DGATE_ARG@);
    constexpr uint32_t cb_pulled_dup = get_compile_time_arg_val(@PULLED_DUP_ARG@);
    CircularBuffer pulled_dgate(cb_pulled_dgate);
    CircularBuffer pulled_dup(cb_pulled_dup);
    const uint32_t first_output_job = node * jobs_per_node;
    for (uint32_t local = 0; local < jobs_per_node; ++local) {
        const uint32_t job = first_output_job + local;
        if (job >= total_jobs) continue;
        const uint32_t producer = job / jobs_per_node;
        const uint32_t activation_node = producer % 10;
        uint32_t local_index = job - producer * jobs_per_node;
        for (uint32_t prior = activation_node; prior < producer; prior += 10) {
            const uint32_t first = prior * jobs_per_node;
            const uint32_t remaining = total_jobs - first;
            local_index += remaining < jobs_per_node
                ? remaining : jobs_per_node;
        }
        const uint32_t bank = local_index / 32;
        const uint32_t slot = local_index - bank * 32;
        volatile tt_l1_ptr uint32_t* dgate_address =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(@OUTPUT_ADDRESS_SEMAPHORE_START@ + 2 * bank));
        volatile tt_l1_ptr uint32_t* dup_address =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(@OUTPUT_ADDRESS_SEMAPHORE_START@ + 2 * bank + 1));
        pulled_dgate.reserve_back(1);
        pulled_dup.reserve_back(1);
        noc_async_read(
            get_noc_addr(
                13, physical_y(activation_node),
                *dgate_address + slot * kTileBytes),
            pulled_dgate.get_write_ptr(), kTileBytes);
        noc_async_read(
            get_noc_addr(
                13, physical_y(activation_node),
                *dup_address + slot * kTileBytes),
            pulled_dup.get_write_ptr(), kTileBytes);
        noc_async_read_barrier();
        pulled_dup.push_back(1);
        pulled_dgate.push_back(1);
    }
    noc_semaphore_inc(
        get_noc_addr(
            13, physical_y(0), get_semaphore(@CONSUMER_DONE_SEMAPHORE@)),
        1);
    noc_async_atomic_barrier();
