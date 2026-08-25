// All-core final/checkpoint writer and rank-H distributor for the layer-owned
// grouped gate/up epoch.  The arena stores only backward-lifetime outputs.

namespace resident_layer_gate_up_writer {

constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kWorkers = @GATE_UP_WORKERS@;
constexpr uint32_t kColumns = @GATE_UP_GRID_COLUMNS@;
constexpr uint32_t kXOffset = @GATE_UP_X_OFFSET@;
constexpr uint8_t kPhysicalX[11] = {
    1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13};
constexpr uint32_t kHPublish = 4;
constexpr uint32_t kHResident = 14;
constexpr uint32_t kOutput = 12;
constexpr bool kFullMLPForward = @FULL_MLP_FORWARD@;
constexpr bool kPairedSwiGLU = @PAIRED_SWIGLU@;
constexpr uint32_t kGateWindow = 17;
constexpr uint32_t kUpWindow = 18;
constexpr uint32_t kDownSavedH = 6;
constexpr uint32_t kLayerOutput = 29;
constexpr uint32_t kRouteWriterCount = 4;
constexpr uint32_t kDownReaderArmed = 5;
constexpr uint32_t kDownPhaseReady = 6;
constexpr uint32_t kGateUpInitReady = 2;

FORCE_INLINE uint64_t worker_l1(uint32_t node, uint32_t address) {
    return get_noc_addr(
        kPhysicalX[kXOffset + node % kColumns],
        static_cast<uint8_t>(node / kColumns + 2), address);
}
FORCE_INLINE uint64_t projection_row_l1(uint32_t row, uint32_t address) {
    return get_noc_addr(
        kPhysicalX[row % 5], static_cast<uint8_t>(row / 5 + 2), address);
}
FORCE_INLINE uint32_t projection_completion_address(uint32_t row) {
    // Q/K/V target-base mailboxes 5/6/7 are dead by the post-normalized
    // phase. Together with semaphore 2 they provide four independent flags
    // without allocating or aliasing another L1 control word.
    return row == 0 ? get_semaphore(2) : get_semaphore(4 + row);
}
FORCE_INLINE uint32_t minimum(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

// A circular buffer is a shared epoch between its producer and consumer
// RISCs.  The attention writer has consumed CB4/CB12 before the joined MLP
// phase reuses those IDs for H publication and the final gate/up output.  A
// new TT-Metal program would initialise these counters for us; an in-program
// phase transition must do it explicitly after proving the old epoch empty.
static __attribute__((noinline, noclone,
                      optimize("Os,no-unroll-loops,no-peel-loops,no-unswitch-loops"))) void
rebase_empty_consumer_cb(uint32_t cb_id) {
    // Every call is dominated by an attention- or gate/up-wide completion
    // fence. That global fence is the authoritative proof that the previous
    // producer finished; CB-local counters may belong to the opposite RISC
    // direction from an earlier aliased epoch.
    *get_cb_tiles_received_ptr(cb_id) = 0;
    *get_cb_tiles_acked_ptr(cb_id) = 0;
    auto& interface = get_local_cb_interface(cb_id);
    const uint32_t base = interface.fifo_limit - interface.fifo_size;
    interface.fifo_rd_ptr = base;
    interface.fifo_wr_ptr = base;
    interface.tiles_acked_received_init = 0;
    interface.fifo_wr_tile_ptr = 0;
}

static __attribute__((noinline, noclone,
                      optimize("Os,no-unroll-loops,no-peel-loops,no-unswitch-loops"))) void
wait_for_x_receivers(uint32_t row) {
    WAYPOINT("PWR0");
    auto* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(2));
    noc_semaphore_wait(ready, 1);
    WAYPOINT("PWR1");
    // The attention reader is the sole producer of this generation word.
    // Clear it only after consuming the publication.  Resetting it at writer
    // startup races the remote NoC write and can erase a ready generation.
    noc_semaphore_set(ready, 0);
    (void)row;
}

static __attribute__((noinline, noclone,
                      optimize("Os,no-unroll-loops,no-peel-loops,no-unswitch-loops"))) void
route_normalized_tile(
    uint32_t source, uint32_t row, uint32_t hidden_tile) {
    if (row == 0) WAYPOINT("PX00");
    else if (row == 1) WAYPOINT("PX10");
    else if (row == 2) WAYPOINT("PX20");
    else WAYPOINT("PX30");
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@;
    constexpr uint32_t intermediate_tiles = @INTERMEDIATE_TILES@;
    constexpr uint32_t jobs_per_row =
        kPairedSwiGLU ? intermediate_tiles : 2 * intermediate_tiles;
    constexpr uint32_t jobs_per_node =
        (@SEQ_TILES@ * jobs_per_row + kWorkers - 1) / kWorkers;
    const uint32_t first_row_job = row * jobs_per_row;
    const uint32_t last_row_job = first_row_job + jobs_per_row - 1;
    const uint32_t first_target = first_row_job / jobs_per_node;
    const uint32_t last_target = last_row_job / jobs_per_node;
    // CB byte layouts 0..14 are deliberately identical on both former
    // attention islands, so one local CB0 base is valid for every worker.
    auto& x_interface = get_local_cb_interface(0);
    const uint32_t x_base = x_interface.fifo_limit - x_interface.fifo_size;
    for (uint32_t target = first_target; target <= last_target; ++target) {
        const uint32_t target_first = target * jobs_per_node;
        const uint32_t target_stop = minimum(
            target_first + jobs_per_node, @SEQ_TILES@ * jobs_per_row);
        if (target_stop <= target_first) continue;
        const uint32_t target_first_row = target_first / jobs_per_row;
        const uint32_t target_last_row = (target_stop - 1) / jobs_per_row;
        if (row < target_first_row || row > target_last_row) continue;
        const uint32_t slot = row - target_first_row;
        noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(
            source,
            worker_l1(
                target, x_base + (slot * hidden_tiles + hidden_tile) * kTileBytes),
            kTileBytes);
        // This L1 handoff is followed by a posted completion atomic from the
        // same source on the same NoC/VC. The root observes all four credits
        // before releasing any consumer, so a return acknowledgement for
        // every 2-KiB data packet is unnecessary and can deadlock Blackhole's
        // cross-island acknowledgement path.
    }
    // Publish one payload-domain completion word to every target after that
    // target's complete normalized-X row.  The data packets and completion
    // word originate on the same NoC and target the same core, so observing
    // the local word proves that the preceding posted payload writes have
    // reached that core.  A former root aggregation released all readers from
    // a different address domain; its small acknowledged completion could
    // overtake posted 2-KiB data packets and exposed partially written rows.
    if (hidden_tile + 1 == hidden_tiles) {
        // Posted L1 writes do not participate in noc_async_write_barrier().
        // Drain their injection queue before publishing the row-complete
        // atomic; otherwise the atomic may become visible while one or more
        // normalized-X packets are still queued (or, under sustained
        // cross-island traffic, fail to make forward progress behind them).
        // This is a departure fence, not a remote acknowledgement, so the
        // handoff remains one-way and cannot recreate the Blackhole ACK
        // cycle that the posted route removed.
        noc_async_posted_writes_flushed();
        if (row == 0) WAYPOINT("PF00");
        else if (row == 1) WAYPOINT("PF10");
        else if (row == 2) WAYPOINT("PF20");
        else WAYPOINT("PF30");
        noc_semaphore_set(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(2)), 1);
        for (uint32_t target = first_target; target <= last_target; ++target) {
            const uint32_t target_first = target * jobs_per_node;
            const uint32_t target_stop = minimum(
                target_first + jobs_per_node, @SEQ_TILES@ * jobs_per_row);
            if (target_stop <= target_first) continue;
            const uint32_t target_first_row = target_first / jobs_per_row;
            const uint32_t target_last_row = (target_stop - 1) / jobs_per_row;
            if (row < target_first_row || row > target_last_row) continue;
            const uint32_t target_cached_rows =
                target_last_row - target_first_row + 1;
            noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(
                get_semaphore(2),
                worker_l1(
                    target,
                    x_base + target_cached_rows * hidden_tiles * kTileBytes),
                sizeof(uint32_t));
        }
        noc_async_posted_writes_flushed();
    }
    if (row == 0) WAYPOINT("PX0D");
    else if (row == 1) WAYPOINT("PX1D");
    else if (row == 2) WAYPOINT("PX2D");
    else WAYPOINT("PX3D");
}

template <typename ArenaAccessor>
static __attribute__((noinline, noclone,
                      optimize("Os,no-unroll-loops,no-peel-loops,no-unswitch-loops")))
void run(
    Noc& noc1, const ArenaAccessor& arena) {
    WAYPOINT("GWS0");
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t intermediate_tiles = @INTERMEDIATE_TILES@;
    constexpr uint32_t total_n_tiles =
        kPairedSwiGLU ? intermediate_tiles : 2 * intermediate_tiles;
    constexpr uint32_t total_jobs = seq_tiles * total_n_tiles;
    constexpr uint32_t jobs_per_node =
        (total_jobs + kWorkers - 1) / kWorkers;
    const uint32_t node =
        get_absolute_logical_y() * kColumns +
        (get_absolute_logical_x() - kXOffset);
    const uint32_t first_job = node * jobs_per_node;
    const uint32_t stop_job = minimum(first_job + jobs_per_node, total_jobs);
    CircularBuffer h_publish(kHPublish), output(kOutput);
    auto& gate_window_interface = get_local_cb_interface(kGateWindow);
    auto& up_window_interface = get_local_cb_interface(kUpWindow);
    const uint32_t gate_window_base =
        gate_window_interface.fifo_limit - gate_window_interface.fifo_size;
    const uint32_t up_window_base =
        up_window_interface.fifo_limit - up_window_interface.fifo_size;
    auto& h_interface = get_local_cb_interface(kHResident);
    const uint32_t h_resident_base =
        h_interface.fifo_limit - h_interface.fifo_size;

    // The gate/up reader clears every reused semaphore/CB epoch and reserves
    // its L1 receive windows before publishing this generation.  This fence
    // must dominate *all* H/output writes and credit atomics below.  Waiting
    // only at the end of the writer phase allowed a fast writer to publish an
    // H credit which the slower reader subsequently erased during startup.
    auto* init_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(kGateUpInitReady));
    auto* reader_armed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(kDownReaderArmed));
    if constexpr (kFullMLPForward) {
        WAYPOINT("GWIR");
        while (true) {
            invalidate_l1_cache();
            if (*init_ready == 1) break;
        }
        WAYPOINT("GWAR");
        while (true) {
            invalidate_l1_cache();
            if (*reader_armed >= 1) break;
        }
    }

    if constexpr (kPairedSwiGLU) {
        for (uint32_t row = 0; row < seq_tiles; ++row) {
            const uint32_t row_first = row * intermediate_tiles;
            const uint32_t row_stop = row_first + intermediate_tiles;
            const uint32_t owner = row_first / jobs_per_node;
            if (node != owner) continue;
            const uint32_t first_target = row_first / jobs_per_node;
            const uint32_t last_target = (row_stop - 1) / jobs_per_node;
            for (uint32_t member = 0; member < 2; ++member) {
                h_publish.wait_front(1);
                noc1.async_write(
                    CoreLocalMem<uint32_t>(h_publish.get_read_ptr()), arena,
                    kTileBytes, {},
                    {.page_id = (member ? @MLP_UP_H_BASE@ : @MLP_GATE_H_BASE@)
                        + row});
                for (uint32_t target = first_target;
                     target <= last_target; ++target) {
                    const uint32_t target_row =
                        (target * jobs_per_node) / intermediate_tiles;
                    const uint32_t slot = (row - target_row) * 2 + member;
                    noc_async_write(
                        h_publish.get_read_ptr(),
                        worker_l1(
                            target, h_resident_base + slot * kTileBytes),
                        kTileBytes);
                }
                noc_async_write_barrier();
                for (uint32_t target = first_target;
                     target <= last_target; ++target)
                    noc_semaphore_inc<true>(
                        worker_l1(target, get_semaphore(0)), 1);
                h_publish.pop_front(1);
            }
        }
    } else {
        for (uint32_t group = 0; group < 2 * seq_tiles; ++group) {
            const uint32_t group_first = group * intermediate_tiles;
            const uint32_t group_stop = group_first + intermediate_tiles;
            const uint32_t owner = group_first / jobs_per_node;
            if (node != owner) continue;
            h_publish.wait_front(1);
            const uint32_t row = group / 2;
            const uint32_t member = group & 1u;
            noc1.async_write(
                CoreLocalMem<uint32_t>(h_publish.get_read_ptr()), arena,
                kTileBytes, {},
                {.page_id = (member ? @MLP_UP_H_BASE@ : @MLP_GATE_H_BASE@)
                    + row});
            const uint32_t first_target = group_first / jobs_per_node;
            const uint32_t last_target = (group_stop - 1) / jobs_per_node;
            for (uint32_t target = first_target;
                 target <= last_target; ++target) {
                const uint32_t target_first_job = target * jobs_per_node;
                const uint32_t target_first_group =
                    target_first_job / intermediate_tiles;
                const uint32_t slot = group - target_first_group;
                noc_async_write(
                    h_publish.get_read_ptr(),
                    worker_l1(
                        target, h_resident_base + slot * kTileBytes),
                    kTileBytes);
            }
            noc_async_write_barrier();
            for (uint32_t target = first_target;
                 target <= last_target; ++target)
                noc_semaphore_inc<true>(
                    worker_l1(target, get_semaphore(0)), 1);
            h_publish.pop_front(1);
        }
    }

    WAYPOINT("GWO0");
    const uint32_t valid_jobs = stop_job - first_job;
    if constexpr (kPairedSwiGLU) {
        if (valid_jobs) output.wait_front(2 * valid_jobs);
    }
    for (uint32_t job = first_job; job < stop_job; ++job) {
        const uint32_t row = job / total_n_tiles;
        const uint32_t n = job - row * total_n_tiles;
        if constexpr (kPairedSwiGLU) {
            const uint32_t local = job - first_job;
            noc1.async_write(
                CoreLocalMem<uint32_t>(
                    output.get_read_ptr() + 2 * local * kTileBytes), arena,
                kTileBytes, {},
                {.page_id = @MLP_GATE_BASE@
                    + row * intermediate_tiles + n});
            noc1.async_write(
                CoreLocalMem<uint32_t>(
                    output.get_read_ptr() + (2 * local + 1) * kTileBytes),
                arena, kTileBytes, {},
                {.page_id = @MLP_UP_BASE@
                    + row * intermediate_tiles + n});
            noc1.async_write_barrier();
        } else {
            const bool up_member = n >= intermediate_tiles;
            const uint32_t member_n = n % intermediate_tiles;
            output.wait_front(1);
            noc1.async_write(
                CoreLocalMem<uint32_t>(output.get_read_ptr()), arena,
                kTileBytes, {},
                {.page_id = (up_member ? @MLP_UP_BASE@ : @MLP_GATE_BASE@)
                    + row * intermediate_tiles + member_n});
            if constexpr (kFullMLPForward) {
            // The exact decomposed SwiGLU consumes gate twice (sigmoid and
            // gate*sigmoid) and up once.  Publish those three logical tiles
            // directly into the next phase's per-core L1 windows.  No CB
            // counter is advanced remotely; the receiving reader owns that
            // publication after the device-wide gate/up completion fence.
            const uint32_t product_job = row * intermediate_tiles + member_n;
            const uint32_t target = product_job % kWorkers;
            const uint32_t slot = product_job / kWorkers;
            if (up_member) {
                noc_async_write(
                    output.get_read_ptr(),
                    worker_l1(
                        target, up_window_base + slot * kTileBytes),
                    kTileBytes);
            } else {
                noc_async_write(
                    output.get_read_ptr(),
                    worker_l1(
                        target, gate_window_base + 2 * slot * kTileBytes),
                    kTileBytes);
                noc_async_write(
                    output.get_read_ptr(),
                    worker_l1(
                        target,
                        gate_window_base + (2 * slot + 1) * kTileBytes),
                    kTileBytes);
            }
        }
            noc1.async_write_barrier();
            if constexpr (kFullMLPForward) {
            noc_async_write_barrier();
            const uint32_t product_job = row * intermediate_tiles + member_n;
            const uint32_t target = product_job % kWorkers;
            noc_semaphore_inc<true>(
                worker_l1(target, get_semaphore(1)), up_member ? 1 : 2);
            }
            output.pop_front(1);
        }
    }
    if constexpr (kPairedSwiGLU) {
        if (valid_jobs) output.pop_front(2 * valid_jobs);
    }
    WAYPOINT("GWO1");

    if constexpr (kFullMLPForward) {
        if constexpr (kPairedSwiGLU) {
            // Product publication is the compute-side completion fence. The
            // paired compute releases normalized X and rank H before pushing
            // these tiles, so observing every local product makes CB0/CB14
            // reuse in the down phase race-free. The down reader remains the
            // sole consumer and will pop the products after NoC distribution.
            CircularBuffer product(3);
            const uint32_t valid_pairs =
                stop_job > first_job ? stop_job - first_job : 0;
            if (valid_pairs) product.wait_front(valid_pairs);
        }
        // Gate/up outputs are routed to product owners on other cores.  A
        // local writer completion is therefore insufficient to begin the
        // down epoch: another writer may still own one of this core's gate or
        // up tiles.  Advancing different cores independently also lets the
        // shared compute program enter two incompatible CB epochs at once.
        //
        // First prove that all reader RISCs completed the phase-semaphore
        // reset.  This global generation word is published by the gate/up
        // reader only after all 60 readers joined its init barrier, so remote
        // writers cannot increment the root count before node zero cleared
        // it.  Then every writer joins one device-side completion barrier
        // after all its writes and credit atomics are visible.  The down
        // phase starts everywhere from one clean generation.
        WAYPOINT("GWS1");
        noc_semaphore_inc<true>(
            worker_l1(0, get_semaphore(kRouteWriterCount)), 1);
        // Do not wait for a per-sender atomic acknowledgement here.  The root
        // waits until all kWorkers increments are visible before publishing
        // kDownPhaseReady, and every sender waits for that publication below.
        // Therefore the ready generation is already the completion fence for
        // this increment.  On Blackhole the redundant atomic-ack barrier can
        // retain all writers in NABD while the down compute RISCs wait on
        // CB30, producing a device-wide cycle on the first invocation.

        auto* writer_count =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(kRouteWriterCount));
        if (node == 0) {
            WAYPOINT("GWCT");
            while (true) {
                invalidate_l1_cache();
                if (*writer_count >= kWorkers) break;
            }
            noc_semaphore_set(writer_count, 0);
            // Advance the one reader-owned generation from ARMED (1) to
            // WRITERS_COMPLETE (2).  Keeping phase readiness on this same
            // monotonic word avoids another reused control semaphore whose
            // value could be cleared by a neighbouring epoch.
            noc_semaphore_set(reader_armed, 2);
            for (uint32_t target = 1; target < kWorkers; ++target) {
                noc_async_write(
                    get_semaphore(kDownReaderArmed),
                    worker_l1(target, get_semaphore(kDownReaderArmed)),
                    sizeof(uint32_t));
                if (target % kColumns == kColumns - 1)
                    noc_async_write_barrier();
            }
            noc_async_write_barrier();
        }
        // The co-located reader consumes state 2, rebases all down-phase input
        // FIFOs, and advances this same generation to state 3.  The writer
        // never clears or consumes the word.
        WAYPOINT("GWS2");

        // These two streams are consumed only by this writer in the next
        // phase.  Reset them before allowing the compute RISC to publish.
        rebase_empty_consumer_cb(kDownSavedH);
        rebase_empty_consumer_cb(kLayerOutput);

        // The down reader advances the same reader-owned generation from 1
        // (armed) to 2 (all input FIFOs rebased and the residual prefetched).
        // Do not allocate a second signal in semaphore 9: that word was the
        // O-projection B-input mailbox in the preceding attention epoch, and
        // reusing it here made the tail depend on another cross-RISC alias.
        // Fold state 2 into the one BRISC -> compute release below instead of
        // reusing CB31 in a fourth producer/consumer direction.
        WAYPOINT("GWRB");
        while (true) {
            invalidate_l1_cache();
            if (*reader_armed >= 3) break;
        }

        WAYPOINT("WR30");
        CircularBuffer compute_release(30);
        compute_release.reserve_back(1);
        auto* release_payload =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                compute_release.get_write_ptr());
        // Compute-side CB pointers use 16-byte address units. Publish the
        // writer-qualified data origins instead of reconstructing them from
        // an aliased descriptor in a later RISC epoch.
        release_payload[0] = gate_window_base >> 4;
        release_payload[1] = up_window_base >> 4;
        compute_release.push_back(1);
        WAYPOINT("WR31");
    }
}

template <typename ArenaAccessor>
static __attribute__((noinline, noclone,
                      optimize("Os,no-unroll-loops,no-peel-loops,no-unswitch-loops")))
void run_full_mlp_tail(Noc& noc1, const ArenaAccessor& arena) {
    if constexpr (!kFullMLPForward) return;
    WAYPOINT("DWT0");
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@;
    constexpr uint32_t output_jobs = @SEQ_TILES@ * hidden_tiles;
    constexpr uint32_t output_jobs_per_node =
        (output_jobs + kWorkers - 1) / kWorkers;
    const uint32_t node =
        get_absolute_logical_y() * kColumns +
        (get_absolute_logical_x() - kXOffset);
    CircularBuffer saved_h(kDownSavedH), output(kLayerOutput);
    CircularBuffer scaled_delta(kUpWindow);
    uint32_t valid_outputs = 0;
    for (uint32_t local = 0; local < output_jobs_per_node; ++local)
        if (node + local * kWorkers < output_jobs) ++valid_outputs;
    if (valid_outputs) scaled_delta.wait_front(valid_outputs);
    // This generation is a device-only completion proof.  No diagnostic copy
    // or framework-visible tensor is materialized; compute consumes the token
    // before it performs the residual add that publishes CB29.
    CircularBuffer delta_checkpoint_ready(30);
    delta_checkpoint_ready.reserve_back(1);
    delta_checkpoint_ready.push_back(1);
    for (uint32_t local = 0; local < output_jobs_per_node; ++local) {
        const uint32_t job = node + local * kWorkers;
        if (job >= output_jobs) continue;
        const uint32_t hidden_tile = job % hidden_tiles;
        if (hidden_tile == 0) {
            // Full MLP backward recomputes rank-H in resident L1; retain no
            // redundant forward checkpoint, but drain the exact producer.
            WAYPOINT("DWH0");
            saved_h.wait_front(1);
            saved_h.pop_front(1);
        }
        WAYPOINT("DWO0");
        output.wait_front(1);
        noc1.async_write(
            CoreLocalMem<uint32_t>(output.get_read_ptr()), arena,
            kTileBytes, {}, {.page_id = @LAYER_OUTPUT_BASE@ + job});
        if constexpr (@PUBLISH_LAYER_OUTPUT_AT_PAGE_ZERO@)
            noc1.async_write(
                CoreLocalMem<uint32_t>(output.get_read_ptr()), arena,
                kTileBytes, {}, {.page_id = job});
        noc1.async_write_barrier();
        output.pop_front(1);
    }
    WAYPOINT("DWTD");
}

}  // namespace resident_layer_gate_up_writer
