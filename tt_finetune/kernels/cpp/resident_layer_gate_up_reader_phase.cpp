// All-core grouped gate/up reader phase appended to the resident attention
// reader kernels.  Post-attention RMSNorm rows arrive directly from the four
// CB22 producers through L1/NoC; only immutable weights are read from DRAM.

namespace resident_layer_gate_up_reader {

constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kWorkers = @GATE_UP_WORKERS@;
constexpr uint32_t kColumns = @GATE_UP_GRID_COLUMNS@;
constexpr uint32_t kXOffset = @GATE_UP_X_OFFSET@;
constexpr uint8_t kPhysicalX[11] = {
    1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13};
constexpr uint32_t kXCache = 0;
constexpr uint32_t kA = 15;
constexpr uint32_t kHResident = 14;
constexpr uint32_t kW = 6;
constexpr uint32_t kB = 9;
constexpr uint32_t kHCredit = 0;
constexpr uint32_t kInitCount = 13;
constexpr uint32_t kInitArmed = 12;
constexpr uint32_t kInitReady = 2;
constexpr uint32_t kXCredit = 3;
constexpr uint32_t kXRouteCount = 7;
constexpr uint32_t kDownReaderArmed = 5;
constexpr bool kFullMLPForward = @FULL_MLP_FORWARD@;
constexpr bool kPairedSwiGLU = @PAIRED_SWIGLU@;

FORCE_INLINE uint64_t worker_l1(uint32_t node, uint32_t address) {
    return get_noc_addr(
        kPhysicalX[kXOffset + node % kColumns],
        static_cast<uint8_t>(node / kColumns + 2), address);
}

FORCE_INLINE uint64_t projection_l1(uint32_t node, uint32_t address) {
    return get_noc_addr(
        kPhysicalX[node % 5], static_cast<uint8_t>(node / 5 + 2), address);
}

FORCE_INLINE uint32_t minimum(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

// The attention epoch used several of the same physical FIFOs.  Reusing an
// empty CB without returning both its counters and ring pointers to the
// physical origin makes remote writers target one address while the local
// producer publishes another.  The reader RISC owns these producer views.
static __attribute__((noinline, noclone)) void
rebase_empty_producer_cb(uint32_t cb_id) {
    volatile uint32_t* received = get_cb_tiles_received_ptr(cb_id);
    const uintptr_t ack_address =
        reinterpret_cast<uintptr_t>(get_cb_tiles_acked_ptr(cb_id));
    const uint16_t pushed = static_cast<uint16_t>(received[0]);
    while (true) {
        invalidate_l1_cache();
        if (static_cast<uint16_t>(reg_read(ack_address)) == pushed) break;
    }
    *get_cb_tiles_received_ptr(cb_id) = 0;
    *get_cb_tiles_acked_ptr(cb_id) = 0;
    auto& interface = get_local_cb_interface(cb_id);
    const uint32_t base = interface.fifo_limit - interface.fifo_size;
    interface.fifo_rd_ptr = base;
    interface.fifo_wr_ptr = base;
    interface.tiles_acked_received_init = 0;
    interface.fifo_wr_tile_ptr = 0;
}

// Arm a reusable all-reader generation before active attention readers and
// idle readers diverge.  The root counter is zero at the end of every prior
// generation; the arrival barrier prevents an idle reader from publishing a
// gate/up reservation while node zero still reuses attention semaphores.
static __attribute__((noinline, noclone,
                      optimize("Os,no-unroll-loops")))
void begin_generation() {
    const uint32_t node =
        get_absolute_logical_y() * kColumns +
        (get_absolute_logical_x() - kXOffset);
    auto* armed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(kInitArmed));
    auto* count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(kInitCount));
    noc_semaphore_set(armed, 0);
    if constexpr (kFullMLPForward) {
        // The projection writer publishes the retained residual base only
        // after attention. Clear the mailbox before that producer can run.
        auto* residual_base =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(8));
        noc_semaphore_set(residual_base, 0);
    }
    noc_semaphore_inc(worker_l1(0, get_semaphore(kInitCount)), 1);
    noc_async_atomic_barrier();
    if (node == 0) {
        noc_semaphore_wait_min(count, kWorkers);
        noc_semaphore_set(count, 0);
        noc_semaphore_set(armed, 1);
        for (uint32_t target = 1; target < kWorkers; ++target)
            noc_async_write(
                get_semaphore(kInitArmed),
                worker_l1(target, get_semaphore(kInitArmed)),
                sizeof(uint32_t));
        noc_async_write_barrier();
    }
    noc_semaphore_wait(armed, 1);
}

template <typename PackedAccessor>
static __attribute__((noinline, noclone,
                      optimize("Os,no-unroll-loops,no-peel-loops,no-unswitch-loops")))
void run(
    const PackedAccessor& packed_parameters) {
    WAYPOINT("GRS0");
    Noc noc0(0);
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@;
    constexpr uint32_t intermediate_tiles = @INTERMEDIATE_TILES@;
    constexpr uint32_t total_n_tiles =
        kPairedSwiGLU ? intermediate_tiles : 2 * intermediate_tiles;
    constexpr uint32_t total_jobs = seq_tiles * total_n_tiles;
    constexpr uint32_t jobs_per_node =
        (total_jobs + kWorkers - 1) / kWorkers;
    constexpr uint32_t base_phase = @GATE_UP_BASE_PHASE@;
    constexpr uint32_t base_blocks = hidden_tiles / base_phase;

    const uint32_t node =
        get_absolute_logical_y() * kColumns +
        (get_absolute_logical_x() - kXOffset);
    const uint32_t first_job = node * jobs_per_node;
    const uint32_t stop_job = minimum(first_job + jobs_per_node, total_jobs);
    const uint32_t valid_jobs = stop_job > first_job ? stop_job - first_job : 0;
    const uint32_t first_row = valid_jobs ? first_job / total_n_tiles : 0;
    const uint32_t last_row = valid_jobs ? (stop_job - 1) / total_n_tiles : 0;
    const uint32_t cached_rows = valid_jobs ? last_row - first_row + 1 : 0;
    const uint32_t first_group = valid_jobs ? first_job / intermediate_tiles : 0;
    const uint32_t last_group = valid_jobs ?
        (stop_job - 1) / intermediate_tiles : 0;
    const uint32_t local_groups = valid_jobs ? (
        kPairedSwiGLU ? 2 * cached_rows : last_group - first_group + 1) : 0;

    CircularBuffer x_cache(kXCache), a_cb(kA), h_resident(kHResident);
    CircularBuffer w_cb(kW), b_cb(kB);
    auto* h_credit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(kHCredit));
    auto* x_credit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(kXCredit));
    auto* init_count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(kInitCount));
    auto* init_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(kInitReady));

    // All four words were used by earlier attention epochs.  Every worker
    // clears its local state before joining the new generation.
    noc_semaphore_set(h_credit, 0);
    noc_semaphore_set(x_credit, 0);
    noc_semaphore_set(init_ready, 0);
    if constexpr (kFullMLPForward) {
        // Semaphores 9 and 10 are still owned by the co-located attention
        // writer here: they carry the O-input base mailbox and O-go edge.
        // The down reader resets/reuses them only after the all-writer phase
        // barrier, which proves the O epoch and gate/up routing complete.
        constexpr uint32_t phase_semaphores[] = {1, 4, 5, 6, 7};
        for (uint32_t sem : phase_semaphores)
            noc_semaphore_set(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(sem)), 0);
        // Writers with no local gate/up jobs can otherwise publish the down
        // phase before this reader finishes clearing the reused semaphore
        // epoch.  Arm the local writer only after every phase word is clean.
        noc_semaphore_set(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(kDownReaderArmed)), 1);
        // kXCredit is semaphore 3 and remains zero until the projection
        // router publishes normalized rows below.
    }
    constexpr uint32_t producer_cbs[] = {
        kXCache, kA, kHResident, kW, kB};
    for (uint32_t cb_id : producer_cbs) rebase_empty_producer_cb(cb_id);
    volatile tt_l1_ptr uint32_t* x_payload_ready = nullptr;
    if (cached_rows) {
        x_cache.reserve_back(cached_rows * hidden_tiles);
        auto& x_interface = get_local_cb_interface(kXCache);
        const uint32_t x_base =
            x_interface.fifo_limit - x_interface.fifo_size;
        x_payload_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            x_base + cached_rows * hidden_tiles * kTileBytes);
        noc_semaphore_set(x_payload_ready, 0);
    }
    noc_semaphore_inc(worker_l1(0, get_semaphore(kInitCount)), 1);
    noc_async_atomic_barrier();
    if (node == 0) {
        noc_semaphore_wait_min(init_count, kWorkers);
        noc_semaphore_set(init_count, 0);
        noc_semaphore_set(init_ready, 1);
        for (uint32_t target = 1; target < kWorkers; ++target)
            noc_async_write(
                get_semaphore(kInitReady),
                worker_l1(target, get_semaphore(kInitReady)),
                sizeof(uint32_t));
        // The four projection row owners publish CB22. They reset this
        // generation before attention work starts and wait locally here.
        for (uint32_t target = 0; target < @SEQ_TILES@; ++target)
            noc_async_write(
                get_semaphore(kInitReady),
                projection_l1(target, get_semaphore(kInitReady)),
                sizeof(uint32_t));
        noc_async_write_barrier();
    }
    while (true) {
        invalidate_l1_cache();
        WAYPOINT("GRIW");
        if (*init_ready == 1) break;
    }
    WAYPOINT("GRI1");

    // Every worker receives a completion word from the same projection row
    // and NoC stream that wrote its normalized-X payload.  This local ordering
    // edge is stronger than a root broadcast from an unrelated address domain.
    if (cached_rows) {
        while (true) {
            invalidate_l1_cache();
            WAYPOINT("GRX0");
            if (*x_payload_ready == 1) break;
        }
        noc_semaphore_set(x_payload_ready, 0);
        x_cache.push_back(cached_rows * hidden_tiles);
    }
    WAYPOINT("GRX1");

    // Pair ownership keeps both rank-H members on the same core and in exact
    // gate-then-up order.  The safe gate/up-only path retains its old grouped
    // schedule unchanged.
    if constexpr (kPairedSwiGLU) {
        for (uint32_t row = 0; row < seq_tiles; ++row) {
            const uint32_t owner =
                (row * intermediate_tiles) / jobs_per_node;
            if (node != owner) continue;
            for (uint32_t member = 0; member < 2; ++member) {
                const uint32_t a_base = member ?
                    @UP_A_PACK_BASE@ : @GATE_A_PACK_BASE@;
                a_cb.reserve_back(hidden_tiles);
                for (uint32_t k = 0; k < hidden_tiles; ++k)
                    noc0.async_read(
                        packed_parameters,
                        CoreLocalMem<uint32_t>(
                            a_cb.get_write_ptr() + k * kTileBytes),
                        kTileBytes, {.page_id = a_base + k}, {});
                noc0.async_read_barrier();
                a_cb.push_back(hidden_tiles);
            }
        }
    } else {
        for (uint32_t group = 0; group < 2 * seq_tiles; ++group) {
            const uint32_t owner =
                (group * intermediate_tiles) / jobs_per_node;
            if (node != owner) continue;
            const bool up_member = (group & 1u) != 0;
            const uint32_t a_base = up_member ?
                @UP_A_PACK_BASE@ : @GATE_A_PACK_BASE@;
            a_cb.reserve_back(hidden_tiles);
            for (uint32_t k = 0; k < hidden_tiles; ++k)
                noc0.async_read(
                    packed_parameters,
                    CoreLocalMem<uint32_t>(
                        a_cb.get_write_ptr() + k * kTileBytes),
                    kTileBytes, {.page_id = a_base + k}, {});
            noc0.async_read_barrier();
            a_cb.push_back(hidden_tiles);
        }
    }

    if (local_groups) h_resident.reserve_back(local_groups);
    if (local_groups) {
        while (true) {
            invalidate_l1_cache();
            WAYPOINT("GRH0");
            if (*h_credit >= local_groups) break;
        }
        noc_semaphore_set(h_credit, 0);
        h_resident.push_back(local_groups);
        WAYPOINT("GRH1");
    }

    WAYPOINT("GRW0");
    for (uint32_t job = first_job; job < stop_job; ++job) {
        const uint32_t logical_n = job % total_n_tiles;
        const uint32_t members = kPairedSwiGLU ? 2 : 1;
        for (uint32_t local_member = 0; local_member < members; ++local_member) {
            const bool up_member = kPairedSwiGLU ?
                local_member != 0 : logical_n >= intermediate_tiles;
            const uint32_t member_n = logical_n % intermediate_tiles;
            const uint32_t packed_n =
                (up_member ? intermediate_tiles : 0) + member_n;
            for (uint32_t block = 0; block < base_blocks; ++block) {
                w_cb.reserve_back(base_phase);
                for (uint32_t inner = 0; inner < base_phase; ++inner) {
                    const uint32_t k = block * base_phase + inner;
                    noc0.async_read(
                        packed_parameters,
                        CoreLocalMem<uint32_t>(
                            w_cb.get_write_ptr() + inner * kTileBytes),
                        kTileBytes,
                        {.page_id = @GATE_UP_WEIGHT_PACK_BASE@
                            + k * (2 * intermediate_tiles) + packed_n}, {});
                }
                noc0.async_read_barrier();
                w_cb.push_back(base_phase);
            }
            b_cb.reserve_back(1);
            const uint32_t b_base = up_member ?
                @UP_B_PACK_BASE@ : @GATE_B_PACK_BASE@;
            noc0.async_read(
                packed_parameters, CoreLocalMem<uint32_t>(b_cb.get_write_ptr()),
                kTileBytes, {.page_id = b_base + member_n}, {});
            noc0.async_read_barrier();
            b_cb.push_back(1);
        }
    }
    WAYPOINT("GRW1");
}

}  // namespace resident_layer_gate_up_reader
