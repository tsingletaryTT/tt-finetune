// Reader/NoC phase for the in-program exact SwiGLU + down projection.
// Gate/up tiles and the residual source are consumed from L1/NoC; only the
// immutable down-projection parameters are streamed from the packed tensor.

namespace resident_layer_swiglu_down_reader {

constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kWorkers = 60;
constexpr uint32_t kColumns = 6;
constexpr uint32_t kXOffset = 5;
constexpr uint8_t kPhysicalX[11] = {
    1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13};

constexpr uint32_t kGate = 17, kUp = 18;
constexpr uint32_t kProductOut = 3, kProductResident = 5;
constexpr uint32_t kA = 7, kHMulticast = 4, kHResident = 14;
constexpr uint32_t kW = 0, kB = 9, kResidual = 28;
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

// run() is entered only after the all-writer gate/up completion barrier. That
// barrier, rather than a CB-local pushed/acked comparison, proves the old
// epoch is finished. Several of these IDs changed producer direction in the
// attention and gate/up phases, so their local counters are not a valid
// cross-epoch completion predicate.
static __attribute__((noinline, noclone)) void
reset_completed_cb_epoch(uint32_t cb_id) {
    *get_cb_tiles_received_ptr(cb_id) = 0;
    *get_cb_tiles_acked_ptr(cb_id) = 0;
    auto& interface = get_local_cb_interface(cb_id);
    const uint32_t base = interface.fifo_limit - interface.fifo_size;
    interface.fifo_rd_ptr = base;
    interface.fifo_wr_ptr = base;
    interface.tiles_acked_received_init = 0;
    interface.fifo_wr_tile_ptr = 0;
}

template <typename PackedAccessor>
static __attribute__((noinline, noclone,
                      optimize("Os,no-unroll-loops,no-peel-loops,no-unswitch-loops")))
void run(const PackedAccessor& packed_parameters) {
    Noc noc0(0);
    constexpr uint32_t m_tiles = @SEQ_TILES@;
    constexpr uint32_t h_tiles = @HIDDEN_TILES@;
    constexpr uint32_t i_tiles = @INTERMEDIATE_TILES@;
    constexpr uint32_t rank_tiles = 1;
    constexpr uint32_t product_jobs = m_tiles * i_tiles;
    constexpr uint32_t product_jobs_per_core =
        (product_jobs + kWorkers - 1) / kWorkers;
    constexpr uint32_t output_jobs = m_tiles * h_tiles;
    constexpr uint32_t output_jobs_per_core =
        (output_jobs + kWorkers - 1) / kWorkers;
    constexpr uint32_t base_phase = @DOWN_BASE_PHASE@;

    const uint32_t node =
        get_absolute_logical_y() * kColumns +
        (get_absolute_logical_x() - kXOffset);
    const uint32_t first_product = kPairedSwiGLU ?
        node * product_jobs_per_core : node;
    const uint32_t product_stop = kPairedSwiGLU ?
        minimum(first_product + product_jobs_per_core, product_jobs) :
        product_jobs;
    uint32_t valid_products = 0;
    if constexpr (kPairedSwiGLU) {
        valid_products = product_stop > first_product ?
            product_stop - first_product : 0;
    } else {
        for (uint32_t local = 0; local < product_jobs_per_core; ++local)
            if (node + local * kWorkers < product_jobs) ++valid_products;
    }
    uint32_t valid_outputs = 0;
    for (uint32_t local = 0; local < output_jobs_per_core; ++local)
        if (node + local * kWorkers < output_jobs) ++valid_outputs;

    auto* reader_phase = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(5));
    WAYPOINT("DRS0");
    while (true) {
        invalidate_l1_cache();
        if (*reader_phase >= 2) break;
    }
    WAYPOINT("DRS1");
    noc_semaphore_set(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), 0);
    noc_semaphore_set(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(10)), 0);

    if constexpr (!kPairedSwiGLU) {
        WAYPOINT("R170"); reset_completed_cb_epoch(kGate);
        WAYPOINT("R180"); reset_completed_cb_epoch(kUp);
    }
    WAYPOINT("R050"); reset_completed_cb_epoch(kProductResident);
    WAYPOINT("R070"); reset_completed_cb_epoch(kA);
    WAYPOINT("R140"); reset_completed_cb_epoch(kHResident);
    WAYPOINT("R000"); reset_completed_cb_epoch(kW);
    WAYPOINT("R090"); reset_completed_cb_epoch(kB);
    WAYPOINT("R280"); reset_completed_cb_epoch(kResidual);
    // Product and H multicast streams travel compute -> reader.
    if constexpr (!kPairedSwiGLU) {
        WAYPOINT("R030"); reset_completed_cb_epoch(kProductOut);
    }
    WAYPOINT("R040"); reset_completed_cb_epoch(kHMulticast);

    CircularBuffer gate(kGate), up(kUp), product_out(kProductOut);
    CircularBuffer product_resident(kProductResident), a(kA);
    CircularBuffer h_multicast(kHMulticast), h_resident(kHResident);
    CircularBuffer w(kW), b(kB), residual(kResidual);

    if constexpr (!kPairedSwiGLU) {
      if (valid_products) {
        gate.reserve_back(2 * valid_products);
        up.reserve_back(valid_products);
        // The all-writer phase barrier above proves both the L1 tile writes
        // and their atomic credits are complete.  The exact credit value is
        // retained as a lifecycle assertion/reset, not as a progress
        // dependency that can split the 60-core program across two epochs.
        auto* routed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(1));
        WAYPOINT("DRG0");
        invalidate_l1_cache();
        ASSERT(*routed == 3 * valid_products);
        noc_semaphore_set(routed, 0);
        gate.push_back(2 * valid_products);
        up.push_back(valid_products);
      }
    }
    WAYPOINT("DRG1");
    if (valid_outputs) {
        if constexpr (kPairedSwiGLU) {
            // This is a phase-owned fixed L1 window, not a streaming FIFO.
            // The all-writer/product fence proves the preceding epoch empty;
            // remote writers below address the physical origin directly. A
            // dynamic reserve consults stale cross-RISC producer credits from
            // the earlier attention alias and can wait forever despite that
            // device-wide proof. Capacity remains a hard safety invariant.
            ASSERT(
                get_local_cb_interface(kProductResident).fifo_num_pages >=
                valid_outputs * i_tiles);
        } else {
            product_resident.reserve_back(valid_outputs * i_tiles);
        }
        h_resident.reserve_back(valid_outputs * rank_tiles);
        residual.reserve_back(valid_outputs);
    }

    // The projection writer retained one residual row in CB13 on each of the
    // four token-row owners and published the common local base in semaphore
    // 8. The enclosing attention completion barrier makes every row visible.
    auto* residual_mailbox =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(8));
    WAYPOINT("DRR0");
    while (true) {
        invalidate_l1_cache();
        if (*residual_mailbox >= 1) break;
    }
    const uint32_t residual_base = *residual_mailbox;
    for (uint32_t local = 0; local < output_jobs_per_core; ++local) {
        const uint32_t job = node + local * kWorkers;
        if (job >= output_jobs) continue;
        const uint32_t row = job / h_tiles;
        const uint32_t h = job - row * h_tiles;
        noc_async_read(
            projection_l1(row, residual_base + h * kTileBytes),
            residual.get_write_ptr() + local * kTileBytes, kTileBytes);
    }
    if (valid_outputs) {
        noc_async_read_barrier();
        residual.push_back(valid_outputs);
    }
    WAYPOINT("DRR1");

    // Semaphore 5 is owned by this reader across the joined MLP transition.
    // Gate/up initialisation published state 1 and the writer root published
    // state 2; advance it monotonically to state 3 only after every
    // reader-owned FIFO has been rebased and the
    // residual is resident.  This avoids reusing semaphore 9, which carried
    // an O-projection mailbox in the immediately preceding attention epoch.
    noc_semaphore_set(reader_phase, 3);
    WAYPOINT("RR91");

    // Distribute each exact product tile into every output owner's local row.
    WAYPOINT("DRP0");
    if (valid_products) product_out.wait_front(valid_products);
    WAYPOINT("DRP1");
    uint32_t row_counts[m_tiles] = {};
    for (uint32_t local = 0; local < product_jobs_per_core; ++local) {
        const uint32_t job = kPairedSwiGLU ?
            first_product + local : node + local * kWorkers;
        if (job >= product_jobs) continue;
        const uint32_t row = job / i_tiles;
        const uint32_t i = job - row * i_tiles;
        ++row_counts[row];
        const uint32_t source = product_out.get_read_ptr() + local * kTileBytes;
        for (uint32_t h = 0; h < h_tiles; ++h) {
            const uint32_t output_job = row * h_tiles + h;
            const uint32_t target = output_job % kWorkers;
            const uint32_t slot = output_job / kWorkers;
            noc_async_write(
                source,
                worker_l1(
                    target,
                    product_resident.get_write_ptr()
                        + (slot * i_tiles + i) * kTileBytes),
                kTileBytes);
        }
    }
    if (valid_products) {
        noc_async_write_barrier();
        for (uint32_t row = 0; row < m_tiles; ++row) {
            if (!row_counts[row]) continue;
            for (uint32_t h = 0; h < h_tiles; ++h) {
                const uint32_t target = (row * h_tiles + h) % kWorkers;
                noc_semaphore_inc(
                    worker_l1(target, get_semaphore(0)), row_counts[row]);
            }
        }
        noc_async_atomic_barrier();
        product_out.pop_front(valid_products);
    }
    if (valid_outputs) {
        auto* product_ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
        WAYPOINT("DRQ0");
        while (true) {
            invalidate_l1_cache();
            if (*product_ready >= valid_outputs * i_tiles) break;
        }
        noc_semaphore_set(product_ready, 0);
        product_resident.push_back(valid_outputs * i_tiles);
    }
    WAYPOINT("DRP2");

    // Feed the rank reduction only on the h=0 owner for each token row.
    for (uint32_t local = 0; local < output_jobs_per_core; ++local) {
        const uint32_t job = node + local * kWorkers;
        if (job >= output_jobs || job % h_tiles != 0) continue;
        for (uint32_t i = 0; i < i_tiles; i += 2) {
            a.reserve_back(2);
            for (uint32_t inner = 0; inner < 2; ++inner)
                noc0.async_read(
                    packed_parameters,
                    CoreLocalMem<uint32_t>(
                        a.get_write_ptr() + inner * kTileBytes),
                    kTileBytes,
                    {.page_id = @DOWN_A_PACK_BASE@ + i + inner}, {});
            noc0.async_read_barrier();
            a.push_back(2);
        }
        WAYPOINT("DRH0");
        h_multicast.wait_front(1);
        WAYPOINT("DRH1");
        const uint32_t row = job / h_tiles;
        for (uint32_t h = 0; h < h_tiles; ++h) {
            const uint32_t output_job = row * h_tiles + h;
            const uint32_t target = output_job % kWorkers;
            const uint32_t slot = output_job / kWorkers;
            noc_async_write(
                h_multicast.get_read_ptr(),
                worker_l1(
                    target,
                    h_resident.get_write_ptr() + slot * kTileBytes),
                kTileBytes);
        }
        noc_async_write_barrier();
        for (uint32_t h = 0; h < h_tiles; ++h)
            noc_semaphore_inc(
                worker_l1(
                    (row * h_tiles + h) % kWorkers, get_semaphore(10)),
                1);
        noc_async_atomic_barrier();
        h_multicast.pop_front(1);
    }
    if (valid_outputs) {
        auto* h_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(10));
        WAYPOINT("DRQ1");
        while (true) {
            invalidate_l1_cache();
            if (*h_ready >= valid_outputs) break;
        }
        noc_semaphore_set(h_ready, 0);
        h_resident.push_back(valid_outputs);
    }

    for (uint32_t local = 0; local < output_jobs_per_core; ++local) {
        const uint32_t job = node + local * kWorkers;
        if (job >= output_jobs) continue;
        const uint32_t h = job % h_tiles;
        for (uint32_t i = 0; i < i_tiles; i += base_phase) {
            w.reserve_back(base_phase);
            for (uint32_t inner = 0; inner < base_phase; ++inner)
                noc0.async_read(
                    packed_parameters,
                    CoreLocalMem<uint32_t>(
                        w.get_write_ptr() + inner * kTileBytes),
                    kTileBytes,
                    {.page_id = @DOWN_WEIGHT_PACK_BASE@
                        + (i + inner) * h_tiles + h}, {});
            noc0.async_read_barrier();
            w.push_back(base_phase);
        }
        b.reserve_back(1);
        noc0.async_read(
            packed_parameters, CoreLocalMem<uint32_t>(b.get_write_ptr()),
            kTileBytes, {.page_id = @DOWN_B_PACK_BASE@ + h}, {});
        noc0.async_read_barrier();
        b.push_back(1);
    }
    WAYPOINT("DRDN");
}

}  // namespace resident_layer_swiglu_down_reader
