// Direct projection-island multicast into query-row-local Q/K/V CB windows.

#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t kTileBytes = 2048, kProjectionColumns = 5;
constexpr uint32_t kProjectionWorkers = 50, kAttentionColumns = 6;
constexpr uint32_t kAttentionXOffset = 5;
constexpr uint8_t kPhysicalX[11] = {1,2,3,4,5,6,7,10,11,12,13};
FORCE_INLINE uint8_t physical_y(uint32_t y) { return static_cast<uint8_t>(y + 2); }
FORCE_INLINE uint64_t projection_l1(uint32_t node, uint32_t address) {
    return get_noc_addr(kPhysicalX[node % kProjectionColumns],
                        physical_y(node / kProjectionColumns), address);
}
FORCE_INLINE uint64_t attention_l1(uint32_t node, uint32_t address) {
    const uint32_t x = kAttentionXOffset + node % kAttentionColumns;
    return get_noc_addr(kPhysicalX[x], physical_y(node / kAttentionColumns), address);
}
}

void kernel_main() {
    constexpr uint32_t cb_q = 0, cb_k = 3, cb_v = 29;
    constexpr uint32_t seq_tiles = @SEQ_TILES@, dim_tiles = @DIM_TILES@;
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@, kv_tiles = @KV_TILES@;
    constexpr uint32_t query_heads = @QUERY_HEADS@, kv_heads = @KV_HEADS@;
    constexpr uint32_t group = query_heads / kv_heads;
    constexpr uint32_t active_attention = query_heads * seq_tiles;
    constexpr uint32_t q_ready_sem = 0, k_ready_sem = 1, v_ready_sem = 2;
    constexpr uint32_t consumer_ready_sem = 3, producer_go_sem = 4;
    const uint32_t x = get_absolute_logical_x(), y = get_absolute_logical_y();
    if (x >= kProjectionColumns) return;
    const uint32_t node = y * kProjectionColumns + x;
    CircularBuffer q(cb_q), k(cb_k), v(cb_v);
    // All 110 workers share these dedicated CB configurations, therefore the
    // initial FIFO address is identical on the projection and attention
    // islands. Capture it before this producer advances its local front.
    const uint32_t q_target_base = q.get_read_ptr();
    const uint32_t k_target_base = k.get_read_ptr();
    const uint32_t v_target_base = v.get_read_ptr();
    volatile tt_l1_ptr uint32_t* go =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(producer_go_sem));
    if (node == 0) {
        volatile tt_l1_ptr uint32_t* ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(consumer_ready_sem));
        noc_semaphore_wait_min(ready, active_attention);
        noc_semaphore_set(ready, 0);
        noc_semaphore_set(go, 1);
        for (uint32_t target = 1; target < kProjectionWorkers; ++target)
            noc_async_write(get_semaphore(producer_go_sem),
                            projection_l1(target, get_semaphore(producer_go_sem)),
                            sizeof(uint32_t));
        noc_async_write_barrier();
    }
    noc_semaphore_wait(go, 1); noc_semaphore_set(go, 0);

    constexpr uint32_t q_jobs = seq_tiles * hidden_tiles;
    constexpr uint32_t q_per_node = (q_jobs + kProjectionWorkers - 1) / kProjectionWorkers;
    for (uint32_t local = 0; local < q_per_node; ++local) {
        const uint32_t job = node * q_per_node + local;
        if (job >= q_jobs) continue;
        const uint32_t token = job / hidden_tiles;
        const uint32_t output = job - token * hidden_tiles;
        const uint32_t head = output / dim_tiles;
        const uint32_t dim = output - head * dim_tiles;
        const uint32_t target = head * seq_tiles + token;
        q.wait_front(1);
        noc_async_write(q.get_read_ptr(),
                        attention_l1(target, q_target_base + dim * kTileBytes),
                        kTileBytes);
        noc_async_write_barrier();
        noc_semaphore_inc(attention_l1(target, get_semaphore(q_ready_sem)), 1);
        noc_async_atomic_barrier(); q.pop_front(1);
    }

    constexpr uint32_t kv_jobs = seq_tiles * kv_tiles;
    constexpr uint32_t kv_per_node = (kv_jobs + kProjectionWorkers - 1) / kProjectionWorkers;
    for (uint32_t kind = 0; kind < 2; ++kind) {
        CircularBuffer& source = kind == 0 ? k : v;
        const uint32_t target_base = kind == 0 ? k_target_base : v_target_base;
        const uint32_t ready_sem = kind == 0 ? k_ready_sem : v_ready_sem;
        for (uint32_t local = 0; local < kv_per_node; ++local) {
            const uint32_t job = node * kv_per_node + local;
            if (job >= kv_jobs) continue;
            const uint32_t token = job / kv_tiles;
            const uint32_t output = job - token * kv_tiles;
            const uint32_t kv_head = output / dim_tiles;
            const uint32_t dim = output - kv_head * dim_tiles;
            const uint32_t slot = token * dim_tiles + dim;
            source.wait_front(1);
            const uint32_t first_head = kv_head * group;
            for (uint32_t member = 0; member < group; ++member)
                for (uint32_t row = 0; row < seq_tiles; ++row) {
                    const uint32_t target = (first_head + member) * seq_tiles + row;
                    noc_async_write(source.get_read_ptr(),
                                    attention_l1(target, target_base + slot * kTileBytes),
                                    kTileBytes);
                }
            noc_async_write_barrier();
            for (uint32_t member = 0; member < group; ++member)
                for (uint32_t row = 0; row < seq_tiles; ++row) {
                    const uint32_t target = (first_head + member) * seq_tiles + row;
                    noc_semaphore_inc(attention_l1(target, get_semaphore(ready_sem)), 1);
                }
            noc_async_atomic_barrier(); source.pop_front(1);
        }
    }
}
