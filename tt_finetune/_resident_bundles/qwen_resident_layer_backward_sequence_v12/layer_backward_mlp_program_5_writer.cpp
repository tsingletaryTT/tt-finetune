#include <cstdint>
#include "api/compile_time_args.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"

namespace resident_layer_phase_mlp_5 {
FORCE_INLINE uint8_t full_product_physical_y(uint32_t y) {
    return static_cast<uint8_t>(y + 2);
}

FORCE_INLINE void full_activation_product_writer_phase() {
    constexpr uint32_t cb_product = get_compile_time_arg_val(52);
    constexpr uint32_t product_jobs = 608;
    constexpr uint32_t jobs_per_node = 61;
    constexpr uint32_t i_tiles = 152;
    constexpr uint32_t m_tiles = 4;
    constexpr uint32_t da_tiles = 2;
    constexpr uint32_t kTileBytes = 2048;
    constexpr uint8_t px[10] = {1,2,3,4,5,6,7,10,11,12};
    const uint32_t node = get_absolute_logical_y();
    CircularBuffer product(cb_product);

    volatile tt_l1_ptr uint32_t* destinations =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(13));
    if (node == 0) {
        noc_semaphore_wait_min(destinations, 100);
        noc_semaphore_set(destinations, 0);
        volatile tt_l1_ptr uint32_t* h_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(14));
        volatile tt_l1_ptr uint32_t* da_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(15));
        noc_semaphore_wait_min(h_addr, 1);
        noc_semaphore_wait_min(da_addr, 1);
        for (uint32_t target = 1; target < 10; ++target) {
            noc_async_write(
                get_semaphore(14),
                get_noc_addr(13, full_product_physical_y(target), get_semaphore(14)),
                sizeof(uint32_t));
            noc_async_write(
                get_semaphore(15),
                get_noc_addr(13, full_product_physical_y(target), get_semaphore(15)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
    volatile tt_l1_ptr uint32_t* h_addr_flag =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(14));
    volatile tt_l1_ptr uint32_t* da_addr_flag =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(15));
    noc_semaphore_wait_min(h_addr_flag, 1);
    noc_semaphore_wait_min(da_addr_flag, 1);
    const uint32_t h_base = *h_addr_flag;
    const uint32_t da_base = *da_addr_flag;
    const uint32_t first = node * jobs_per_node;
    for (uint32_t local = 0; local < jobs_per_node; ++local) {
        const uint32_t job = first + local;
        if (job >= product_jobs) continue;
        const uint32_t m = job / i_tiles;
        const uint32_t k = job - m * i_tiles;
        product.wait_front(1);
        const uint32_t source = product.get_read_ptr();
        noc_async_write(
            source,
            get_noc_addr(px[m % 10], full_product_physical_y(m / 10),
                         h_base + k * kTileBytes),
            kTileBytes);
        const uint32_t owner = k / da_tiles;
        const uint32_t slot = (k % da_tiles) * m_tiles + m;
        noc_async_write(
            source,
            get_noc_addr(px[owner % 10], full_product_physical_y(owner / 10),
                         da_base + slot * kTileBytes),
            kTileBytes);
        noc_async_write_barrier();
        product.pop_front(1);
    }
    noc_semaphore_inc(
        get_noc_addr(13, full_product_physical_y(0), get_semaphore(13)), 1);
    noc_async_atomic_barrier();
    if (node == 0) {
        noc_semaphore_wait_min(destinations, 10);
        noc_semaphore_set(destinations, 0);
        volatile tt_l1_ptr uint32_t* ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(14));
        noc_semaphore_set(ready, 1);
        for (uint32_t target = 0; target < 100; ++target) {
            noc_async_write(
                get_semaphore(14),
                get_noc_addr(px[target % 10], full_product_physical_y(target / 10),
                             get_semaphore(14)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }
}

// Keep exact dGate/dUp outputs banked in activation-core L1 and publish the
// bank addresses to the matmul island. The matmul readers consume the tiles
// directly over NoC and acknowledge completion before these CBs are released.



namespace {
constexpr uint32_t kMatmulCols = 10;
constexpr uint32_t kMatmulNodes = 100;
constexpr uint32_t kActivationNodes = 10;
constexpr uint8_t kMatmulPhysicalX[kMatmulCols] = {
    1, 2, 3, 4, 5, 6, 7, 10, 11, 12};
FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
}  // namespace

FORCE_INLINE void full_activation_backward_writer_phase() {
    constexpr uint32_t cb_dgate0 = get_compile_time_arg_val(35);
    constexpr uint32_t cb_dup0 = get_compile_time_arg_val(36);
    constexpr uint32_t total_jobs = 608;
    constexpr uint32_t producer_jobs = 7;
    constexpr uint32_t bank_count = 2;
    constexpr uint32_t address_start = 0;
    constexpr uint32_t output_ready = 4;
    constexpr uint32_t consumer_done = 5;
    constexpr uint32_t release = 6;
    const uint32_t node = get_absolute_logical_y();
    uint32_t activation_jobs = 0;
    for (uint32_t producer = node; producer < kMatmulNodes;
         producer += kActivationNodes) {
        const uint32_t first = producer * producer_jobs;
        if (first >= total_jobs) continue;
        const uint32_t remaining = total_jobs - first;
        activation_jobs += remaining < producer_jobs
            ? remaining : producer_jobs;
    }
    const uint32_t bank0_jobs =
        activation_jobs < 32 ? activation_jobs : 32;
    const uint32_t bank1_jobs =
        activation_jobs > 32 ? activation_jobs - 32 : 0;

    CircularBuffer dgate0(cb_dgate0);
    CircularBuffer dup0(cb_dup0);
    dgate0.wait_front(bank0_jobs);
    dup0.wait_front(bank0_jobs);
    uint32_t dgate_addresses[2] = {dgate0.get_read_ptr(), 0};
    uint32_t dup_addresses[2] = {dup0.get_read_ptr(), 0};
    CircularBuffer scaled_dgate0(
        get_compile_time_arg_val(53));
    CircularBuffer scaled_dup0(
        get_compile_time_arg_val(54));
    scaled_dgate0.wait_front(bank0_jobs);
    scaled_dup0.wait_front(bank0_jobs);
    uint32_t scaled_dgate_addresses[2] = {scaled_dgate0.get_read_ptr(), 0};
    uint32_t scaled_dup_addresses[2] = {scaled_dup0.get_read_ptr(), 0};

    if constexpr (bank_count == 2) {
        constexpr uint32_t cb_dgate1 = get_compile_time_arg_val(38);
        constexpr uint32_t cb_dup1 = get_compile_time_arg_val(39);
        CircularBuffer dgate1(cb_dgate1);
        CircularBuffer dup1(cb_dup1);
        dgate1.wait_front(bank1_jobs);
        dup1.wait_front(bank1_jobs);
        dgate_addresses[1] = dgate1.get_read_ptr();
        dup_addresses[1] = dup1.get_read_ptr();
        CircularBuffer scaled_dgate1(
            get_compile_time_arg_val(55));
        CircularBuffer scaled_dup1(
            get_compile_time_arg_val(56));
        scaled_dgate1.wait_front(bank1_jobs);
        scaled_dup1.wait_front(bank1_jobs);
        scaled_dgate_addresses[1] = scaled_dgate1.get_read_ptr();
        scaled_dup_addresses[1] = scaled_dup1.get_read_ptr();
    }

    noc_semaphore_inc(
        get_noc_addr(13, physical_y(0), get_semaphore(output_ready)), 1);
    noc_async_atomic_barrier();

    if (node == 0) {
        volatile tt_l1_ptr uint32_t* ready_flag =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(output_ready));
        noc_semaphore_wait_min(ready_flag, kActivationNodes);
        noc_semaphore_set(ready_flag, 0);

        // CB allocations are identical on every core in one descriptor
        // range, so core zero publishes one address per bank and output.
        for (uint32_t target = 0; target < kMatmulNodes; ++target) {
            const uint32_t target_x = target % kMatmulCols;
            const uint32_t target_y = target / kMatmulCols;
            for (uint32_t bank = 0; bank < bank_count; ++bank) {
                volatile tt_l1_ptr uint32_t* dgate_mailbox =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        get_semaphore(address_start + 2 * bank));
                volatile tt_l1_ptr uint32_t* dup_mailbox =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        get_semaphore(address_start + 2 * bank + 1));
                noc_semaphore_set(dgate_mailbox, dgate_addresses[bank]);
                noc_semaphore_set(dup_mailbox, dup_addresses[bank]);
                noc_async_write(
                    get_semaphore(address_start + 2 * bank),
                    get_noc_addr(
                        kMatmulPhysicalX[target_x], physical_y(target_y),
                        get_semaphore(address_start + 2 * bank)),
                    sizeof(uint32_t));
                noc_async_write(
                    get_semaphore(address_start + 2 * bank + 1),
                    get_noc_addr(
                        kMatmulPhysicalX[target_x], physical_y(target_y),
                        get_semaphore(address_start + 2 * bank + 1)),
                    sizeof(uint32_t));
                constexpr uint32_t scaled_address_start = 7;
                volatile tt_l1_ptr uint32_t* scaled_dgate_mailbox =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        get_semaphore(scaled_address_start + 2 * bank));
                volatile tt_l1_ptr uint32_t* scaled_dup_mailbox =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        get_semaphore(scaled_address_start + 2 * bank + 1));
                noc_semaphore_set(
                    scaled_dgate_mailbox, scaled_dgate_addresses[bank]);
                noc_semaphore_set(
                    scaled_dup_mailbox, scaled_dup_addresses[bank]);
                noc_async_write(
                    get_semaphore(scaled_address_start + 2 * bank),
                    get_noc_addr(
                        kMatmulPhysicalX[target_x], physical_y(target_y),
                        get_semaphore(scaled_address_start + 2 * bank)),
                    sizeof(uint32_t));
                noc_async_write(
                    get_semaphore(scaled_address_start + 2 * bank + 1),
                    get_noc_addr(
                        kMatmulPhysicalX[target_x], physical_y(target_y),
                        get_semaphore(scaled_address_start + 2 * bank + 1)),
                    sizeof(uint32_t));
            }
            volatile tt_l1_ptr uint32_t* target_ready =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(output_ready));
            noc_semaphore_set(target_ready, 1);
            noc_async_write(
                get_semaphore(output_ready),
                get_noc_addr(
                    kMatmulPhysicalX[target_x], physical_y(target_y),
                    get_semaphore(output_ready)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();

        volatile tt_l1_ptr uint32_t* done_flag =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(consumer_done));
        noc_semaphore_wait_min(done_flag, kMatmulNodes);
        noc_semaphore_set(done_flag, 0);
        volatile tt_l1_ptr uint32_t* release_flag =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(release));
        noc_semaphore_set(release_flag, 1);
        for (uint32_t target = 1; target < kActivationNodes; ++target) {
            noc_async_write(
                get_semaphore(release),
                get_noc_addr(13, physical_y(target), get_semaphore(release)),
                sizeof(uint32_t));
        }
        noc_async_write_barrier();
    }

    volatile tt_l1_ptr uint32_t* release_flag =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(release));
    noc_semaphore_wait(release_flag, 1);
    noc_semaphore_set(release_flag, 0);
    dgate0.pop_front(bank0_jobs);
    dup0.pop_front(bank0_jobs);
    scaled_dgate0.pop_front(bank0_jobs);
    scaled_dup0.pop_front(bank0_jobs);
    if constexpr (bank_count == 2) {
        CircularBuffer dgate1(get_compile_time_arg_val(38));
        CircularBuffer dup1(get_compile_time_arg_val(39));
        dgate1.pop_front(bank1_jobs);
        dup1.pop_front(bank1_jobs);
        CircularBuffer scaled_dgate1(
            get_compile_time_arg_val(55));
        CircularBuffer scaled_dup1(
            get_compile_time_arg_val(56));
        scaled_dgate1.pop_front(bank1_jobs);
        scaled_dup1.pop_front(bank1_jobs);
    }
}

void run() {
    full_activation_product_writer_phase();
    full_activation_backward_writer_phase();
}
}  // namespace resident_layer_phase_mlp_5

void kernel_main() {
    resident_layer_phase_mlp_5::run();
}
