#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#include "ckernel.h"
#include "ckernel_defs.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(1);
    constexpr uint32_t cb_result = get_compile_time_arg_val(3);
    constexpr uint32_t k_tiles = @K_TILES@;

    CircularBuffer x(cb_x);
    CircularBuffer weight(cb_weight);
    CircularBuffer result(cb_result);
    compute_kernel_hw_startup<SrcOrder::Reverse>(
        cb_x, cb_weight, cb_result);
    matmul_block_init(cb_x, cb_weight, false, 1, 1, k_tiles);

    uint32_t active_jobs = 0;
    UNPACK(active_jobs = mailbox_read(ckernel::ThreadId::BriscThreadId);)
    MATH(active_jobs = mailbox_read(ckernel::ThreadId::BriscThreadId);)
    PACK(active_jobs = mailbox_read(ckernel::ThreadId::BriscThreadId);)
    for (uint32_t job = 0; job < active_jobs; ++job) {
        x.wait_front(k_tiles);
        weight.wait_front(k_tiles);
        tile_regs_acquire();
        for (uint32_t k = 0; k < k_tiles; ++k) {
            matmul_block(
                cb_x, cb_weight, k, k, 0, false, 1, 1, k_tiles);
        }
        tile_regs_commit();
        result.reserve_back(1);
        tile_regs_wait();
        pack_tile_block(0, cb_result, 1);
        tile_regs_release();
        result.push_back(1);
        weight.pop_front(k_tiles);
        x.pop_front(k_tiles);
    }
}
