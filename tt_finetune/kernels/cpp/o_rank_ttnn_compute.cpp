// Minimal TTNN-schedule LoRA rank producer: H = X @ A.T.
//
// This deliberately mirrors the one-output reduction path in
// bmm_large_block_zm_fused_bias_activation.cpp.  Keeping this as a separate
// compute child also keeps its CB/data-format context identical to a plain
// matmul instead of sharing configuration with base, delta and add epochs.

#define FP32_DEST_ACC_EN 1

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_a = get_compile_time_arg_val(1);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t k_tiles = @K_TILES@;
    constexpr uint32_t k_block_tiles = 2;
    static_assert(k_tiles % k_block_tiles == 0);
    constexpr uint32_t k_blocks = k_tiles / k_block_tiles;
    constexpr uint32_t m_tiles = @M_TILES@;
    constexpr uint32_t grid_cols = @GRID_COLS@;

    const uint32_t job =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();
    if (job >= m_tiles) {
        return;
    }

    CircularBuffer x(cb_x), a(cb_a), partial(cb_partial), out(cb_out);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_a, cb_partial);
    matmul_block_init(cb_x, cb_a, false, 1, 1, k_block_tiles);
    bool enable_reload = false;
    for (uint32_t block = 0; block < k_blocks; ++block) {
        const bool last = block == k_blocks - 1;
        x.wait_front(k_block_tiles);
        a.wait_front(k_block_tiles);
        tile_regs_acquire();
        if (enable_reload) {
            copy_tile_to_dst_init_short_with_dt(cb_a, cb_partial);
            partial.wait_front(1);
            copy_block_matmul_partials(cb_partial, 0, 0, 1);
            partial.pop_front(1);
            reconfig_data_format_srca(cb_partial, cb_a);
            matmul_block_init(
                cb_x, cb_a, false, 1, 1, k_block_tiles);
        }
        for (uint32_t inner = 0; inner < k_block_tiles; ++inner) {
            matmul_block(
                cb_x,
                cb_a,
                inner,
                inner,
                0,
                false,
                1,
                1,
                k_block_tiles);
        }
        tile_regs_commit();
        const uint32_t target = last ? cb_out : cb_partial;
        if (last) {
            out.reserve_back(1);
        } else {
            partial.reserve_back(1);
        }
        tile_regs_wait();
        PACK((pack_reconfig_data_format(target)));
        pack_tile_block(0, target, 1);
        tile_regs_release();
        if (last) {
            out.push_back(1);
        } else {
            partial.push_back(1);
            enable_reload = true;
        }
        x.pop_front(k_block_tiles);
        a.pop_front(k_block_tiles);
    }
}
