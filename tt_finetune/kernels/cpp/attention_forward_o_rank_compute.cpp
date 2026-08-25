// Exact four-core O-LoRA rank projection: H = X @ A.T.
//
// This reproduces TTNN's mcast-in1 large-block compute schedule: two K tiles
// per destination accumulation, FP32 spill/reload between blocks, HiFi4 and
// approximate-mode enabled by the kernel descriptor.  In particular, the
// packer is initialized for the FP32 partial CB by hw_startup and is *not*
// reconfigured for every intermediate spill.  TTNN only reconfigures it when
// the last block changes the pack target to the BF16 output CB.  That hardware
// state transition is part of the canonical numerical contract on Blackhole.
// The raw A tensor is in canonical [rank,K] tile layout, so the RHS tile is
// transposed in matmul.

#define FP32_DEST_ACC_EN 1

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t kTiles = @HIDDEN_TILES@;
    constexpr uint32_t kBlockTiles = 2;
    static_assert(kTiles % kBlockTiles == 0);
    constexpr uint32_t kBlocks = kTiles / kBlockTiles;
    constexpr uint32_t sequenceTiles = @SEQ_TILES@;
    const uint32_t row = get_absolute_logical_x();
    if (row >= sequenceTiles) return;

    CircularBuffer x(0), a(1), partial(2), out(3);
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 2);
    matmul_block_init(0, 1, true, 1, 1, kBlockTiles);
    bool enable_reload = false;
    for (uint32_t block = 0; block < kBlocks; ++block) {
        const bool last = block == kBlocks - 1;
        x.wait_front(kBlockTiles);
        a.wait_front(kBlockTiles);
        tile_regs_acquire();
        if (enable_reload) {
            copy_tile_to_dst_init_short_with_dt(1, 2);
            partial.wait_front(1);
            copy_block_matmul_partials(2, 0, 0, 1);
            partial.pop_front(1);
            reconfig_data_format_srca(2, 1);
            matmul_block_init(0, 1, true, 1, 1, kBlockTiles);
        }
        for (uint32_t inner = 0; inner < kBlockTiles; ++inner)
            matmul_block(
                0, 1, inner, inner, 0, true, 1, 1, kBlockTiles);
        tile_regs_commit();
        const uint32_t target = last ? 3 : 2;
        if (last) out.reserve_back(1); else partial.reserve_back(1);
        tile_regs_wait();
        // Match bmm_large_block_zm_fused_bias_activation exactly: startup
        // already configured the partial pack target.  Reconfigure only for
        // the final FP32-DEST -> BF16 output pack.
        if (last) PACK((pack_reconfig_data_format(target)));
        pack_tile_block(0, target, 1);
        tile_regs_release();
        if (last) out.push_back(1);
        else {
            partial.push_back(1);
            enable_reload = true;
        }
        x.pop_front(kBlockTiles);
        a.pop_front(kBlockTiles);
    }
}
