// Exact paired gate/up frozen+LoRA projection and register-resident SwiGLU.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

namespace {

ALWI void add_tiles_bf16(
    uint32_t lhs, uint32_t rhs, uint32_t lhs_tile, uint32_t rhs_tile,
    uint32_t destination) {
    UNPACK((llk_unpack_AB(lhs, rhs, lhs_tile, rhs_tile)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWADD, BroadcastType::NONE, false,
          MathFidelity::LoFi, EltwiseBinaryReuseDestType::NONE>(
        lhs, rhs, destination, false)));
}

ALWI void commit_bf16() { MATH((llk_math_dest_section_done<false>())); }
ALWI void pack_reconfig_bf16(uint32_t cb) {
    PACK((llk_pack_reconfig_data_format<false>(cb)));
}
ALWI void pack_bf16(uint32_t source, uint32_t cb, uint32_t target) {
    PACK((llk_pack<false, false, PackMode::Default>(source, cb, target)));
}
ALWI void release_bf16() { PACK((llk_pack_dest_section_done<false>())); }

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_x_cache = 0;
    constexpr uint32_t cb_a = 1;
    constexpr uint32_t cb_h_partial = 2;
    constexpr uint32_t cb_h_publish = 3;
    constexpr uint32_t cb_h_resident = 5;
    constexpr uint32_t cb_w = 6;
    constexpr uint32_t cb_base_partial = 7;
    constexpr uint32_t cb_base = 8;
    constexpr uint32_t cb_b = 9;
    constexpr uint32_t cb_delta = 10;
    constexpr uint32_t cb_scaled = 11;
    constexpr uint32_t cb_gate_up = 12;
    constexpr uint32_t cb_product = 13;

    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@;
    constexpr uint32_t intermediate_tiles = @INTERMEDIATE_TILES@;
    constexpr uint32_t total_pairs = seq_tiles * intermediate_tiles;
    constexpr uint32_t kWorkers = @WORKERS@;
    constexpr uint32_t kColumns = @GRID_COLUMNS@;
    constexpr uint32_t pairs_per_node =
        (total_pairs + kWorkers - 1) / kWorkers;
    constexpr uint32_t base_phase = @BASE_PHASE@;
    constexpr uint32_t base_blocks = hidden_tiles / base_phase;
    constexpr uint32_t h_blocks = hidden_tiles / 2;
    constexpr uint32_t scale_bits = @SCALE_BITS@;

    const uint32_t node =
        get_absolute_logical_y() * kColumns + get_absolute_logical_x();
    const uint32_t first_pair = node * pairs_per_node;
    const uint32_t stop_pair =
        first_pair + pairs_per_node < total_pairs
        ? first_pair + pairs_per_node : total_pairs;
    const uint32_t valid_pairs =
        stop_pair > first_pair ? stop_pair - first_pair : 0;
    const uint32_t first_row =
        valid_pairs ? first_pair / intermediate_tiles : 0;
    const uint32_t last_row =
        valid_pairs ? (stop_pair - 1) / intermediate_tiles : 0;
    const uint32_t cached_rows =
        valid_pairs ? last_row - first_row + 1 : 0;

    CircularBuffer x_cache(cb_x_cache), a(cb_a), h_partial(cb_h_partial);
    CircularBuffer h_publish(cb_h_publish), h_resident(cb_h_resident);
    CircularBuffer w(cb_w), base_partial(cb_base_partial), base(cb_base);
    CircularBuffer b(cb_b), delta(cb_delta), scaled(cb_scaled);
    CircularBuffer gate_up(cb_gate_up), product(cb_product);

    if (cached_rows) x_cache.wait_front(cached_rows * hidden_tiles);

    // One owner per token row computes both rank-H members in gate/up order.
    for (uint32_t row = 0; row < seq_tiles; ++row) {
        const uint32_t owner = (row * intermediate_tiles) / pairs_per_node;
        if (node != owner) continue;
        const uint32_t x_offset = (row - first_row) * hidden_tiles;
        for (uint32_t member = 0; member < 2; ++member) {
            a.wait_front(hidden_tiles);
            compute_kernel_hw_startup<SrcOrder::Reverse>(
                cb_x_cache, cb_a, cb_h_partial);
            matmul_block_init(cb_x_cache, cb_a, true, 1, 1, 2);
            for (uint32_t block = 0; block < h_blocks; ++block) {
                tile_regs_acquire();
                if (block > 0) {
                    copy_tile_to_dst_init_short_with_dt(cb_a, cb_h_partial);
                    h_partial.wait_front(1);
                    copy_tile(cb_h_partial, 0, 0);
                    h_partial.pop_front(1);
                    reconfig_data_format_srca(cb_h_partial, cb_a);
                    matmul_block_init(cb_x_cache, cb_a, true, 1, 1, 2);
                }
                for (uint32_t inner = 0; inner < 2; ++inner) {
                    matmul_block(
                        cb_x_cache, cb_a,
                        x_offset + block * 2 + inner,
                        block * 2 + inner, 0, true, 1, 1, 2);
                }
                tile_regs_commit();
                const bool last = block + 1 == h_blocks;
                if (last) h_publish.reserve_back(1);
                else h_partial.reserve_back(1);
                tile_regs_wait();
                PACK((pack_reconfig_data_format(
                    last ? cb_h_publish : cb_h_partial)));
                pack_tile<true>(
                    0, last ? cb_h_publish : cb_h_partial, 0);
                tile_regs_release();
                if (last) h_publish.push_back(1);
                else h_partial.push_back(1);
            }
            a.pop_front(hidden_tiles);
        }
    }

    if (cached_rows) h_resident.wait_front(2 * cached_rows);

    // Produce exact FP32 base/scaled pairs in [gate, up] order.
    for (uint32_t pair = first_pair; pair < stop_pair; ++pair) {
        const uint32_t row = pair / intermediate_tiles;
        const uint32_t x_offset = (row - first_row) * hidden_tiles;
        const uint32_t h_base = (row - first_row) * 2;
        for (uint32_t member = 0; member < 2; ++member) {
            compute_kernel_hw_startup<SrcOrder::Reverse>(
                cb_x_cache, cb_w, cb_base_partial);
            matmul_block_init(
                cb_x_cache, cb_w, false, 1, 1, base_phase);
            for (uint32_t block = 0; block < base_blocks; ++block) {
                const bool last = block + 1 == base_blocks;
                w.wait_front(base_phase);
                tile_regs_acquire();
                if (block > 0) {
                    copy_tile_to_dst_init_short_with_dt(cb_w, cb_base_partial);
                    base_partial.wait_front(1);
                    copy_block_matmul_partials(cb_base_partial, 0, 0, 1);
                    base_partial.pop_front(1);
                    reconfig_data_format_srca(cb_base_partial, cb_w);
                    matmul_block_init(
                        cb_x_cache, cb_w, false, 1, 1, base_phase);
                }
                for (uint32_t inner = 0; inner < base_phase; ++inner) {
                    matmul_block(
                        cb_x_cache, cb_w,
                        x_offset + block * base_phase + inner,
                        inner, 0, false, 1, 1, base_phase);
                }
                tile_regs_commit();
                if (last) base.reserve_back(1);
                else base_partial.reserve_back(1);
                tile_regs_wait();
                PACK((pack_reconfig_data_format(
                    last ? cb_base : cb_base_partial)));
                pack_tile<true>(
                    0, last ? cb_base : cb_base_partial, 0);
                tile_regs_release();
                if (last) base.push_back(1);
                else base_partial.push_back(1);
                w.pop_front(base_phase);
            }

            b.wait_front(1);
            delta.reserve_back(1);
            compute_kernel_hw_startup<SrcOrder::Reverse>(
                cb_h_resident, cb_b, cb_delta);
            matmul_block_init(cb_h_resident, cb_b, true, 1, 1, 1);
            tile_regs_acquire();
            matmul_block(
                cb_h_resident, cb_b, h_base + member, 0, 0,
                true, 1, 1, 1);
            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(cb_delta)));
            pack_tile<true>(0, cb_delta, 0);
            tile_regs_release();
            delta.push_back(1);
            b.pop_front(1);

            delta.wait_front(1);
            scaled.reserve_back(1);
            init_sfpu(cb_delta, cb_scaled);
            tile_regs_acquire();
            copy_tile_init(cb_delta);
            copy_tile(cb_delta, 0, 0);
            binop_with_scalar_tile_init();
            { volatile int32_t scalar = scale_bits; mul_unary_tile(0, scalar); }
            tile_regs_commit();
            tile_regs_wait();
            PACK((pack_reconfig_data_format(cb_scaled)));
            pack_tile<true>(0, cb_scaled, 0);
            tile_regs_release();
            scaled.push_back(1);
            delta.pop_front(1);
        }
    }

    // Base and LoRA reductions no longer need normalized X or rank H. Release
    // those reader-owned windows before publishing the final product. The
    // writer's product-ready fence can then prove the old CB14/CB0 epochs are
    // completely quiescent before the down reader reuses them.
    if (cached_rows) {
        h_resident.pop_front(2 * cached_rows);
        x_cache.pop_front(cached_rows * hidden_tiles);
    }

    // Keep every exact BF16 boundary in one acquired DEST transaction:
    // gate/up add -> sigmoid -> gate*sigmoid -> silu*up. No value is packed
    // merely to be unpacked by another operation or compute epoch.
    add_tiles_init(cb_base, cb_scaled);
    disable_fp32_dest_acc();
    if (valid_pairs) {
        gate_up.reserve_back(2 * valid_pairs);
        product.reserve_back(valid_pairs);
    }
    for (uint32_t local = 0; local < valid_pairs; ++local) {
        base.wait_front(2);
        scaled.wait_front(2);
        tile_regs_acquire();
        add_tiles_bf16(cb_base, cb_scaled, 0, 0, 0);
        add_tiles_bf16(cb_base, cb_scaled, 1, 1, 1);

        copy_dest_values_init();
        copy_dest_values<DataFormat::Float16_b>(0, 2);
        MATH((SFPU_UNARY_INIT_FN(sigmoid, sfpu::sigmoid_init, (false))));
        MATH((SFPU_UNARY_CALL(
            DST_SYNC_MODE, false, calculate_sigmoid,
            (false, false, 8), 2, VectorMode::RC)));
        MATH((SFPU_BINARY_INIT_FN(
            unused, sfpu::sfpu_binary_init,
            (false, ckernel::BinaryOp::MUL))));
        MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE, false, calculate_sfpu_binary_mul,
            (false, ckernel::BinaryOp::MUL, 8, false),
            0, 2, 3, VectorMode::RC)));
        MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE, false, calculate_sfpu_binary_mul,
            (false, ckernel::BinaryOp::MUL, 8, false),
            3, 1, 4, VectorMode::RC)));
        commit_bf16();
        tile_regs_wait();
        pack_reconfig_bf16(cb_gate_up);
        pack_bf16(0, cb_gate_up, 0);
        pack_bf16(1, cb_gate_up, 1);
        pack_reconfig_bf16(cb_product);
        pack_bf16(4, cb_product, 0);
        release_bf16();
        base.pop_front(2);
        scaled.pop_front(2);
    }
    if (valid_pairs) {
        gate_up.push_back(2 * valid_pairs);
        product.push_back(valid_pairs);
    }
    enable_fp32_dest_acc();

}
