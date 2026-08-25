// Exact 50-core row-cached grouped gate/up frozen+LoRA forward.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

ALWI void resident_add_tiles_bf16(
    uint32_t lhs, uint32_t rhs, uint32_t lhs_tile, uint32_t rhs_tile,
    uint32_t destination) {
    UNPACK((llk_unpack_AB(lhs, rhs, lhs_tile, rhs_tile)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWADD, BroadcastType::NONE, false,
          MathFidelity::LoFi, EltwiseBinaryReuseDestType::NONE>(
        lhs, rhs, destination, false)));
}

ALWI void resident_commit_bf16() { MATH((llk_math_dest_section_done<false>())); }
ALWI void resident_pack_reconfig_bf16(uint32_t cb) {
    PACK((llk_pack_reconfig_data_format<false>(cb)));
}
ALWI void resident_pack_bf16(uint32_t source, uint32_t cb, uint32_t target) {
    PACK((llk_pack<false, false, PackMode::Default>(source, cb, target)));
}
ALWI void resident_release_bf16() { PACK((llk_pack_dest_section_done<false>())); }

void kernel_main() {
    WAYPOINT("GCS0");
    constexpr uint32_t cb_x_cache = get_compile_time_arg_val(0);
    constexpr uint32_t cb_a = get_compile_time_arg_val(1);
    constexpr uint32_t cb_h_partial = get_compile_time_arg_val(2);
    constexpr uint32_t cb_h_publish = get_compile_time_arg_val(3);
    constexpr uint32_t cb_h_resident = get_compile_time_arg_val(5);
    constexpr uint32_t cb_w = get_compile_time_arg_val(6);
    constexpr uint32_t cb_base_partial = get_compile_time_arg_val(7);
    constexpr uint32_t cb_base = get_compile_time_arg_val(8);
    constexpr uint32_t cb_b = get_compile_time_arg_val(9);
    constexpr uint32_t cb_delta = get_compile_time_arg_val(10);
    constexpr uint32_t cb_scaled = get_compile_time_arg_val(11);
    constexpr uint32_t cb_output = get_compile_time_arg_val(12);

    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@;
    constexpr uint32_t intermediate_tiles = @INTERMEDIATE_TILES@;
    constexpr uint32_t total_n_tiles = 2 * intermediate_tiles;
    constexpr uint32_t total_jobs = seq_tiles * total_n_tiles;
    constexpr uint32_t kWorkers = @WORKERS@;
    constexpr uint32_t kColumns = @GRID_COLUMNS@;
    constexpr uint32_t jobs_per_node =
        (total_jobs + kWorkers - 1) / kWorkers;
    constexpr uint32_t base_phase = @BASE_PHASE@;
    constexpr uint32_t base_blocks = hidden_tiles / base_phase;
    constexpr uint32_t h_blocks = hidden_tiles / 2;
    constexpr uint32_t scale_bits = @SCALE_BITS@;

    const uint32_t node =
        get_absolute_logical_y() * kColumns + get_absolute_logical_x();
    const uint32_t first_job = node * jobs_per_node;
    const uint32_t stop_job =
        first_job + jobs_per_node < total_jobs
        ? first_job + jobs_per_node : total_jobs;
    const uint32_t valid_jobs = stop_job > first_job ? stop_job - first_job : 0;
    const uint32_t first_row = valid_jobs ? first_job / total_n_tiles : 0;
    const uint32_t last_row = valid_jobs ? (stop_job - 1) / total_n_tiles : 0;
    const uint32_t cached_rows = valid_jobs ? last_row - first_row + 1 : 0;
    const uint32_t first_group = valid_jobs ? first_job / intermediate_tiles : 0;
    const uint32_t last_group = valid_jobs ? (stop_job - 1) / intermediate_tiles : 0;
    const uint32_t local_groups = valid_jobs ? last_group - first_group + 1 : 0;

    CircularBuffer x_cache(cb_x_cache), a(cb_a), h_partial(cb_h_partial);
    CircularBuffer h_publish(cb_h_publish), h_resident(cb_h_resident);
    CircularBuffer w(cb_w), base_partial(cb_base_partial), base(cb_base);
    CircularBuffer b(cb_b), delta(cb_delta), scaled(cb_scaled), output(cb_output);

    if (cached_rows) {
        WAYPOINT("GCX0");
        x_cache.wait_front(cached_rows * hidden_tiles);
        WAYPOINT("GCX1");
    }

    // The owner of each logical group performs the exact two-tile FP32-spill
    // reduction once. All output columns consume the resulting BF16 H tile.
    WAYPOINT("GCH0");
    for (uint32_t group = 0; group < 2 * seq_tiles; ++group) {
        const uint32_t owner = (group * intermediate_tiles) / jobs_per_node;
        if (node != owner) continue;
        const uint32_t row = group / 2;
        const uint32_t x_offset = (row - first_row) * hidden_tiles;
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
            PACK((pack_reconfig_data_format(last ? cb_h_publish : cb_h_partial)));
            pack_tile<true>(0, last ? cb_h_publish : cb_h_partial, 0);
            tile_regs_release();
            if (last) h_publish.push_back(1);
            else h_partial.push_back(1);
        }
        a.pop_front(hidden_tiles);
    }
    WAYPOINT("GCH1");

    if (local_groups) {
        WAYPOINT("GCR0");
        h_resident.wait_front(local_groups);
        WAYPOINT("GCR1");
    }
    WAYPOINT("GCB0");
    for (uint32_t job = first_job; job < stop_job; ++job) {
        const uint32_t row = job / total_n_tiles;
        const uint32_t group = job / intermediate_tiles;
        const uint32_t x_offset = (row - first_row) * hidden_tiles;
        const uint32_t h_slot = group - first_group;

        compute_kernel_hw_startup<SrcOrder::Reverse>(
            cb_x_cache, cb_w, cb_base_partial);
        matmul_block_init(cb_x_cache, cb_w, false, 1, 1, base_phase);
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
                matmul_block_init(cb_x_cache, cb_w, false, 1, 1, base_phase);
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
            PACK((pack_reconfig_data_format(last ? cb_base : cb_base_partial)));
            pack_tile<true>(0, last ? cb_base : cb_base_partial, 0);
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
        matmul_block(cb_h_resident, cb_b, h_slot, 0, 0, true, 1, 1, 1);
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
    WAYPOINT("GCB1");

    add_tiles_init(cb_base, cb_scaled);
    disable_fp32_dest_acc();
    WAYPOINT("GCO0");
    for (uint32_t job = first_job; job < stop_job; ++job) {
        base.wait_front(1);
        scaled.wait_front(1);
        output.reserve_back(1);
        tile_regs_acquire();
        resident_add_tiles_bf16(cb_base, cb_scaled, 0, 0, 0);
        resident_commit_bf16();
        tile_regs_wait();
        resident_pack_reconfig_bf16(cb_output);
        resident_pack_bf16(0, cb_output, 0);
        resident_release_bf16();
        output.push_back(1);
        scaled.pop_front(1);
        base.pop_front(1);
    }
    WAYPOINT("GCO1");
    enable_fp32_dest_acc();
    if (local_groups) h_resident.pop_front(local_groups);
    if (cached_rows) x_cache.pop_front(cached_rows * hidden_tiles);
}
