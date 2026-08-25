// Exact specialization of TTNN layernorm.cpp for RMSNorm + gamma.

#define FP32_DEST_ACC_EN 1

#include <cstdint>
#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/layernorm.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/operations/normalization/kernel_util/compute/numeric.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "ttnn/operations/normalization/kernel_util/compute/policies.h"

namespace generic = norm::kernel_util::generic;
namespace numeric = norm::kernel_util::compute::numeric;
namespace policies = norm::kernel_util::compute::policies;

namespace {

// `unary_op_init_common` begins with a cross-TRISC state_configure barrier.
// That barrier is valid at operation startup, but not after this resident
// kernel has already completed reduction, rsqrt, and broadcast phases: the
// unpack thread can advance while math/pack remain on the previous epoch.
// Reproduce every hardware configuration performed by the helper, excluding
// only that barrier.
FORCE_INLINE void unary_op_reinit_no_barrier(
    uint32_t input_id, uint32_t output_id) {
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(input_id)));
    UNPACK((llk_unpack_A_init<
        BroadcastType::NONE, false,
        EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
            false, false, input_id)));

    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(output_id)));
    PACK((llk_pack_init(output_id)));
    PACK((llk_pack_dest_init<
        DST_ACCUM_MODE, PackMode::Default>()));

    MATH((llk_math_eltwise_unary_datacopy_init<
        DataCopyType::A2D, DST_ACCUM_MODE,
        BroadcastType::NONE>(input_id)));
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(
        input_id, input_id)));
    MATH((ckernel::math::_configure_unary_preserve_zero_flag_state_()));
}

template <bool Add>
FORCE_INLINE void exact_sfpu_binary_tile(
    CircularBuffer& lhs, CircularBuffer& rhs, CircularBuffer& output,
    uint32_t lhs_id, uint32_t rhs_id, uint32_t output_id) {
    lhs.wait_front(1);
    rhs.wait_front(1);
    output.reserve_back(1);

    // Reproduce eltwise_binary_sfpu.cpp's setup and destination-register
    // schedule.  In particular both operands unpack as true FP32, the lhs and
    // rhs occupy adjacent DEST tiles, and SFPU writes its result back to the
    // lhs tile before the pack boundary.
    unary_op_init_common(lhs_id, output_id);
    if constexpr (Add) {
        add_binary_tile_init();
    } else {
        mul_binary_tile_init();
    }
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(rhs_id, lhs_id);
    copy_tile(lhs_id, 0, 0);
    copy_tile_to_dst_init_short_with_dt(lhs_id, rhs_id);
    copy_tile(rhs_id, 0, 1);
    if constexpr (Add) {
        add_binary_tile(0, 1, 0);
    } else {
        mul_binary_tile(0, 1, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, output_id);
    tile_regs_release();

    output.push_back(1);
    lhs.pop_front(1);
}

FORCE_INLINE void publish_column_diagnostic(
    CircularBuffer& source, CircularBuffer& output,
    uint32_t source_id, uint32_t output_id,
    uint32_t hidden_tiles, uint32_t block_size) {
    source.wait_front(1);
    reconfig_data_format_srca(source_id);
    pack_reconfig_data_format(output_id);
    unary_bcast_init<BroadcastType::COL>(source_id, output_id);
    for (auto block : generic::blocks(hidden_tiles, block_size)) {
        output.reserve_back(block.full_block_size());
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < block.full_block_size(); ++tile) {
            unary_bcast<BroadcastType::COL>(source_id, 0, tile);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t tile = 0; tile < block.full_block_size(); ++tile) {
            pack_tile(tile, output_id);
        }
        tile_regs_release();
        output.push_back(block.full_block_size());
    }
    source.pop_front(1);
}

}  // namespace

void kernel_main() {
    // Standalone owner defaults. Resident composition removes/replaces this
    // block with explicit phase arguments; keeping the raw source
    // self-contained is required for safe-path compilation and diagnostics.
    constexpr bool publish_post_rstd = false;
    constexpr bool resident_reentry = false;
    constexpr uint32_t cb_x_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight_id = get_compile_time_arg_val(5);
    constexpr uint32_t cb_scaler_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_eps_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_inverse_hidden_id = 4;
    constexpr uint32_t cb_mean_id = 17;
    constexpr uint32_t cb_variance_eps_id = 18;
    constexpr uint32_t cb_square_id = get_compile_time_arg_val(20);
    constexpr uint32_t cb_variance_id = get_compile_time_arg_val(19);
    constexpr uint32_t cb_rstd_id = get_compile_time_arg_val(21);
    constexpr uint32_t cb_normalized_id = get_compile_time_arg_val(22);
    constexpr uint32_t cb_normalized_fp32_id = get_compile_time_arg_val(7);
    constexpr uint32_t cb_rstd_expanded_id = get_compile_time_arg_val(8);
    constexpr uint32_t cb_output_id = get_compile_time_arg_val(16);
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@;
    constexpr uint32_t padded_tiles = @PADDED_TILES@;
    constexpr uint32_t token_tiles = @TOKEN_TILES@;
    constexpr uint32_t block_size = 4;
    // Each SFPU output uses an adjacent pair of FP32 destination tiles
    // (activation, expanded rstd). Blackhole can safely hold two such pairs
    // per destination transaction; four pairs wrap into the alternate bank
    // and corrupt tiles 2..7 in a repeating pattern.
    constexpr uint32_t suffix_block_size = 2;
    constexpr uint32_t hidden = @HIDDEN_SIZE@;
    const uint32_t node =
        get_absolute_logical_y() * 11 + get_absolute_logical_x();
    if (node >= token_tiles) return;

    CircularBuffer x(cb_x_id), weight(cb_weight_id), scaler(cb_scaler_id);
    CircularBuffer eps(cb_eps_id), inverse_hidden(cb_inverse_hidden_id);
    CircularBuffer square(cb_square_id), sum(cb_variance_id);
    CircularBuffer mean(cb_mean_id), variance_eps(cb_variance_eps_id);
    CircularBuffer variance(cb_variance_id), rstd(cb_rstd_id);
    CircularBuffer normalized(cb_normalized_id), output(cb_output_id);

    // Hardware startup is legal exactly once and must precede every Compute
    // API call.  Start in the power producer's configuration; subsequent
    // operation families use the ordinary reconfiguration APIs.  Calling
    // startup again between power and reduction is explicitly unsafe and can
    // leave the three TRISCs observing different state.
    compute_kernel_hw_startup(cb_x_id, cb_x_id, cb_square_id);
#if @OUTPUT_STAGE@ == 3
    // Gamma is drained directly by the writer on this diagnostic path.  The
    // compute RISC only balances the other producer-owned FIFOs.
    x.wait_front(padded_tiles);
    scaler.wait_front(1);
    eps.wait_front(1);
    inverse_hidden.wait_front(1);
    x.pop_front(padded_tiles);
    scaler.pop_front(1);
    eps.pop_front(1);
    inverse_hidden.pop_front(1);
    return;
#endif
    eps.wait_front(1);
    inverse_hidden.wait_front(1);

#if @OUTPUT_STAGE@ == 9
    // Reproduce the fused ``ttnn.rms_norm`` selected by TT-XLA for the real
    // Hugging Face graph.  This is intentionally not the explicit
    // FP32-mean/typecast/gamma sequence below: the fused kernel squares BF16
    // input through the FPU, keeps its normalization product in an FP32
    // fusion FIFO, and applies gamma before the final BF16 pack.
    reconfig_data_format(cb_x_id, cb_x_id);
    pack_reconfig_data_format(cb_square_id);
    mul_tiles_init(cb_x_id, cb_x_id);
    for (auto block : generic::blocks(hidden_tiles, block_size)) {
        if (!resident_reentry) {
            x.wait_front(block.start() + block.full_block_size());
        }
        tile_regs_acquire();
        for (auto i : block.local()) {
            const auto global_i = block.to_global(i);
            mul_tiles(cb_x_id, cb_x_id, global_i, global_i, i);
        }
        tile_regs_commit();
        square.reserve_back(block.full_block_size());
        tile_regs_wait();
        for (auto i : block.local()) pack_tile(i, cb_square_id);
        tile_regs_release();
        square.push_back(block.full_block_size());
    }

    numeric::row_wise_mean<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        true,
        policies::FullBlockWithPopPolicy>(
            square, scaler, mean, hidden, hidden_tiles,
            block_size, 32);

    mean.wait_front(1);
    reconfig_data_format(cb_mean_id, cb_eps_id);
    tile_regs_acquire();
    add_tiles_init(cb_mean_id, cb_eps_id);
    add_tiles(cb_mean_id, cb_eps_id, 0, 0, 0);
    rsqrt_tile_init<false>();
    rsqrt_tile<false>(0);
    tile_regs_commit();
    mean.pop_front(1);
    rstd.reserve_back(1);
    pack_reconfig_data_format(cb_rstd_id);
    tile_regs_wait();
    pack_tile(0, cb_rstd_id);
    tile_regs_release();
    rstd.push_back(1);

#if @OUTPUT_STAGE@ == 9
    // Publish the exact statistic from the production fused RMSNorm path.
    // Unlike the explicit diagnostic reduction below, this tile has the
    // same row-value contract consumed by fused x*rstd. Expand it across
    // columns once, then let the writer replicate the raw FP32 tile across
    // hidden pages for RMSNorm backward.
    CircularBuffer fused_rstd_expanded(cb_rstd_expanded_id);
    rstd.wait_front(1);
    fused_rstd_expanded.reserve_back(1);
    reconfig_data_format_srca(cb_rstd_id);
    pack_reconfig_data_format(cb_rstd_id, cb_rstd_expanded_id);
    unary_bcast_init<BroadcastType::COL>(
        cb_rstd_id, cb_rstd_expanded_id);
    tile_regs_acquire();
    unary_bcast<BroadcastType::COL>(cb_rstd_id, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_rstd_expanded_id);
    tile_regs_release();
    fused_rstd_expanded.push_back(1);
    rstd.pop_front(1);
    x.pop_front(padded_tiles);
    weight.pop_front(padded_tiles);
    scaler.pop_front(1);
    eps.pop_front(1);
    inverse_hidden.pop_front(1);
    return;
#endif

    // Keep this exact line stable: the resident owner injects its optional
    // rstd checkpoint handshake immediately after this publication.
    rstd.wait_front(1);
    CircularBuffer fusion(cb_normalized_fp32_id);
    for (auto block : generic::blocks(hidden_tiles, block_size)) {
        reconfig_data_format(cb_x_id, cb_rstd_id);
        pack_reconfig_data_format(cb_normalized_fp32_id);
        fusion.reserve_back(block.full_block_size());
        tile_regs_acquire();
        mul_bcast_cols_init_short(cb_x_id, cb_rstd_id);
        for (auto i : block.local()) {
            mul_tiles_bcast_cols(
                cb_x_id, cb_rstd_id, block.to_global(i), 0, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (auto i : block.local()) {
            pack_tile(i, cb_normalized_fp32_id);
        }
        tile_regs_release();
        fusion.push_back(block.full_block_size());

        weight.wait_front(block.start() + block.full_block_size());
        fusion.wait_front(block.full_block_size());
        output.reserve_back(block.full_block_size());
        reconfig_data_format_srca(cb_x_id, cb_normalized_fp32_id);
        reconfig_data_format_srcb(cb_rstd_id, cb_weight_id);
        pack_reconfig_data_format(cb_output_id);
        tile_regs_acquire();
        mul_bcast_rows_init_short(cb_normalized_fp32_id, cb_weight_id);
        for (auto i : block.local()) {
            mul_tiles_bcast_rows(
                cb_normalized_fp32_id, cb_weight_id,
                i, block.to_global(i), i);
        }
        tile_regs_commit();
        fusion.pop_front(block.full_block_size());
        tile_regs_wait();
        for (auto i : block.local()) pack_tile(i, cb_output_id);
        tile_regs_release();
        output.push_back(block.full_block_size());
    }
    rstd.pop_front(1);
    x.pop_front(padded_tiles);
    weight.pop_front(padded_tiles);
    scaler.pop_front(1);
    eps.pop_front(1);
    inverse_hidden.pop_front(1);
    return;
#endif

    // `ttnn.pow(x, 2.0)` recognizes the integral exponent and selects the
    // iterative SFPU primitive.  The general logarithmic POWER primitive is
    // close but differs in most FP32 mantissas.
    init_sfpu(cb_x_id, cb_square_id);
    for (auto block : generic::blocks(hidden_tiles, block_size)) {
        if (!resident_reentry) {
            x.wait_front(block.start() + block.full_block_size());
        }
        tile_regs_acquire();
        for (auto i : block.local()) {
            const auto global_i = block.to_global(i);
            copy_tile_init(cb_x_id);
            copy_tile(cb_x_id, global_i, i);
            power_iterative_tile_init();
            power_iterative_tile(i, 2);
        }
        tile_regs_commit();
        square.reserve_back(block.full_block_size());
        tile_regs_wait();
        for (auto i : block.local()) pack_tile(i, cb_square_id);
        tile_regs_release();
        square.push_back(block.full_block_size());
    }

#if @OUTPUT_STAGE@ == 4
    // The writer drains the FP32 power result directly.
    x.pop_front(padded_tiles);
    weight.pop_front(padded_tiles);
    scaler.pop_front(1);
    eps.pop_front(1);
    inverse_hidden.pop_front(1);
    return;
#endif

    // StableHLO recognizes ``sum(square) * (1 / hidden)`` as ``ttnn.mean``.
    // For FP32 input that operation deliberately selects the accurate SFPU
    // reduction: full-FP32 cross-tile folding, an intra-tile SFPU sum, then
    // the 1/N scalar multiply in the same destination register before pack.
    // The former resident implementation used the fast TF32/FPU SUM and a
    // separate binary multiply.  It happened to round identically for the
    // post-attention qualification tensor but diverges on real embedding-scale
    // input, so production must reproduce the actual mean program.
#if @OUTPUT_STAGE@ == 0 || @OUTPUT_STAGE@ == 5
    compute_kernel_lib::reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_square_id,
        cb_scaler_id,
        cb_variance_id,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT,
        ReduceFp32Mode::Fast>(
            compute_kernel_lib::ReduceInputBlockShape::of(
                1, hidden_tiles, 1),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            compute_kernel_lib::NoOp{});
#else
    compute_kernel_lib::reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_square_id,
        cb_scaler_id,
        cb_mean_id,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT,
        ReduceFp32Mode::Accurate>(
            compute_kernel_lib::ReduceInputBlockShape::of(
                1, hidden_tiles, 1),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            [](uint32_t dst_idx) {
                compute_kernel_lib::detail::reduce_post_mul_tile<
                    DataFormat::Float32>(
                        dst_idx, uint32_t{@INVERSE_HIDDEN_BITS@});
            });
#endif
    // The power producer publishes a block-aligned FIFO.  The reference sum
    // consumes only the logical hidden tiles, so explicitly release the
    // producer's unused tail before this CB is reused by a later layer phase.
    if constexpr (padded_tiles > hidden_tiles) {
        square.pop_front(padded_tiles - hidden_tiles);
    }

#if @OUTPUT_STAGE@ == 5
    // The writer drains the compact FP32 row-reduction tile directly.  Do not
    // expand it through an FPU broadcast: that diagnostic operation would
    // truncate the value to TF32 and falsely attribute the mismatch to sum.
    x.pop_front(padded_tiles);
    weight.pop_front(padded_tiles);
    scaler.pop_front(1);
    eps.pop_front(1);
    inverse_hidden.pop_front(1);
    return;
#endif

#if @OUTPUT_STAGE@ == 0
    // The accepted full-layer owner follows the generic StableHLO sequence:
    // a fast FP32 SUM followed by a distinct FP32 multiply by 1/hidden.  The
    // newer fused-mean schedule is mathematically equivalent but changes the
    // reduction/rounding contract and cannot be substituted inside the
    // bit-exact resident forward ABI.
    exact_sfpu_binary_tile<false>(
        sum, inverse_hidden, mean,
        cb_variance_id, cb_inverse_hidden_id, cb_mean_id);
#endif

#if @OUTPUT_STAGE@ == 6
    // Compact scalar diagnostic; writer consumes `mean` directly.
    x.pop_front(padded_tiles);
    weight.pop_front(padded_tiles);
    scaler.pop_front(1);
    eps.pop_front(1);
    inverse_hidden.pop_front(1);
    return;
#endif

    exact_sfpu_binary_tile<true>(
        mean, eps, variance_eps,
        cb_mean_id, cb_eps_id, cb_variance_eps_id);

    // The rsqrt is a third explicit unary operation and therefore reloads the
    // packed FP32 variance+epsilon tile before executing.
    variance_eps.wait_front(1);
    reconfig_data_format_srca(cb_variance_eps_id);
    pack_reconfig_data_format(cb_rstd_id);
    tile_regs_acquire();
    copy_tile_init(cb_variance_eps_id);
    copy_tile(cb_variance_eps_id, 0, 0);
    rsqrt_tile_init<false>();
    rsqrt_tile<false>(0);
    tile_regs_commit();
    variance_eps.pop_front(1);
    rstd.reserve_back(1);
    tile_regs_wait();
    pack_tile(0, cb_rstd_id);
    tile_regs_release();
    rstd.push_back(1);
    if (publish_post_rstd) {
        CircularBuffer post_rstd_ready(29), post_rstd_copied(28);
        post_rstd_ready.reserve_back(1);
        post_rstd_ready.push_back(1);
        post_rstd_copied.wait_front(1);
        post_rstd_copied.pop_front(1);
    }

#if @OUTPUT_STAGE@ == 7
    // Compact scalar diagnostic; writer consumes `rstd` directly.
    x.pop_front(padded_tiles);
    weight.pop_front(padded_tiles);
    scaler.pop_front(1);
    eps.pop_front(1);
    inverse_hidden.pop_front(1);
    return;
#endif

    constexpr uint32_t multiply_output_id =
#if @OUTPUT_STAGE@ == 2
        cb_output_id;
#else
        cb_normalized_fp32_id;
#endif

    // Match binary_ng's device-side column expansion, but retain its result
    // in a dedicated FP32 L1 FIFO for the following SFPU multiply. Keeping
    // this producer/consumer chain on the compute engine avoids the reverse
    // compute-to-reader CB dependency, whose credit wakeup is not reliable on
    // Blackhole.
    CircularBuffer rstd_expanded(cb_rstd_expanded_id);
    rstd.wait_front(1);
    rstd_expanded.reserve_back(1);
    reconfig_data_format_srca(cb_rstd_id);
    pack_reconfig_data_format(cb_rstd_id, cb_rstd_expanded_id);
    unary_bcast_init<BroadcastType::COL>(
        cb_rstd_id, cb_rstd_expanded_id);
    tile_regs_acquire();
    unary_bcast<BroadcastType::COL>(cb_rstd_id, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_rstd_expanded_id);
    tile_regs_release();
    rstd_expanded.push_back(1);
    rstd.pop_front(1);

#if @OUTPUT_STAGE@ == 8
    // Backward consumes the column-broadcast FP32 statistic at every hidden
    // tile. The writer replicates this one exact broadcast tile through NoC;
    // do not run the following x*rstd multiply or introduce another pack.
    x.pop_front(padded_tiles);
    weight.pop_front(padded_tiles);
    scaler.pop_front(1);
    eps.pop_front(1);
    inverse_hidden.pop_front(1);
    return;
#endif
    unary_op_reinit_no_barrier(cb_x_id, multiply_output_id);
    mul_binary_tile_init();
    rstd_expanded.wait_front(1);

#if @OUTPUT_STAGE@ != 2
    // The explicit graph has a real FP32 -> BF16 typecast boundary between
    // x*rstd and the gamma multiply.  Keep it in a dedicated FP32 FIFO so a
    // reduction CB never changes producer/consumer role in one invocation.
    // makes that numerical boundary explicit without leaving L1.  Packing the
    // binary SFPU result directly to a BF16 CB after the FP32 reduction
    // pipeline can leave Blackhole's pack/unpack RISCs in incompatible format
    // state and deadlock the writer.
    CircularBuffer normalized_fp32(cb_normalized_fp32_id);
#endif

    for (auto block : generic::blocks(hidden_tiles, suffix_block_size)) {
        // The previous resident suffix may have left the packer targeting
        // BF16 normalized/final output.  Return it to the FP32 x*rstd FIFO
        // with the ordinary in-program reconfiguration path.
        pack_reconfig_data_format(multiply_output_id);
        mul_binary_tile_init();
#if @OUTPUT_STAGE@ == 2
        output.reserve_back(block.full_block_size());
#else
        normalized_fp32.reserve_back(block.full_block_size());
#endif
        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(
            cb_rstd_expanded_id, cb_x_id);
        for (auto i : block.local()) {
            copy_tile(cb_x_id, block.to_global(i), i * 2);
        }
        copy_tile_to_dst_init_short_with_dt(
            cb_x_id, cb_rstd_expanded_id);
        for (auto i : block.local()) {
            copy_tile(cb_rstd_expanded_id, 0, i * 2 + 1);
            mul_binary_tile(i * 2, i * 2 + 1, i * 2);
        }
#if @OUTPUT_STAGE@ == 1
        // The standalone typecast is a BF16 pack boundary. Destination tiles
        // are already FP32, so packing them into this BF16 CB performs the
        // same RNE conversion without mixing two SFPU instruction families in
        // one destination-register transaction.
#endif
        tile_regs_commit();
        tile_regs_wait();
        for (auto i : block.local()) {
            pack_tile(
                i * 2,
#if @OUTPUT_STAGE@ == 2
                cb_output_id
#else
                cb_normalized_fp32_id
#endif
            );
        }
        tile_regs_release();
#if @OUTPUT_STAGE@ == 2
        output.push_back(block.full_block_size());
#else
        normalized_fp32.push_back(block.full_block_size());

        // Match the standalone TTNN FP32->BF16 typecast operation exactly,
        // but consume and publish its tiles inside this resident program.
        constexpr uint32_t normalized_output_id = cb_normalized_id;
        normalized_fp32.wait_front(block.full_block_size());
        normalized.reserve_back(block.full_block_size());
        // This is a later phase of one resident program, not a fresh unary
        // program.  Follow TTNN's multi-stage normalization kernels: change
        // SrcA through the short copy initializer and switch only the pack
        // destination after DEST is committed.  Re-entering a startup helper
        // here deadlocks because the three TRISCs are no longer at its common
        // initialization barrier.
        copy_tile_to_dst_init_short_with_dt(
            cb_rstd_expanded_id, cb_normalized_fp32_id);
        tile_regs_acquire();
        for (auto i : block.local()) {
            copy_tile(cb_normalized_fp32_id, i, i);
            typecast_tile_init<
                static_cast<std::underlying_type_t<DataFormat>>(
                    DataFormat::Float32),
                static_cast<std::underlying_type_t<DataFormat>>(
                    DataFormat::Float16_b)>();
            typecast_tile<
                static_cast<std::underlying_type_t<DataFormat>>(
                    DataFormat::Float32),
                static_cast<std::underlying_type_t<DataFormat>>(
                    DataFormat::Float16_b)>(i);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(normalized_output_id);
        for (auto i : block.local()) {
            pack_tile(i, normalized_output_id);
        }
        tile_regs_release();
        normalized_fp32.pop_front(block.full_block_size());
        normalized.push_back(block.full_block_size());
#if @OUTPUT_STAGE@ == 0

        // The final BF16 x BF16 gamma multiply follows TTNN's FPU binary
        // path.  Using the mixed-format SFPU helper here is close but not
        // exact (it perturbs products near BF16 rounding ties).  Gamma is a
        // complete physical tile in the owner reader, so ordinary tilewise
        // FPU multiplication reproduces the standalone `ttnn.multiply`.
        if (!resident_reentry) {
            weight.wait_front(block.start() + block.full_block_size());
        }
        normalized.wait_front(block.full_block_size());
        output.reserve_back(block.full_block_size());
        reconfig_data_format_srca(
            cb_normalized_fp32_id, cb_normalized_id);
        reconfig_data_format_srcb(cb_weight_id);
        pack_reconfig_data_format(cb_output_id);
        mul_tiles_init(cb_normalized_id, cb_weight_id);
        tile_regs_acquire();
        for (auto i : block.local()) {
            mul_tiles(
                cb_normalized_id, cb_weight_id,
                i, block.to_global(i), i);
        }
        // Preserve the accepted standalone binary/typecast boundary.  BF16
        // products are exact in FP32, so halfway cases are common and the
        // explicit RNE conversion is part of the production exact contract.
        typecast_tile_init<
            static_cast<std::underlying_type_t<DataFormat>>(
                DataFormat::Float32),
            static_cast<std::underlying_type_t<DataFormat>>(
                DataFormat::Float16_b)>();
        for (auto i : block.local()) {
            typecast_tile<
                static_cast<std::underlying_type_t<DataFormat>>(
                    DataFormat::Float32),
                static_cast<std::underlying_type_t<DataFormat>>(
                    DataFormat::Float16_b)>(i);
        }
        tile_regs_commit();
        normalized.pop_front(block.full_block_size());
        tile_regs_wait();
        for (auto i : block.local()) {
            pack_tile(i, cb_output_id);
        }
        tile_regs_release();
        output.push_back(block.full_block_size());
#endif
#endif
    }
    rstd_expanded.pop_front(1);
    x.pop_front(padded_tiles);
    weight.pop_front(padded_tiles);
    scaler.pop_front(1);
    eps.pop_front(1);
    inverse_hidden.pop_front(1);
}
