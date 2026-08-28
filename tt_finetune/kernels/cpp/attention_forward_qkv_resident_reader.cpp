// Serial exact Q/K/V/O projection reader on the 50-core projection island.
//
// Q/K/V consume the framework hidden tensor.  O is different: its complete
// token-major input is assembled directly in local CB15/CB20 by the 60-core
// attention writer.  The reader publishes those two real L1 addresses only
// after every projection core has reserved its complete local window.

#include <cstdint>
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/layernorm_dataflow_utils.h"

namespace generic = norm::kernel_util::generic;
namespace layernorm_dataflow_utils =
    norm::layernorm::device::kernels::dataflow;

namespace {
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kWorkers = @PROJECTION_WORKERS@;
constexpr uint32_t kOWorkers = @O_WORKERS@;
constexpr uint32_t kColumns = 5;
constexpr uint32_t cb_h_x = 0, cb_a = 1, cb_base_x = 5;
constexpr uint32_t cb_w = 6, cb_b = 9, cb_bias = 13;
constexpr uint32_t cb_o_h_x = @O_H_X_CB@, cb_o_a = @O_A_CB@;
constexpr uint32_t cb_o_h = @O_H_CB@;
constexpr uint32_t cb_o_base_x = @O_BASE_X_CB@;
constexpr uint32_t cb_o_w = @O_W_CB@, cb_o_b = @O_B_CB@;
constexpr uint32_t kOInputReady = 13;
constexpr uint32_t kInvocationWriterCount = 14;
constexpr uint32_t kInvocationDone = 15;
constexpr bool kInputRMSNorm = @INPUT_RMSNORM@;
constexpr bool kPrefixOnly = @PROJECTION_ROPE_PREFIX_ONLY@;
constexpr bool kDedicatedORank = @DEDICATED_O_RANK@;
constexpr bool kPostAttentionBlock = @POST_ATTENTION_BLOCK@;
constexpr uint32_t kPostDiagnosticStage = @POST_DIAGNOSTIC_STAGE@;
constexpr uint32_t kNormX = 15, kNormWeight = 16;
constexpr uint32_t kNormScaler = 17, kNormEpsilon = 18;
constexpr uint32_t kNormInverseHidden = 19, kNormalizedHidden = 22;
// Blackhole exposes 16 semaphore IDs. Reuse the later O-handoff mailboxes
// during the earlier normalization epoch; their consumers cannot run until
// Q/K/V and attention have completed.
constexpr uint32_t kNormReadyCount = 8, kNormReady = 9;
constexpr uint8_t kPhysicalX[11] = {1,2,3,4,5,6,7,10,11,12,13};

FORCE_INLINE uint8_t physical_y(uint32_t y) {
    return static_cast<uint8_t>(y + 2);
}

FORCE_INLINE uint64_t projection_l1(uint32_t node, uint32_t address) {
    return get_noc_addr(kPhysicalX[node % kColumns],
                        physical_y(node / kColumns), address);
}

FORCE_INLINE uint64_t attention_l1(uint32_t node, uint32_t address) {
    constexpr uint32_t attention_columns = 6;
    constexpr uint32_t attention_x_offset = 5;
    const uint32_t x = attention_x_offset + node % attention_columns;
    return get_noc_addr(kPhysicalX[x], physical_y(node / attention_columns), address);
}

// A local CB is allowed to wrap only when one push lands exactly on its
// physical limit.  The resident Q/K/V and O epochs intentionally reuse CB
// identities, but Q/K/V assigns a different number of jobs to tail cores.
// Once compute has acknowledged that the CB is empty, re-arm the producer
// view at the physical origin instead of carrying that partial-cycle pointer
// into O.  This is the same empty-CB rebase contract used by Metal's fused
// multi-epoch kernels: reset the shared stream counters on the data-movement
// RISC and reset the consumer-local interface separately on TRISC.
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

// Keep the O-handoff waits distinguishable in Watcher traces.  Metal's
// generic noc_semaphore_wait_min helper records only ``NSMW``; when a
// multi-epoch owner stalls that loses the identity of the generation that
// failed to arrive.  These are the same cache-invalidating >= waits with a
// phase-specific waypoint, and compile to the ordinary loop when Watcher is
// disabled.
FORCE_INLINE void wait_o_input_ready(
    volatile tt_l1_ptr uint32_t* semaphore, uint32_t target) {
    WAYPOINT("OIRW");
    do { invalidate_l1_cache(); } while (*semaphore < target);
    WAYPOINT("OIRD");
}

// The RMSNorm owner has a reader-side invocation barrier before any Q/K/V
// work begins.  Use that barrier to reset one readiness flag on each
// projection core, then let the root poll the distributed flags after every
// core has reserved its complete O-input window.  This keeps reset and
// publication on the same NCRISC and avoids the lost-update race caused by a
// writer-RISC reset of one shared atomic counter on repeated submissions.
FORCE_INLINE void wait_all_o_input_ready_flags() {
    const uint32_t scratch_address = get_semaphore(kOInputReady);
    auto* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        scratch_address);
    // Node zero reaches this routine only after publishing its own local
    // readiness flag, so it needs to poll the remaining workers only.  Its
    // local flag is now free to serve as the one-word NoC read scratchpad.
    for (uint32_t target = 1; target < kWorkers; ++target) {
        while (true) {
            noc_semaphore_set(scratch, 0);
            noc_async_read(
                projection_l1(
                    target, get_semaphore(kOInputReady)),
                scratch_address, sizeof(uint32_t));
            noc_async_read_barrier();
            invalidate_l1_cache();
            if (*scratch >= 1) break;
        }
    }
}

FORCE_INLINE void wait_o_h_credit(
    volatile tt_l1_ptr uint32_t* semaphore, uint32_t target) {
    WAYPOINT("OHCW");
    do { invalidate_l1_cache(); } while (*semaphore < target);
    WAYPOINT("OHCD");
}

FORCE_INLINE void wait_o_base_credit(
    volatile tt_l1_ptr uint32_t* semaphore, uint32_t target) {
    WAYPOINT("OBCW");
    do { invalidate_l1_cache(); } while (*semaphore < target);
    WAYPOINT("OBCD");
}

static __attribute__((noinline, noclone)) void prepare_full_fp32_scalar(
    CircularBuffer& cb, uint32_t scalar_bits) {
    cb.reserve_back(1);
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        cb.get_write_ptr());
    for (uint32_t i = 0; i < 1024; ++i) ptr[i] = scalar_bits;
    cb.push_back(1);
}

static __attribute__((noinline, noclone)) void zero_tile_memory(uint32_t address) {
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
    for (uint32_t word = 0; word < kTileBytes / sizeof(uint32_t); ++word)
        words[word] = 0;
}

template <typename XAccessor>
FORCE_INLINE void read_hidden_tile(
    Noc& noc0, const XAccessor& input, uint32_t token, uint32_t hidden_tile,
    uint32_t destination) {
    if constexpr (kInputRMSNorm) {
        CircularBuffer normalized(kNormalizedHidden);
        noc_async_read(
            projection_l1(
                token,
                normalized.get_read_ptr() + hidden_tile * kTileBytes),
            destination, kTileBytes);
    } else {
        noc0.async_read(
            input, CoreLocalMem<uint32_t>(destination), kTileBytes,
            {.page_id = @HIDDEN_PAGE_BASE@ +
                token * @HIDDEN_TILES@ + hidden_tile}, {});
    }
}

template <typename XAccessor, typename WAccessor, typename BiasAccessor,
          typename AAccessor, typename BAccessor>
FORCE_INLINE void read_projection(
    Noc& noc0, uint32_t node, uint32_t output_tiles,
    uint32_t base_tile_offset, uint32_t h_rank_tile, uint32_t b_rank_tile,
    const XAccessor& input, const WAccessor& weight,
    const BiasAccessor& bias, const AAccessor& a, const BAccessor& b
) {
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@;
    constexpr uint32_t base_phase = @BASE_PHASE@;
    constexpr uint32_t qkv_rank_tiles = @QKV_RANK_TILES@;
    constexpr uint32_t qkv_storage_rank_tiles = @QKV_STORAGE_RANK_TILES@;
    constexpr bool precomputed_qkv_h = @PRECOMPUTED_QKV_H@;
    constexpr bool precomputed_qkv_base = @PRECOMPUTED_QKV_BASE@;
    constexpr bool precomputed_qkv_delta = @PRECOMPUTED_QKV_DELTA@;
    constexpr uint32_t total_qkv_tiles = hidden_tiles + 2 * @KV_TILES@;
    constexpr bool has_bias = @HAS_BIAS@;
    constexpr uint32_t lora_phase = @LORA_PHASE@;
    constexpr uint32_t h_blocks = hidden_tiles / lora_phase;
    constexpr uint32_t base_blocks = hidden_tiles / base_phase;
    const uint32_t total_jobs = seq_tiles * output_tiles;
    const uint32_t jobs_per_node = (total_jobs + kWorkers - 1) / kWorkers;
    CircularBuffer hx(cb_h_x), aa(cb_a), h(3), saved_h(4);
    CircularBuffer bx(cb_base_x), ww(cb_w), base(8), bb(cb_b);
    CircularBuffer bias_cb(cb_bias);
    for (uint32_t local = 0; local < jobs_per_node; ++local) {
        const uint32_t job = node * jobs_per_node + local;
        if (job >= total_jobs) continue;
        const uint32_t token = job / output_tiles;
        const uint32_t output = job - token * output_tiles;
        if constexpr (precomputed_qkv_h) {
            if constexpr (!precomputed_qkv_delta) {
                h.reserve_back(qkv_rank_tiles);
                for (uint32_t rank_tile = 0; rank_tile < qkv_rank_tiles;
                     ++rank_tile) {
                    noc0.async_read(
                        a, CoreLocalMem<uint32_t>(h.get_write_ptr() +
                            rank_tile * kTileBytes), kTileBytes,
                        {.page_id = token * qkv_storage_rank_tiles +
                            h_rank_tile + rank_tile}, {});
                }
            }
            if (output == 0) {
                saved_h.reserve_back(qkv_rank_tiles);
                for (uint32_t rank_tile = 0; rank_tile < qkv_rank_tiles;
                     ++rank_tile) {
                    noc0.async_read(
                        a, CoreLocalMem<uint32_t>(saved_h.get_write_ptr() +
                            rank_tile * kTileBytes), kTileBytes,
                        {.page_id = token * qkv_storage_rank_tiles +
                            h_rank_tile + rank_tile}, {});
                }
            }
            noc0.async_read_barrier();
            if constexpr (!precomputed_qkv_delta) {
                h.push_back(qkv_rank_tiles);
            }
            if (output == 0) saved_h.push_back(qkv_rank_tiles);
        } else {
            for (uint32_t block = 0; block < h_blocks; ++block) {
                hx.reserve_back(lora_phase);
                aa.reserve_back(qkv_rank_tiles * lora_phase);
                for (uint32_t tile = 0; tile < lora_phase; ++tile) {
                    const uint32_t k = block * lora_phase + tile;
                    read_hidden_tile(
                        noc0, input, token, k,
                        hx.get_write_ptr() + tile * kTileBytes);
                    for (uint32_t rank_tile = 0;
                         rank_tile < qkv_rank_tiles; ++rank_tile) {
                        noc0.async_read(a,
                            CoreLocalMem<uint32_t>(aa.get_write_ptr() +
                                (rank_tile * lora_phase + tile) * kTileBytes),
                            kTileBytes,
                            {.page_id = rank_tile * hidden_tiles + k}, {});
                    }
                }
                noc0.async_read_barrier();
                hx.push_back(lora_phase);
                aa.push_back(qkv_rank_tiles * lora_phase);
            }
        }
        if constexpr (precomputed_qkv_base) {
            base.reserve_back(1);
            noc0.async_read(
                weight, CoreLocalMem<uint32_t>(base.get_write_ptr()),
                kTileBytes,
                {.page_id = token * total_qkv_tiles +
                    base_tile_offset + output}, {});
            noc0.async_read_barrier();
            base.push_back(1);
        } else {
            for (uint32_t block = 0; block < base_blocks; ++block) {
                bx.reserve_back(base_phase); ww.reserve_back(base_phase);
                for (uint32_t tile = 0; tile < base_phase; ++tile) {
                    const uint32_t k = block * base_phase + tile;
                    read_hidden_tile(
                        noc0, input, token, k,
                        bx.get_write_ptr() + tile * kTileBytes);
                    noc0.async_read(weight,
                        CoreLocalMem<uint32_t>(
                            ww.get_write_ptr() + tile * kTileBytes),
                        kTileBytes,
                        {.page_id = k * output_tiles + output}, {});
                }
                noc0.async_read_barrier();
                bx.push_back(base_phase);
                ww.push_back(base_phase);
            }
        }
        if constexpr (precomputed_qkv_delta) {
            // In the separate-Q/K/V exact path the outer owner has already
            // run TTNN's shape-selected H@B.T program.  Operand B is remapped
            // to that token-major delta tensor, so publish one final tile
            // directly to the compute-side delta queue.
            bb.reserve_back(1);
            noc0.async_read(
                b, CoreLocalMem<uint32_t>(bb.get_write_ptr()),
                kTileBytes,
                {.page_id = token * output_tiles + output}, {});
            noc0.async_read_barrier();
            bb.push_back(1);
        } else {
            bb.reserve_back(qkv_rank_tiles);
            for (uint32_t rank_tile = 0; rank_tile < qkv_rank_tiles;
                 ++rank_tile) {
                noc0.async_read(
                    b, CoreLocalMem<uint32_t>(bb.get_write_ptr() +
                        rank_tile * kTileBytes), kTileBytes,
                    {.page_id = output * qkv_storage_rank_tiles +
                        b_rank_tile + rank_tile}, {});
            }
            noc0.async_read_barrier(); bb.push_back(qkv_rank_tiles);
        }
        if constexpr (has_bias && !precomputed_qkv_base) {
            bias_cb.reserve_back(1);
            noc0.async_read(bias,
                CoreLocalMem<uint32_t>(bias_cb.get_write_ptr()),
                kTileBytes, {.page_id = output}, {});
            noc0.async_read_barrier(); bias_cb.push_back(1);
        }
    }
}

template <typename WAccessor, typename AAccessor, typename BAccessor>
FORCE_INLINE void read_o_projection_weights(
    Noc& noc0, uint32_t node, const WAccessor& weight,
    const AAccessor& a, const BAccessor& b
) {
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@;
    constexpr uint32_t base_phase = @BASE_PHASE@;
    constexpr uint32_t lora_phase = @O_LORA_PHASE@;
    constexpr uint32_t rank_tiles = @RANK_TILES@;
    constexpr uint32_t h_blocks = hidden_tiles / lora_phase;
    constexpr uint32_t base_blocks = hidden_tiles / base_phase;
    constexpr uint32_t total_jobs = seq_tiles * hidden_tiles;
    constexpr uint32_t jobs_per_node =
        (total_jobs + kOWorkers - 1) / kOWorkers;
    CircularBuffer aa(cb_o_a), ww(cb_o_w), bb(cb_o_b);
    for (uint32_t local = 0; local < jobs_per_node; ++local) {
        const uint32_t job = node * jobs_per_node + local;
        if (job >= total_jobs) continue;
        const uint32_t output = job % hidden_tiles;
        if constexpr (!kDedicatedORank) {
            for (uint32_t block = 0; block < h_blocks; ++block) {
                aa.reserve_back(rank_tiles * lora_phase);
                for (uint32_t tile = 0; tile < lora_phase; ++tile) {
                    const uint32_t k = block * lora_phase + tile;
                    for (uint32_t rank_tile = 0; rank_tile < rank_tiles;
                         ++rank_tile) {
                        noc0.async_read(a,
                            CoreLocalMem<uint32_t>(aa.get_write_ptr() +
                                (rank_tile * lora_phase + tile) * kTileBytes),
                            kTileBytes,
                            {.page_id = rank_tile * hidden_tiles + k}, {});
                    }
                }
                noc0.async_read_barrier();
                aa.push_back(rank_tiles * lora_phase);
            }
        }
        for (uint32_t block = 0; block < base_blocks; ++block) {
            ww.reserve_back(base_phase);
            for (uint32_t tile = 0; tile < base_phase; ++tile) {
                const uint32_t k = block * base_phase + tile;
                noc0.async_read(weight,
                    CoreLocalMem<uint32_t>(ww.get_write_ptr() + tile * kTileBytes),
                    kTileBytes, {.page_id = k * hidden_tiles + output}, {});
            }
            noc0.async_read_barrier();
            ww.push_back(base_phase);
        }
        bb.reserve_back(rank_tiles);
        for (uint32_t rank_tile = 0; rank_tile < rank_tiles; ++rank_tile) {
            noc0.async_read(b,
                CoreLocalMem<uint32_t>(bb.get_write_ptr() +
                    rank_tile * kTileBytes), kTileBytes,
                {.page_id = output * rank_tiles + rank_tile}, {});
        }
        noc0.async_read_barrier();
        bb.push_back(rank_tiles);
    }
}
}

__attribute__((optimize("Os"))) void kernel_main() {
    const uint32_t x = get_absolute_logical_x();
    if (x >= kColumns) return;
    const uint32_t node = get_absolute_logical_y() * kColumns + x;
    volatile tt_l1_ptr uint32_t* invocation_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(kInvocationDone));
    noc_semaphore_set(invocation_done, 0);
    if (node == 0) {
        volatile tt_l1_ptr uint32_t* writer_count =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(kInvocationWriterCount));
        noc_semaphore_set(writer_count, 0);
    }
    const auto hidden = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, @TOTAL_CBS@>(), 0>(),
        get_common_arg_val<uint32_t>(0), kTileBytes);
    const auto qw = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, @TOTAL_CBS@>(), 1>(),
        get_common_arg_val<uint32_t>(1), kTileBytes);
    const auto qbias = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, @TOTAL_CBS@>(), 2>(),
        get_common_arg_val<uint32_t>(2), kTileBytes);
    const auto qa = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<3, @TOTAL_CBS@>(), 3>(),
        get_common_arg_val<uint32_t>(3), kTileBytes);
    const auto qb = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<4, @TOTAL_CBS@>(), 4>(),
        get_common_arg_val<uint32_t>(4), kTileBytes);
    const auto kw = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<5, @TOTAL_CBS@>(), 5>(),
        get_common_arg_val<uint32_t>(5), kTileBytes);
    const auto kbias = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<6, @TOTAL_CBS@>(), 6>(),
        get_common_arg_val<uint32_t>(6), kTileBytes);
    const auto ka = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<7, @TOTAL_CBS@>(), 7>(),
        get_common_arg_val<uint32_t>(7), kTileBytes);
    const auto kb = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<8, @TOTAL_CBS@>(), 8>(),
        get_common_arg_val<uint32_t>(8), kTileBytes);
    const auto vw = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<9, @TOTAL_CBS@>(), 9>(),
        get_common_arg_val<uint32_t>(9), kTileBytes);
    const auto vbias = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<10, @TOTAL_CBS@>(), 10>(),
        get_common_arg_val<uint32_t>(10), kTileBytes);
    const auto va = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<11, @TOTAL_CBS@>(), 11>(),
        get_common_arg_val<uint32_t>(11), kTileBytes);
    const auto vb = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<12, @TOTAL_CBS@>(), 12>(),
        get_common_arg_val<uint32_t>(12), kTileBytes);
    const auto ow = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<13, @TOTAL_CBS@>(), 13>(),
        get_common_arg_val<uint32_t>(13), kTileBytes);
    const auto oa = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<14, @TOTAL_CBS@>(), 14>(),
        get_common_arg_val<uint32_t>(14), kTileBytes);
    const auto ob = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<15, @TOTAL_CBS@>(), 15>(),
        get_common_arg_val<uint32_t>(15), kTileBytes);
    const auto input_norm_weight = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<19, @TOTAL_CBS@>(), 16>(),
        get_common_arg_val<uint32_t>(16), kTileBytes);
    const auto post_norm_weight = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<20, @TOTAL_CBS@>(), 17>(),
        get_common_arg_val<uint32_t>(17), kTileBytes);
    Noc noc0(0);
    if constexpr (!kInputRMSNorm) {
        // Non-normalized projection variants still need an invocation-start
        // barrier before distributed O-readiness flags can be reused.  The
        // same semaphore pair is otherwise idle until the later O/post-block
        // epoch, so establish the barrier here without introducing another
        // physical control allocation.
        volatile tt_l1_ptr uint32_t* init_ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(kNormReady));
        volatile tt_l1_ptr uint32_t* o_ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(kOInputReady));
        noc_semaphore_set(init_ready, 0);
        noc_semaphore_set(o_ready, 0);
        noc_semaphore_inc(
            projection_l1(0, get_semaphore(kNormReadyCount)), 1);
        noc_async_atomic_barrier();
        if (node == 0) {
            volatile tt_l1_ptr uint32_t* count =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(kNormReadyCount));
            noc_semaphore_wait_min(count, kWorkers);
            noc_semaphore_set(count, 0);
            noc_semaphore_set(init_ready, 1);
            for (uint32_t target = 1; target < kWorkers; ++target)
                noc_async_write(
                    get_semaphore(kNormReady),
                    projection_l1(target, get_semaphore(kNormReady)),
                    sizeof(uint32_t));
            noc_async_write_barrier();
        }
        noc_semaphore_wait_min(init_ready, 1);
        noc_semaphore_set(init_ready, 0);
    }
    if constexpr (kInputRMSNorm) {
        // Every participant clears its local generation words before joining
        // the root initialization count. This prevents a previous invocation
        // from satisfying either phase wait.
        volatile tt_l1_ptr uint32_t* norm_ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(kNormReady));
        volatile tt_l1_ptr uint32_t* o_ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(kOInputReady));
        noc_semaphore_set(norm_ready, 0);
        noc_semaphore_set(o_ready, 0);
        noc_semaphore_inc(
            projection_l1(0, get_semaphore(kNormReadyCount)), 1);
        noc_async_atomic_barrier();
        if (node == 0) {
            volatile tt_l1_ptr uint32_t* count =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(kNormReadyCount));
            noc_semaphore_wait_min(count, kWorkers);
            noc_semaphore_set(count, 0);
            noc_semaphore_set(norm_ready, 1);
            for (uint32_t target = 1; target < kWorkers; ++target)
                noc_async_write(
                    get_semaphore(kNormReady),
                    projection_l1(target, get_semaphore(kNormReady)),
                    sizeof(uint32_t));
            noc_async_write_barrier();
        }
        noc_semaphore_wait_min(norm_ready, 1);

        constexpr uint32_t hidden_tiles = @HIDDEN_TILES@;
        constexpr uint32_t padded_tiles = @PADDED_HIDDEN_TILES@;
        constexpr uint32_t token_tiles = @SEQ_TILES@;
        if (node < token_tiles) {
            dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
                kNormScaler, ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
            CircularBuffer inverse_hidden(kNormInverseHidden);
            CircularBuffer epsilon(kNormEpsilon);
            constexpr float inverse_hidden_value =
                1.0f / @HIDDEN_FEATURES@;
            prepare_full_fp32_scalar(
                inverse_hidden,
                __builtin_bit_cast(uint32_t, inverse_hidden_value));
            prepare_full_fp32_scalar(
                epsilon, uint32_t{@RMSNORM_EPSILON_BITS@});

            CircularBuffer norm_x(kNormX), gamma(kNormWeight);
            for (auto block : generic::blocks(hidden_tiles, uint32_t{4})) {
                layernorm_dataflow_utils::read_block_to_cb(
                    noc0, norm_x, hidden, kTileBytes,
                    @HIDDEN_PAGE_BASE@ +
                        node * hidden_tiles + block.start(), block);
                layernorm_dataflow_utils::read_block_to_cb(
                    noc0, gamma, input_norm_weight, kTileBytes,
                    block.start(), block);
            }
        }
        // The writer RISC consumes the compute-produced normalized CB and
        // publishes generation 2.  Reader-side wait_front on a CB produced
        // by compute is an unreliable reverse wake-up direction on
        // Blackhole; the ordinary compute -> writer CB edge is supported.
        noc_semaphore_wait_min(norm_ready, 2);
    }
    read_projection(noc0, node, @HIDDEN_TILES@, 0, 0, 0,
                    hidden, qw, qbias, qa, qb);
    read_projection(noc0, node, @KV_TILES@, @HIDDEN_TILES@, 0, 0,
                    hidden, kw, kbias, ka, kb);
    read_projection(noc0, node, @KV_TILES@,
                    @HIDDEN_TILES@ + @KV_TILES@,
                    @GROUPED_QKV_LORA@ == 1 ? 1 : 0,
                    @GROUPED_QKV_LORA@ == 1 ? 1 : 0,
                    hidden, vw, vbias, va, vb);

    // In the aliased layer owner, O reuses Q/K/V CB identities. Wait until
    // compute has consumed every Q/K/V reader tile before reserving O input.
    if constexpr (@ALIASED_PROJECTION_CBS@ && !kPrefixOnly) {
        CircularBuffer qkv_consumed(30), qkv_reuse_ack(31);
        qkv_consumed.wait_front(1);
        qkv_consumed.pop_front(1);
        rebase_empty_producer_cb(cb_h_x);
        rebase_empty_producer_cb(cb_a);
        rebase_empty_producer_cb(cb_base_x);
        rebase_empty_producer_cb(cb_w);
        rebase_empty_producer_cb(cb_b);
        if constexpr (kPostAttentionBlock) {
            // Input RMSNorm has fully consumed these FIFOs.  Rebase their
            // reader-owned views before CB15/16 become the post-O staging and
            // original-residual streams and CB17..19 become the second norm's
            // scalar inputs.
            rebase_empty_producer_cb(15);
            rebase_empty_producer_cb(16);
            rebase_empty_producer_cb(17);
            rebase_empty_producer_cb(18);
            rebase_empty_producer_cb(19);
            rebase_empty_producer_cb(cb_bias);
        }
        qkv_reuse_ack.reserve_back(1);
        qkv_reuse_ack.push_back(1);
    }

    if constexpr (!kPrefixOnly) {
    // Reserve the complete post-attention row before any O writer targets it.
    // All 50 reader RISCs join the readiness count so no late core can clear a
    // freshly published generation.  Only the four sequence-row owners hold
    // data; identical CB allocation makes node zero's CB13 address valid on
    // every row owner.
    CircularBuffer post_attention_output(13), layer_input(16);
    if constexpr (kPostAttentionBlock && kPostDiagnosticStage != 6) {
        constexpr uint32_t hidden_tiles = @POST_HIDDEN_TILES@;
        constexpr uint32_t padded_tiles = @POST_PADDED_HIDDEN_TILES@;
        constexpr uint32_t token_tiles = @POST_SEQ_TILES@;
        if constexpr (kPostDiagnosticStage != 6) if (node < token_tiles) {
            volatile tt_l1_ptr uint32_t* credits =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(7));
            noc_semaphore_set(credits, 0);
            if constexpr (kPostDiagnosticStage != 7 &&
                          kPostDiagnosticStage != 8) {
                post_attention_output.reserve_back(padded_tiles);
                layer_input.reserve_back(padded_tiles);
                for (uint32_t tile = 0; tile < hidden_tiles; ++tile)
                    noc0.async_read(
                        hidden,
                        CoreLocalMem<uint32_t>(
                            layer_input.get_write_ptr() + tile * kTileBytes),
                        kTileBytes,
                        {.page_id = @HIDDEN_PAGE_BASE@ +
                            node * hidden_tiles + tile}, {});
                noc0.async_read_barrier();
                for (uint32_t tile = hidden_tiles; tile < padded_tiles; ++tile) {
                    zero_tile_memory(
                        post_attention_output.get_write_ptr() + tile * kTileBytes);
                    zero_tile_memory(layer_input.get_write_ptr() + tile * kTileBytes);
                }
                layer_input.push_back(padded_tiles);
            }
        }
    }

    // Reserve the whole local O input epoch before advertising its address.
    // Every valid O job consumes the same hidden-width head row twice: once
    // for LoRA H and once for the frozen-base projection.  Keeping the two
    // exact streams distinct preserves the accepted projection schedule.
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@;
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t total_o_jobs = seq_tiles * hidden_tiles;
    constexpr uint32_t jobs_per_node =
        (total_o_jobs + kOWorkers - 1) / kOWorkers;
    const uint32_t first_job = node * jobs_per_node;
    const uint32_t local_jobs = first_job >= total_o_jobs ? 0 :
        ((total_o_jobs - first_job) < jobs_per_node ?
            (total_o_jobs - first_job) : jobs_per_node);
    const uint32_t input_tiles = local_jobs * hidden_tiles;
    CircularBuffer o_hx(cb_o_h_x), o_h(cb_o_h), o_bx(cb_o_base_x);
    if (input_tiles != 0) {
        if constexpr (!kDedicatedORank) o_hx.reserve_back(input_tiles);
        else o_h.reserve_back(local_jobs);
        o_bx.reserve_back(input_tiles);
    }
    volatile tt_l1_ptr uint32_t* h_credit =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(11));
    volatile tt_l1_ptr uint32_t* base_credit =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(12));
    noc_semaphore_set(h_credit, 0);
    noc_semaphore_set(base_credit, 0);
    // Reaching this point proves this core has reserved both destination
    // windows and reset its per-core credits.  Every variant has now passed
    // an invocation-start reader barrier, so readiness is always published
    // through the distributed per-core flag protocol.
    volatile tt_l1_ptr uint32_t* local_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(kOInputReady));
    noc_semaphore_set(local_ready, 1);
    if (node == 0) {
        wait_all_o_input_ready_flags();
        volatile tt_l1_ptr uint32_t* h_address =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(4));
        volatile tt_l1_ptr uint32_t* base_address =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(5));
        noc_semaphore_set(
            h_address,
            kDedicatedORank ? o_h.get_write_ptr() : o_hx.get_write_ptr());
        noc_semaphore_set(base_address, o_bx.get_write_ptr());
        constexpr uint32_t active_attention = @QUERY_HEADS@ * seq_tiles;
        for (uint32_t target = 0; target < active_attention; ++target) {
            if constexpr (!kDedicatedORank) noc_async_write(
                get_semaphore(4), attention_l1(target, get_semaphore(8)),
                sizeof(uint32_t));
            noc_async_write(
                get_semaphore(5), attention_l1(target, get_semaphore(9)),
                sizeof(uint32_t));
        }
        if constexpr (kDedicatedORank) {
            for (uint32_t row = 0; row < seq_tiles; ++row) {
                const uint32_t producer = 45 + row;
                noc_async_write(
                    get_semaphore(4),
                    projection_l1(producer, get_semaphore(1)),
                    sizeof(uint32_t));
            }
        }
        noc_async_write_barrier();
        if constexpr (kDedicatedORank) {
            for (uint32_t row = 0; row < seq_tiles; ++row)
                noc_semaphore_inc(
                    projection_l1(45 + row, get_semaphore(2)), 1);
            noc_async_atomic_barrier();
        }
        for (uint32_t target = 0; target < active_attention; ++target) {
            noc_semaphore_inc(attention_l1(target, get_semaphore(10)), 1);
        }
        noc_async_atomic_barrier();
    }
    if (input_tiles != 0) {
        wait_o_h_credit(
            h_credit, kDedicatedORank ? local_jobs : input_tiles);
        wait_o_base_credit(base_credit, input_tiles);
        noc_semaphore_set(h_credit, 0);
        noc_semaphore_set(base_credit, 0);
        if constexpr (!kDedicatedORank) o_hx.push_back(input_tiles);
        else o_h.push_back(local_jobs);
        o_bx.push_back(input_tiles);
    }
    read_o_projection_weights(noc0, node, ow, oa, ob);
    if constexpr (kPostAttentionBlock) {
        constexpr uint32_t hidden_tiles = @POST_HIDDEN_TILES@;
        constexpr uint32_t padded_tiles = @POST_PADDED_HIDDEN_TILES@;
        constexpr uint32_t token_tiles = @POST_SEQ_TILES@;
        if constexpr (kPostDiagnosticStage < 5 ||
                      kPostDiagnosticStage == 9 ||
                      kPostDiagnosticStage == 10 ||
                      kPostDiagnosticStage == 11 ||
                      kPostDiagnosticStage == 12) if (node < token_tiles) {
            volatile tt_l1_ptr uint32_t* credits =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(7));
            noc_semaphore_wait_min(
                credits,
                kPostDiagnosticStage == 4 ? uint32_t{1} : hidden_tiles);
            noc_semaphore_set(credits, 0);
            post_attention_output.push_back(padded_tiles);

            // Residual compute owns CB16 until its last add.  Its CB30 token
            // proves the original layer-input row has been consumed before
            // gamma is streamed into the same physical CB for the shared
            // RMSNorm routine's second invocation.
            CircularBuffer o_consumed(30), post_gamma_ready(31);
            o_consumed.wait_front(1);
            o_consumed.pop_front(1);
            if constexpr (kPostDiagnosticStage < 5)
                rebase_empty_producer_cb(16);
            if constexpr (kPostDiagnosticStage != 4) {
                volatile tt_l1_ptr uint32_t* residual_written =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        get_semaphore(12));
                noc_semaphore_wait_min(residual_written, 1);
                noc_semaphore_set(residual_written, 0);
            }
            if constexpr (kPostDiagnosticStage != 1 &&
                          kPostDiagnosticStage != 10 &&
                          kPostDiagnosticStage != 4) {
                dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
                    kNormScaler, ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW,
                    dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
                CircularBuffer inverse_hidden(kNormInverseHidden);
                CircularBuffer epsilon(kNormEpsilon);
                constexpr float inverse_hidden_value =
                    1.0f / @HIDDEN_FEATURES@;
                prepare_full_fp32_scalar(
                    inverse_hidden,
                    __builtin_bit_cast(uint32_t, inverse_hidden_value));
                prepare_full_fp32_scalar(
                    epsilon, uint32_t{@POST_RMSNORM_EPSILON_BITS@});
                CircularBuffer gamma(16);
                for (auto block : generic::blocks(hidden_tiles, uint32_t{4}))
                    layernorm_dataflow_utils::read_block_to_cb(
                        noc0, gamma, post_norm_weight, kTileBytes,
                        block.start(), block);
                post_gamma_ready.reserve_back(1);
                post_gamma_ready.push_back(1);
            }
        }
    }
    }
    // A later resident owner reuses these fixed L1 semaphore/mailbox
    // addresses.  Keep this reader RISC alive until every writer in both
    // islands has completed the current invocation.
    noc_semaphore_wait_min(invocation_done, 2);
    if constexpr (@ALIASED_PROJECTION_CBS@) {
        rebase_empty_producer_cb(cb_h_x);
        rebase_empty_producer_cb(cb_a);
        rebase_empty_producer_cb(cb_base_x);
        rebase_empty_producer_cb(cb_w);
        rebase_empty_producer_cb(cb_b);
        rebase_empty_producer_cb(cb_bias);
        if constexpr (kPostAttentionBlock) {
            // Stages 1/2 deliberately stop before the residual stream in
            // CB15 is consumed by RMSNorm; do not run the empty-CB audit on
            // that diagnostic-only retained stream.
            if constexpr (kPostDiagnosticStage == 0 ||
                          kPostDiagnosticStage == 3)
                rebase_empty_producer_cb(15);
            if constexpr (kPostDiagnosticStage != 2 &&
                          kPostDiagnosticStage != 9 &&
                          kPostDiagnosticStage != 10 &&
                          kPostDiagnosticStage != 12) {
                if constexpr (kPostDiagnosticStage != 5 &&
                              kPostDiagnosticStage != 6)
                    rebase_empty_producer_cb(16);
                rebase_empty_producer_cb(17);
                rebase_empty_producer_cb(18);
                rebase_empty_producer_cb(19);
            }
        }
    }
}
