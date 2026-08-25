// All-worker exact Q/K/V frozen+LoRA projection reader.

#include <cstdint>
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kWorkers = 110;
constexpr uint32_t kColumns = 11;
constexpr uint32_t cb_h_x = 0, cb_a = 1, cb_base_x = 5;
constexpr uint32_t cb_w = 6, cb_b = 9, cb_bias = 13;

template <typename XAccessor, typename WAccessor, typename BiasAccessor,
          typename AAccessor, typename BAccessor>
FORCE_INLINE void read_projection(
    Noc& noc0, uint32_t node, uint32_t output_tiles,
    const XAccessor& input, const WAccessor& weight,
    const BiasAccessor& bias, const AAccessor& a, const BAccessor& b
) {
    constexpr uint32_t seq_tiles = @SEQ_TILES@;
    constexpr uint32_t hidden_tiles = @HIDDEN_TILES@;
    constexpr uint32_t base_phase = @BASE_PHASE@;
    constexpr bool has_bias = @HAS_BIAS@;
    constexpr uint32_t lora_phase = @LORA_PHASE@;
    constexpr uint32_t h_blocks = hidden_tiles / lora_phase;
    constexpr uint32_t base_blocks = hidden_tiles / base_phase;
    const uint32_t total_jobs = seq_tiles * output_tiles;
    const uint32_t jobs_per_node = (total_jobs + kWorkers - 1) / kWorkers;
    CircularBuffer hx(cb_h_x), aa(cb_a), bx(cb_base_x), ww(cb_w), bb(cb_b);
    CircularBuffer bias_cb(cb_bias);
    for (uint32_t local = 0; local < jobs_per_node; ++local) {
        const uint32_t job = node * jobs_per_node + local;
        if (job >= total_jobs) continue;
        const uint32_t token = job / output_tiles;
        const uint32_t output = job - token * output_tiles;
        for (uint32_t block = 0; block < h_blocks; ++block) {
            hx.reserve_back(lora_phase);
            aa.reserve_back(lora_phase);
            for (uint32_t tile = 0; tile < lora_phase; ++tile) {
                const uint32_t k = block * lora_phase + tile;
                noc0.async_read(input,
                    CoreLocalMem<uint32_t>(hx.get_write_ptr() + tile * kTileBytes),
                    kTileBytes, {.page_id = token * hidden_tiles + k}, {});
                noc0.async_read(a,
                    CoreLocalMem<uint32_t>(aa.get_write_ptr() + tile * kTileBytes),
                    kTileBytes, {.page_id = k}, {});
            }
            noc0.async_read_barrier();
            hx.push_back(lora_phase);
            aa.push_back(lora_phase);
        }
        for (uint32_t block = 0; block < base_blocks; ++block) {
            bx.reserve_back(base_phase); ww.reserve_back(base_phase);
            for (uint32_t tile = 0; tile < base_phase; ++tile) {
                const uint32_t k = block * base_phase + tile;
                noc0.async_read(input,
                    CoreLocalMem<uint32_t>(bx.get_write_ptr() + tile * kTileBytes),
                    kTileBytes, {.page_id = token * hidden_tiles + k}, {});
                noc0.async_read(weight,
                    CoreLocalMem<uint32_t>(ww.get_write_ptr() + tile * kTileBytes),
                    kTileBytes, {.page_id = k * output_tiles + output}, {});
            }
            noc0.async_read_barrier(); bx.push_back(base_phase); ww.push_back(base_phase);
        }
        bb.reserve_back(1);
        noc0.async_read(b, CoreLocalMem<uint32_t>(bb.get_write_ptr()),
                        kTileBytes, {.page_id = output}, {});
        noc0.async_read_barrier(); bb.push_back(1);
        if constexpr (has_bias) {
            bias_cb.reserve_back(1);
            noc0.async_read(bias, CoreLocalMem<uint32_t>(bias_cb.get_write_ptr()),
                            kTileBytes, {.page_id = output}, {});
            noc0.async_read_barrier(); bias_cb.push_back(1);
        }
    }
}
}

void kernel_main() {
    const uint32_t node =
        get_absolute_logical_y() * kColumns + get_absolute_logical_x();
    // Preserve the runtime-address ordinal emitted by the qualified
    // standalone TT-Lang projection.  TensorAccessorArgs' first ordinal is
    // the logical operation operand while its second ordinal selects the
    // kernel's compact common-argument list.  These ordinals need not be the
    // same: TT-Lang orders bias, weight and input addresses as 2, 1, 0.
    const auto hidden = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 15>(), 2>(),
        get_common_arg_val<uint32_t>(2), kTileBytes);
    const auto qw = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 15>(), 1>(),
        get_common_arg_val<uint32_t>(1), kTileBytes);
    const auto qbias = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 15>(), 0>(),
        get_common_arg_val<uint32_t>(0), kTileBytes);
    const auto qa = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<3, 15>(), 3>(),
        get_common_arg_val<uint32_t>(3), kTileBytes);
    const auto qb = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<4, 15>(), 4>(),
        get_common_arg_val<uint32_t>(4), kTileBytes);
    const auto kw = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<5, 15>(), 6>(),
        get_common_arg_val<uint32_t>(6), kTileBytes);
    const auto kbias = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<6, 15>(), 5>(),
        get_common_arg_val<uint32_t>(5), kTileBytes);
    const auto ka = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<7, 15>(), 7>(),
        get_common_arg_val<uint32_t>(7), kTileBytes);
    const auto kb = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<8, 15>(), 8>(),
        get_common_arg_val<uint32_t>(8), kTileBytes);
    const auto vw = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<9, 15>(), 10>(),
        get_common_arg_val<uint32_t>(10), kTileBytes);
    const auto vbias = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<10, 15>(), 9>(),
        get_common_arg_val<uint32_t>(9), kTileBytes);
    const auto va = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<11, 15>(), 11>(),
        get_common_arg_val<uint32_t>(11), kTileBytes);
    const auto vb = TensorAccessor(TensorAccessorArgs<
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<12, 15>(), 12>(),
        get_common_arg_val<uint32_t>(12), kTileBytes);
    Noc noc0(0);
    read_projection(noc0, node, @HIDDEN_TILES@, hidden, qw, qbias, qa, qb);
    if constexpr (@PHASE_COUNT@ > 1)
        read_projection(noc0, node, @KV_TILES@, hidden, kw, kbias, ka, kb);
    if constexpr (@PHASE_COUNT@ > 2)
        read_projection(noc0, node, @KV_TILES@, hidden, vw, vbias, va, vb);
}
