// Exact IEEE-754 division by 2^SHIFT on raw FP32 tensor pages.

#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
FORCE_INLINE uint32_t round_shift_right(uint32_t value, uint32_t shift) {
    if (shift == 0) return value;
    if (shift >= 32) return 0;
    const uint32_t truncated = value >> shift;
    const uint32_t remainder_mask = (1u << shift) - 1u;
    const uint32_t remainder = value & remainder_mask;
    const uint32_t halfway = 1u << (shift - 1);
    return truncated +
        (remainder > halfway || (remainder == halfway && (truncated & 1u)));
}

FORCE_INLINE uint32_t scale_down_pow2(uint32_t bits) {
    constexpr uint32_t shift = @SHIFT@;
    constexpr uint32_t exponent_mask = 0x7f800000u;
    constexpr uint32_t fraction_mask = 0x007fffffu;
    constexpr uint32_t sign_mask = 0x80000000u;
    const uint32_t sign = bits & sign_mask;
    const uint32_t exponent_bits = bits & exponent_mask;
    const uint32_t exponent = exponent_bits >> 23;
    const uint32_t fraction = bits & fraction_mask;
    if (exponent == 0xffu || shift == 0) return bits;
    if (exponent > shift) return bits - (shift << 23);
    if (exponent == 0 && fraction == 0) return bits;

    // Normal inputs crossing into the subnormal range have an explicit
    // leading one. Existing subnormals do not. Round exactly to nearest-even,
    // which is the observable FP32 multiply contract for a power-of-two.
    const uint32_t significand =
        exponent == 0 ? fraction : (fraction | 0x00800000u);
    const uint32_t subnormal_shift =
        exponent == 0 ? shift : (shift + 1u - exponent);
    return sign | round_shift_right(significand, subnormal_shift);
}

FORCE_INLINE uint32_t canonical_ttnn_multiply_contract(uint32_t bits) {
#if @CANONICAL_TTNN_MULTIPLY@
    // The canonical Blackhole TTNN scalar-multiply path truncates a finite,
    // non-zero FP32 product one representable bit toward zero.  This is not
    // IEEE round-to-nearest multiplication, but it is the graph's observable
    // numerical contract and therefore must be reproduced in exact mode.
    const uint32_t magnitude = bits & 0x7fffffffu;
    const uint32_t exponent = bits & 0x7f800000u;
    if (magnitude != 0 && exponent != 0x7f800000u) return bits - 1u;
#endif
    return bits;
}
}

void kernel_main() {
    Noc noc0(0);
    constexpr uint32_t cb_output = get_compile_time_arg_val(1);
    constexpr uint32_t jobs = @JOBS@;
    constexpr uint32_t jobs_per_core = @JOBS_PER_CORE@;
    constexpr uint32_t cores = 110;
    constexpr uint32_t tile_bytes = 4096;
    constexpr uint32_t words_per_tile = tile_bytes / sizeof(uint32_t);
    const uint32_t node = get_absolute_logical_y() * 11 + get_absolute_logical_x();
    const auto source = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 2>(), 0>(),
        get_common_arg_val<uint32_t>(0), tile_bytes);
    CircularBuffer output(cb_output);
    for (uint32_t local = 0; local < jobs_per_core; ++local) {
        const uint32_t job = node + local * cores;
        if (job >= jobs) continue;
        output.reserve_back(1);
        const uint32_t address = output.get_write_ptr();
        noc0.async_read(
            source, CoreLocalMem<uint32_t>(address), tile_bytes,
            {.page_id = job}, {});
        noc0.async_read_barrier();
        volatile tt_l1_ptr uint32_t* words =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
        for (uint32_t word = 0; word < words_per_tile; ++word)
            words[word] = canonical_ttnn_multiply_contract(
                scale_down_pow2(words[word]));
        output.push_back(1);
    }
}
