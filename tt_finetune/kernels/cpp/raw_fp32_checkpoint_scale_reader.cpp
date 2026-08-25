// Reassemble an opaque FP32 arena tile and divide it exactly by 2^SHIFT.

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
    const uint32_t remainder = value & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1);
    return truncated +
        (remainder > halfway || (remainder == halfway && (truncated & 1u)));
}

FORCE_INLINE uint32_t scale_down_pow2(uint32_t bits) {
    constexpr uint32_t shift = @SHIFT@;
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t exponent = (bits & 0x7f800000u) >> 23;
    const uint32_t fraction = bits & 0x007fffffu;
    if (exponent == 0xffu || shift == 0) return bits;
    if (exponent > shift) return bits - (shift << 23);
    if (exponent == 0 && fraction == 0) return bits;
    const uint32_t significand =
        exponent == 0 ? fraction : (fraction | 0x00800000u);
    const uint32_t subnormal_shift =
        exponent == 0 ? shift : (shift + 1u - exponent);
    return sign | round_shift_right(significand, subnormal_shift);
}
}

void kernel_main() {
    Noc noc0(0);
    constexpr uint32_t cb_output = get_compile_time_arg_val(1);
    constexpr uint32_t jobs = @JOBS@;
    constexpr uint32_t jobs_per_core = @JOBS_PER_CORE@;
    constexpr uint32_t cores = 110;
    constexpr uint32_t bf16_page_bytes = 2048;
    constexpr uint32_t fp32_tile_bytes = 4096;
    constexpr uint32_t words_per_tile = fp32_tile_bytes / sizeof(uint32_t);
    const uint32_t node = get_absolute_logical_y() * 11 + get_absolute_logical_x();
    const auto arena = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 2>(), 0>(),
        get_common_arg_val<uint32_t>(0), bf16_page_bytes);
    CircularBuffer output(cb_output);
    for (uint32_t local = 0; local < jobs_per_core; ++local) {
        const uint32_t job = node + local * cores;
        if (job >= jobs) continue;
        output.reserve_back(1);
        const uint32_t address = output.get_write_ptr();
        noc0.async_read(
            arena, CoreLocalMem<uint32_t>(address), bf16_page_bytes,
            {.page_id = @PAGE_OFFSET@ + 2 * job}, {});
        noc0.async_read(
            arena, CoreLocalMem<uint32_t>(address + bf16_page_bytes),
            bf16_page_bytes,
            {.page_id = @PAGE_OFFSET@ + 2 * job + 1}, {});
        noc0.async_read_barrier();
        volatile tt_l1_ptr uint32_t* words =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
        for (uint32_t word = 0; word < words_per_tile; ++word)
            words[word] = scale_down_pow2(words[word]);
        output.push_back(1);
    }
}
