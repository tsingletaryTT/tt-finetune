// Receive resident max/logsum, replay logits, and synthesize the exact
// cross-entropy seed from one compact INT32 metadata tile per 32 tokens.

#include <cstdint>
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint8_t kMaxPhysicalX = 1;
constexpr uint8_t kLogPhysicalX = 3;
constexpr uint32_t kBf16TileBytes = 2048;
constexpr uint32_t kWordTileBytes = 4096;
constexpr uint32_t kFace = 16;
constexpr uint32_t kFaceElements = kFace * kFace;

FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}

FORCE_INLINE uint32_t tile_offset(uint32_t row, uint32_t column) {
    const uint32_t face = ((row >> 4) << 1) | (column >> 4);
    return face * kFaceElements + (row & 15u) * kFace + (column & 15u);
}

template <bool DeviceCount>
FORCE_INLINE void produce_seed_tile(
    const volatile tt_l1_ptr uint32_t* metadata_words,
    uint32_t vocabulary_tile,
    uint32_t device_scale_bits,
    DataflowBuffer seed) {
    seed.reserve_back(1);
    auto* seed_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        seed.get_write_ptr());
    for (uint32_t element = 0; element < 1024; ++element)
        seed_words[element] = 0;
    const uint32_t vocabulary_start = vocabulary_tile * 32;
    for (uint32_t logical_row = 0; logical_row < 32; ++logical_row) {
        const int32_t target = static_cast<int32_t>(
            metadata_words[tile_offset(logical_row, 0)]);
        const uint32_t scale_bits = DeviceCount
            ? device_scale_bits
            : metadata_words[tile_offset(logical_row, 1)];
        if (target >= static_cast<int32_t>(vocabulary_start) &&
            target < static_cast<int32_t>(vocabulary_start + 32)) {
            seed_words[tile_offset(
                logical_row,
                static_cast<uint32_t>(target) - vocabulary_start)] =
                scale_bits;
        }
    }
    seed.push_back(1);
}

template <bool DeviceCount>
FORCE_INLINE void produce_row_seed_tile(
    const volatile tt_l1_ptr uint32_t* metadata_words,
    uint32_t device_row_scale_bits,
    DataflowBuffer row_seed) {
    row_seed.reserve_back(1);
    auto* destination = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        row_seed.get_write_ptr());
    for (uint32_t logical_row = 0; logical_row < 32; ++logical_row) {
        const int32_t target = static_cast<int32_t>(
            metadata_words[tile_offset(logical_row, 0)]);
        const uint32_t bits = DeviceCount
            ? (target >= 0 ? device_row_scale_bits : 0u)
            : metadata_words[tile_offset(logical_row, 2)];
        for (uint32_t logical_column = 0; logical_column < 32;
             ++logical_column) {
            destination[tile_offset(logical_row, logical_column)] = bits;
        }
    }
    row_seed.push_back(1);
}

FORCE_INLINE void produce_negative_target_tile(
    const volatile tt_l1_ptr uint32_t* metadata_words,
    uint32_t vocabulary_tile,
    DataflowBuffer target) {
    target.reserve_back(1);
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        target.get_write_ptr());
    for (uint32_t element = 0; element < 1024; ++element)
        words[element] = 0;
    const uint32_t vocabulary_start = vocabulary_tile * 32;
    for (uint32_t logical_row = 0; logical_row < 32; ++logical_row) {
        const int32_t label = static_cast<int32_t>(
            metadata_words[tile_offset(logical_row, 0)]);
        if (label >= static_cast<int32_t>(vocabulary_start) &&
            label < static_cast<int32_t>(vocabulary_start + 32)) {
            words[tile_offset(
                logical_row,
                static_cast<uint32_t>(label) - vocabulary_start)] =
                0xbf800000u;
        }
    }
    target.push_back(1);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t wt = @VOCAB_TILES@;
    constexpr uint32_t vocabulary_size = @VOCAB_SIZE@;
    constexpr uint32_t row_tiles = @ROW_TILES@;
    constexpr bool device_count = @DEVICE_COUNT@;
    const uint32_t row_tile = get_absolute_logical_y();

    auto* max_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(1));
    auto* max_permission = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(9));
    auto* log_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(7));
    auto* log_permission = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(8));
    noc_semaphore_set(max_ready, 0);
    noc_semaphore_set(max_permission, 0);
    noc_semaphore_set(log_ready, 0);
    noc_semaphore_set(log_permission, 0);

    noc_semaphore_wait(max_permission, 1);
    noc_semaphore_set(max_permission, 0);
    DataflowBuffer maximum(27);
    maximum.reserve_back(1);
    auto* max_mailbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(0));
    noc_semaphore_set(max_mailbox, maximum.get_write_ptr());
    noc_async_write(
        get_semaphore(0),
        get_noc_addr(
            kMaxPhysicalX, physical_y(row_tile), get_semaphore(0)),
        sizeof(uint32_t));
    noc_async_write_barrier();
    noc_semaphore_set(max_mailbox, 0);
    noc_semaphore_wait(max_ready, 1);
    noc_semaphore_set(max_ready, 0);
    maximum.push_back(1);

    const auto logits = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                0, 32>(),
            0>(),
        get_common_arg_val<uint32_t>(0), kBf16TileBytes);
    Noc noc;
    DataflowBuffer input(7);
    const uint32_t first_page = row_tile * wt;

    // StableHLO's first exponential pass, consumed directly by SUM.
    for (uint32_t tile = 0; tile < wt; ++tile) {
        input.reserve_back(1);
        noc.async_read(
            logits, input, kBf16TileBytes,
            {.page_id = first_page + tile}, {.offset_bytes = 0});
        noc.async_read_barrier();
        input.push_back(1);
    }

    // Return log(sum(exp)) into c25 without publishing it through PJRT.
    noc_semaphore_wait(log_permission, 1);
    noc_semaphore_set(log_permission, 0);
    DataflowBuffer logsum(25);
    logsum.reserve_back(1);
    auto* log_mailbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(6));
    noc_semaphore_set(log_mailbox, logsum.get_write_ptr());
    noc_async_write(
        get_semaphore(6),
        get_noc_addr(
            kLogPhysicalX, physical_y(row_tile), get_semaphore(6)),
        sizeof(uint32_t));
    noc_async_write_barrier();
    noc_semaphore_set(log_mailbox, 0);
    noc_semaphore_wait(log_ready, 1);
    noc_semaphore_set(log_ready, 0);
    logsum.push_back(1);

    // Metadata page layout is [target_index, full_seed_bits,
    // tf32_row_seed_bits, ...] for every logical row. It is only 4 KiB per
    // 32 tokens and replaces a
    // vocabulary-sized one-hot tensor.
    const auto metadata_accessor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                1, 32>(),
            1>(),
        get_common_arg_val<uint32_t>(1), kWordTileBytes);

    uint32_t valid_count = 0;
    if constexpr (device_count) {
        // Labels are tiny compared with logits and vocabulary weights.  Each
        // row owner scans the compact label pages to derive the same global
        // count without host scales or a new cross-core synchronization
        // boundary.  At the supported maximum this is only 40 KiB per owner.
        DataflowBuffer count_metadata(19);
        for (uint32_t page = 0; page < row_tiles; ++page) {
            count_metadata.reserve_back(1);
            noc.async_read(
                metadata_accessor, count_metadata, kWordTileBytes,
                {.page_id = page}, {.offset_bytes = 0});
            noc.async_read_barrier();
            count_metadata.push_back(1);
            const auto* count_words =
                reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(
                    count_metadata.get_read_ptr());
            for (uint32_t logical_row = 0; logical_row < 32; ++logical_row) {
                const int32_t target = static_cast<int32_t>(
                    count_words[tile_offset(logical_row, 0)]);
                valid_count += target >= 0 &&
                    target < static_cast<int32_t>(vocabulary_size);
            }
            count_metadata.pop_front(1);
        }
    }

    DataflowBuffer metadata(31);
    metadata.reserve_back(1);
    noc.async_read(
        metadata_accessor, metadata, kWordTileBytes,
        {.page_id = row_tile}, {.offset_bytes = 0});
    noc.async_read_barrier();
    metadata.push_back(1);
    const auto* metadata_words =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            metadata.get_read_ptr());

    uint32_t device_seed_bits = 0;
    uint32_t device_row_seed_bits = 0;
    if constexpr (device_count) {
        const float denominator = static_cast<float>(
            valid_count == 0 ? 1 : valid_count);
        const float scale = -1.0f / denominator;
        __builtin_memcpy(&device_seed_bits, &scale, sizeof(scale));
        device_row_seed_bits = device_seed_bits & 0xffffe000u;
    }

    // StableHLO's elementwise divide keeps the full FP32 seed, whereas its
    // following fast row reduction publishes TF32-truncated bits.  The
    // device-count owner derives both from the global compact-label scan.
    produce_row_seed_tile<device_count>(
        metadata_words, device_row_seed_bits, DataflowBuffer(18));
    DataflowBuffer seed(28);
    DataflowBuffer negative_target(17);
    for (uint32_t tile = 0; tile < wt; ++tile) {
        produce_seed_tile<device_count>(
            metadata_words, tile, device_seed_bits, seed);
        produce_negative_target_tile(
            metadata_words, tile, negative_target);

        // Keep the input CB shallow: each replay tile follows its seed tiles
        // and is consumed before the next iteration.
        input.reserve_back(1);
        noc.async_read(
            logits, input, kBf16TileBytes,
            {.page_id = first_page + tile}, {.offset_bytes = 0});
        noc.async_read_barrier();
        input.push_back(1);
    }
    metadata.pop_front(1);
}
