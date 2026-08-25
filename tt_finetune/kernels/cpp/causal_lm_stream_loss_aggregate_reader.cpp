// Gather one scalar partial per token-row tile and derive the global count.

#include <cstdint>
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {
constexpr uint8_t kLossPhysicalX = 4;
constexpr uint32_t kWordTileBytes = 4096;
constexpr uint32_t kFp32TileBytes = 4096;
constexpr uint32_t kFace = 16;
constexpr uint32_t kFaceElements = kFace * kFace;
FORCE_INLINE uint8_t physical_y(uint32_t logical_y) {
    return static_cast<uint8_t>(logical_y + 2);
}
FORCE_INLINE uint32_t tile_offset(uint32_t row, uint32_t column) {
    const uint32_t face = ((row >> 4) << 1) | (column >> 4);
    return face * kFaceElements + (row & 15u) * kFace + (column & 15u);
}
}

void kernel_main() {
    constexpr uint32_t row_tiles = @ROW_TILES@;
    constexpr uint32_t vocabulary_size = @VOCAB_SIZE@;
    auto* arrivals = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(15));
    noc_semaphore_set(arrivals, 0);

    const auto metadata = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                1, 32>(),
            0>(),
        get_common_arg_val<uint32_t>(0), kWordTileBytes);
    Noc noc;
    uint32_t valid_count = 0;
    DataflowBuffer metadata_tile(19);
    for (uint32_t page = 0; page < row_tiles; ++page) {
        metadata_tile.reserve_back(1);
        noc.async_read(
            metadata, metadata_tile, kWordTileBytes,
            {.page_id = page}, {.offset_bytes = 0});
        noc.async_read_barrier();
        metadata_tile.push_back(1);
        const auto* words =
            reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(
                metadata_tile.get_read_ptr());
        for (uint32_t row = 0; row < 32; ++row) {
            const int32_t target = static_cast<int32_t>(
                words[tile_offset(row, 0)]);
            valid_count += target >= 0 &&
                target < static_cast<int32_t>(vocabulary_size);
        }
        metadata_tile.pop_front(1);
    }
    noc_semaphore_set(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(14)),
        valid_count == 0 ? 1 : valid_count);
    const float count_value = static_cast<float>(
        valid_count == 0 ? 1 : valid_count);
    uint32_t count_bits;
    __builtin_memcpy(&count_bits, &count_value, sizeof(count_bits));
    DataflowBuffer count_tile(18);
    count_tile.reserve_back(1);
    auto* count_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        count_tile.get_write_ptr());
    for (uint32_t element = 0; element < 1024; ++element)
        count_words[element] = count_bits;
    count_tile.push_back(1);

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        9, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL>();
    noc_semaphore_wait(arrivals, row_tiles);

    DataflowBuffer local_numerator(15), partials(16);
    const uint32_t remote_address = local_numerator.get_read_ptr();
    for (uint32_t row = 0; row < row_tiles; ++row) {
        partials.reserve_back(1);
        noc_async_read(
            get_noc_addr(
                kLossPhysicalX, physical_y(row), remote_address),
            partials.get_write_ptr(), kFp32TileBytes);
        noc_async_read_barrier();
        partials.push_back(1);
        noc_semaphore_inc(
            get_noc_addr(
                kLossPhysicalX, physical_y(row), get_semaphore(14)), 1);
        noc_async_atomic_barrier();
    }
}
