#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {

constexpr uint32_t kJobsPerNode = @JOBS_PER_NODE@;
constexpr uint32_t kTotalJobs = @TOTAL_JOBS@;
constexpr uint32_t kBlockCount = @BLOCK_COUNT@;
constexpr uint32_t kExpertCount = @EXPERT_COUNT@;
constexpr uint32_t kKTiles = @N_TILES@;
constexpr uint32_t kNTiles = @K_TILES@;
constexpr uint32_t kReductionTiles = @REDUCTION_TILES_PER_PHASE@;
constexpr uint32_t kReductionBlocks = kKTiles / kReductionTiles;
constexpr uint32_t kGridCols = @GRID_COLS@;
constexpr uint32_t kBf16TileBytes = 2048;
constexpr uint32_t kInt32TileBytes = 4096;

inline uint32_t faced_row_zero_offset(uint32_t column) {
    constexpr uint32_t face_elements = 256;
    return (column / 16) * face_elements + column % 16;
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(1);
    constexpr uint32_t cb_experts = get_compile_time_arg_val(4);
    CircularBuffer x(cb_x);
    CircularBuffer weight(cb_weight);
    CircularBuffer experts(cb_experts);
    Noc noc0(0);

    const uint32_t input_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t weight_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t expert_addr = get_common_arg_val<uint32_t>(0);
    const auto input_accessor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 5>(),
            2>(),
        input_addr,
        kBf16TileBytes);
    const auto weight_accessor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 5>(),
            1>(),
        weight_addr,
        kBf16TileBytes);
    const auto expert_accessor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 5>(),
            0>(),
        expert_addr,
        kInt32TileBytes);

    constexpr uint32_t expert_tiles = (kBlockCount + 31) / 32;
    experts.reserve_back(expert_tiles);
    const uint32_t expert_l1 = experts.get_write_ptr();
    for (uint32_t tile = 0; tile < expert_tiles; ++tile) {
        noc0.async_read(
            expert_accessor,
            CoreLocalMem<uint32_t>(expert_l1 + tile * kInt32TileBytes),
            expert_accessor.get_aligned_page_size(),
            {.page_id = tile},
            {});
    }
    noc0.async_read_barrier();
    const auto* expert_words =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(expert_l1);

    const uint32_t node =
        get_absolute_logical_y() * kGridCols + get_absolute_logical_x();
    for (uint32_t local = 0; local < kJobsPerNode; ++local) {
        const uint32_t job = node * kJobsPerNode + local;
        if (job >= kTotalJobs) {
            continue;
        }
        const uint32_t block = job / kNTiles;
        const uint32_t n_tile = job % kNTiles;
        const uint32_t id_tile = block / 32;
        const uint32_t id_column = block % 32;
        const uint32_t expert = expert_words[
            id_tile * 1024 + faced_row_zero_offset(id_column)];
        if (expert >= kExpertCount) {
            continue;
        }
        for (uint32_t phase = 0; phase < kReductionBlocks; ++phase) {
            x.reserve_back(kReductionTiles);
            weight.reserve_back(kReductionTiles);
            const uint32_t x_l1 = x.get_write_ptr();
            const uint32_t weight_l1 = weight.get_write_ptr();
            for (uint32_t inner = 0; inner < kReductionTiles; ++inner) {
                const uint32_t k = phase * kReductionTiles + inner;
                noc0.async_read(
                    input_accessor,
                    CoreLocalMem<uint32_t>(
                        x_l1 + inner * kBf16TileBytes),
                    input_accessor.get_aligned_page_size(),
                    {.page_id = block * kKTiles + k},
                    {});
                noc0.async_read(
                    weight_accessor,
                    CoreLocalMem<uint32_t>(
                        weight_l1 + inner * kBf16TileBytes),
                    weight_accessor.get_aligned_page_size(),
                    {.page_id =
                         (expert * kKTiles + k) * kNTiles + n_tile},
                    {});
            }
            noc0.async_read_barrier();
            weight.push_back(kReductionTiles);
            x.push_back(kReductionTiles);
        }
    }
}
