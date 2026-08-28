#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ckernel.h"
#include "ckernel_defs.h"

namespace {

constexpr uint32_t kJobsPerNode = @JOBS_PER_NODE@;
constexpr uint32_t kTotalJobs = @TOTAL_JOBS@;
constexpr uint32_t kExperts = @EXPERTS@;
constexpr uint32_t kKTiles = @K_TILES@;
constexpr uint32_t kNTiles = @N_TILES@;
constexpr uint32_t kGridCols = @GRID_COLS@;
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kFaceWidth = 16;
constexpr uint32_t kFaceElements = 256;

inline uint32_t faced_offset(uint32_t row, uint32_t col) {
    const uint32_t face = (row / kFaceWidth) * 2 + col / kFaceWidth;
    return face * kFaceElements + (row % kFaceWidth) * kFaceWidth + col % kFaceWidth;
}

inline bool job_is_active(
    const volatile tt_l1_ptr uint16_t* routing, uint32_t job) {
    const uint32_t block = job / kNTiles;
    const uint32_t token_block = block / kExperts;
    const uint32_t expert = block % kExperts;
    return routing[faced_offset(token_block, expert)] != 0;
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(1);
    constexpr uint32_t cb_routing = get_compile_time_arg_val(2);

    CircularBuffer x(cb_x);
    CircularBuffer weight(cb_weight);
    CircularBuffer routing_cb(cb_routing);
    Noc noc0(0);

    const uint32_t input_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t weight_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t routing_addr = get_common_arg_val<uint32_t>(2);
    const auto input_accessor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 4>(),
            1>(),
        input_addr,
        kTileBytes);
    const auto weight_accessor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 4>(),
            0>(),
        weight_addr,
        kTileBytes);
    const auto routing_accessor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 4>(),
            2>(),
        routing_addr,
        kTileBytes);

    routing_cb.reserve_back(1);
    noc0.async_read(
        routing_accessor,
        CoreLocalMem<uint32_t>(routing_cb.get_write_ptr()),
        routing_accessor.get_aligned_page_size(),
        {.page_id = 0},
        {});
    noc0.async_read_barrier();
    const auto* routing =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
            routing_cb.get_write_ptr());

    const uint32_t node =
        get_absolute_logical_y() * kGridCols + get_absolute_logical_x();
    const uint32_t first_job = node * kJobsPerNode;
    const uint32_t last_job =
        first_job + kJobsPerNode < kTotalJobs
            ? first_job + kJobsPerNode
            : kTotalJobs;
    uint32_t active_jobs = 0;
    for (uint32_t job = first_job; job < last_job; ++job) {
        active_jobs += job_is_active(routing, job);
    }
    routing_cb.push_back(1);

    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, active_jobs);
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, active_jobs);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, active_jobs);

    for (uint32_t job = first_job; job < last_job; ++job) {
        if (!job_is_active(routing, job)) {
            continue;
        }
        const uint32_t block = job / kNTiles;
        const uint32_t n_tile = job % kNTiles;
        const uint32_t expert = block % kExperts;

        x.reserve_back(kKTiles);
        weight.reserve_back(kKTiles);
        const uint32_t x_l1 = x.get_write_ptr();
        const uint32_t weight_l1 = weight.get_write_ptr();
        for (uint32_t k = 0; k < kKTiles; ++k) {
            noc0.async_read(
                input_accessor,
                CoreLocalMem<uint32_t>(x_l1 + k * kTileBytes),
                input_accessor.get_aligned_page_size(),
                {.page_id = block * kKTiles + k},
                {});
            noc0.async_read(
                weight_accessor,
                CoreLocalMem<uint32_t>(weight_l1 + k * kTileBytes),
                weight_accessor.get_aligned_page_size(),
                {.page_id =
                     (expert * kKTiles + k) * kNTiles + n_tile},
                {});
        }
        noc0.async_read_barrier();
        weight.push_back(kKTiles);
        x.push_back(kKTiles);
    }
}
