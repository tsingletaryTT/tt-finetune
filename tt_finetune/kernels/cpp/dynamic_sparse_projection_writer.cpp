#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {

constexpr uint32_t kJobsPerNode = @JOBS_PER_NODE@;
constexpr uint32_t kTotalJobs = @TOTAL_JOBS@;
constexpr uint32_t kExperts = @EXPERTS@;
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
    constexpr uint32_t cb_routing = get_compile_time_arg_val(2);
    constexpr uint32_t cb_result = get_compile_time_arg_val(3);
    CircularBuffer routing_cb(cb_routing);
    CircularBuffer result(cb_result);
    Noc noc1(1);

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const auto output_accessor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<3, 4>(),
            0>(),
        output_addr,
        kTileBytes);
    routing_cb.wait_front(1);
    const auto* routing =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
            routing_cb.get_read_ptr());

    const uint32_t node =
        get_absolute_logical_y() * kGridCols + get_absolute_logical_x();
    const uint32_t first_job = node * kJobsPerNode;
    const uint32_t last_job =
        first_job + kJobsPerNode < kTotalJobs
            ? first_job + kJobsPerNode
            : kTotalJobs;
    for (uint32_t job = first_job; job < last_job; ++job) {
        if (!job_is_active(routing, job)) {
            continue;
        }
        result.wait_front(1);
        noc1.async_write(
            CoreLocalMem<uint32_t>(result.get_read_ptr()),
            output_accessor,
            output_accessor.get_aligned_page_size(),
            {},
            {.page_id = job});
        noc1.async_write_barrier();
        result.pop_front(1);
    }
    routing_cb.pop_front(1);
}
