// Exact BF16 StableHLO SwiGLU arithmetic inside one TT-Metal program.
//
// Stage 0: gate * up (diagnostic)
// Stage 1: sigmoid(gate) (diagnostic)
// Stage 2: gate * sigmoid(gate) (diagnostic)
// Stage 3: gate * sigmoid(gate) * up (production prototype)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace {

constexpr uint32_t cb_gate = 0;
constexpr uint32_t cb_gate_saved = 1;
constexpr uint32_t cb_up = 2;
constexpr uint32_t cb_sigmoid = 3;
constexpr uint32_t cb_silu = 4;
constexpr uint32_t cb_output = 5;

ALWI void run_sigmoid(uint32_t input_id, uint32_t output_id, uint32_t jobs) {
    CircularBuffer input(input_id);
    CircularBuffer output(output_id);
    init_sfpu(input_id, output_id);
    for (uint32_t job = 0; job < jobs; ++job) {
        input.wait_front(1);
        output.reserve_back(1);
        tile_regs_acquire();
        copy_tile_init(input_id);
        copy_tile(input_id, 0, 0);
        sigmoid_tile_init();
        sigmoid_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, output_id, 0);
        tile_regs_release();
        output.push_back(1);
        input.pop_front(1);
    }
}

ALWI void run_silu(uint32_t input_id, uint32_t output_id, uint32_t jobs) {
    CircularBuffer input(input_id);
    CircularBuffer output(output_id);
    init_sfpu(input_id, output_id);
    for (uint32_t job = 0; job < jobs; ++job) {
        input.wait_front(1);
        output.reserve_back(1);
        tile_regs_acquire();
        copy_tile_init(input_id);
        copy_tile(input_id, 0, 0);
        silu_tile_init();
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, output_id, 0);
        tile_regs_release();
        output.push_back(1);
        input.pop_front(1);
    }
}

ALWI void run_sfpu_multiply(
    uint32_t lhs_id,
    uint32_t rhs_id,
    uint32_t output_id,
    uint32_t jobs
) {
    CircularBuffer lhs(lhs_id);
    CircularBuffer rhs(rhs_id);
    CircularBuffer output(output_id);

    // This is the same BF16 path selected by BinaryNgDeviceOperation when
    // fast_and_approximate_mode=false. In particular, SFPU conversion uses
    // round-to-nearest-even for exact halfway products; the FPU mul_tiles
    // convenience path does not reproduce that contract.
    unary_op_init_common(lhs_id, output_id);
    mul_binary_tile_init();
    for (uint32_t job = 0; job < jobs; ++job) {
        lhs.wait_front(1);
        rhs.wait_front(1);
        output.reserve_back(1);
        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(rhs_id, lhs_id);
        copy_tile(lhs_id, 0, 0);
        copy_tile_to_dst_init_short_with_dt(lhs_id, rhs_id);
        copy_tile(rhs_id, 0, 1);
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, output_id, 0);
        tile_regs_release();
        output.push_back(1);
        lhs.pop_front(1);
        rhs.pop_front(1);
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t total_jobs = @TOTAL_JOBS@;
    constexpr uint32_t jobs_per_node = @JOBS_PER_NODE@;
    constexpr uint32_t grid_cols = @GRID_COLS@;
    constexpr uint32_t stage = @STAGE_ID@;
    constexpr bool starts_fp32 = @STARTS_FP32@;
    const uint32_t node =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();
    const uint32_t first_job = node * jobs_per_node;
    const uint32_t remaining =
        first_job >= total_jobs ? 0 : total_jobs - first_job;
    const uint32_t local_jobs =
        remaining < jobs_per_node ? remaining : jobs_per_node;

    if constexpr (starts_fp32) {
        disable_fp32_dest_acc();
    }

    if constexpr (stage == 0) {
        DeviceZoneScopedN("swiglu_exact_multiply");
        run_sfpu_multiply(cb_gate, cb_up, cb_output, local_jobs);
        if constexpr (starts_fp32) {
            enable_fp32_dest_acc();
        }
        return;
    }
    if constexpr (stage == 4) {
        DeviceZoneScopedN("swiglu_builtin_silu");
        run_silu(cb_gate, cb_output, local_jobs);
        if constexpr (starts_fp32) {
            enable_fp32_dest_acc();
        }
        return;
    }
    if constexpr (stage == 5) {
        {
            DeviceZoneScopedN("swiglu_builtin_silu");
            run_silu(cb_gate, cb_silu, local_jobs);
        }
        {
            DeviceZoneScopedN("swiglu_builtin_up_multiply");
            run_sfpu_multiply(cb_silu, cb_up, cb_output, local_jobs);
        }
        if constexpr (starts_fp32) {
            enable_fp32_dest_acc();
        }
        return;
    }

    {
        DeviceZoneScopedN("swiglu_exact_sigmoid");
        run_sigmoid(
            cb_gate,
            stage == 1 ? cb_output : cb_sigmoid,
            local_jobs);
    }
    if constexpr (stage >= 2) {
        DeviceZoneScopedN("swiglu_exact_gate_multiply");
        run_sfpu_multiply(
            cb_gate_saved,
            cb_sigmoid,
            stage == 2 ? cb_output : cb_silu,
            local_jobs);
    }
    if constexpr (stage == 3) {
        DeviceZoneScopedN("swiglu_exact_up_multiply");
        run_sfpu_multiply(cb_silu, cb_up, cb_output, local_jobs);
    }
    if constexpr (starts_fp32) {
        enable_fp32_dest_acc();
    }
}
