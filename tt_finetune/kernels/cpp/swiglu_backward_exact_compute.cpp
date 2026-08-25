// Exact BF16 activation backward for product = silu(gate) * up.
//
// dProduct, gate, and sigmoid remain at the compute RISC's CB front across
// all dependent operations. Every StableHLO-visible arithmetic edge still
// packs to BF16, but no input or sigmoid tile is copied merely to gain a
// second consumer.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace {

constexpr uint32_t cb_dproduct = 0;
constexpr uint32_t cb_gate = 1;
constexpr uint32_t cb_up = 2;
constexpr uint32_t cb_pregradient = 3;
constexpr uint32_t cb_sigmoid = 4;
constexpr uint32_t cb_one_minus_sigmoid = 5;
constexpr uint32_t cb_gate_times_one_minus = 6;
constexpr uint32_t cb_derivative_inner = 7;
constexpr uint32_t cb_derivative = 8;
constexpr uint32_t cb_silu = 9;
constexpr uint32_t cb_dgate = 10;
constexpr uint32_t cb_dup = 11;
constexpr uint32_t cb_one = 12;

static __attribute__((noinline, noclone)) void exact_multiply_one(
    uint32_t lhs_id,
    uint32_t rhs_id,
    uint32_t output_id,
    bool lhs_already_ready,
    bool rhs_already_ready
) {
    CircularBuffer lhs(lhs_id);
    CircularBuffer rhs(rhs_id);
    CircularBuffer output(output_id);
    if (!lhs_already_ready) {
        lhs.wait_front(1);
    }
    if (!rhs_already_ready) {
        rhs.wait_front(1);
    }
    output.reserve_back(1);
    unary_op_init_common(lhs_id, output_id);
    mul_binary_tile_init();
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
}

static __attribute__((noinline, noclone)) void sigmoid_one_ready_input(
    uint32_t input_id,
    uint32_t output_id
) {
    CircularBuffer output(output_id);
    output.reserve_back(1);
    init_sfpu(input_id, output_id);
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
}

static __attribute__((noinline, noclone)) void fill_one_tile(
    uint32_t output_id
) {
    CircularBuffer output(output_id);
    output.reserve_back(1);
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 1.0f);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, output_id, 0);
    tile_regs_release();
    output.push_back(1);
}

static __attribute__((noinline, noclone)) void subtract_one_minus(
    uint32_t one_id,
    uint32_t sigmoid_id,
    uint32_t output_id
) {
    CircularBuffer one(one_id);
    CircularBuffer output(output_id);
    one.wait_front(1);
    output.reserve_back(1);
    binary_op_init_common(one_id, sigmoid_id, output_id);
    tile_regs_acquire();
    sub_tiles_init(one_id, sigmoid_id);
    sub_tiles(one_id, sigmoid_id, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, output_id, 0);
    tile_regs_release();
    output.push_back(1);
    one.pop_front(1);
}

static __attribute__((noinline, noclone)) void add_one(
    uint32_t input_id,
    uint32_t one_id,
    uint32_t output_id
) {
    CircularBuffer input(input_id);
    CircularBuffer one(one_id);
    CircularBuffer output(output_id);
    input.wait_front(1);
    one.wait_front(1);
    output.reserve_back(1);
    binary_op_init_common(input_id, one_id, output_id);
    tile_regs_acquire();
    add_tiles_init(input_id, one_id);
    add_tiles(input_id, one_id, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, output_id, 0);
    tile_regs_release();
    output.push_back(1);
    one.pop_front(1);
}

static __attribute__((noinline, noclone)) void copy_one(
    uint32_t input_id,
    uint32_t output_id,
    bool input_already_ready
) {
    CircularBuffer input(input_id);
    CircularBuffer output(output_id);
    if (!input_already_ready) {
        input.wait_front(1);
    }
    output.reserve_back(1);
    copy_tile_init(input_id);
    tile_regs_acquire();
    copy_tile(input_id, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, output_id, 0);
    tile_regs_release();
    output.push_back(1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t total_jobs = @TOTAL_JOBS@;
    constexpr uint32_t jobs_per_node = @JOBS_PER_NODE@;
    constexpr uint32_t grid_cols = @GRID_COLS@;
    constexpr uint32_t diagnostic_stage = @DIAGNOSTIC_STAGE@;
    const uint32_t node =
        get_absolute_logical_y() * grid_cols + get_absolute_logical_x();
    const uint32_t first_job = node * jobs_per_node;
    const uint32_t remaining =
        first_job >= total_jobs ? 0 : total_jobs - first_job;
    const uint32_t local_jobs =
        remaining < jobs_per_node ? remaining : jobs_per_node;

    CircularBuffer dproduct(cb_dproduct);
    CircularBuffer gate(cb_gate);
    CircularBuffer up(cb_up);
    CircularBuffer pregradient(cb_pregradient);
    CircularBuffer sigmoid(cb_sigmoid);
    CircularBuffer one_minus(cb_one_minus_sigmoid);
    CircularBuffer gate_one_minus(cb_gate_times_one_minus);
    CircularBuffer inner(cb_derivative_inner);
    CircularBuffer derivative(cb_derivative);
    CircularBuffer silu(cb_silu);

    for (uint32_t job = 0; job < local_jobs; ++job) {
        dproduct.wait_front(1);
        gate.wait_front(1);
        up.wait_front(1);

        {
            DeviceZoneScopedN("swiglu_backward_input_multiply");
            exact_multiply_one(
                cb_dproduct, cb_up, cb_pregradient, true, true);
        }
        up.pop_front(1);
        if constexpr (diagnostic_stage == 0) {
            copy_one(cb_pregradient, cb_dgate, false);
        }

        {
            DeviceZoneScopedN("swiglu_backward_sigmoid");
            sigmoid_one_ready_input(cb_gate, cb_sigmoid);
        }
        sigmoid.wait_front(1);
        if constexpr (diagnostic_stage == 1) {
            copy_one(cb_sigmoid, cb_dgate, true);
        }

        {
            DeviceZoneScopedN("swiglu_backward_one_minus_sigmoid");
            fill_one_tile(cb_one);
            subtract_one_minus(cb_one, cb_sigmoid, cb_one_minus_sigmoid);
        }
        if constexpr (diagnostic_stage == 2) {
            copy_one(cb_one_minus_sigmoid, cb_dgate, false);
        }
        {
            DeviceZoneScopedN("swiglu_backward_derivative_inner");
            exact_multiply_one(
                cb_gate,
                cb_one_minus_sigmoid,
                cb_gate_times_one_minus,
                true,
                diagnostic_stage == 2);
            one_minus.pop_front(1);
            if constexpr (diagnostic_stage == 3) {
                copy_one(cb_gate_times_one_minus, cb_dgate, false);
            }
            fill_one_tile(cb_one);
            add_one(
                cb_gate_times_one_minus, cb_one, cb_derivative_inner);
            gate_one_minus.pop_front(1);
            if constexpr (diagnostic_stage == 4) {
                copy_one(cb_derivative_inner, cb_dgate, false);
            }
            exact_multiply_one(
                cb_sigmoid,
                cb_derivative_inner,
                cb_derivative,
                true,
                diagnostic_stage == 4);
            inner.pop_front(1);
            if constexpr (diagnostic_stage == 5) {
                copy_one(cb_derivative, cb_dgate, false);
            }
        }
        {
            DeviceZoneScopedN("swiglu_backward_gate_gradient");
            if constexpr (diagnostic_stage == 6) {
                exact_multiply_one(
                    cb_pregradient,
                    cb_derivative,
                    cb_dgate,
                    false,
                    false);
            }
            pregradient.pop_front(1);
            derivative.pop_front(1);
        }
        {
            DeviceZoneScopedN("swiglu_backward_up_gradient");
            exact_multiply_one(
                cb_gate, cb_sigmoid, cb_silu, true, true);
            exact_multiply_one(
                cb_dproduct, cb_silu, cb_dup, true, false);
            silu.pop_front(1);
        }

        sigmoid.pop_front(1);
        dproduct.pop_front(1);
        gate.pop_front(1);
    }
}
