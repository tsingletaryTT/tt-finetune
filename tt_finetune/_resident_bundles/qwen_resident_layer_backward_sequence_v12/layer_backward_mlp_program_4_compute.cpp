#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"
#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/fill.h"

namespace resident_layer_phase_mlp_4 {
// Exact BF16 StableHLO SwiGLU arithmetic inside one TT-Metal program.
//
// Stage 0: gate * up (diagnostic)
// Stage 1: sigmoid(gate) (diagnostic)
// Stage 2: gate * sigmoid(gate) (diagnostic)
// Stage 3: gate * sigmoid(gate) * up (production prototype)


namespace {

constexpr uint32_t full_product_cb_gate = get_compile_time_arg_val(47);
constexpr uint32_t full_product_cb_gate_saved = get_compile_time_arg_val(48);
constexpr uint32_t full_product_cb_up = get_compile_time_arg_val(49);
constexpr uint32_t full_product_cb_sigmoid = get_compile_time_arg_val(50);
constexpr uint32_t full_product_cb_silu = get_compile_time_arg_val(51);
constexpr uint32_t full_product_cb_output = get_compile_time_arg_val(52);

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

FORCE_INLINE void full_activation_product_compute_phase() {
    constexpr uint32_t total_jobs = 608;
    constexpr uint32_t jobs_per_node = 61;
    constexpr uint32_t grid_cols = 1;
    const uint32_t node = get_absolute_logical_y();
    const uint32_t first_job = node * jobs_per_node;
    const uint32_t remaining =
        first_job >= total_jobs ? 0 : total_jobs - first_job;
    const uint32_t local_jobs =
        remaining < jobs_per_node ? remaining : jobs_per_node;
    for (uint32_t job = 0; job < local_jobs; ++job) {
        run_sigmoid(full_product_cb_gate, full_product_cb_sigmoid, 1);
        run_sfpu_multiply(full_product_cb_gate_saved, full_product_cb_sigmoid, full_product_cb_silu, 1);
        run_sfpu_multiply(full_product_cb_silu, full_product_cb_up, full_product_cb_output, 1);
    }
}

// Exact BF16 activation backward for product = silu(gate) * up.
//
// dProduct, gate, and sigmoid remain at the compute RISC's CB front across
// all dependent operations. Every StableHLO-visible arithmetic edge still
// packs to BF16, but no input or sigmoid tile is copied merely to gain a
// second consumer.



namespace {

constexpr uint32_t cb_dproduct = get_compile_time_arg_val(25);
constexpr uint32_t cb_gate = get_compile_time_arg_val(26);
constexpr uint32_t cb_up = get_compile_time_arg_val(27);
constexpr uint32_t cb_pregradient = get_compile_time_arg_val(28);
constexpr uint32_t cb_sigmoid = get_compile_time_arg_val(29);
constexpr uint32_t cb_one_minus_sigmoid = get_compile_time_arg_val(30);
constexpr uint32_t cb_gate_times_one_minus = get_compile_time_arg_val(31);
constexpr uint32_t cb_derivative_inner = get_compile_time_arg_val(32);
constexpr uint32_t cb_derivative = get_compile_time_arg_val(33);
constexpr uint32_t cb_silu = get_compile_time_arg_val(34);
constexpr uint32_t cb_dgate0 = get_compile_time_arg_val(35);
constexpr uint32_t cb_dgate1 = get_compile_time_arg_val(38);
constexpr uint32_t cb_dup0 = get_compile_time_arg_val(36);
constexpr uint32_t cb_dup1 = get_compile_time_arg_val(39);
constexpr uint32_t cb_one = get_compile_time_arg_val(37);

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


static __attribute__((noinline, noclone)) void
full_scale_resident_gradient_one(
    uint32_t input_id,
    uint32_t input_tile,
    uint32_t output_id,
    uint32_t scale_bits
) {
    CircularBuffer input(input_id);
    CircularBuffer output(output_id);
    // Raw dGate/dUp remain retained for frozen-base dX and dB. This indexed
    // consumer observes readiness but deliberately does not acknowledge it.
    input.wait_front(input_tile + 1);
    output.reserve_back(1);
    init_sfpu(input_id, output_id);
    tile_regs_acquire();
    copy_tile_init(input_id);
    copy_tile(input_id, input_tile, 0);
    binop_with_scalar_tile_init();
    volatile int32_t scalar = static_cast<int32_t>(scale_bits);
    mul_unary_tile(0, scalar);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, output_id, 0);
    tile_regs_release();
    output.push_back(1);
}

}  // namespace

FORCE_INLINE void full_activation_backward_compute_phase() {
    constexpr uint32_t total_jobs = 608;
    constexpr uint32_t producer_jobs = 7;
    constexpr uint32_t producer_nodes = 100;
    constexpr uint32_t activation_nodes = 10;
    constexpr uint32_t diagnostic_stage = 6;
    const uint32_t node = get_absolute_logical_y();
    uint32_t local_jobs = 0;
    for (uint32_t producer = node; producer < producer_nodes;
         producer += activation_nodes) {
        const uint32_t first = producer * producer_jobs;
        if (first >= total_jobs) continue;
        const uint32_t remaining = total_jobs - first;
        local_jobs += remaining < producer_jobs ? remaining : producer_jobs;
    }

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
        const uint32_t cb_dgate = job < 32 ? cb_dgate0 : cb_dgate1;
        const uint32_t cb_dup = job < 32 ? cb_dup0 : cb_dup1;
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

        {
            DeviceZoneScopedN("swiglu_backward_scaled_gradient_handoff");
            const uint32_t scaled_gate_cb = job < 32 ? get_compile_time_arg_val(53) : get_compile_time_arg_val(55);
            const uint32_t scaled_up_cb = job < 32 ? get_compile_time_arg_val(54) : get_compile_time_arg_val(56);
            const uint32_t retained_tile = job < 32 ? job : job - 32;
            full_scale_resident_gradient_one(
                cb_dgate, retained_tile, scaled_gate_cb, 1073741824u);
            full_scale_resident_gradient_one(
                cb_dup, retained_tile, scaled_up_cb, 1073741824u);
        }

        sigmoid.pop_front(1);
        dproduct.pop_front(1);
        gate.pop_front(1);
    }
}

void run() {
    full_activation_product_compute_phase();
    full_activation_backward_compute_phase();
}
}  // namespace resident_layer_phase_mlp_4

void kernel_main() {
    resident_layer_phase_mlp_4::run();
}
