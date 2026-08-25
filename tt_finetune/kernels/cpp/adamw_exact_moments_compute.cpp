// Exact BF16 capturable-AdamW moments with L1-resident numerical edges.

#define FP32_DEST_ACC_EN 1

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/eltwise_unary/lerp.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace {
constexpr uint32_t cb_param = 0, cb_grad = 1, cb_m = 2, cb_v = 3;
constexpr uint32_t cb_param_out = 4, cb_m_out = 5, cb_v_out = 6,
                   cb_sqrt_out = 7;
constexpr uint32_t cb_m_delta = 8, cb_m_scaled = 9, cb_g_square = 10,
                   cb_g_scaled = 11, cb_v_scaled = 12, cb_v_next = 13;
constexpr uint32_t cb_one_minus_beta1 = 14, cb_beta2 = 15,
                   cb_one_minus_beta2 = 16;

ALWI float bits_to_float(uint32_t bits) {
    float value;
    __builtin_memcpy(&value, &bits, sizeof(value));
    return value;
}

ALWI void make_bf16_constant(uint32_t output_id, uint32_t bits) {
    CircularBuffer output(output_id);
    output.reserve_back(1);
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, bits_to_float(bits));
    tile_regs_commit();
    tile_regs_wait();
    PACK((pack_reconfig_data_format(output_id)));
    pack_tile<true>(0, output_id, 0);
    tile_regs_release();
    output.push_back(1);
}

template <uint32_t Op>
ALWI void binary_to_cb(uint32_t left_id, uint32_t right_id,
                       uint32_t output_id, uint32_t tiles,
                       bool pop_left, bool pop_right,
                       bool right_broadcast = false) {
    CircularBuffer left(left_id), right(right_id), output(output_id);
    left.wait_front(tiles);
    right.wait_front(right_broadcast ? 1 : tiles);
    output.reserve_back(tiles);
    if constexpr (Op == 0) {
        unary_op_init_common(left_id, output_id);
        mul_binary_tile_init();
    } else if constexpr (Op == 1) {
        sub_tiles_init(left_id, right_id);
        reconfig_data_format(left_id, right_id);
        PACK((pack_reconfig_data_format(output_id)));
    } else {
        add_tiles_init(left_id, right_id);
        reconfig_data_format(left_id, right_id);
        PACK((pack_reconfig_data_format(output_id)));
    }
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        if constexpr (Op == 0) {
            copy_tile_to_dst_init_short_with_dt(right_id, left_id);
            copy_tile(left_id, tile, 0);
            copy_tile_to_dst_init_short_with_dt(left_id, right_id);
            copy_tile(right_id, right_broadcast ? 0 : tile, 1);
            mul_binary_tile(0, 1, 0);
        } else if constexpr (Op == 1) {
            sub_tiles(left_id, right_id, tile,
                      right_broadcast ? 0 : tile, 0);
        } else {
            add_tiles(left_id, right_id, tile,
                      right_broadcast ? 0 : tile, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, tile);
        tile_regs_release();
    }
    output.push_back(tiles);
    if (pop_left) left.pop_front(tiles);
    if (pop_right) right.pop_front(right_broadcast ? 1 : tiles);
}

ALWI void scalar_mul_to_cb(uint32_t input_id, uint32_t output_id,
                           uint32_t scalar_bits, uint32_t tiles,
                           bool pop_input) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(tiles);
    output.reserve_back(tiles);
    init_sfpu(input_id, output_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        copy_tile_init(input_id);
        copy_tile(input_id, tile, 0);
        binop_with_scalar_tile_init();
        volatile int32_t scalar = static_cast<int32_t>(scalar_bits);
        mul_unary_tile(0, scalar);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, tile);
        tile_regs_release();
    }
    output.push_back(tiles);
    if (pop_input) input.pop_front(tiles);
}

ALWI void copy_to_cb(uint32_t input_id, uint32_t output_id,
                     uint32_t tiles, bool pop_input) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(tiles);
    output.reserve_back(tiles);
    init_sfpu(input_id, output_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        copy_tile_init(input_id);
        copy_tile(input_id, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, tile);
        tile_regs_release();
    }
    output.push_back(tiles);
    if (pop_input) input.pop_front(tiles);
}

ALWI void sqrt_to_cb(uint32_t input_id, uint32_t output_id,
                     uint32_t tiles, bool pop_input) {
    CircularBuffer input(input_id), output(output_id);
    input.wait_front(tiles);
    output.reserve_back(tiles);
    init_sfpu(input_id, output_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        copy_tile_init(input_id);
        copy_tile(input_id, tile, 0);
        sqrt_tile_init();
        sqrt_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, tile);
        tile_regs_release();
    }
    output.push_back(tiles);
    if (pop_input) input.pop_front(tiles);
}

ALWI void lerp_scalar_to_cb(uint32_t start_id, uint32_t end_id,
                            uint32_t output_id, uint32_t weight_bits,
                            uint32_t tiles) {
    CircularBuffer start(start_id), end(end_id), output(output_id);
    start.wait_front(tiles);
    end.wait_front(tiles);
    output.reserve_back(tiles);
    unary_op_init_common(start_id, output_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        copy_tile_init(start_id);
        copy_tile(start_id, tile, 0);
        copy_tile_init(end_id);
        copy_tile(end_id, tile, 1);
        fill_tile_init();
        fill_tile(2, bits_to_float(weight_bits));
        lerp_tile_init();
        lerp_tile<DataFormat::Float16_b>(0, 1, 2, 0);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, tile);
        tile_regs_release();
    }
    output.push_back(tiles);
}

ALWI void addcmul_to_cb(uint32_t input_id, uint32_t tensor1_id,
                        uint32_t tensor2_id, uint32_t output_id,
                        uint32_t scalar_bits, uint32_t tiles) {
    CircularBuffer input(input_id), tensor1(tensor1_id), tensor2(tensor2_id),
        output(output_id);
    input.wait_front(tiles);
    tensor1.wait_front(tiles);
    tensor2.wait_front(tiles);
    output.reserve_back(tiles);
    unary_op_init_common(input_id, output_id);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        copy_tile_init(input_id);
        copy_tile(input_id, tile, 0);
        copy_tile_init(tensor1_id);
        copy_tile(tensor1_id, tile, 1);
        copy_tile_init(tensor2_id);
        copy_tile(tensor2_id, tile, 2);
        addcmul_tile_init();
        addcmul_tile<DataFormat::Float16_b>(0, 1, 2, 0, scalar_bits);
        tile_regs_commit();
        tile_regs_wait();
        PACK((pack_reconfig_data_format(output_id)));
        pack_tile<true>(0, output_id, tile);
        tile_regs_release();
    }
    output.push_back(tiles);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t total_blocks = @TOTAL_BLOCKS@;
    constexpr uint32_t blocks_per_node = @BLOCKS_PER_NODE@;
    constexpr uint32_t rows = @ROWS_PER_BLOCK@;
    constexpr uint32_t grid_columns = 8;
    const uint32_t node =
        get_absolute_logical_y() * grid_columns + get_absolute_logical_x();
    CircularBuffer param(cb_param), grad(cb_grad), m(cb_m), v(cb_v);

    // Compile both destination modes so SFPU binary kernels retain the full
    // four-face tile path, then execute these BF16 checkpoints in 16-bit DST.
    disable_fp32_dest_acc();

    make_bf16_constant(cb_one_minus_beta1, @ONE_MINUS_BETA1_BITS@);
    make_bf16_constant(cb_beta2, @BETA2_BITS@);
    make_bf16_constant(cb_one_minus_beta2, @ONE_MINUS_BETA2_BITS@);

    DeviceZoneScopedN("adamw_exact_moments_l1");
    for (uint32_t local = 0; local < blocks_per_node; ++local) {
        if (node * blocks_per_node + local >= total_blocks) continue;
        copy_to_cb(cb_param, cb_param_out, rows, false);

        binary_to_cb<1>(cb_grad, cb_m, cb_m_delta, rows, false, false);
        binary_to_cb<0>(cb_m_delta, cb_one_minus_beta1, cb_m_scaled,
                        rows, true, false, true);
        binary_to_cb<2>(cb_m, cb_m_scaled, cb_m_out,
                        rows, false, true);

        // The captured StableHLO lowers the second moment into explicit BF16
        // multiply/add checkpoints.  The native addcmul SFPU has a different
        // scalar contract and already differs on the first invocation.
        binary_to_cb<0>(cb_grad, cb_grad, cb_g_square,
                        rows, false, false);
        binary_to_cb<0>(cb_g_square, cb_one_minus_beta2, cb_g_scaled,
                        rows, true, false, true);
        binary_to_cb<0>(cb_v, cb_beta2, cb_v_scaled,
                        rows, false, false, true);
        binary_to_cb<2>(cb_v_scaled, cb_g_scaled, cb_v_next,
                        rows, true, true);
        copy_to_cb(cb_v_next, cb_v_out, rows, false);
        sqrt_to_cb(cb_v_next, cb_sqrt_out, rows, true);

        v.pop_front(rows);
        m.pop_front(rows);
        grad.pop_front(rows);
        param.pop_front(rows);
    }
    CircularBuffer(cb_one_minus_beta2).pop_front(1);
    CircularBuffer(cb_beta2).pop_front(1);
    CircularBuffer(cb_one_minus_beta1).pop_front(1);
    enable_fp32_dest_acc();
}
