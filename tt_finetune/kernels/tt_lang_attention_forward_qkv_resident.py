"""Exact Q/K/V frozen+LoRA projection into resident causal attention.

This is the first production-compute expansion of the attention-forward
owner.  A 50-core projection island computes Q, K and V with the accepted
Blackhole reduction/rounding schedule.  Final BF16 tiles move directly to a
60-core attention island through L1/NoC; no Q/K/V framework tensor or DRAM
round trip exists between projection and attention.
"""

from __future__ import annotations

import hashlib
import inspect
import re
import struct
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch

from .full_attention_forward_planner import plan_full_attention_forward
from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_attention_forward_resident import (
    _ALL_WORKERS_RANGE,
    _ATTENTION_RANGE,
    _PROJECTION_RANGE,
    _resident_compute_source,
)
from .tt_lang_frozen_linear_lora import (
    BLACKHOLE_P150A_GRID,
    _exact_bf16_bias_then_lora_add_source,
    _exact_biasless_add_source,
)
from .tt_lang_rmsnorm_forward import _ttnn_fused_rmsnorm_compute_source


ttl = None
tt_lang_operation = None
_ATTENTION_CBS = 32
_TOTAL_CBS = 62
_DEDICATED_O_MAIN_RANGE = {"start": [0, 0], "end": [4, 8]}
_DEDICATED_O_RANK_RANGE = {"start": [0, 9], "end": [3, 9]}


_LAYER_GATE_UP_CB_MAP = {
    0: 0,    # normalized row cache
    1: 15,   # LoRA A
    2: 23,   # FP32 H partial
    3: 4,    # H publication
    4: 3,    # unused skeleton slot
    5: 14,   # resident H
    6: 6,    # frozen W
    7: 24,   # FP32 base partial
    8: 16,   # base result
    9: 9,    # LoRA B
    10: 10,  # delta
    11: 13,  # scaled delta
    12: 12,  # final gate/up output
}

_LAYER_PAIRED_GATE_UP_SWIGLU_CB_MAP = {
    0: 0,    # normalized row cache
    1: 15,   # LoRA A
    2: 23,   # FP32 H partial
    3: 4,    # H publication
    5: 14,   # resident gate/up H pair
    6: 6,    # frozen W
    7: 24,   # FP32 base partial
    8: 16,   # paired base results
    9: 9,    # LoRA B
    10: 10,  # delta
    11: 13,  # paired scaled deltas
    12: 12,  # gate/up checkpoints
    13: 3,   # exact resident SwiGLU product
}


def _strip_cpp_includes(source: str) -> str:
    """Return a composable kernel body without duplicate include directives."""

    return "\n".join(
        line for line in source.splitlines()
        if not line.lstrip().startswith("#include")
    ) + "\n"


def _canonical_ttnn_o_rank_compute_source(hidden_tiles: int) -> str:
    """Specialize TTNN's real mcast-1D large-block compute body.

    The canonical training graph selects the generic TTNN
    ``bmm_large_block_zm_fused_bias_activation`` program for the O adapter's
    long-K rank projection.  A mathematically equivalent, shortened compute
    loop does not reproduce that program's Blackhole BF16 contract.  Keep the
    complete upstream body and specialize only the factory-provided compile
    arguments/named CB identities.  This makes the resident producer use the
    same control structure, spill/reload boundaries and pack state as TTNN
    while its inputs and result remain private to the attention owner.
    """

    if hidden_tiles <= 0 or hidden_tiles % 2:
        raise ValueError("canonical O-rank schedule requires even hidden tiles")
    source_path = (
        Path(__file__).parents[2]
        / "vendor"
        / "tt-metal-source"
        / "ttnn"
        / "cpp"
        / "ttnn"
        / "operations"
        / "matmul"
        / "device"
        / "kernels"
        / "compute"
        / "bmm_large_block_zm_fused_bias_activation.cpp"
    )
    source = source_path.read_text(encoding="utf-8")
    positional = (
        2,                  # in0_block_w
        1,                  # in0_num_subblocks
        2,                  # in0_block_num_tiles
        2,                  # in0_subblock_num_tiles
        1,                  # in1_num_subblocks
        2,                  # in1_block_num_tiles
        1,                  # in1 block width
        hidden_tiles // 2,  # K blocks
        1,                  # output blocks in N
        1,                  # output blocks in M
        1, 1, 1,            # output subblock H/W/tiles
        1,                  # batch
        1,                  # output block tiles
        0,                  # tiled output
        0,                  # batch is compile-time known
        0,                  # do not transpose input zero
    )
    for index, value in enumerate(positional):
        marker = f"get_compile_time_arg_val({index})"
        if marker not in source:
            raise RuntimeError(
                f"canonical TTNN matmul compile argument {index} changed"
            )
        source = source.replace(marker, str(value))
    named = {
        "cb_in0": 0,
        "cb_in1": 1,
        "cb_bias": 3,
        "cb_out": 4,
        "cb_intermed0": 5,
        "cb_in0_transposed": 10,
        "bias_ntiles": 1,
    }
    for name, value in named.items():
        marker = f'get_named_compile_time_arg_val("{name}")'
        if marker not in source:
            raise RuntimeError(
                f"canonical TTNN matmul named argument {name!r} changed"
            )
        source = source.replace(marker, str(value))
    # These are the two factory defines selected by transpose_b=True and an
    # FP32 destination accumulator. Embedding them before the untouched
    # upstream body gives the JIT the same preprocessor contract.
    return "#define IN1_TRANSPOSE_TILE 1\n#define FP32_DEST_ACC_EN 1\n" + source


def _layer_gate_up_compute_source() -> str:
    """Adapt the qualified standalone arithmetic to the layer CB epoch."""

    source = (
        Path(__file__).with_name("cpp")
        / "resident_grouped_gate_up_compute.cpp"
    ).read_text(encoding="utf-8")
    source = _strip_cpp_includes(source)
    source = source.replace(
        "void kernel_main()", "static __attribute__((noinline, noclone, "
        "optimize(\"Os,no-unroll-loops,no-peel-loops,no-unswitch-loops\"))) "
        "void run_resident_layer_gate_up_compute()", 1,
    )
    # The standalone schedule's generic placeholders share names with the
    # attention program (notably SCALE_BITS and BASE_PHASE) but have different
    # values.  Give the resident epoch its own namespace before the common
    # replacement pass and fix the qualified all-core topology here.
    source = source.replace("@WORKERS@", "@GATE_UP_WORKERS@")
    source = source.replace("@GRID_COLUMNS@", "@GATE_UP_GRID_COLUMNS@")
    source = source.replace(
        "get_absolute_logical_y() * kColumns + get_absolute_logical_x()",
        "get_absolute_logical_y() * kColumns + "
        "(get_absolute_logical_x() - @GATE_UP_X_OFFSET@)",
        1,
    )
    source = source.replace("@SCALE_BITS@", "@LORA_SCALE_BITS@")
    source = source.replace("@BASE_PHASE@", "@GATE_UP_BASE_PHASE@")
    for old, new in (
        ("resident_add_tiles_bf16", "layer_gate_up_add_tiles_bf16"),
        ("resident_commit_bf16", "layer_gate_up_commit_bf16"),
        ("resident_pack_reconfig_bf16", "layer_gate_up_pack_reconfig_bf16"),
        ("resident_pack_bf16", "layer_gate_up_pack_bf16"),
        ("resident_release_bf16", "layer_gate_up_release_bf16"),
    ):
        source = source.replace(old, new)
    source = re.sub(
        r"get_compile_time_arg_val\((\d+)\)",
        lambda match: str(
            _LAYER_GATE_UP_CB_MAP.get(
                int(match.group(1)), int(match.group(1))
            )
        ),
        source,
    )
    cb_declarations = (
        "    CircularBuffer b(cb_b), delta(cb_delta), scaled(cb_scaled), "
        "output(cb_output);\n"
    )
    if source.count(cb_declarations) != 1:
        raise RuntimeError("gate/up compute CB declaration marker changed")
    rebase = (
        cb_declarations
        + "\n    // Begin a clean compute epoch after attention. Reader-owned "
        "input counters and writer-owned output counters are already reset "
        "before this local release.\n"
        + "    UNPACK({\n"
        + "        constexpr uint32_t input_cbs[] = {0,15,14,6,9};\n"
        + "        for (uint32_t cb_id : input_cbs) {\n"
        + "            auto& interface = get_local_cb_interface(cb_id);\n"
        + "            interface.fifo_rd_ptr = interface.fifo_limit - "
        "interface.fifo_size;\n"
        + "            interface.tiles_acked_received_init = 0;\n"
        + "        }\n"
        + "        constexpr uint32_t private_cbs[] = {2,7,13,10,11};\n"
        + "        for (uint32_t cb_id : private_cbs) {\n"
        + "            *get_cb_tiles_received_ptr(cb_id) = 0;\n"
        + "            *get_cb_tiles_acked_ptr(cb_id) = 0;\n"
        + "            auto& interface = get_local_cb_interface(cb_id);\n"
        + "            interface.fifo_rd_ptr = interface.fifo_limit - "
        "interface.fifo_size;\n"
        + "            interface.tiles_acked_received_init = 0;\n"
        + "        }\n"
        + "    });\n"
        + "    PACK({\n"
        + "        constexpr uint32_t output_cbs[] = {2,4,7,13,10,11,12};\n"
        + "        for (uint32_t cb_id : output_cbs) {\n"
        + "            auto& interface = get_local_cb_interface(cb_id);\n"
        + "            interface.fifo_wr_ptr = interface.fifo_limit - "
        "interface.fifo_size;\n"
        + "            interface.fifo_wr_tile_ptr = 0;\n"
        + "            interface.tiles_acked_received_init = 0;\n"
        + "        }\n"
        + "    });\n"
    )
    source = source.replace(cb_declarations, rebase, 1)
    return source


def _layer_paired_gate_up_swiglu_compute_source() -> str:
    """Adapt the exact pair-owned projection/SwiGLU operation to the layer."""

    source = (
        Path(__file__).with_name("cpp")
        / "resident_paired_gate_up_swiglu_compute.cpp"
    ).read_text(encoding="utf-8")
    source = _strip_cpp_includes(source)
    source = source.replace(
        "void kernel_main()",
        "static __attribute__((noinline, noclone, "
        "optimize(\"Os,no-unroll-loops,no-peel-loops,no-unswitch-loops\"))) "
        "void run_resident_layer_paired_gate_up_swiglu()",
        1,
    )
    source = source.replace("@WORKERS@", "@GATE_UP_WORKERS@")
    source = source.replace("@GRID_COLUMNS@", "@GATE_UP_GRID_COLUMNS@")
    source = source.replace(
        "get_absolute_logical_y() * kColumns + get_absolute_logical_x()",
        "get_absolute_logical_y() * kColumns + "
        "(get_absolute_logical_x() - @GATE_UP_X_OFFSET@)",
        1,
    )
    source = source.replace("@SCALE_BITS@", "@LORA_SCALE_BITS@")
    source = source.replace("@BASE_PHASE@", "@GATE_UP_BASE_PHASE@")
    for old, new in (
        ("add_tiles_bf16", "layer_paired_add_tiles_bf16"),
        ("commit_bf16", "layer_paired_commit_bf16"),
        ("pack_reconfig_bf16", "layer_paired_pack_reconfig_bf16"),
        ("pack_bf16", "layer_paired_pack_bf16"),
        ("release_bf16", "layer_paired_release_bf16"),
    ):
        source = source.replace(old, new)
    source = re.sub(
        r"constexpr uint32_t (cb_[a-z_]+) = (\d+);",
        lambda match: (
            f"constexpr uint32_t {match.group(1)} = "
            f"{_LAYER_PAIRED_GATE_UP_SWIGLU_CB_MAP.get(int(match.group(2)), int(match.group(2)))};"
        ),
        source,
    )
    declarations = (
        "    CircularBuffer b(cb_b), delta(cb_delta), scaled(cb_scaled);\n"
        "    CircularBuffer gate_up(cb_gate_up), product(cb_product);\n"
    )
    if source.count(declarations) != 1:
        raise RuntimeError("paired gate/up compute CB declaration marker changed")
    rebase = declarations + """
    // Attention and post-norm used these physical FIFOs in earlier epochs.
    // The reader/writer device barrier proves those epochs complete; reset the
    // local Tensix views before the pair-owned projection starts.
    UNPACK({
        constexpr uint32_t input_cbs[] = {0,15,14,6,9};
        for (uint32_t cb_id : input_cbs) {
            auto& interface = get_local_cb_interface(cb_id);
            interface.fifo_rd_ptr = interface.fifo_limit - interface.fifo_size;
            interface.tiles_acked_received_init = 0;
        }
        constexpr uint32_t private_cbs[] = {23,24,16,10,13};
        for (uint32_t cb_id : private_cbs) {
            *get_cb_tiles_received_ptr(cb_id) = 0;
            *get_cb_tiles_acked_ptr(cb_id) = 0;
            auto& interface = get_local_cb_interface(cb_id);
            interface.fifo_rd_ptr = interface.fifo_limit - interface.fifo_size;
            interface.tiles_acked_received_init = 0;
        }
    });
    PACK({
        constexpr uint32_t output_cbs[] = {23,4,24,16,10,13,12,3};
        for (uint32_t cb_id : output_cbs) {
            auto& interface = get_local_cb_interface(cb_id);
            interface.fifo_wr_ptr = interface.fifo_limit - interface.fifo_size;
            interface.fifo_wr_tile_ptr = 0;
            interface.tiles_acked_received_init = 0;
        }
    });
"""
    return source.replace(declarations, rebase, 1)


def _layer_swiglu_down_compute_source() -> str:
    """Adapt the exact 110-core MLP tail to the 60-core layer island."""

    source = (
        Path(__file__).with_name("cpp")
        / "resident_swiglu_down_compute.cpp"
    ).read_text(encoding="utf-8")
    source = _strip_cpp_includes(source)
    # This helper is embedded in an attention program whose own
    # ``@SCALE_BITS@`` is 1/sqrt(head_dim).  Rename the down-LoRA scalar before
    # applying the outer replacement table so the two independent numerical
    # contracts cannot alias (0.125 vs 2.0 for the SmolLM2 qualification).
    source = source.replace("@SCALE_BITS@", "@LORA_SCALE_BITS@")
    standalone_bf16_init = """ALWI void bf16_unary_init(uint32_t input_id, uint32_t output_id) {
#ifndef ARCH_QUASAR
    state_configure<Operand::SRCA, Operand::PACK>(
        input_id, output_id, __builtin_LINE());
    UNPACK((llk_unpack_hw_configure<false>(input_id)));
    UNPACK((llk_unpack_A_init<
            BroadcastType::NONE,
            false,
            EltwiseBinaryReuseDestType::NONE,
            UnpackToDestEn>(false, false, input_id)));
    PACK((llk_pack_hw_configure<false>(output_id)));
    PACK((llk_pack_init(output_id)));
    PACK((llk_pack_dest_init<false, PackMode::Default>()));
    MATH((llk_math_eltwise_unary_datacopy_init<
          DataCopyType::A2D,
          false,
          BroadcastType::NONE>(input_id)));
    MATH((llk_math_pack_sync_init<false>()));
    MATH((llk_math_hw_configure<false>(input_id, input_id)));
    MATH((ckernel::math::_configure_unary_preserve_zero_flag_state_()));
#endif
}
"""
    resident_bf16_init = """ALWI void bf16_unary_init(uint32_t input_id, uint32_t output_id) {
#ifndef ARCH_QUASAR
    // Resident continuation: keep the already-balanced BF16 destination
    // epoch and change only the live source/packer formats and operation MOPs.
    UNPACK((llk_unpack_reconfig_data_format_srca<
            false, p_dim_stride_target::IGNORE>(input_id)));
    UNPACK((llk_unpack_A_init<
            BroadcastType::NONE,
            false,
            EltwiseBinaryReuseDestType::NONE,
            UnpackToDestEn>(false, false, input_id)));
    PACK((llk_pack_reconfig_data_format<false>(output_id)));
    MATH((llk_math_eltwise_unary_datacopy_init<
          DataCopyType::A2D,
          false,
          BroadcastType::NONE>(input_id)));
    MATH((llk_math_reconfig_data_format_srca<false>(input_id)));
    MATH((ckernel::math::_configure_unary_preserve_zero_flag_state_()));
#endif
}
"""
    if source.count(standalone_bf16_init) != 1:
        raise RuntimeError("standalone BF16 SwiGLU initializer changed")
    source = source.replace(
        standalone_bf16_init, resident_bf16_init, 1,
    )
    cb_map = {
        0: 17, 1: 18, 2: 1, 3: 3, 4: 5, 5: 7,
        6: 19, 7: 4, 8: 14, 9: 6, 10: 0, 11: 20,
        12: 8, 13: 9, 14: 10, 15: 11, 16: 12, 17: 13,
    }
    source = re.sub(
        r"get_compile_time_arg_val\((\d+)\)",
        lambda match: str(cb_map[int(match.group(1))]),
        source,
    )
    # CB18 is an input window only until the paired gate/up epoch retires. In
    # the down epoch it carries a private copy of every scaled LoRA delta. The
    # writer observes that complete generation and releases compute before the
    # final residual add, preventing CB29's aliased writer view from consuming
    # the preceding O-output generation.
    source = source.replace(
        "    CircularBuffer scaled(cb_scaled);\n"
        "    CircularBuffer output(cb_output);",
        "    CircularBuffer scaled(cb_scaled);\n"
        "    CircularBuffer scaled_checkpoint(18);\n"
        "    CircularBuffer output(cb_output);",
        1,
    )
    source = source.replace(
        "        scaled.reserve_back(1);\n"
        "        init_sfpu(cb_delta, cb_scaled);",
        "        scaled.reserve_back(1);\n"
        "        scaled_checkpoint.reserve_back(1);\n"
        "        init_sfpu(cb_delta, cb_scaled);",
        1,
    )
    source = source.replace(
        "        PACK((pack_reconfig_data_format(cb_scaled)));\n"
        "        pack_tile<true>(0, cb_scaled, 0);\n"
        "        tile_regs_release();\n"
        "        scaled.push_back(1);",
        "        PACK((pack_reconfig_data_format(cb_scaled)));\n"
        "        pack_tile<true>(0, cb_scaled, 0);\n"
        "        PACK((pack_reconfig_data_format(18)));\n"
        "        pack_tile<true>(0, 18, 0);\n"
        "        tile_regs_release();\n"
        "        scaled.push_back(1);\n"
        "        scaled_checkpoint.push_back(1);",
        1,
    )
    paired_marker = "    constexpr bool decomposed_activation = @DECOMPOSED_ACTIVATION@;\n"
    if source.count(paired_marker) != 1:
        raise RuntimeError("SwiGLU/down activation mode marker changed")
    source = source.replace(
        paired_marker,
        paired_marker + "    constexpr bool paired_swiglu = @PAIRED_SWIGLU@;\n",
        1,
    )
    source = source.replace(
        "constexpr uint32_t kCoreCount = 110;",
        "constexpr uint32_t kCoreCount = 60;",
        1,
    ).replace(
        "get_absolute_logical_y() * 11 + get_absolute_logical_x()",
        "get_absolute_logical_y() * 6 + (get_absolute_logical_x() - 5)",
        1,
    ).replace(
        "void kernel_main()",
        "static __attribute__((noinline, noclone, "
        "optimize(\"Os,no-unroll-loops,no-peel-loops,no-unswitch-loops\"))) "
        "void run()",
        1,
    )
    source = re.sub(r"\s*DeviceZoneScopedN\([^;]+;", "", source)
    declarations = "    CircularBuffer product_resident(cb_product_resident);\n"
    if source.count(declarations) != 1:
        raise RuntimeError("SwiGLU/down compute declaration marker changed")
    clean_epoch = (
        "    CircularBuffer writer_release(30);\n"
        "    WAYPOINT(\"DCW0\");\n"
        "    writer_release.wait_front(1);\n"
        "    WAYPOINT(\"DCW1\");\n"
        "    uint32_t down_epoch_token = 0;\n"
        "    UNPACK({\n"
        "        down_epoch_token = 0x444f574eu;\n"
        "        ckernel::mailbox_write(\n"
        "            ckernel::ThreadId::MathThreadId, down_epoch_token);\n"
        "        ckernel::mailbox_write(\n"
        "            ckernel::ThreadId::PackThreadId, down_epoch_token);\n"
        "    });\n"
        "    MATH(down_epoch_token = ckernel::mailbox_read(\n"
        "        ckernel::ThreadId::UnpackThreadId););\n"
        "    PACK(down_epoch_token = ckernel::mailbox_read(\n"
        "        ckernel::ThreadId::UnpackThreadId););\n"
        "    ASSERT(down_epoch_token == 0x444f574eu);\n"
        "    UNPACK({\n"
        "        const uint32_t release_address = "
        "get_local_cb_interface(30).fifo_rd_ptr << 4;\n"
        "        auto* release_payload = "
        "reinterpret_cast<volatile tt_l1_ptr uint32_t*>(release_address);\n"
        "        auto& gate_interface = get_local_cb_interface(17);\n"
        "        gate_interface.fifo_rd_ptr = release_payload[0];\n"
        "        gate_interface.tiles_acked_received_init = 0;\n"
        "        auto& up_interface = get_local_cb_interface(18);\n"
        "        up_interface.fifo_rd_ptr = release_payload[1];\n"
        "        up_interface.tiles_acked_received_init = 0;\n"
        "        constexpr uint32_t input_cbs[] = {5,7,14,0,9,28};\n"
        "        for (uint32_t cb_id : input_cbs) {\n"
        "            auto& interface = get_local_cb_interface(cb_id);\n"
        "            interface.fifo_rd_ptr = interface.fifo_limit - "
        "interface.fifo_size;\n"
        "            interface.tiles_acked_received_init = 0;\n"
        "        }\n"
        "        constexpr uint32_t private_cbs[] = {1,19,20,8,10,11,12,13};\n"
        "        for (uint32_t cb_id : private_cbs) {\n"
        "            *get_cb_tiles_received_ptr(cb_id) = 0;\n"
        "            *get_cb_tiles_acked_ptr(cb_id) = 0;\n"
        "            auto& interface = get_local_cb_interface(cb_id);\n"
        "            interface.fifo_rd_ptr = interface.fifo_limit - "
        "interface.fifo_size;\n"
        "            interface.tiles_acked_received_init = 0;\n"
        "        }\n"
        "    });\n"
        "    writer_release.pop_front(1);\n"
        "    PACK({\n"
        "        constexpr uint32_t output_cbs[] = "
        "{1,19,4,6,20,8,10,11,12,13,29,31};\n"
        "        for (uint32_t cb_id : output_cbs) {\n"
        "            auto& interface = get_local_cb_interface(cb_id);\n"
        "            interface.fifo_wr_ptr = interface.fifo_limit - "
        "interface.fifo_size;\n"
        "            interface.fifo_wr_tile_ptr = 0;\n"
        "            interface.tiles_acked_received_init = 0;\n"
        "        }\n"
        "        if constexpr (!paired_swiglu) {\n"
        "            auto& interface = get_local_cb_interface(3);\n"
        "            interface.fifo_wr_ptr = interface.fifo_limit - "
        "interface.fifo_size;\n"
        "            interface.fifo_wr_tile_ptr = 0;\n"
        "            interface.tiles_acked_received_init = 0;\n"
        "        }\n"
        "    });\n"
    )
    source = source.replace(declarations, clean_epoch + declarations, 1)
    activation_start = (
        "    for (uint32_t local = 0; local < valid_products; ++local) {\n"
    )
    if source.count(activation_start) != 1:
        raise RuntimeError("SwiGLU/down activation loop marker changed")
    source = source.replace(
        activation_start,
        "    if constexpr (!paired_swiglu) {\n"
        + activation_start,
        1,
    )
    activation_end = (
        "    // The explicit BF16 helpers configure the three Tensix threads"
    )
    if source.count(activation_end) != 1:
        raise RuntimeError("SwiGLU/down activation end marker changed")
    source = source.replace(
        activation_end,
        "    }\n" + activation_end,
        1,
    )
    phase_markers = {
        "    for (uint32_t local = 0; local < valid_products; ++local) {\n":
            "    WAYPOINT(\"DCA0\");\n"
            "    for (uint32_t local = 0; local < valid_products; ++local) {\n",
        "    // The explicit BF16 helpers configure the three Tensix threads":
            "    WAYPOINT(\"DCA1\");\n"
            "    // The explicit BF16 helpers configure the three Tensix threads",
        "    if (valid_outputs != 0) {\n"
        "        product_resident.wait_front(valid_outputs * i_tiles);\n"
        "    }\n":
            "    WAYPOINT(\"DCP0\");\n"
            "    if (valid_outputs != 0) {\n"
            "        product_resident.wait_front(valid_outputs * i_tiles);\n"
            "    }\n"
            "    WAYPOINT(\"DCP1\");\n",
        "    // One owner per token row computes H = product @ A.T.":
            "    WAYPOINT(\"DCH0\");\n"
            "    // One owner per token row computes H = product @ A.T.",
        "    if (valid_outputs != 0) {\n"
        "        h_resident.wait_front(valid_outputs * rank_tiles);\n"
        "    }\n":
            "    WAYPOINT(\"DCH1\");\n"
            "    if (valid_outputs != 0) {\n"
            "        h_resident.wait_front(valid_outputs * rank_tiles);\n"
            "    }\n"
            "    WAYPOINT(\"DCB0\");\n",
    }
    for marker, replacement in phase_markers.items():
        if source.count(marker) != 1:
            raise RuntimeError("SwiGLU/down compute phase marker changed")
        source = source.replace(marker, replacement, 1)
    final_marker = (
        "    enable_fp32_dest_acc();\n"
        "    if (valid_outputs != 0) {\n"
        "        h_resident.pop_front(valid_outputs * rank_tiles);"
    )
    if source.count(final_marker) != 1:
        raise RuntimeError("SwiGLU/down final-add marker changed")
    final_residual = (
        "    WAYPOINT(\"DCR0\");\n"
        "    writer_release.wait_front(1);\n"
        "    writer_release.pop_front(1);\n"
        "    CircularBuffer residual(28), layer_output(29);\n"
        "    add_tiles_init(cb_output, 28);\n"
        "    for (uint32_t local = 0; local < valid_outputs; ++local) {\n"
        "        output.wait_front(1);\n"
        "        residual.wait_front(1);\n"
        "        layer_output.reserve_back(1);\n"
        "        tile_regs_acquire();\n"
        "        add_tiles_bf16(cb_output, 28, 0, 0, 0);\n"
        "        tile_regs_commit_bf16();\n"
        "        tile_regs_wait();\n"
        "        pack_reconfig_data_format_bf16(29);\n"
        "        pack_tile_bf16(0, 29, 0);\n"
        "        tile_regs_release_bf16();\n"
        "        layer_output.push_back(1);\n"
        "        output.pop_front(1);\n"
        "        residual.pop_front(1);\n"
        "    }\n"
        "    WAYPOINT(\"DCR1\");\n"
        + final_marker
    )
    source = source.replace(final_marker, final_residual, 1)
    return "namespace resident_layer_swiglu_down_compute {\n" + source + (
        "\n}  // namespace resident_layer_swiglu_down_compute\n"
    )


def _layer_gate_up_reader_source() -> str:
    return _strip_cpp_includes(
        (
            Path(__file__).with_name("cpp")
            / "resident_layer_gate_up_reader_phase.cpp"
        ).read_text(encoding="utf-8")
    )


def _layer_swiglu_down_reader_source() -> str:
    return _strip_cpp_includes(
        (
            Path(__file__).with_name("cpp")
            / "resident_layer_swiglu_down_reader_phase.cpp"
        ).read_text(encoding="utf-8")
    )


def _layer_gate_up_writer_source() -> str:
    return _strip_cpp_includes(
        (
            Path(__file__).with_name("cpp")
            / "resident_layer_gate_up_writer_phase.cpp"
        ).read_text(encoding="utf-8")
    )


def _layer_gate_up_router_source() -> str:
    """Writer subset needed by the projection-side CB22 NoC producer."""

    source = _layer_gate_up_writer_source()
    phase = source.index("template <typename ArenaAccessor>")
    namespace_close = source.rindex("}  // namespace resident_layer_gate_up_writer")
    return source[:phase] + source[namespace_close:]


def _kernel_declaration_start(source: str) -> int:
    """Return the start of kernel_main, including a same-line attribute."""

    kernel = source.index("void kernel_main()")
    line_start = source.rfind("\n", 0, kernel) + 1
    declaration_prefix = source[line_start:kernel]
    return line_start if "__attribute__" in declaration_prefix else kernel


def _append_gate_up_compute_helpers(source: str) -> str:
    """Append the accepted arithmetic body before an owner's kernel_main."""

    kernel = _kernel_declaration_start(source)
    return source[:kernel] + _layer_gate_up_compute_source() + source[kernel:]


def _append_full_mlp_compute_helpers(source: str) -> str:
    kernel = _kernel_declaration_start(source)
    helpers = (
        _layer_paired_gate_up_swiglu_compute_source()
        + _layer_swiglu_down_compute_source()
    )
    return (
        '#include "api/compute/copy_dest_values.h"\n'
        + source[:kernel] + helpers + source[kernel:]
    )


def _make_attention_compute_all_core(source: str) -> str:
    """Keep idle attention-island TRISCs alive for the joined MLP epoch."""

    marker = "    if (node >= jobs) return;"
    kernel = source.index("void kernel_main()")
    prefix, kernel_source = source[:kernel], source[kernel:]
    # The composed source also contains the reusable run_attention_tail helper,
    # which has its own identical active-worker guard.  Only kernel_main must
    # be opened here: active workers still invoke the unchanged tail, while
    # inactive workers bypass the whole attention prefix and reach the common
    # gate/up continuation after the completion fence.
    if kernel_source.count(marker) != 1:
        raise RuntimeError("attention compute active-worker marker changed")
    kernel_source = kernel_source.replace(
        "void kernel_main()",
        "__attribute__((optimize(\"Os\"))) void kernel_main()",
        1,
    )
    kernel_source = kernel_source.replace(marker, "    if (node < jobs) {", 1)
    body, separator, suffix = kernel_source.rpartition("}")
    if not separator:
        raise RuntimeError("attention compute terminator changed")
    return prefix + body + "    }\n}\n" + suffix


def _compose_gate_up_reader(
    source: str, *, attention_island: bool, total_cbs: int,
    full_mlp_forward: bool = False,
) -> str:
    helper = _layer_gate_up_reader_source()
    if full_mlp_forward:
        helper += _layer_swiglu_down_reader_source()
    kernel = _kernel_declaration_start(source)
    source = source[:kernel] + helper + source[kernel:]
    marker = "    noc_semaphore_wait_min(invocation_done, 2);"
    if source.count(marker) != 1:
        raise RuntimeError("attention reader completion marker changed")
    if attention_island:
        source = source.replace(
            "void kernel_main()",
            "__attribute__((optimize(\"Os\"))) void kernel_main()",
            1,
        )
        active = "    if (node >= jobs) return;"
        if source.count(active) != 1:
            raise RuntimeError("attention reader active-worker marker changed")
        done_declaration = (
            "    volatile tt_l1_ptr uint32_t* invocation_done =\n"
            "        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(\n"
            "            get_semaphore(invocation_done_sem));\n"
            "    noc_semaphore_set(invocation_done, 0);\n"
        )
        if source.count(done_declaration) != 1:
            raise RuntimeError("attention reader invocation marker changed")
        early_o_reset = ""
        if full_mlp_forward:
            # The gate/up generation barrier can outlive projection-side O
            # input preparation.  Reset the writer's O-go word before that
            # barrier, otherwise a fast projection reader may publish O-go
            # while this RISC is still in begin_generation() and the original
            # later reset erases the live handoff.
            late_o_reset = "    @RESET_O_GO@\n"
            if source.count(late_o_reset) != 1:
                raise RuntimeError("attention reader O-go reset marker changed")
            source = source.replace(late_o_reset, "", 1)
            early_o_reset = late_o_reset
        source = source.replace(done_declaration, "", 1)
        source = source.replace(
            active,
            done_declaration
            + early_o_reset
            + "    resident_layer_gate_up_reader::begin_generation();\n"
            + "    if (node < jobs) {",
            1,
        )
        mask_accessor = (
            "tensor_accessor::detail::get_tensor_accessor_args_cta_offset<"
            "5, 32>(), 2>()"
        )
        if source.count(mask_accessor) != 1:
            raise RuntimeError("attention reader mask accessor marker changed")
        source = source.replace(
            mask_accessor,
            "tensor_accessor::detail::get_tensor_accessor_args_cta_offset<"
            f"20, {total_cbs}>(), 2>()",
            1,
        )
        mask_page = ".page_id = row * seq_tiles + column"
        if source.count(mask_page) != 1:
            raise RuntimeError("attention reader mask page marker changed")
        source = source.replace(
            mask_page,
            ".page_id = @MASK_PACK_BASE@ + row * seq_tiles + column",
            1,
        )
        call = (
            "    const auto layer_parameters = TensorAccessor(TensorAccessorArgs<\n"
            "        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<"
            f"20, {total_cbs}>(), 2>(),\n"
            "        get_common_arg_val<uint32_t>(2), 2048);\n"
            "    resident_layer_gate_up_reader::run(layer_parameters);\n"
            + (
            "    resident_layer_swiglu_down_reader::run(layer_parameters);\n"
                if full_mlp_forward else ""
            )
        )
    else:
        call = (
            "    resident_layer_gate_up_reader::run(post_norm_weight);\n"
        )
    if attention_island:
        # The continuation performs its receive-window setup before the final
        # attention-reader wait. Its W/B phase contains the narrower state-1
        # fence, avoiding a cycle with projection-side normalized-X routing.
        return source.replace(marker, "    }\n" + call + marker, 1)
    return source.replace(marker, call + marker, 1)


def _compose_gate_up_writer(
    source: str, *, projection_island: bool,
    full_mlp_forward: bool = False,
) -> str:
    helper = (
        _layer_gate_up_router_source()
        if projection_island else _layer_gate_up_writer_source()
    )
    kernel = _kernel_declaration_start(source)
    source = source[:kernel] + helper + source[kernel:]
    if projection_island:
        loop = (
            "                for (uint32_t tile = 0; "
            "tile < hidden_tiles; ++tile) {\n"
            "                    wait_post_normalized(kNormalizedHidden);"
        )
        if source.count(loop) != 1:
            raise RuntimeError("post-normalized routing marker changed")
        routed = (
            "                resident_layer_gate_up_writer::"
            "wait_for_x_receivers(node);\n"
            + loop
            + "\n                    resident_layer_gate_up_writer::"
            "route_normalized_tile(\n"
            "                        post_normalized.get_read_ptr(), node, tile);"
        )
        source = source.replace(loop, routed, 1)
        broadcast = (
            "        for (uint32_t target = 0; "
            "target < active_attention; ++target)"
        )
        if source.count(broadcast) != 1:
            raise RuntimeError("attention completion broadcast marker changed")
        source = source.replace(
            broadcast,
            "        for (uint32_t target = 0; target < 60; ++target)",
            1,
        )
    if not projection_island:
        source = source.replace(
            "void kernel_main()",
            "__attribute__((optimize(\"Os\"))) void kernel_main()",
            1,
        )
        active = "    if (node >= jobs) return;"
        if source.count(active) != 1:
            raise RuntimeError("attention writer active-worker marker changed")
        arena_declaration = (
            "    const auto arena = TensorAccessor(TensorAccessorArgs<\n"
            "        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<6, 32>(), 0>(),\n"
            "        get_common_arg_val<uint32_t>(0), tile_bytes);\n"
            "    Noc noc1(1);\n"
        )
        if source.count(arena_declaration) != 1:
            raise RuntimeError("attention writer arena marker changed")
        source = source.replace(arena_declaration, "", 1)
        source = source.replace(
            active, arena_declaration + "    if (node < jobs) {", 1
        )
        count_marker = "    noc_semaphore_inc(\n        projection_l1(0, get_semaphore(invocation_writer_count_sem)), 1);"
        if source.count(count_marker) != 1:
            raise RuntimeError("attention writer completion-count marker changed")
        counted = count_marker + "\n    noc_async_atomic_barrier();"
        if source.count(counted) != 1:
            raise RuntimeError("attention writer completion barrier changed")
        # Inactive attention workers join the resident gate/up continuation,
        # but they must not inflate the preceding attention writer count.
        source = source.replace(counted, counted + "\n    }", 1)
        release_cbs = (
            "    CircularBuffer invocation_release(0), "
            "invocation_acknowledgement(1);"
        )
        if source.count(release_cbs) != 1:
            raise RuntimeError("attention writer release-CB marker changed")
        # CB0 is the resident normalized-X cache in the continuation.  Using
        # it as the old completion token consumes an X tile (or deadlocks on
        # its reservation).  CB28/29 are empty after attention and mirror the
        # projection-island completion fence.
        source = source.replace(
            release_cbs,
            "    CircularBuffer invocation_release(28), "
            f"invocation_acknowledgement({31 if full_mlp_forward else 29});",
            1,
        )
    release = (
        "    invocation_release.reserve_back(1);\n"
        "    invocation_release.push_back(1);"
    )
    if source.count(release) != 1:
        raise RuntimeError("attention writer release marker changed")
    continuation = (
        "\n    resident_layer_gate_up_writer::run(noc1, arena);"
        "\n    resident_layer_gate_up_writer::run_full_mlp_tail(noc1, arena);"
        if not projection_island else ""
    )
    preparation = (
        "    // CB3/CB4/CB12 and the reused control CB30 were consumed by the "
        "attention writer. Reset their shared counters and BRISC read "
        "pointers before releasing the co-located gate/up compute RISC.\n"
        + (
            "    resident_layer_gate_up_writer::rebase_empty_consumer_cb(3);\n"
            if full_mlp_forward else ""
        )
        +
        "    resident_layer_gate_up_writer::rebase_empty_consumer_cb(4);\n"
        "    resident_layer_gate_up_writer::rebase_empty_consumer_cb(12);\n"
        "    resident_layer_gate_up_writer::rebase_empty_consumer_cb(30);\n"
        + (
            "    // CB29 carries layer data; reserve a clean CB31 generation "
            "for the final compute acknowledgement.\n"
            "    resident_layer_gate_up_writer::rebase_empty_consumer_cb(31);\n"
            if full_mlp_forward else ""
        )
        if not projection_island else ""
    )
    return source.replace(
        release, preparation + release + continuation, 1
    )


def _divisor_worker_count(total_jobs: int, limit: int = 50) -> int:
    """Largest active worker count that leaves no partial tile-job tail."""

    if total_jobs <= 0 or limit <= 0:
        raise ValueError("worker planning requires positive jobs and limit")
    return max(
        workers for workers in range(1, min(limit, total_jobs) + 1)
        if total_jobs % workers == 0
    )


def _compute_completion_fence(
    source: str, *, release_cb: str, acknowledgement_cb: str,
    rebase_projection_cbs: bool = False,
    resident_continuation: str = "",
) -> str:
    """Keep the compute RISC inside the current owner completion epoch."""

    body, separator, suffix = source.rpartition("}")
    if not separator:
        raise RuntimeError("compute source has no kernel_main terminator")
    rebase = (
        "    UNPACK({\n"
        "        constexpr uint32_t read_cbs[] = {0,1,2,3,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27};\n"
        "        for (uint32_t cb_id : read_cbs) {\n"
        "            auto& interface = get_local_cb_interface(cb_id);\n"
        "            interface.fifo_rd_ptr = interface.fifo_limit - interface.fifo_size;\n"
        "            interface.tiles_acked_received_init = 0;\n"
        "        }\n"
        "        constexpr uint32_t private_norm_cbs[] = {20,21,23,24,25,26,27};\n"
        "        for (uint32_t cb_id : private_norm_cbs) {\n"
        "            *get_cb_tiles_received_ptr(cb_id) = 0;\n"
        "            *get_cb_tiles_acked_ptr(cb_id) = 0;\n"
        "        }\n"
        "    });\n"
        "    PACK({\n"
        "        constexpr uint32_t write_cbs[] = {2,3,4,7,8,10,11,12,14,20,21,22,23,24,25,26,27};\n"
        "        for (uint32_t cb_id : write_cbs) {\n"
        "            auto& interface = get_local_cb_interface(cb_id);\n"
        "            interface.fifo_wr_ptr = interface.fifo_limit - interface.fifo_size;\n"
        "            interface.tiles_acked_received_init = 0;\n"
        "            interface.fifo_wr_tile_ptr = 0;\n"
        "        }\n"
        "    });\n"
        if rebase_projection_cbs else ""
    )
    return (
        body
        + f"    CircularBuffer invocation_release({release_cb});\n"
        + f"    CircularBuffer invocation_acknowledgement({acknowledgement_cb});\n"
        + "    invocation_release.wait_front(1);\n"
        + "    invocation_release.pop_front(1);\n"
        + resident_continuation
        + rebase
        + "    invocation_acknowledgement.reserve_back(1);\n"
        + "    invocation_acknowledgement.push_back(1);\n"
        + "}\n"
        + suffix
    )


def _projection_compute_source(
    *, sequence_tiles: int, hidden_tiles: int, q_tiles: int,
    kv_tiles: int, base_phase: int, scale_bits: int, has_bias: bool,
    lora_phase: int = 2,
    o_lora_phase: int | None = None,
    o_lora_init_schedule: int = 0,
    bias_after_bf16_base: bool = False,
    projection_workers: int = 50,
    projection_grid_cols: int = 5,
    include_o_projection: bool = True,
    force_inline_projection_phases: bool = True,
    qkv_phase_count: int = 3,
    direct_single_projection_kernel: bool = False,
    grouped_qkv_rank_tiles: int = 1,
    o_projection_workers: int = 50,
    precomputed_grouped_qkv_h: bool = False,
    precomputed_grouped_qkv_base: bool = False,
    precomputed_grouped_qkv_delta: bool = False,
    o_cb_offset: int = 15,
    qkv_o_phase_barrier: bool = False,
    precomputed_o_rank_h: bool = False,
) -> str:
    """Compose three exact projection epochs under one compute kernel."""

    template = (
        Path(__file__).with_name("cpp")
        / "frozen_linear_lora_forward_exact_compute.cpp"
    ).read_text()
    # Projection and attention occupy disjoint core ranges. Both islands can
    # therefore use hardware-local CB IDs starting at zero even though the
    # artifact contains 47 range-qualified descriptors in total.

    rounded_bias = has_bias and bias_after_bf16_base
    bias_replacements = {
        "@BIAS_DECLARATIONS@": (
            "constexpr uint32_t cb_bias = get_compile_time_arg_val(13);\n"
            "    constexpr uint32_t cb_biased_base = "
            "get_compile_time_arg_val(14);"
            if has_bias else ""
        ),
        "@BIAS_BUFFER_OBJECTS@": (
            "CircularBuffer bias(cb_bias);\n"
            "    CircularBuffer biased_base(cb_biased_base);"
            if has_bias else ""
        ),
        "@BASE_BIAS_WAIT@": "",
        "@BASE_BIAS_ACCUMULATE@": "",
        "@BASE_BIAS_POP@": "",
        "@BASE_FINAL_TARGET@": (
            "cb_base_partial" if has_bias and not rounded_bias
            else "(last ? cb_base : cb_base_partial)"
        ),
        "@BASE_RESERVE@": (
            "base_partial.reserve_back(n_tiles_per_block);"
            if has_bias and not rounded_bias else
            "if (last) { base.reserve_back(n_tiles_per_block); } else { "
            "base_partial.reserve_back(n_tiles_per_block); }"
        ),
        "@BASE_PUSH@": (
            "base_partial.push_back(n_tiles_per_block);"
            if has_bias and not rounded_bias else
            "if (last) { base.push_back(n_tiles_per_block); } else { "
            "base_partial.push_back(n_tiles_per_block); }"
        ),
        "@BASE_BIAS_FINISH@": (
            "if (last) {\n"
            "                base_partial.wait_front(n_tiles_per_block);\n"
            "                bias.wait_front(n_tiles_per_block);\n"
            "                base.reserve_back(n_tiles_per_block);\n"
            "                reconfig_data_format(cb_w, cb_base_partial, "
            "cb_base_x, cb_bias);\n"
            "                add_tiles_init(cb_base_partial, cb_bias);\n"
            "                tile_regs_acquire();\n"
            "                for (uint32_t tile = 0; tile < "
            "n_tiles_per_block; ++tile) {\n"
            "                    add_tiles(cb_base_partial, cb_bias, tile, "
            "tile, tile);\n"
            "                }\n"
            "                tile_regs_commit();\n"
            "                tile_regs_wait();\n"
            "                PACK((pack_reconfig_data_format(cb_base)));\n"
            "                pack_tile_block(0, cb_base, "
            "n_tiles_per_block);\n"
            "                tile_regs_release();\n"
            "                base.push_back(n_tiles_per_block);\n"
            "                bias.pop_front(n_tiles_per_block);\n"
            "                base_partial.pop_front(n_tiles_per_block);\n"
            "            }"
            if has_bias and not rounded_bias else ""
        ),
        "@FINAL_ADD_BODY@": (
            _exact_bf16_bias_then_lora_add_source()
            if rounded_bias else _exact_biasless_add_source()
        ),
    }
    biasless_replacements = {
        "@BIAS_DECLARATIONS@": "",
        "@BIAS_BUFFER_OBJECTS@": "",
        "@BASE_BIAS_WAIT@": "",
        "@BASE_BIAS_ACCUMULATE@": "",
        "@BASE_BIAS_POP@": "",
        "@BASE_FINAL_TARGET@": "(last ? cb_base : cb_base_partial)",
        "@BASE_RESERVE@": (
            "if (last) { base.reserve_back(n_tiles_per_block); } else { "
            "base_partial.reserve_back(n_tiles_per_block); }"
        ),
        "@BASE_PUSH@": (
            "if (last) { base.push_back(n_tiles_per_block); } else { "
            "base_partial.push_back(n_tiles_per_block); }"
        ),
        "@BASE_BIAS_FINISH@": "",
        "@FINAL_ADD_BODY@": _exact_biasless_add_source(),
    }

    shared_prefix = template[: template.index("void kernel_main()")]
    o_lora_phase = lora_phase if o_lora_phase is None else o_lora_phase
    if qkv_phase_count not in (1, 2, 3):
        raise ValueError("QKV phase count must be 1, 2 or 3")
    phase_specs = (("q", q_tiles), ("k", kv_tiles), ("v", kv_tiles))
    if lora_phase <= 0 or hidden_tiles % lora_phase:
        raise ValueError("invalid LoRA projection K phase")
    if o_lora_phase <= 0 or hidden_tiles % o_lora_phase:
        raise ValueError("invalid O LoRA projection K phase")
    functions: list[str] = []
    for name, output_tiles in phase_specs[:qkv_phase_count]:
        total_jobs = sequence_tiles * output_tiles
        replacements = {
            "@NUM_K_BLOCKS@": str(hidden_tiles // lora_phase),
            "@H_K_BLOCK_TILES@": str(lora_phase),
            "@RANK_TILES@": str(grouped_qkv_rank_tiles),
            "@DELTA_RANK_TILES@": str(grouped_qkv_rank_tiles),
            "@DELTA_H_TILE_OFFSET@": "0",
            "@BASE_K_BLOCK_TILES@": str(base_phase),
            "@NUM_BASE_K_BLOCKS@": str(hidden_tiles // base_phase),
            "@BASE_TRANSPOSE_RHS@": "false",
            "@NUM_N_BLOCKS@": str(output_tiles),
            "@N_TILES_PER_BLOCK@": "1",
            "@JOBS_PER_NODE@": str(-(-total_jobs // projection_workers)),
            "@TOTAL_JOBS@": str(total_jobs),
            "@GRID_COLS@": str(projection_grid_cols),
            "@SCALE_BITS@": str(scale_bits),
            "@LORA_INIT_SCHEDULE@": "0",
            **bias_replacements,
        }
        body = template[template.index("void kernel_main()"):]
        phase_declaration = (
            f"FORCE_INLINE void run_{name}_projection()"
            if force_inline_projection_phases
            else (
                "static __attribute__((noinline, noclone)) void "
                f"run_{name}_projection()"
            )
        )
        body = body.replace("void kernel_main()", phase_declaration, 1)
        for marker, value in replacements.items():
            body = body.replace(marker, value)
        if precomputed_grouped_qkv_h:
            # Remove the complete rank-projection branch, including its
            # schedule-4 ``if/else`` wrapper.  Starting at the inner
            # ``Exact H`` comment leaves the outer branch unbalanced and only
            # fails later in the Tensix C++ compiler.
            h_start = body.index(
                "        // Schedule 4 is a literal one-output adaptation"
            )
            base_start = body.index(
                "        // Exact base = X @ W.T for one <=4-tile N block."
            )
            body = body[:h_start] + body[base_start:]
        if precomputed_grouped_qkv_base:
            base_start = body.index(
                "        // Exact base = X @ W.T for one <=4-tile N block."
            )
            delta_start = body.index("        // delta = H @ B.T.")
            body = body[:base_start] + body[delta_start:]
        if precomputed_grouped_qkv_delta:
            delta_start = body.index("        // delta = H @ B.T.")
            scale_start = body.index(
                "        // Preserve BF16(delta * scale)."
            )
            scale_tail = body[scale_start:]
            # Keep the precomputed tile on the reader-produced B channel.
            # CB-delta is normally produced and consumed by the compute RISC;
            # changing its producer to the reader creates an unsupported CB
            # epoch and can leave projection cores waiting forever.
            scale_tail = re.sub(r"\bcb_delta\b", "cb_b", scale_tail)
            scale_tail = re.sub(r"\bdelta\.", "b.", scale_tail)
            body = body[:delta_start] + scale_tail
        if re.search(r"@[A-Z0-9_]+@", body):
            raise RuntimeError(f"unexpanded exact {name} projection placeholder")
        if (
            direct_single_projection_kernel
            and qkv_phase_count == 1
            and not include_o_projection
        ):
            return shared_prefix + body.replace(
                phase_declaration, "void kernel_main()", 1
            )
        functions.append(body)
    if not include_o_projection:
        calls = "".join(
            f"    run_{name}_projection();\n"
            for name, _ in phase_specs[:qkv_phase_count]
        )
        return (
            shared_prefix
            + "\n".join(functions)
            + "\nvoid kernel_main() {\n"
            + calls
            + "}\n"
        )
    o_template = re.sub(
        r"get_compile_time_arg_val\((\d+)\)",
        lambda match: (
            f"get_compile_time_arg_val({o_cb_offset + int(match.group(1))})"
        ),
        template,
    )
    o_total_jobs = sequence_tiles * q_tiles
    o_replacements = {
        "@NUM_K_BLOCKS@": str(hidden_tiles // o_lora_phase),
        "@H_K_BLOCK_TILES@": str(o_lora_phase),
        "@RANK_TILES@": str(grouped_qkv_rank_tiles),
        "@DELTA_RANK_TILES@": str(grouped_qkv_rank_tiles),
        "@DELTA_H_TILE_OFFSET@": "0",
        "@BASE_K_BLOCK_TILES@": str(base_phase),
        "@NUM_BASE_K_BLOCKS@": str(hidden_tiles // base_phase),
        "@BASE_TRANSPOSE_RHS@": "false",
        "@NUM_N_BLOCKS@": str(q_tiles),
        "@N_TILES_PER_BLOCK@": "1",
        "@JOBS_PER_NODE@": str(-(-o_total_jobs // o_projection_workers)),
        "@TOTAL_JOBS@": str(o_total_jobs),
        "@GRID_COLS@": str(projection_grid_cols),
        "@SCALE_BITS@": str(scale_bits),
        "@LORA_INIT_SCHEDULE@": str(o_lora_init_schedule),
        **biasless_replacements,
    }
    o_declaration = (
        "FORCE_INLINE void run_o_projection()"
        if force_inline_projection_phases
        else (
            "static __attribute__((noinline, noclone)) void "
            "run_o_projection()"
        )
    )
    o_body = o_template[o_template.index("void kernel_main()"):].replace(
        "void kernel_main()", o_declaration, 1
    )
    for marker, value in o_replacements.items():
        o_body = o_body.replace(marker, value)
    if precomputed_o_rank_h:
        h_start = o_body.index(
            "        // Schedule 4 is a literal one-output adaptation"
        )
        base_start = o_body.index(
            "        // Exact base = X @ W.T for one <=4-tile N block."
        )
        o_body = o_body[:h_start] + o_body[base_start:]
    if re.search(r"@[A-Z0-9_]+@", o_body):
        raise RuntimeError("unexpanded exact O projection placeholder")
    functions.append(o_body)
    qkv_o_barrier = (
        "    CircularBuffer qkv_consumed(30), qkv_reuse_ack(31);\n"
        "    qkv_consumed.reserve_back(1);\n"
        "    qkv_consumed.push_back(1);\n"
        "    qkv_reuse_ack.wait_front(1);\n"
        "    qkv_reuse_ack.pop_front(1);\n"
        "    CircularBuffer qkv_outputs_drained(28), qkv_outputs_reuse_ack(29);\n"
        "    qkv_outputs_drained.wait_front(1);\n"
        "    qkv_outputs_drained.pop_front(1);\n"
        "    UNPACK({\n"
        "        constexpr uint32_t read_cbs[] = {0,1,2,3,5,6,7,8,9,10,11,13,15,16,17,18,19,20,21,22,23,24,25,26,27};\n"
        "        for (uint32_t cb_id : read_cbs) {\n"
        "            auto& interface = get_local_cb_interface(cb_id);\n"
        "            interface.fifo_rd_ptr = interface.fifo_limit - interface.fifo_size;\n"
        "            interface.tiles_acked_received_init = 0;\n"
        "        }\n"
        "    });\n"
        "    PACK({\n"
        "        constexpr uint32_t write_cbs[] = {2,3,4,7,8,10,11,12,15,20,21,22,23,24,25,26,27};\n"
        "        for (uint32_t cb_id : write_cbs) {\n"
        "            auto& interface = get_local_cb_interface(cb_id);\n"
        "            interface.fifo_wr_ptr = interface.fifo_limit - interface.fifo_size;\n"
        "            interface.tiles_acked_received_init = 0;\n"
        "            interface.fifo_wr_tile_ptr = 0;\n"
        "        }\n"
        "    });\n"
        "    qkv_outputs_reuse_ack.reserve_back(1);\n"
        "    qkv_outputs_reuse_ack.push_back(1);\n"
        if qkv_o_phase_barrier else ""
    )
    return (
        shared_prefix
        + "\n".join(functions)
        + "\n__attribute__((optimize(\"Os\"))) void kernel_main() {\n"
        + "    run_q_projection();\n"
        + "    run_k_projection();\n"
        + "    run_v_projection();\n"
        + qkv_o_barrier
        + "    run_o_projection();\n"
        + "}\n"
    )


def _input_rmsnorm_compute_source(
    *, hidden_features: int, sequence_length: int, epsilon: float,
) -> str:
    """Embed the qualified exact RMSNorm as a projection-core phase.

    IDs 15..27 are freed by aliasing the later O projection onto the drained
    Q/K/V CB map. The normalized BF16 row is retained in c22 until every
    projection reader has fetched it over NoC.
    """

    hidden_tiles = hidden_features // 32
    padded_tiles = ((hidden_tiles + 3) // 4) * 4
    token_tiles = sequence_length // 32
    source = (
        Path(__file__).with_name("cpp") / "rmsnorm_forward_compute.cpp"
    ).read_text()
    cb_map = {
        0: 15, 5: 16, 2: 17, 3: 18, 4: 19,
        7: 20, 8: 21, 16: 22, 17: 23, 18: 24,
        19: 25, 20: 26, 21: 27, 22: 14,
    }
    source = re.sub(
        r"get_compile_time_arg_val\((\d+)\)",
        lambda match: str(cb_map[int(match.group(1))]),
        source,
    )
    source = source.replace(
        "constexpr uint32_t cb_inverse_hidden_id = 4;",
        "constexpr uint32_t cb_inverse_hidden_id = 19;",
    ).replace(
        "constexpr uint32_t cb_mean_id = 17;",
        "constexpr uint32_t cb_mean_id = 23;",
    ).replace(
        "constexpr uint32_t cb_variance_eps_id = 18;",
        "constexpr uint32_t cb_variance_eps_id = 24;",
    ).replace(
        "get_absolute_logical_y() * 11 + get_absolute_logical_x()",
        "get_absolute_logical_y() * 5 + get_absolute_logical_x()",
    ).replace(
        "void kernel_main()",
        "static __attribute__((noinline, noclone)) void "
        "run_resident_rmsnorm(bool publish_post_rstd, bool resident_reentry)",
        1,
    )
    standalone_defaults = (
        "    // Standalone owner defaults. Resident composition removes/replaces this\n"
        "    // block with explicit phase arguments; keeping the raw source\n"
        "    // self-contained is required for safe-path compilation and diagnostics.\n"
        "    constexpr bool publish_post_rstd = false;\n"
        "    constexpr bool resident_reentry = false;\n"
    )
    if standalone_defaults not in source:
        raise RuntimeError("resident RMSNorm standalone-default marker changed")
    source = source.replace(standalone_defaults, "", 1)
    source = source.replace(
        "    rstd.push_back(1);\n",
        "    rstd.push_back(1);\n"
        "    if (publish_post_rstd) {\n"
        "        CircularBuffer post_rstd_ready(29), post_rstd_copied(28);\n"
        "        post_rstd_ready.reserve_back(1);\n"
        "        post_rstd_ready.push_back(1);\n"
        "        post_rstd_copied.wait_front(1);\n"
        "        post_rstd_copied.pop_front(1);\n"
        "    }\n",
        1,
    )
    # This helper is invoked twice when it owns the full attention block:
    # once before Q/K/V and once after O + residual.  Hardware startup is a
    # one-time cross-TRISC rendezvous, not an operation reconfiguration API;
    # entering it again in the post-attention epoch can leave the three
    # compute RISCs waiting at different generations.  The SFPU/reduction/
    # pack init calls in the body already reconfigure every operation needed
    # by the second invocation.
    source = source.replace(
        "    compute_kernel_hw_startup(cb_x_id, cb_x_id, cb_square_id);\n",
        "    if (!resident_reentry) {\n"
        "        compute_kernel_hw_startup(cb_x_id, cb_x_id, cb_square_id);\n"
        "    } else {\n"
        "        unary_op_reinit_no_barrier(cb_x_id, cb_square_id);\n"
        "    }\n",
        1,
    )
    # CB15 changes producer ownership at the post-attention boundary: the
    # reader produced the first norm input, whereas PACK produces the exact
    # residual row.  The latter is complete before this helper is called, but
    # Blackhole does not propagate a fresh credit generation across that
    # dynamic role change.  Consume the resident bytes in program order and
    # explicitly return the stream to a fresh epoch after the final read.
    source = source.replace(
        "        x.wait_front(block.start() + block.full_block_size());\n",
        "        if (!resident_reentry) {\n"
        "            x.wait_front(block.start() + block.full_block_size());\n"
        "        }\n",
        1,
    )
    source = source.replace(
        "        weight.wait_front(block.start() + block.full_block_size());\n",
        "        if (!resident_reentry) {\n"
        "            weight.wait_front(block.start() + block.full_block_size());\n"
        "        }\n",
        1,
    )
    resident_cleanup = (
        "    if (!resident_reentry) {\n"
        "        x.pop_front(padded_tiles);\n"
        "    } else {\n"
        "        UNPACK({\n"
        "            *get_cb_tiles_received_ptr(cb_x_id) = 0;\n"
        "            *get_cb_tiles_acked_ptr(cb_x_id) = 0;\n"
        "            auto& interface = get_local_cb_interface(cb_x_id);\n"
        "            interface.fifo_rd_ptr = "
        "interface.fifo_limit - interface.fifo_size;\n"
        "            interface.tiles_acked_received_init = 0;\n"
        "            *get_cb_tiles_received_ptr(cb_weight_id) = 0;\n"
        "            *get_cb_tiles_acked_ptr(cb_weight_id) = 0;\n"
        "            auto& weight_interface = "
        "get_local_cb_interface(cb_weight_id);\n"
        "            weight_interface.fifo_rd_ptr = "
        "weight_interface.fifo_limit - weight_interface.fifo_size;\n"
        "            weight_interface.tiles_acked_received_init = 0;\n"
        "        });\n"
        "        PACK({\n"
        "            auto& interface = get_local_cb_interface(cb_x_id);\n"
        "            interface.fifo_wr_ptr = "
        "interface.fifo_limit - interface.fifo_size;\n"
        "            interface.fifo_wr_tile_ptr = 0;\n"
        "            interface.tiles_acked_received_init = 0;\n"
        "        });\n"
        "    }\n"
        "    if (!resident_reentry) {\n"
        "        weight.pop_front(padded_tiles);\n"
        "    }\n"
        "    scaler.pop_front(1);\n"
        "    eps.pop_front(1);\n"
        "    inverse_hidden.pop_front(1);\n"
    )
    # OUTPUT_STAGE=0 is a self-contained fused RMSNorm implementation and
    # returns before the legacy diagnostic implementation below.  Keep its
    # producer/consumer epoch cleanup identical to the legacy exit; otherwise
    # the post-attention invocation observes stale X/gamma CB credits even
    # though the isolated standalone kernel is exact.
    fused_cleanup = (
        "    rstd.pop_front(1);\n"
        "    x.pop_front(padded_tiles);\n"
        "    weight.pop_front(padded_tiles);\n"
        "    scaler.pop_front(1);\n"
        "    eps.pop_front(1);\n"
        "    inverse_hidden.pop_front(1);\n"
        "    return;\n"
    )
    if fused_cleanup not in source:
        raise RuntimeError("resident fused RMSNorm cleanup marker changed")
    source = source.replace(
        fused_cleanup,
        "    WAYPOINT(\"RC00\");\n"
        "    rstd.pop_front(1);\n"
        "    WAYPOINT(\"RC01\");\n"
        + resident_cleanup.replace(
            "    if (!resident_reentry) {\n"
            "        x.pop_front(padded_tiles);\n",
            "    if (!resident_reentry) {\n"
            "        WAYPOINT(\"RCX0\");\n"
            "        x.pop_front(padded_tiles);\n"
            "        WAYPOINT(\"RCX1\");\n",
            1,
        ).replace(
            "    if (!resident_reentry) {\n"
            "        weight.pop_front(padded_tiles);\n",
            "    if (!resident_reentry) {\n"
            "        WAYPOINT(\"RCW0\");\n"
            "        weight.pop_front(padded_tiles);\n"
            "        WAYPOINT(\"RCW1\");\n",
            1,
        )
        + "    WAYPOINT(\"RCRD\");\n"
        "    return;\n",
        1,
    )
    source = source.replace(
        "    x.pop_front(padded_tiles);\n"
        "    weight.pop_front(padded_tiles);\n"
        "    scaler.pop_front(1);\n"
        "    eps.pop_front(1);\n"
        "    inverse_hidden.pop_front(1);\n"
        "}\n",
        resident_cleanup + "}\n",
        1,
    )
    replacements = {
        "@HIDDEN_TILES@": str(hidden_tiles),
        "@PADDED_TILES@": str(padded_tiles),
        "@TOKEN_TILES@": str(token_tiles),
        "@HIDDEN_SIZE@": str(hidden_features),
        "@EPSILON_BITS@": str(
            struct.unpack("<I", struct.pack("<f", float(epsilon)))[0]
        ),
        "@INVERSE_HIDDEN_BITS@": str(
            struct.unpack(
                "<I", struct.pack("<f", 1.0 / hidden_features)
            )[0]
        ),
        "@OUTPUT_STAGE@": "0",
    }
    for marker, value in replacements.items():
        source = source.replace(marker, value)
    if re.search(r"@[A-Z0-9_]+@", source):
        raise RuntimeError("unexpanded resident input RMSNorm marker")
    return source


def _ttnn_input_rmsnorm_compute_source(
    *, hidden_features: int, sequence_length: int, epsilon: float,
) -> str:
    """Compose TTNN's exact RMSNorm arithmetic into the projection island.

    The old resident helper had drifted to a different BF16 normalization
    boundary.  This mapping retains TTNN's physical reduction/packing body
    and changes only its CB addresses and resident lifecycle.  CB22 is the
    normalized BF16 row consumed by Q/K/V.  CB27 retains the FP32 rstd long
    enough for the existing post-attention writer handshake.
    """

    del epsilon  # Epsilon bits are produced by the resident reader.
    cb_ids = {
        "cb_in": 15,
        "cb_inb": 1,
        "cb_scaler": 17,
        "cb_eps": 18,
        "cb_gamma": 16,
        "cb_beta": 6,
        "cb_out": 22,
        "cb_ex": 23,
        "cb_ex2": 24,
        "cb_xmm2": 26,
        "cb_ex2pe": 27,
        "cb_fusion": 25,
        "cb_x": 15,
        "cb_xmm": 15,
        "cb_in_rm": 15,
        "cb_out_rm": 22,
    }
    return _ttnn_fused_rmsnorm_compute_source(
        sequence_length=sequence_length,
        hidden_size=hidden_features,
        cb_ids=cb_ids,
        logical_grid_width=5,
        function_signature=(
            "static __attribute__((noinline, noclone)) void "
            "run_resident_rmsnorm(bool publish_post_rstd, "
            "bool resident_reentry)"
        ),
        balance_resident_inputs=True,
        publish_rstd=True,
    )


def _post_attention_rmsnorm_compute_source(
    *, hidden_features: int, sequence_length: int, epsilon: float,
) -> str:
    """Exact post-attention RMSNorm body over resident residual CB22.

    The shared helper definitions come from the input-RMSNorm source. Only
    this second phase's function body is emitted here. Its final BF16 stream
    uses CB12, the established compute-to-writer output FIFO.
    """

    hidden_tiles = hidden_features // 32
    padded_tiles = ((hidden_tiles + 3) // 4) * 4
    token_tiles = sequence_length // 32
    source = (
        Path(__file__).with_name("cpp") / "rmsnorm_forward_compute.cpp"
    ).read_text()
    body = source[source.index("void kernel_main()") :]
    standalone_defaults = (
        "    // Standalone owner defaults. Resident composition removes/replaces this\n"
        "    // block with explicit phase arguments; keeping the raw source\n"
        "    // self-contained is required for safe-path compilation and diagnostics.\n"
        "    constexpr bool publish_post_rstd = false;\n"
        "    constexpr bool resident_reentry = false;\n"
    )
    if standalone_defaults not in body:
        raise RuntimeError("post RMSNorm standalone-default marker changed")
    body = body.replace(
        standalone_defaults,
        "    constexpr bool publish_post_rstd = true;\n"
        "    constexpr bool resident_reentry = true;\n",
        1,
    )
    cb_map = {
        0: 22, 5: 0, 2: 17, 3: 18, 4: 19,
        7: 20, 8: 21, 16: 12, 17: 23, 18: 24,
        19: 25, 20: 26, 21: 27, 22: 14,
    }
    body = re.sub(
        r"get_compile_time_arg_val\((\d+)\)",
        lambda match: str(cb_map[int(match.group(1))]),
        body,
    ).replace(
        "constexpr uint32_t cb_inverse_hidden_id = 4;",
        "constexpr uint32_t cb_inverse_hidden_id = 19;",
    ).replace(
        "constexpr uint32_t cb_mean_id = 17;",
        "constexpr uint32_t cb_mean_id = 23;",
    ).replace(
        "constexpr uint32_t cb_variance_eps_id = 18;",
        "constexpr uint32_t cb_variance_eps_id = 24;",
    ).replace(
        "get_absolute_logical_y() * 11 + get_absolute_logical_x()",
        "get_absolute_logical_y() * 5 + get_absolute_logical_x()",
    ).replace(
        "void kernel_main()", "FORCE_INLINE void run_post_attention_rmsnorm()", 1,
    )
    # Hardware startup contains a one-time cross-TRISC barrier.  This body is
    # the third compute epoch in the resident program, so re-entering startup
    # would deadlock RISCs that reached the O/residual boundary at different
    # times.  The ordinary per-operation init immediately below performs the
    # required source/packer reconfiguration without another startup barrier.
    body = body.replace(
        "    compute_kernel_hw_startup(cb_x_id, cb_x_id, cb_square_id);\n",
        "    // Resident epoch: kernel startup already ran in input RMSNorm.\n",
        1,
    )
    # Backward needs the exact row rstd.  Keep the producer-owned FP32 tile in
    # CB27 while the writer copies its raw 4096 bits into two arena pages.  A
    # CB29/CB28 handshake prevents compute from popping the tile early and
    # does not expose an intermediate PJRT tensor.
    body = body.replace(
        "    rstd.push_back(1);\n",
        "    rstd.push_back(1);\n"
        "    CircularBuffer post_rstd_ready(29), post_rstd_copied(28);\n"
        "    post_rstd_ready.reserve_back(1);\n"
        "    post_rstd_ready.push_back(1);\n"
        "    post_rstd_copied.wait_front(1);\n"
        "    post_rstd_copied.pop_front(1);\n",
        1,
    )
    replacements = {
        "@HIDDEN_TILES@": str(hidden_tiles),
        "@PADDED_TILES@": str(padded_tiles),
        "@TOKEN_TILES@": str(token_tiles),
        "@HIDDEN_SIZE@": str(hidden_features),
        "@EPSILON_BITS@": str(
            struct.unpack("<I", struct.pack("<f", float(epsilon)))[0]
        ),
        "@INVERSE_HIDDEN_BITS@": str(
            struct.unpack(
                "<I", struct.pack("<f", 1.0 / hidden_features)
            )[0]
        ),
        "@OUTPUT_STAGE@": "0",
    }
    for marker, value in replacements.items():
        body = body.replace(marker, value)
    if re.search(r"@[A-Z0-9_]+@", body):
        raise RuntimeError("unexpanded post-attention RMSNorm marker")
    return body


def _compose_input_rmsnorm_projection_compute(
    rmsnorm_source: str, projection_source: str,
    post_attention_block: bool = False,
    post_diagnostic_stage: int = 0,
) -> str:
    """Run RMSNorm then the accepted Q/K/V/O phase chain in one kernel."""

    projection_source = projection_source.replace(
        "void kernel_main() {", "FORCE_INLINE void run_projection_owner() {", 1
    )
    post_phase = ""
    if post_attention_block:
        post_phase = (
            "\nFORCE_INLINE void run_post_attention_residual() {\n"
            "    constexpr uint32_t hidden_tiles = @POST_HIDDEN_TILES@;\n"
            "    constexpr uint32_t padded_tiles = @POST_PADDED_HIDDEN_TILES@;\n"
            "    constexpr uint32_t seq_tiles = @POST_SEQ_TILES@;\n"
            "    const uint32_t node = get_absolute_logical_y() * 5 + "
            "get_absolute_logical_x();\n"
            "    if (node >= seq_tiles) return;\n"
            "    CircularBuffer attention_output(13), layer_input(16);\n"
            "    CircularBuffer residual(15), residual_checkpoint(12);\n"
            "    add_tiles_init(13, 16);\n"
            "    disable_fp32_dest_acc();\n"
            "    for (uint32_t tile = 0; tile < padded_tiles; ++tile) {\n"
            "        attention_output.wait_front(1);\n"
            "        layer_input.wait_front(1);\n"
            "        residual.reserve_back(1);\n"
            "        if (tile < hidden_tiles) residual_checkpoint.reserve_back(1);\n"
            "        tile_regs_acquire();\n"
            "        add_tiles_bf16(13, 16, 0, 0, 0);\n"
            "        tile_regs_commit_bf16();\n"
            "        tile_regs_wait();\n"
            "        PACK((pack_reconfig_data_format_bf16(15)));\n"
            "        pack_tile_bf16(0, 15);\n"
            "        if (tile < hidden_tiles) {\n"
            "            PACK((pack_reconfig_data_format_bf16(12)));\n"
            "            pack_tile_bf16(0, 12);\n"
            "        }\n"
            "        tile_regs_release_bf16();\n"
            "        residual.push_back(1);\n"
            "        if (tile < hidden_tiles) residual_checkpoint.push_back(1);\n"
            "        attention_output.pop_front(1);\n"
            "        layer_input.pop_front(1);\n"
            "    }\n"
            "    enable_fp32_dest_acc();\n"
            "    CircularBuffer o_consumed(30), post_gamma_ready(31);\n"
            "    o_consumed.reserve_back(1);\n"
            "    o_consumed.push_back(1);\n"
            + ("    post_gamma_ready.wait_front(1);\n"
               "    post_gamma_ready.pop_front(1);\n"
               "    UNPACK({\n"
               "        auto& gamma_interface = get_local_cb_interface(16);\n"
               "        gamma_interface.fifo_rd_ptr = "
               "gamma_interface.fifo_limit - gamma_interface.fifo_size;\n"
               "        gamma_interface.tiles_acked_received_init = 0;\n"
               "    });\n"
               if post_diagnostic_stage not in {1, 10} else "")
            + "}\n"
        )
        if post_diagnostic_stage == 4:
            # Route every logical O tile into the reserved row-owner FIFO,
            # then drain only that FIFO and the original residual input. This
            # cut validates NoC routing and CB lifecycle without executing
            # residual arithmetic or publishing another writer stream.
            post_phase = (
                "\nFORCE_INLINE void run_post_attention_residual() {\n"
                "    constexpr uint32_t padded_tiles = "
                "@POST_PADDED_HIDDEN_TILES@;\n"
                "    constexpr uint32_t seq_tiles = @POST_SEQ_TILES@;\n"
                "    const uint32_t node = get_absolute_logical_y() * 5 + "
                "get_absolute_logical_x();\n"
                "    if (node >= seq_tiles) return;\n"
                "    CircularBuffer attention_output(13), layer_input(16);\n"
                "    attention_output.wait_front(padded_tiles);\n"
                "    layer_input.wait_front(padded_tiles);\n"
                "    attention_output.pop_front(padded_tiles);\n"
                "    layer_input.pop_front(padded_tiles);\n"
                "    CircularBuffer o_consumed(30);\n"
                "    o_consumed.reserve_back(1);\n"
                "    o_consumed.push_back(1);\n"
                "}\n"
            )
        elif post_diagnostic_stage in {5, 6, 7, 8}:
            post_phase = ""
        if post_diagnostic_stage == 9:
            post_phase += (
                "\nFORCE_INLINE void run_post_norm_input_probe() {\n"
                "    constexpr uint32_t padded_tiles = "
                "@POST_PADDED_HIDDEN_TILES@;\n"
                "    constexpr uint32_t seq_tiles = @POST_SEQ_TILES@;\n"
                "    const uint32_t node = get_absolute_logical_y() * 5 + "
                "get_absolute_logical_x();\n"
                "    if (node >= seq_tiles) return;\n"
                "    CircularBuffer residual(15), gamma(16), scaler(17);\n"
                "    CircularBuffer epsilon(18), inverse_hidden(19);\n"
                "    residual.wait_front(padded_tiles);\n"
                "    gamma.wait_front(padded_tiles);\n"
                "    scaler.wait_front(1);\n"
                "    epsilon.wait_front(1);\n"
                "    inverse_hidden.wait_front(1);\n"
                "    residual.pop_front(padded_tiles);\n"
                "    gamma.pop_front(padded_tiles);\n"
                "    scaler.pop_front(1);\n"
                "    epsilon.pop_front(1);\n"
                "    inverse_hidden.pop_front(1);\n"
                "}\n"
            )
        if post_diagnostic_stage == 10:
            post_phase += (
                "\nFORCE_INLINE void run_post_residual_input_probe() {\n"
                "    constexpr uint32_t padded_tiles = "
                "@POST_PADDED_HIDDEN_TILES@;\n"
                "    constexpr uint32_t seq_tiles = @POST_SEQ_TILES@;\n"
                "    const uint32_t node = get_absolute_logical_y() * 5 + "
                "get_absolute_logical_x();\n"
                "    if (node >= seq_tiles) return;\n"
                "    CircularBuffer residual(15);\n"
                "    residual.wait_front(padded_tiles);\n"
                "    residual.pop_front(padded_tiles);\n"
                "}\n"
            )
        if post_diagnostic_stage == 11:
            post_phase += (
                "\nFORCE_INLINE void run_post_gamma_scalar_probe() {\n"
                "    constexpr uint32_t padded_tiles = "
                "@POST_PADDED_HIDDEN_TILES@;\n"
                "    constexpr uint32_t seq_tiles = @POST_SEQ_TILES@;\n"
                "    const uint32_t node = get_absolute_logical_y() * 5 + "
                "get_absolute_logical_x();\n"
                "    if (node >= seq_tiles) return;\n"
                "    CircularBuffer gamma(16), scaler(17), epsilon(18);\n"
                "    CircularBuffer inverse_hidden(19);\n"
                "    gamma.wait_front(padded_tiles);\n"
                "    scaler.wait_front(1);\n"
                "    epsilon.wait_front(1);\n"
                "    inverse_hidden.wait_front(1);\n"
                "    gamma.pop_front(padded_tiles);\n"
                "    scaler.pop_front(1);\n"
                "    epsilon.pop_front(1);\n"
                "    inverse_hidden.pop_front(1);\n"
                "}\n"
            )
        if post_diagnostic_stage == 12:
            post_phase += (
                "\nFORCE_INLINE void run_post_gamma_probe() {\n"
                "    constexpr uint32_t padded_tiles = "
                "@POST_PADDED_HIDDEN_TILES@;\n"
                "    constexpr uint32_t seq_tiles = @POST_SEQ_TILES@;\n"
                "    const uint32_t node = get_absolute_logical_y() * 5 + "
                "get_absolute_logical_x();\n"
                "    if (node >= seq_tiles) return;\n"
                "    CircularBuffer gamma(16);\n"
                "    gamma.wait_front(padded_tiles);\n"
                "    gamma.pop_front(padded_tiles);\n"
                "}\n"
            )
    composed = (
        rmsnorm_source
        + "\n"
        + projection_source
        + post_phase
        + "\n__attribute__((optimize(\"Os\"))) void kernel_main() {\n"
        + "    WAYPOINT(\"CMR0\");\n"
        + "    run_resident_rmsnorm(false, false);\n"
        + "    WAYPOINT(\"CMR1\");\n"
        + "    run_projection_owner();\n"
        + "    WAYPOINT(\"CMP1\");\n"
        + ("    run_post_attention_residual();\n" if post_phase else "")
        + ("    run_resident_rmsnorm(false, true);\n"
           if post_phase and post_diagnostic_stage == 3 else "")
        + ("    run_resident_rmsnorm(true, true);\n"
           if post_phase and post_diagnostic_stage == 0 else "")
        + ("    run_post_norm_input_probe();\n"
           if post_phase and post_diagnostic_stage == 9 else "")
        + ("    run_post_residual_input_probe();\n"
           if post_phase and post_diagnostic_stage == 10 else "")
        + ("    run_post_gamma_scalar_probe();\n"
           if post_phase and post_diagnostic_stage == 11 else "")
        + ("    run_post_gamma_probe();\n"
           if post_phase and post_diagnostic_stage == 12 else "")
        + "}\n"
    )
    return composed


@lru_cache(maxsize=8)
def build_tt_lang_attention_forward_qkv_resident(
    *, sequence_length: int, hidden_features: int, kv_features: int,
    query_heads: int, kv_heads: int, head_dim: int, physical_rank: int,
    lora_scale: float, attention_scale: float,
    base_k_tiles_per_phase: int = 1,
    lora_k_tiles_per_phase: int | None = None,
    o_lora_k_tiles_per_phase: int | None = None,
    o_lora_init_schedule: int = 0,
    has_bias: bool = False,
    bias_after_bf16_base: bool = False,
    grouped_qkv_lora: bool = False,
    active_lora_roles: tuple[str, ...] = ("q", "k", "v", "o"),
    smol_grouped_qkv_lora: bool = False,
    separate_qkv_lora: bool = False,
    precomputed_separate_qkv_delta: bool = False,
    input_rmsnorm_epsilon: float | None = None,
    post_attention_rmsnorm_epsilon: float | None = None,
    post_attention_diagnostic_stage: int = 0,
    resident_gate_up_intermediate_features: int | None = None,
    resident_gate_up_base_k_tiles_per_phase: int = 1,
    resident_full_mlp_forward: bool = False,
    resident_down_base_k_tiles_per_phase: int = 1,
    hidden_page_base: int = 0,
    publish_layer_output_at_page_zero: bool = False,
    debug_input_rmsnorm_checkpoint: bool = False,
    rebase_projection_cbs: bool = False,
    dedicated_o_rank_producer: bool = False,
    publish_raw_score_checkpoint: bool = False,
    projection_rope_prefix_only: bool = False,
) -> Callable:
    plan = plan_full_attention_forward(
        tokens=sequence_length,
        hidden_features=hidden_features,
        kv_features=kv_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        physical_rank=physical_rank,
        qkv_bias=has_bias,
        include_attention_block=post_attention_rmsnorm_epsilon is not None,
    )
    attention_roles = {"q", "k", "v", "o"}
    active_lora_roles = tuple(dict.fromkeys(active_lora_roles))
    if not set(active_lora_roles) <= attention_roles:
        raise ValueError("active attention LoRA roles must be a canonical subset")
    if physical_rank <= 0 or physical_rank % 32 or physical_rank > 128:
        raise ValueError(
            "QKV resident owner requires tiled physical rank up to 128"
        )
    # Qwen's exact reference rank projection is currently produced by the
    # adjacent TTNN 2-D grouped matmul and passed as this owner's first A-slot.
    # The resident program consumes one selected physical H tile per member.
    # This closes numerical parity first; the next step absorbs that same 2-D
    # program into this TT-Metal invocation and removes the small DRAM handoff.
    if sum(map(bool, (
        grouped_qkv_lora, smol_grouped_qkv_lora, separate_qkv_lora,
    ))) > 1:
        raise ValueError(
            "Qwen grouped, Smol grouped and separate-QKV layouts are "
            "mutually exclusive"
        )
    if precomputed_separate_qkv_delta and not separate_qkv_lora:
        raise ValueError(
            "precomputed separate Q/K/V delta requires separate-QKV layout"
        )
    precomputed_grouped_qkv_h = (
        grouped_qkv_lora or smol_grouped_qkv_lora or separate_qkv_lora
    )
    precomputed_grouped_qkv_base = grouped_qkv_lora
    rank_tiles = physical_rank // 32
    if (grouped_qkv_lora or smol_grouped_qkv_lora) and rank_tiles != 1:
        raise ValueError(
            "legacy packed grouped-QKV layouts require one physical rank tile"
        )
    grouped_qkv_rank_tiles = rank_tiles
    grouped_qkv_storage_rank_tiles = (
        2 if grouped_qkv_lora else rank_tiles
    )
    seq_tiles = sequence_length // 32
    hidden_tiles = hidden_features // 32
    kv_tiles = kv_features // 32
    dim_tiles = head_dim // 32
    input_rmsnorm = input_rmsnorm_epsilon is not None
    if input_rmsnorm and float(input_rmsnorm_epsilon) <= 0.0:
        raise ValueError("input RMSNorm epsilon must be positive")
    post_attention_block = post_attention_rmsnorm_epsilon is not None
    if projection_rope_prefix_only and (
        post_attention_block or resident_gate_up_intermediate_features is not None
    ):
        raise ValueError(
            "projection/RoPE prefix-only execution cannot own the post-attention block"
        )
    if post_attention_block and not input_rmsnorm:
        raise ValueError("post-attention ownership requires resident input RMSNorm")
    if post_attention_block and float(post_attention_rmsnorm_epsilon) <= 0.0:
        raise ValueError("post-attention RMSNorm epsilon must be positive")
    if post_attention_diagnostic_stage not in {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    }:
        raise ValueError(
            "post-attention diagnostic stage must be 0 through 12"
        )
    if post_attention_diagnostic_stage and not post_attention_block:
        raise ValueError("post-attention diagnostic stage requires block ownership")
    resident_gate_up = resident_gate_up_intermediate_features is not None
    full_mlp_forward = bool(resident_full_mlp_forward)
    if hidden_page_base < 0:
        raise ValueError("hidden page base must be non-negative")
    if full_mlp_forward and not resident_gate_up:
        raise ValueError("resident full MLP requires resident gate/up")
    if resident_gate_up and not post_attention_block:
        raise ValueError(
            "resident gate/up requires the post-attention RMSNorm owner"
        )
    # Stage 3 preserves the complete residual/normalization dataflow while
    # omitting only the backward-lifetime rstd checkpoint handshake.  Keep it
    # available as a resident gate/up lifecycle diagnostic.
    if resident_gate_up and post_attention_diagnostic_stage not in {0, 3}:
        raise ValueError(
            "resident gate/up requires the complete post-attention path"
        )
    intermediate_tiles = (
        int(resident_gate_up_intermediate_features or 0) // 32
    )
    if resident_gate_up and (
        int(resident_gate_up_intermediate_features or 0) <= 0
        or int(resident_gate_up_intermediate_features or 0) % 32
    ):
        raise ValueError("resident gate/up intermediate width must be positive /32")
    if resident_gate_up and (
        resident_gate_up_base_k_tiles_per_phase <= 0
        or hidden_tiles % resident_gate_up_base_k_tiles_per_phase
    ):
        raise ValueError("invalid resident gate/up frozen-base K phase")
    if full_mlp_forward and (
        resident_down_base_k_tiles_per_phase <= 0
        or intermediate_tiles % resident_down_base_k_tiles_per_phase
        or intermediate_tiles % 2
    ):
        raise ValueError("invalid resident down-projection K phase")
    padded_hidden_tiles = ((hidden_tiles + 3) // 4) * 4
    total_o_jobs = seq_tiles * hidden_tiles
    # A partial final worker leaves a producer/consumer ring on a different
    # physical epoch from the full workers.  Select the largest <=50 divisor
    # so every active O worker owns the same complete number of jobs.  This is
    # 36x2 for SmolLM2 and 28x4 for Qwen at sequence 128.
    o_projection_workers = _divisor_worker_count(total_o_jobs)
    o_jobs_per_worker = total_o_jobs // o_projection_workers
    total_cbs = 64 if input_rmsnorm else _TOTAL_CBS
    artifact_total_cbs = total_cbs + (4 if dedicated_o_rank_producer else 0)
    projection_workers = 45 if dedicated_o_rank_producer else 50
    # TTNN's full-graph LoRA rank projection selects a one-K-tile spill for
    # SmolLM2's 576-wide rank-8 geometry and the previously qualified
    # two-K-tile spill for the wider rank-16/Qwen geometry. This is an
    # observable BF16 contract, not an interchangeable tuning choice.
    lora_phase = (
        1
        if (
            lora_k_tiles_per_phase is None
            and hidden_features == 576
            and rank_tiles == 1
        )
        else 2 if lora_k_tiles_per_phase is None
        else int(lora_k_tiles_per_phase)
    )
    o_lora_phase = (
        lora_phase
        if o_lora_k_tiles_per_phase is None
        else int(o_lora_k_tiles_per_phase)
    )
    if lora_phase <= 0 or hidden_tiles % lora_phase:
        raise ValueError("invalid QKV LoRA K phase")
    if o_lora_phase <= 0 or hidden_tiles % o_lora_phase:
        raise ValueError("invalid O LoRA K phase")
    if o_lora_init_schedule not in (0, 1, 2, 4):
        raise ValueError("O LoRA init schedule must be 0, 1, 2 or 4")
    if hidden_tiles % base_k_tiles_per_phase:
        raise ValueError("invalid QKV frozen-base K phase")

    # The optional MLP epoch extends only the private page arena.  Existing
    # attention offsets remain stable, which keeps the accepted backward ABI
    # and all safe-path diagnostics unchanged.
    gate_base = plan.arena.used_pages
    up_base = gate_base + seq_tiles * intermediate_tiles
    gate_h_base = up_base + seq_tiles * intermediate_tiles
    up_h_base = gate_h_base + seq_tiles
    layer_output_base = up_h_base + seq_tiles
    layer_used_pages = (
        layer_output_base + seq_tiles * hidden_tiles
        if full_mlp_forward else (
            up_h_base + seq_tiles
            if resident_gate_up else plan.arena.used_pages
        )
    )
    layer_allocated_pages = (
        -(-layer_used_pages // hidden_tiles) * hidden_tiles
        if resident_gate_up else plan.arena.allocated_pages
    )
    input_rmsnorm_debug_base = layer_allocated_pages
    if debug_input_rmsnorm_checkpoint:
        layer_allocated_pages += seq_tiles * hidden_tiles
    layer_arena_shape = (
        (layer_allocated_pages // hidden_tiles) * 32,
        hidden_features,
    )
    # post_norm_weight is a page-packed immutable forward-parameter tensor on
    # the experimental layer path. Gamma remains first, so the already exact
    # RMSNorm reader is unchanged.
    mask_pack_base = hidden_tiles
    gate_up_weight_pack_base = mask_pack_base + seq_tiles * seq_tiles
    gate_a_pack_base = gate_up_weight_pack_base + hidden_tiles * 2 * intermediate_tiles
    gate_b_pack_base = gate_a_pack_base + hidden_tiles
    up_a_pack_base = gate_b_pack_base + intermediate_tiles
    up_b_pack_base = up_a_pack_base + hidden_tiles
    down_weight_pack_base = up_b_pack_base + intermediate_tiles
    down_a_pack_base = (
        down_weight_pack_base + intermediate_tiles * hidden_tiles
    )
    down_b_pack_base = down_a_pack_base + intermediate_tiles

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("resident QKV attention requires TT-XLA") from exc
        ttl = ttl_module
        tt_lang_operation = decorator

    options: dict[str, object] = {
        "grid": BLACKHOLE_P150A_GRID,
        "fp32_dest_acc_en": True,
        "options": "--ttl-matmul-full-fp32",
    }
    if "math_approx_mode" in inspect.signature(ttl.operation).parameters:
        options["math_approx_mode"] = True

    @ttl.operation(**options)
    def _program_paired_v23(
        hidden, q_weight, q_bias, q_a, q_b,
        k_weight, k_bias, k_a, k_b,
        v_weight, v_bias, v_a, v_b,
        o_weight, o_a, o_b,
        cosine, sine, mask, input_norm_weight, post_norm_weight, arena,
    ):
        # TT-Lang's AST capture does not support Python containers of DFBs.
        # Keep this deliberately explicit; the descriptors are replaced by
        # exact per-CB formats/capacities before artifact serialization.
        b0 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b1 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b2 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b3 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b4 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b5 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b6 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b7 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b8 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b9 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b10 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b11 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b12 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b13 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b14 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b15 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b16 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b17 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b18 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b19 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b20 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b21 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b22 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b23 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b24 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b25 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b26 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b27 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b28 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b29 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b30 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        # Revision token for the six-kernel resident owner. The TT-Lang
        # skeleton cache keys the captured placeholder program separately
        # from our generated C++ sources; changing this harmless control-CB
        # capacity forces it to materialize the current v23 source/config set.
        b31 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=18)

        @ttl.datamovement()
        def read():
            with b0.reserve() as block:
                tx = ttl.copy(hidden[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(q_weight[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(q_bias[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(q_a[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(q_b[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(k_weight[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(k_bias[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(k_a[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(k_b[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(v_weight[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(v_bias[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(v_a[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(v_b[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(o_weight[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(o_a[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(o_b[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(cosine[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(sine[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(mask[0:1, 0:1], block); tx.wait()
            with b0.reserve() as block:
                tx = ttl.copy(input_norm_weight[0:1, 0:1], block); tx.wait()
        @ttl.compute()
        def compute():
            for _ in range(20):
                with b0.wait() as item:
                    with b31.reserve() as result:
                        result.store(item)
            x, y = ttl.node(dims=2)
            if x + y < 0:
                # Capture every CB in the skeleton artifact.
                with b1.wait() as _: pass
                with b2.wait() as _: pass
                with b3.wait() as _: pass
                with b4.wait() as _: pass
                with b5.wait() as _: pass
                with b6.wait() as _: pass
                with b7.wait() as _: pass
                with b8.wait() as _: pass
                with b9.wait() as _: pass
                with b10.wait() as _: pass
                with b11.wait() as _: pass
                with b12.wait() as _: pass
                with b13.wait() as _: pass
                with b14.wait() as _: pass
                with b15.wait() as _: pass
                with b16.wait() as _: pass
                with b17.wait() as _: pass
                with b18.wait() as _: pass
                with b19.wait() as _: pass
                with b20.wait() as _: pass
                with b21.wait() as _: pass
                with b22.wait() as _: pass
                with b23.wait() as _: pass
                with b24.wait() as _: pass
                with b25.wait() as _: pass
                with b26.wait() as _: pass
                with b27.wait() as _: pass
                with b28.wait() as _: pass
                with b29.wait() as _: pass
                with b30.wait() as _: pass

        @ttl.datamovement()
        def write():
            for _ in range(20):
                with b31.wait() as item:
                    tx = ttl.copy(item, arena[0:1, 0:1]); tx.wait()

    lora_scale_bf16 = float(
        torch.tensor(lora_scale, dtype=torch.bfloat16).float().item()
    )
    lora_scale_bits = struct.unpack(
        "<I", struct.pack("<f", lora_scale_bf16)
    )[0]
    attention_scale_bits = struct.unpack(
        "<I", struct.pack("<f", float(attention_scale))
    )[0]
    variant = hashlib.sha256(
        (
            f"full-resident-v54-canonical-bmm-fifo:{sequence_length}:{hidden_features}:{kv_features}:"
            f"{query_heads}:{kv_heads}:{head_dim}:{physical_rank}:"
            f"{lora_scale_bf16:.17g}:{attention_scale:.17g}:"
            f"{base_k_tiles_per_phase}:{lora_phase}:{o_lora_phase}:"
            f"o_init={o_lora_init_schedule}:"
            f"{int(has_bias)}:"
            f"{int(bias_after_bf16_base)}:{int(grouped_qkv_lora)}:"
            f"targets={','.join(active_lora_roles)}:"
            f"{int(smol_grouped_qkv_lora)}:"
            f"{int(separate_qkv_lora)}:"
            f"predelta={int(precomputed_separate_qkv_delta)}:"
            f"{input_rmsnorm_epsilon!r}:{post_attention_rmsnorm_epsilon!r}:"
            f"{post_attention_diagnostic_stage}:"
            f"gate_up={resident_gate_up_intermediate_features!r}:"
            f"gate_up_phase={resident_gate_up_base_k_tiles_per_phase}:"
            f"full_mlp={int(full_mlp_forward)}:"
            f"down_phase={resident_down_base_k_tiles_per_phase}:"
            f"hidden_page_base={hidden_page_base}:"
            f"publish_page0={int(publish_layer_output_at_page_zero)}:"
            f"debug_input_norm={int(debug_input_rmsnorm_checkpoint)}:"
            f"rebase_projection_cbs={int(rebase_projection_cbs)}"
            f":dedicated_o_rank={int(dedicated_o_rank_producer)}"
            f":raw_score_checkpoint={int(publish_raw_score_checkpoint)}"
            f":rope_prefix_only={int(projection_rope_prefix_only)}"
        ).encode()
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.attention_forward_qkv_resident.{variant}",
        arg_roles=("in",) * 21 + ("out",),
        version_tag=f"attention-forward-full-resident-{variant}",
    )
    def operation(
        hidden, q_weight, q_bias, q_a, q_b,
        k_weight, k_bias, k_a, k_b,
        v_weight, v_bias, v_a, v_b,
        o_weight, o_a, o_b,
        cosine, sine, mask, input_norm_weight, post_norm_weight, arena,
    ):
        _clear_ttl_compile_wrapper_cache(_program_paired_v23)
        result = _program_paired_v23(
            hidden, q_weight, q_bias, q_a, q_b,
            k_weight, k_bias, k_a, k_b,
            v_weight, v_bias, v_a, v_b,
            o_weight, o_a, o_b,
            cosine, sine, mask, input_norm_weight, post_norm_weight, arena,
        )
        closure = _program_paired_v23.__closure__ or ()
        for name, cell in zip(
            _program_paired_v23.__code__.co_freevars, closure
        ):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != _ATTENTION_CBS:
                    raise RuntimeError(
                        f"QKV resident skeleton expected {_ATTENTION_CBS} CBs, "
                        f"got {len(compiled.cb_configs)}"
                    )
                replacements = {
                    "@SEQ_TILES@": str(seq_tiles),
                    "@DIM_TILES@": str(dim_tiles),
                    "@HIDDEN_TILES@": str(hidden_tiles),
                    "@HIDDEN_PAGE_BASE@": str(hidden_page_base),
                    "@KV_TILES@": str(kv_tiles),
                    "@O_WORKERS@": str(o_projection_workers),
                    "@QUERY_HEADS@": str(query_heads),
                    "@KV_HEADS@": str(kv_heads),
                    "@SCALE_BITS@": str(attention_scale_bits),
                    "@FP32_CANONICAL_ATTENTION@": "1",
                    "@PUBLISH_RAW_SCORE_CHECKPOINT@": (
                        "1" if publish_raw_score_checkpoint else "0"
                    ),
                    "@PROJECTION_ROPE_PREFIX_ONLY@": (
                        "1" if projection_rope_prefix_only else "0"
                    ),
                    "@ATTENTION_CHECKPOINT_CB@": (
                        "23" if publish_raw_score_checkpoint else "27"
                    ),
                    "@PV_PHASE_TILES@": "1",
                    "@TOTAL_CBS@": str(artifact_total_cbs),
                    "@PROJECTION_WORKERS@": str(projection_workers),
                    "@DEDICATED_O_RANK@": (
                        "1" if dedicated_o_rank_producer else "0"
                    ),
                    # The DPS result address is kernel tensor 21, supplied by
                    # the writer's compact common-arg slot.  TT-Lang's
                    # multi-program artifact does not emit a stable output
                    # accessor descriptor at CTA table slot 21, however.
                    # Tensor 20 is the packed immutable BF16 page tensor and
                    # has the identical interleaved 2-KiB tile geometry, so
                    # retain its accepted descriptor for arena page math.
                    "@ARENA_TENSOR_INDEX@": "21",
                    "@LAYER_PROBE_PAGE@": str(layer_allocated_pages - 2),
                    "@LAYER_CONSTANT_PROBE_PAGE@": str(layer_allocated_pages - 4),
                    "@INPUT_RMSNORM_DEBUG_BASE@": str(input_rmsnorm_debug_base),
                    "@INPUT_RMSNORM@": "1" if input_rmsnorm else "0",
                    "@POST_ATTENTION_BLOCK@": (
                        "1" if post_attention_block else "0"
                    ),
                    "@POST_DIAGNOSTIC_STAGE@": str(
                        post_attention_diagnostic_stage
                    ),
                    "@POST_HIDDEN_TILES@": str(hidden_tiles),
                    "@POST_PADDED_HIDDEN_TILES@": str(padded_hidden_tiles),
                    "@POST_SEQ_TILES@": str(seq_tiles),
                    "@INTERMEDIATE_TILES@": str(intermediate_tiles),
                    "@MASK_PACK_BASE@": str(mask_pack_base),
                    "@GATE_UP_WORKERS@": "60",
                    "@GATE_UP_GRID_COLUMNS@": "6",
                    "@GATE_UP_X_OFFSET@": "5",
                    "@GATE_UP_BASE_PHASE@": str(
                        resident_gate_up_base_k_tiles_per_phase
                    ),
                    "@LORA_SCALE_BITS@": str(lora_scale_bits),
                    "@FULL_MLP_FORWARD@": "1" if full_mlp_forward else "0",
                    "@PAIRED_SWIGLU@": "1" if full_mlp_forward else "0",
                    "@LAYER_OUTPUT_BASE@": str(
                        layer_output_base if full_mlp_forward else 0
                    ),
                    "@PUBLISH_LAYER_OUTPUT_AT_PAGE_ZERO@": (
                        "1" if publish_layer_output_at_page_zero else "0"
                    ),
                    "@DOWN_BASE_PHASE@": str(
                        resident_down_base_k_tiles_per_phase
                    ),
                    "@DOWN_WEIGHT_PACK_BASE@": str(down_weight_pack_base),
                    "@DOWN_A_PACK_BASE@": str(down_a_pack_base),
                    "@DOWN_B_PACK_BASE@": str(down_b_pack_base),
                    "@PRODUCT_JOBS@": str(seq_tiles * intermediate_tiles),
                    "@PRODUCT_JOBS_PER_CORE@": str(
                        -(-(seq_tiles * intermediate_tiles) // 60)
                    ),
                    "@OUTPUT_JOBS@": str(seq_tiles * hidden_tiles),
                    "@OUTPUT_JOBS_PER_CORE@": str(
                        -(-(seq_tiles * hidden_tiles) // 60)
                    ),
                    "@M_TILES@": str(seq_tiles),
                    "@H_TILES@": str(hidden_tiles),
                    "@I_TILES@": str(intermediate_tiles),
                    "@RANK_TILES@": str(rank_tiles),
                    "@BASE_K_BLOCK_TILES@": str(
                        resident_down_base_k_tiles_per_phase
                    ),
                    "@NUM_BASE_K_BLOCKS@": str(
                        intermediate_tiles
                        // resident_down_base_k_tiles_per_phase
                    ),
                    "@NUM_H_K_BLOCKS@": str(intermediate_tiles // 2),
                    "@H_K_BLOCK_TILES@": "2",
                    "@FP32_ACTIVATION@": "false",
                    "@DECOMPOSED_ACTIVATION@": "true",
                    "@GATE_UP_WEIGHT_PACK_BASE@": str(
                        gate_up_weight_pack_base
                    ),
                    "@GATE_A_PACK_BASE@": str(gate_a_pack_base),
                    "@GATE_B_PACK_BASE@": str(gate_b_pack_base),
                    "@UP_A_PACK_BASE@": str(up_a_pack_base),
                    "@UP_B_PACK_BASE@": str(up_b_pack_base),
                    "@MLP_GATE_BASE@": str(gate_base),
                    "@MLP_UP_BASE@": str(up_base),
                    "@MLP_GATE_H_BASE@": str(gate_h_base),
                    "@MLP_UP_H_BASE@": str(up_h_base),
                    "@ALIASED_PROJECTION_CBS@": (
                        "1" if input_rmsnorm else "0"
                    ),
                    "@PADDED_HIDDEN_TILES@": str(padded_hidden_tiles),
                    "@HIDDEN_FEATURES@": str(hidden_features),
                    "@RMSNORM_EPSILON_BITS@": str(
                        struct.unpack(
                            "<I", struct.pack(
                                "<f", float(input_rmsnorm_epsilon or 1.0)
                            )
                        )[0]
                    ),
                    "@POST_RMSNORM_EPSILON_BITS@": str(
                        struct.unpack(
                            "<I", struct.pack(
                                "<f", float(
                                    post_attention_rmsnorm_epsilon or 1.0
                                )
                            )
                        )[0]
                    ),
                    "@O_H_X_CB@": "0" if input_rmsnorm else "15",
                    "@O_A_CB@": "1" if input_rmsnorm else "16",
                    "@O_H_CB@": "3" if input_rmsnorm else "18",
                    "@O_BASE_X_CB@": "5" if input_rmsnorm else "20",
                    "@O_W_CB@": "6" if input_rmsnorm else "21",
                    "@O_B_CB@": "9" if input_rmsnorm else "24",
                    "@O_SAVED_H_CB@": "4" if input_rmsnorm else "19",
                    "@O_OUTPUT_CB@": "12" if input_rmsnorm else "27",
                    "@HAS_BIAS@": "1" if has_bias else "0",
                    "@QKV_RANK_TILES@": str(grouped_qkv_rank_tiles),
                    "@QKV_STORAGE_RANK_TILES@": str(
                        grouped_qkv_storage_rank_tiles
                    ),
                    "@GROUPED_QKV_LORA@": "1" if grouped_qkv_lora else "0",
                    "@SMOL_GROUPED_QKV_LORA@": (
                        "1" if smol_grouped_qkv_lora else "0"
                    ),
                    "@PRECOMPUTED_QKV_H@": (
                        "1" if precomputed_grouped_qkv_h else "0"
                    ),
                    "@PRECOMPUTED_QKV_BASE@": (
                        "1" if precomputed_grouped_qkv_base else "0"
                    ),
                    "@PRECOMPUTED_QKV_DELTA@": (
                        "1" if precomputed_separate_qkv_delta else "0"
                    ),
                    "@BASE_PHASE@": str(base_k_tiles_per_phase),
                    "@LORA_PHASE@": str(lora_phase),
                    "@O_LORA_PHASE@": str(o_lora_phase),
                    "@Q_BASE@": str(plan.arena.slice("query").page_offset),
                    "@K_BASE@": str(plan.arena.slice("key").page_offset),
                    "@V_BASE@": str(plan.arena.slice("value").page_offset),
                    "@HEAD_BASE@": str(plan.arena.slice("head_output").page_offset),
                    "@PROBABILITY_BASE@": str(
                        plan.arena.slice("probability_fp32").page_offset
                    ),
                    "@HQ_BASE@": str(plan.arena.slice("q_rank_hidden").page_offset),
                    "@HK_BASE@": str(plan.arena.slice("k_rank_hidden").page_offset),
                    "@HV_BASE@": str(plan.arena.slice("v_rank_hidden").page_offset),
                    "@HO_BASE@": str(plan.arena.slice("o_rank_hidden").page_offset),
                    "@OUTPUT_BASE@": str(plan.arena.slice("output").page_offset),
                    "@POST_RESIDUAL_BASE@": str(
                        plan.arena.slice("post_attention_residual").page_offset
                        if post_attention_block else 0
                    ),
                    "@POST_NORMALIZED_BASE@": str(
                        plan.arena.slice("post_attention_normalized").page_offset
                        if post_attention_block else 0
                    ),
                    "@POST_RSTD_BASE@": str(
                        plan.arena.slice("post_attention_rstd_fp32").page_offset
                        if post_attention_block else 0
                    ),
                    "@RESET_O_GO@": (
                        "volatile tt_l1_ptr uint32_t* o_go =\n"
                        "        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(\n"
                        "            get_semaphore(10));\n"
                        "    noc_semaphore_set(o_go, 0);"
                    ),
                    "@PUBLISH_TARGET_MAILBOX@": (
                        "if (node == 0) {\n"
                        "        volatile tt_l1_ptr uint32_t* q_mailbox =\n"
                        "            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(\n"
                        "                get_semaphore(5));\n"
                        "        volatile tt_l1_ptr uint32_t* k_mailbox =\n"
                        "            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(\n"
                        "                get_semaphore(6));\n"
                        "        volatile tt_l1_ptr uint32_t* v_mailbox =\n"
                        "            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(\n"
                        "                get_semaphore(7));\n"
                        "        noc_semaphore_set(q_mailbox, q.get_write_ptr());\n"
                        "        noc_semaphore_set(k_mailbox, k.get_write_ptr());\n"
                        "        noc_semaphore_set(v_mailbox, v.get_write_ptr());\n"
                        "        noc_async_write(get_semaphore(5),\n"
                        "            projection_zero_l1(get_semaphore(5)), sizeof(uint32_t));\n"
                        "        noc_async_write(get_semaphore(6),\n"
                        "            projection_zero_l1(get_semaphore(6)), sizeof(uint32_t));\n"
                        "        noc_async_write(get_semaphore(7),\n"
                        "            projection_zero_l1(get_semaphore(7)), sizeof(uint32_t));\n"
                        "        noc_async_write_barrier();\n"
                        "    }"
                    ),
                    "@HEAD_TO_O_HANDOFF@": (
                        "volatile tt_l1_ptr uint32_t* o_go =\n"
                        "        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(\n"
                        "            get_semaphore(10));\n"
                        "    noc_semaphore_wait(o_go, 1);\n"
                        "    noc_semaphore_set(o_go, 0);\n"
                        f"    constexpr uint32_t hidden_tiles = {hidden_tiles};\n"
                        "    constexpr uint32_t o_jobs = seq_tiles * hidden_tiles;\n"
                        f"    constexpr uint32_t o_per_node = {o_jobs_per_worker};\n"
                        "    const uint32_t o_hx_base =\n"
                        "        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(\n"
                        "            get_semaphore(8));\n"
                        "    const uint32_t o_bx_base =\n"
                        "        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(\n"
                        "            get_semaphore(9));\n"
                        "    for (uint32_t dim = 0; dim < dim_tiles; ++dim) {\n"
                        "        const uint32_t k_tile = head_index * dim_tiles + dim;\n"
                        "        const uint32_t source = head.get_read_ptr() + dim * tile_bytes;\n"
                        "        for (uint32_t output_tile = 0; output_tile < hidden_tiles;\n"
                        "             ++output_tile) {\n"
                        "            const uint32_t job = row * hidden_tiles + output_tile;\n"
                        "            const uint32_t target = job / o_per_node;\n"
                        "            const uint32_t slot = job - target * o_per_node;\n"
                        "            const uint32_t offset =\n"
                        "                (slot * hidden_tiles + k_tile) * tile_bytes;\n"
                        "            const uint32_t o_hx_address = o_hx_base + offset;\n"
                        "            const uint32_t o_bx_address = o_bx_base + offset;\n"
                        "            noc_async_write(source, projection_l1(target, o_hx_address),\n"
                        "                            tile_bytes);\n"
                        "            noc_async_write(source, projection_l1(target, o_bx_address),\n"
                        "                            tile_bytes);\n"
                        "        }\n"
                        "    }\n"
                        "    noc_async_write_barrier();\n"
                        "    for (uint32_t output_tile = 0; output_tile < hidden_tiles;\n"
                        "         ++output_tile) {\n"
                        "        const uint32_t job = row * hidden_tiles + output_tile;\n"
                        "        const uint32_t target = job / o_per_node;\n"
                        "        for (uint32_t dim = 0; dim < dim_tiles; ++dim) {\n"
                        "            noc_semaphore_inc(projection_l1(target, get_semaphore(11)), 1);\n"
                        "            noc_semaphore_inc(projection_l1(target, get_semaphore(12)), 1);\n"
                        "        }\n"
                        "    }\n"
                        "    noc_async_atomic_barrier();"
                    ),
                }
                if dedicated_o_rank_producer:
                    replacements["@HEAD_TO_O_HANDOFF@"] = (
                        "volatile tt_l1_ptr uint32_t* o_go =\n"
                        "        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(\n"
                        "            get_semaphore(10));\n"
                        "    volatile tt_l1_ptr uint32_t* h_ready =\n"
                        "        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(\n"
                        "            get_semaphore(12));\n"
                        "    noc_semaphore_wait(o_go, 1);\n"
                        "    noc_semaphore_set(o_go, 0);\n"
                        "    noc_semaphore_wait(h_ready, 1);\n"
                        "    noc_semaphore_set(h_ready, 0);\n"
                        f"    constexpr uint32_t hidden_tiles = {hidden_tiles};\n"
                        "    constexpr uint32_t o_jobs = seq_tiles * hidden_tiles;\n"
                        f"    constexpr uint32_t o_per_node = {o_jobs_per_worker};\n"
                        "    const uint32_t h_input_base =\n"
                        "        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(\n"
                        "            get_semaphore(8));\n"
                        "    const uint32_t o_bx_base =\n"
                        "        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(\n"
                        "            get_semaphore(9));\n"
                        "    for (uint32_t dim = 0; dim < dim_tiles; ++dim) {\n"
                        "        const uint32_t k_tile = head_index * dim_tiles + dim;\n"
                        "        const uint32_t source = head.get_read_ptr() + dim * tile_bytes;\n"
                        "        noc_async_write(\n"
                        "            source, projection_l1(45 + row,\n"
                        "                h_input_base + k_tile * tile_bytes), tile_bytes);\n"
                        "        for (uint32_t output_tile = 0; output_tile < hidden_tiles;\n"
                        "             ++output_tile) {\n"
                        "            const uint32_t job = row * hidden_tiles + output_tile;\n"
                        "            const uint32_t target = job / o_per_node;\n"
                        "            const uint32_t slot = job - target * o_per_node;\n"
                        "            const uint32_t offset =\n"
                        "                (slot * hidden_tiles + k_tile) * tile_bytes;\n"
                        "            noc_async_write(\n"
                        "                source, projection_l1(target, o_bx_base + offset),\n"
                        "                tile_bytes);\n"
                        "        }\n"
                        "    }\n"
                        "    noc_async_write_barrier();\n"
                        "    for (uint32_t dim = 0; dim < dim_tiles; ++dim)\n"
                        "        noc_semaphore_inc(\n"
                        "            projection_l1(45 + row, get_semaphore(0)), 1);\n"
                        "    for (uint32_t output_tile = 0; output_tile < hidden_tiles;\n"
                        "         ++output_tile) {\n"
                        "        const uint32_t job = row * hidden_tiles + output_tile;\n"
                        "        const uint32_t target = job / o_per_node;\n"
                        "        for (uint32_t dim = 0; dim < dim_tiles; ++dim)\n"
                        "            noc_semaphore_inc(\n"
                        "                projection_l1(target, get_semaphore(12)), 1);\n"
                        "    }\n"
                        "    noc_async_atomic_barrier();"
                    )
                cpp = Path(__file__).with_name("cpp")
                projection_compute = _projection_compute_source(
                            sequence_tiles=seq_tiles,
                            hidden_tiles=hidden_tiles,
                            q_tiles=hidden_tiles,
                            kv_tiles=kv_tiles,
                            base_phase=base_k_tiles_per_phase,
                            lora_phase=lora_phase,
                            o_lora_phase=o_lora_phase,
                            o_lora_init_schedule=o_lora_init_schedule,
                            scale_bits=lora_scale_bits,
                            has_bias=(
                                has_bias and not precomputed_grouped_qkv_base
                            ),
                            bias_after_bf16_base=bias_after_bf16_base,
                            grouped_qkv_rank_tiles=grouped_qkv_rank_tiles,
                            o_projection_workers=o_projection_workers,
                            projection_workers=projection_workers,
                            precomputed_grouped_qkv_h=(
                                precomputed_grouped_qkv_h
                            ),
                            precomputed_grouped_qkv_base=(
                                precomputed_grouped_qkv_base
                            ),
                            precomputed_grouped_qkv_delta=(
                                precomputed_separate_qkv_delta
                            ),
                            o_cb_offset=0 if input_rmsnorm else 15,
                            qkv_o_phase_barrier=input_rmsnorm,
                            precomputed_o_rank_h=dedicated_o_rank_producer,
                            include_o_projection=not projection_rope_prefix_only,
                        )
                if input_rmsnorm:
                    projection_compute = _compose_input_rmsnorm_projection_compute(
                        _input_rmsnorm_compute_source(
                            hidden_features=hidden_features,
                            sequence_length=sequence_length,
                            epsilon=float(input_rmsnorm_epsilon),
                        ),
                        projection_compute,
                        post_attention_block,
                        post_attention_diagnostic_stage,
                    )
                projection_reader = (
                    cpp / "attention_forward_qkv_resident_reader.cpp"
                ).read_text()
                projection_writer = (
                    cpp / "attention_forward_qkv_resident_writer.cpp"
                ).read_text()
                if resident_gate_up:
                    projection_writer = _compose_gate_up_writer(
                        projection_writer, projection_island=True,
                        full_mlp_forward=full_mlp_forward,
                    )
                projection_sources = (
                    projection_reader,
                    _compute_completion_fence(
                        projection_compute,
                        release_cb="get_compile_time_arg_val(28)",
                        acknowledgement_cb="get_compile_time_arg_val(29)",
                        # Compact transformer stacks invoke this projection
                        # topology repeatedly inside one artifact-v4 epoch.
                        # Its reader/writer views already end empty; opt in so
                        # the three compute RISCs also return their CB views
                        # to the physical origin when RMSNorm is a separate
                        # preceding child.
                        rebase_projection_cbs=(
                            input_rmsnorm or rebase_projection_cbs
                        ),
                    ),
                    projection_writer,
                )
                attention_compute_body = _resident_compute_source(
                    replacements,
                    projection_rope_prefix_only=projection_rope_prefix_only,
                ).replace(
                    "get_absolute_logical_y() * 11 + get_absolute_logical_x()",
                    "get_absolute_logical_y() * 6 + "
                    "(get_absolute_logical_x() - 5)",
                )
                if resident_gate_up:
                    attention_compute_body = _make_attention_compute_all_core(
                        attention_compute_body
                    )
                    attention_compute_body = (
                        _append_full_mlp_compute_helpers(attention_compute_body)
                        if full_mlp_forward else
                        _append_gate_up_compute_helpers(attention_compute_body)
                    )
                attention_compute = _compute_completion_fence(
                    attention_compute_body,
                    release_cb="28" if resident_gate_up else "0",
                    acknowledgement_cb=(
                        "31" if resident_gate_up and full_mlp_forward else
                        "29" if resident_gate_up else "1"
                    ),
                    resident_continuation=(
                        (
                            "    run_resident_layer_paired_gate_up_swiglu();\n"
                            "    resident_layer_swiglu_down_compute::run();\n"
                            if full_mlp_forward else
                            "    run_resident_layer_gate_up_compute();\n"
                        )
                        if resident_gate_up else ""
                    ),
                )
                attention_reader = (
                    cpp / "attention_forward_handoff_resident_reader.cpp"
                ).read_text()
                attention_writer = (
                    cpp / "attention_forward_resident_writer.cpp"
                ).read_text().replace(
                    "get_absolute_logical_y() * 11 + get_absolute_logical_x()",
                    "get_absolute_logical_y() * 6 + "
                    "(get_absolute_logical_x() - 5)",
                )
                if resident_gate_up:
                    attention_reader = _compose_gate_up_reader(
                        attention_reader,
                        attention_island=True,
                        total_cbs=total_cbs,
                        full_mlp_forward=full_mlp_forward,
                    )
                    attention_writer = _compose_gate_up_writer(
                        attention_writer, projection_island=False,
                        full_mlp_forward=full_mlp_forward,
                    )
                attention_reader = attention_reader.replace(
                    "get_tensor_accessor_args_cta_offset<3, 32>()",
                    f"get_tensor_accessor_args_cta_offset<16, {total_cbs}>()",
                ).replace(
                    "get_tensor_accessor_args_cta_offset<4, 32>()",
                    f"get_tensor_accessor_args_cta_offset<17, {total_cbs}>()",
                ).replace(
                    "get_tensor_accessor_args_cta_offset<5, 32>()",
                    f"get_tensor_accessor_args_cta_offset<18, {total_cbs}>()",
                )
                attention_writer = attention_writer.replace(
                    "get_tensor_accessor_args_cta_offset<6, 32>()",
                    f"get_tensor_accessor_args_cta_offset<21, {total_cbs}>()",
                ).replace(
                    "constexpr uint32_t q_pages = query_heads * seq_tiles * dim_tiles;",
                    "constexpr uint32_t q_pages = query_heads * seq_tiles * dim_tiles;\n"
                    "    constexpr uint32_t q_base = @Q_BASE@;\n"
                    "    constexpr uint32_t k_base = @K_BASE@;",
                ).replace(
                    "constexpr uint32_t head_base = q_pages + k_pages;",
                    "constexpr uint32_t head_base = @HEAD_BASE@;",
                ).replace(
                    "constexpr uint32_t probability_base = head_base + head_pages;",
                    "constexpr uint32_t probability_base = @PROBABILITY_BASE@;",
                ).replace(
                    ".page_id = node * dim_tiles + dim",
                    ".page_id = q_base + node * dim_tiles + dim",
                ).replace(
                    ".page_id = q_pages + kv_head * seq_tiles * dim_tiles + tile",
                    ".page_id = k_base + kv_head * seq_tiles * dim_tiles + tile",
                )
                debug_marker = (
                    "            if (node == 0) {\n"
                    "                noc1.async_write(\n"
                    "                    CoreLocalMem<uint32_t>(normalized.get_read_ptr()), arena,\n"
                    "                    kTileBytes, {}, {.page_id = @LAYER_PROBE_PAGE@});\n"
                    "                noc1.async_write_barrier();\n"
                    "            }\n"
                )
                debug_body = ""
                if debug_input_rmsnorm_checkpoint:
                    debug_body = (
                        "            for (uint32_t tile = 0; tile < hidden_tiles; ++tile) {\n"
                        "                noc1.async_write(\n"
                        "                    CoreLocalMem<uint32_t>(normalized.get_read_ptr() + tile * kTileBytes),\n"
                        "                    arena, kTileBytes, {},\n"
                        "                    {.page_id = @INPUT_RMSNORM_DEBUG_BASE@ + node * hidden_tiles + tile});\n"
                        "            }\n"
                        "            noc1.async_write_barrier();\n"
                    )
                # The compact TT-Lang child artifact is reused while resolving
                # the paired direct/base owner graph.  Its C++ sources are
                # intentionally patched in place, so the second resolution can
                # already contain the requested debug body (or no body at all).
                # Treat that state as idempotent instead of rejecting an
                # otherwise identical second custom call.
                if debug_marker in attention_writer:
                    attention_writer = attention_writer.replace(
                        debug_marker, debug_body, 1
                    )
                elif debug_body and debug_body not in attention_writer:
                    raise RuntimeError("input RMSNorm debug marker changed")
                source_specs = [
                    ("attention_forward_qkv_projection_reader.cpp", "noc", projection_sources[0]),
                    ("attention_forward_qkv_projection_compute.cpp", "compute", projection_sources[1]),
                    ("attention_forward_qkv_projection_writer.cpp", "noc", projection_sources[2]),
                    ("attention_forward_qkv_attention_reader.cpp", "noc", attention_reader),
                    ("attention_forward_qkv_attention_compute.cpp", "compute", attention_compute),
                    ("attention_forward_qkv_attention_writer.cpp", "noc", attention_writer),
                ]
                if dedicated_o_rank_producer:
                    source_specs.extend(
                        [
                            (
                                "attention_forward_o_rank_reader.cpp",
                                "noc",
                                (cpp / "attention_forward_o_rank_reader.cpp").read_text(),
                            ),
                            (
                                "attention_forward_o_rank_compute.cpp",
                                "compute",
                                _canonical_ttnn_o_rank_compute_source(
                                    hidden_tiles
                                ),
                            ),
                            (
                                "attention_forward_o_rank_writer.cpp",
                                "noc",
                                (cpp / "attention_forward_o_rank_writer.cpp").read_text(),
                            ),
                        ]
                    )
                generated = Path(compiled.kernel_paths[0][0]).parent
                paths = []
                for filename, thread, source in source_specs:
                    for marker, value in replacements.items():
                        source = source.replace(marker, value)
                    unresolved = sorted(set(re.findall(r"@[A-Z0-9_]+@", source)))
                    if unresolved:
                        raise RuntimeError(
                            "unexpanded QKV resident marker in "
                            f"{filename}: {unresolved}"
                        )
                    path = generated / filename
                    path.write_text(source, encoding="utf-8")
                    paths.append((str(path), thread))

                from ttl.dataflow_buffer import CompilerAllocatedDFBConfig
                attention_capacities = (
                    dim_tiles, dim_tiles, dim_tiles,
                    seq_tiles * dim_tiles, seq_tiles * dim_tiles,
                    seq_tiles * dim_tiles, dim_tiles, dim_tiles, dim_tiles,
                    dim_tiles, seq_tiles * dim_tiles, seq_tiles * dim_tiles,
                    seq_tiles * dim_tiles, seq_tiles * dim_tiles,
                    seq_tiles * dim_tiles, seq_tiles, seq_tiles, seq_tiles,
                    seq_tiles, 1, seq_tiles, 1, 1, seq_tiles, 1, 1,
                    seq_tiles, seq_tiles, seq_tiles,
                    seq_tiles * dim_tiles, dim_tiles, dim_tiles,
                )
                attention_fp32 = {19, 20, 21, 22, 23, 24, 25, 26, 27, 30}
                q_jobs = seq_tiles * hidden_tiles
                max_jobs = max(2, -(-q_jobs // projection_workers))
                projection_capacities = (
                    2 * lora_phase,
                    2 * lora_phase * grouped_qkv_rank_tiles,
                    2 * grouped_qkv_rank_tiles,
                    2 * grouped_qkv_rank_tiles,
                    grouped_qkv_rank_tiles,
                    2 * base_k_tiles_per_phase,
                    2 * base_k_tiles_per_phase,
                    2, max_jobs, 2 * grouped_qkv_rank_tiles, 2,
                    max_jobs, 2,
                    max_jobs, max_jobs,
                )
                o_projection_capacities = (
                    o_jobs_per_worker * hidden_tiles,
                    2 * o_lora_phase * rank_tiles,
                    2 * rank_tiles, 2 * rank_tiles, rank_tiles,
                    o_jobs_per_worker * hidden_tiles,
                    2 * base_k_tiles_per_phase, 2,
                    o_jobs_per_worker, 2 * rank_tiles, 2,
                    o_jobs_per_worker, 2,
                    o_jobs_per_worker, o_jobs_per_worker,
                )
                if input_rmsnorm:
                    aliased_projection = tuple(
                        max(qkv, o) for qkv, o in zip(
                            projection_capacities, o_projection_capacities
                        )
                    )
                    projection_physical_capacities = (
                        aliased_projection
                        + (
                            padded_hidden_tiles, padded_hidden_tiles,
                            # The accepted generic RMSNorm schedule uses FP32
                            # scalar FIFOs for both reduction scale and epsilon.
                            # Leave a second raw-tile window so the write is not
                            # a full-ring generation in this resident program.
                            2, 2, 2, 8, 1, padded_hidden_tiles,
                            2, 2, 2, padded_hidden_tiles, 8,
                            1, 1, 1, 1,
                        )
                    )
                    if post_attention_block:
                        post_capacities = list(
                            projection_physical_capacities
                        )
                        post_capacities[13] = max(
                            post_capacities[13], padded_hidden_tiles
                        )
                        projection_physical_capacities = tuple(
                            post_capacities
                        )
                    projection_fp32 = {
                        2, 7, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27,
                    }
                else:
                    projection_physical_capacities = (
                        projection_capacities + o_projection_capacities
                    )
                    projection_fp32 = {2, 7, 17, 22}
                attention_capacities = list(attention_capacities)
                # The canonical model path promotes RoPE Q/K and V to FP32,
                # keeps QK/mask/softmax/P@V in FP32, and packs only the final
                # head. CB19/20/23/30 are deliberately reused between phases;
                # retain a spare tile so a full-ring generation is never
                # mistaken for an empty producer view.
                attention_capacities[18] = max(
                    attention_capacities[18], seq_tiles + 1
                )
                if publish_raw_score_checkpoint:
                    # CB16 is unused by the canonical FP32 path.  Preserve an
                    # exact copy of raw QK there until the writer publishes it
                    # into the private probability-sized arena region.
                    attention_capacities[16] = max(
                        attention_capacities[16], seq_tiles
                    )
                    attention_fp32.add(16)
                attention_capacities[19] = max(
                    attention_capacities[19],
                    dim_tiles + seq_tiles * dim_tiles + 1,
                )
                attention_capacities[20] = max(
                    attention_capacities[20],
                    2 * seq_tiles * dim_tiles + seq_tiles + 1,
                )
                attention_capacities[23] = max(
                    attention_capacities[23], 2 * seq_tiles + 1
                )
                attention_capacities[30] = max(
                    attention_capacities[30], dim_tiles + 1
                )
                attention_fp32.add(18)
                attention_fp32.add(28)
                projection_physical_capacities = list(
                    projection_physical_capacities
                )
                if resident_gate_up:
                    gate_jobs = seq_tiles * (
                        intermediate_tiles if full_mlp_forward
                        else 2 * intermediate_tiles
                    )
                    gate_jobs_per_core = -(-gate_jobs // 60)
                    maximum_cached_rows = 0
                    maximum_local_groups = 0
                    for worker in range(60):
                        first = worker * gate_jobs_per_core
                        stop = min(first + gate_jobs_per_core, gate_jobs)
                        if stop <= first:
                            continue
                        maximum_cached_rows = max(
                            maximum_cached_rows,
                            (stop - 1) // (
                                intermediate_tiles if full_mlp_forward
                                else 2 * intermediate_tiles
                            )
                            - first // (
                                intermediate_tiles if full_mlp_forward
                                else 2 * intermediate_tiles
                            ) + 1,
                        )
                        maximum_local_groups = max(
                            maximum_local_groups,
                            (
                                2 * (
                                    (stop - 1) // intermediate_tiles
                                    - first // intermediate_tiles + 1
                                )
                                if full_mlp_forward else
                                (stop - 1) // intermediate_tiles
                                - first // intermediate_tiles + 1
                            ),
                        )
                    gate_required_blocks = {
                        # One payload-domain generation word follows each
                        # target's complete normalized-X cache. Keeping it in
                        # the same CB0 L1 range makes the posted data/sentinel
                        # ordering valid across the projection/attention grids.
                        0: maximum_cached_rows * hidden_tiles + 1,
                        4: 1,
                        6: 2 * resident_gate_up_base_k_tiles_per_phase,
                        9: 2,
                        10: 2,
                        12: 2,
                        13: max(
                            2,
                            2 * gate_jobs_per_core
                            if full_mlp_forward else gate_jobs_per_core,
                        ),
                        14: maximum_local_groups,
                        15: hidden_tiles,
                        16: max(
                            2,
                            2 * gate_jobs_per_core
                            if full_mlp_forward else gate_jobs_per_core,
                        ),
                        23: 2,
                        24: 2,
                    }
                    for cb_index, blocks in gate_required_blocks.items():
                        attention_capacities[cb_index] = max(
                            attention_capacities[cb_index], blocks
                        )
                if full_mlp_forward:
                    product_jobs_per_core = -(
                        -(seq_tiles * intermediate_tiles) // 60
                    )
                    output_jobs_per_core = -(
                        -(seq_tiles * hidden_tiles) // 60
                    )
                    down_required_blocks = {
                        0: 2 * resident_down_base_k_tiles_per_phase,
                        1: 2,
                        3: max(2, product_jobs_per_core + 1),
                        4: 1,
                        # Keep one physical tile outside the largest down-row
                        # reservation. Reused full-ring CB generations can be
                        # indistinguishable from empty to the producer-side
                        # modulo pointer check when the reservation exactly
                        # equals capacity, even after counters are rebased.
                        5: output_jobs_per_core * intermediate_tiles + 1,
                        6: 1,
                        7: 4,
                        8: max(2, output_jobs_per_core),
                        9: 2,
                        10: 2,
                        11: max(2, output_jobs_per_core),
                        12: max(
                            2 * product_jobs_per_core + 1,
                            output_jobs_per_core,
                        ),
                        13: 2,
                        14: output_jobs_per_core,
                        17: 2 * product_jobs_per_core,
                        18: product_jobs_per_core,
                        19: 2,
                        20: 2,
                        28: output_jobs_per_core,
                        29: max(2, output_jobs_per_core),
                    }
                    for cb_index, blocks in down_required_blocks.items():
                        attention_capacities[cb_index] = max(
                            attention_capacities[cb_index], blocks
                        )
                if input_rmsnorm:
                    # Q/K/V and O reuse projection CB12. O publishes one tile
                    # per local job; avoid an exact full-ring generation after
                    # the in-program QKV rebase.
                    projection_physical_capacities[12] = max(
                        projection_physical_capacities[12],
                        max_jobs + 1,
                        o_jobs_per_worker + 1,
                    )
                for index, blocks in enumerate(attention_capacities):
                    fp32 = index in attention_fp32
                    config = CompilerAllocatedDFBConfig(
                        dfb_index=index,
                        num_tiles=1,
                        data_format="float32" if fp32 else "bfloat16",
                        block_count=blocks,
                    )
                    config._cb_index = index
                    compiled.cb_configs[index] = config
                for local_index, blocks in enumerate(
                    projection_physical_capacities
                ):
                    if local_index in (28, 29, 30, 31):
                        blocks = 1
                    config = CompilerAllocatedDFBConfig(
                        dfb_index=local_index,
                        num_tiles=1,
                        data_format=(
                            "float32" if local_index in projection_fp32
                            else "bfloat16"
                        ),
                        block_count=blocks,
                    )
                    config._cb_index = local_index
                    compiled.cb_configs.append(config)
                if dedicated_o_rank_producer:
                    # Use TTNN's canonical logical matmul ports: input zero,
                    # input one, BF16 output and FP32 partial are CB0/1/4/5.
                    # These identities are part of the compiled compute
                    # contract; the core range is disjoint from the other
                    # resident islands, so duplicate logical IDs are safe.
                    for local_index, blocks, fp32 in (
                        (0, hidden_tiles, False),
                        (1, hidden_tiles, False),
                        (4, 1, False),
                        (5, 2, True),
                    ):
                        config = CompilerAllocatedDFBConfig(
                            dfb_index=local_index,
                            num_tiles=1,
                            data_format="float32" if fp32 else "bfloat16",
                            block_count=blocks,
                        )
                        config._cb_index = local_index
                        compiled.cb_configs.append(config)

                import ttnn

                compiled.kernel_paths = paths
                compiled.kernel_configs = [
                    ttnn.ReaderConfigDescriptor(),
                    ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        math_approx_mode=False,
                        fp32_dest_acc_en=True,
                    ),
                    ttnn.WriterConfigDescriptor(),
                    ttnn.ReaderConfigDescriptor(),
                    ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        math_approx_mode=True,
                        fp32_dest_acc_en=True,
                    ),
                    ttnn.WriterConfigDescriptor(),
                ]
                if dedicated_o_rank_producer:
                    compiled.kernel_configs.extend(
                        [
                            ttnn.ReaderConfigDescriptor(),
                            ttnn.ComputeConfigDescriptor(
                                math_fidelity=ttnn.MathFidelity.HiFi4,
                                math_approx_mode=True,
                                fp32_dest_acc_en=True,
                            ),
                            ttnn.WriterConfigDescriptor(),
                        ]
                    )
                if input_rmsnorm:
                    # Preserve the accepted standalone RMSNorm numerical
                    # descriptor after remapping its physical CB identities
                    # into the projection island.  These FP32 values are
                    # consumed by SFPU/copy-to-DEST operations and must use
                    # direct FP32 unpack.  The accurate mean also consumes
                    # remapped square CB26 through SFPU; only its scaler CB
                    # retains the default reduction-protocol intake mode.
                    projection_compute_config = compiled.kernel_configs[1]
                    while len(projection_compute_config.unpack_to_dest_mode) < 64:
                        projection_compute_config.unpack_to_dest_mode.append(
                            ttnn.UnpackToDestMode.Default
                        )
                    # Exact accepted descriptor: CB17 and CB26 retain the
                    # reduction/FPU intake mode; scalar SFPU intermediates use
                    # direct FP32-to-DEST unpack.
                    rmsnorm_full_fp32_cbs = {
                        18, 19, 20, 21, 23, 24, 25, 27,
                    }
                    for cb_index in rmsnorm_full_fp32_cbs:
                        projection_compute_config.unpack_to_dest_mode[cb_index] = (
                            ttnn.UnpackToDestMode.UnpackToDestFp32
                        )
                compiled.kernel_tensor_indices = [
                    list(range(16)) + [19, 20], [], [21],
                    ([16, 17, 20] if resident_gate_up
                     else [16, 17, 18]), [], [21],
                ]
                if dedicated_o_rank_producer:
                    compiled.kernel_tensor_indices.extend(
                        [[14], [], [21]]
                    )
                print(
                    "TT_LANG_DENSE_TABLE_DEBUG",
                    "num_tensors=", getattr(compiled, "num_tensors", None),
                    "kernel_tensor_indices=", compiled.kernel_tensor_indices,
                    flush=True,
                )
                projection_range = (
                    _DEDICATED_O_MAIN_RANGE
                    if dedicated_o_rank_producer else _PROJECTION_RANGE
                )
                compiled.kernel_core_ranges = [
                    projection_range, projection_range, projection_range,
                    _ATTENTION_RANGE, _ATTENTION_RANGE, _ATTENTION_RANGE,
                ]
                if dedicated_o_rank_producer:
                    compiled.kernel_core_ranges.extend(
                        [_DEDICATED_O_RANK_RANGE] * 3
                    )
                compiled.kernel_noc_roles = [
                    "reader", None, "writer", "reader", None, "writer",
                ]
                if dedicated_o_rank_producer:
                    compiled.kernel_noc_roles.extend(
                        ["reader", None, "writer"]
                    )
                compiled.cb_core_ranges = (
                    [_ATTENTION_RANGE] * _ATTENTION_CBS
                    + [projection_range] * len(
                        projection_physical_capacities
                    )
                    + (
                        [_DEDICATED_O_RANK_RANGE] * 4
                        if dedicated_o_rank_producer else []
                    )
                )
                compiled.semaphore_configs = [
                    {
                        "id": semaphore,
                        "core_type": "worker",
                        "core_range": _ALL_WORKERS_RANGE,
                        "initial_value": 0,
                    }
                    for semaphore in range(16)
                ]
                compiled.num_pipe_sync_semaphores = 16
                compiled.kernel_arg_specs = [
                    [] for _ in range(len(compiled.kernel_paths))
                ]
            break
        return result

    operation.output_shape = (
        layer_arena_shape if resident_gate_up else plan.arena.physical_shape
    )
    operation.forward_plan = plan
    operation.intermediate_dram_bytes = 0
    operation.checkpoint_dram_bytes = (
        (layer_used_pages - plan.arena.output_pages) * 2048
        if resident_gate_up else plan.arena.checkpoint_dram_bytes
    )
    operation.reverse_handoff_noc_bytes = (
        (
            query_heads * seq_tiles * dim_tiles * hidden_tiles
            + query_heads * seq_tiles * dim_tiles
            + seq_tiles * hidden_tiles
        ) * 2048
        if dedicated_o_rank_producer else
        query_heads * seq_tiles * dim_tiles * hidden_tiles * 2 * 2048
    )
    local_jobs = max(
        2, -(-(seq_tiles * hidden_tiles) // projection_workers)
    )
    qkv_caps = (
        2 * lora_phase, 2 * lora_phase * grouped_qkv_rank_tiles,
        2 * grouped_qkv_rank_tiles,
        2 * grouped_qkv_rank_tiles,
        grouped_qkv_rank_tiles,
        2 * base_k_tiles_per_phase, 2 * base_k_tiles_per_phase,
        2, local_jobs, 2 * grouped_qkv_rank_tiles, 2,
        local_jobs, 2, local_jobs, local_jobs,
    )
    o_caps = (
        o_jobs_per_worker * hidden_tiles,
        2 * o_lora_phase * rank_tiles,
        2 * rank_tiles, 2 * rank_tiles, rank_tiles,
        o_jobs_per_worker * hidden_tiles, 2 * base_k_tiles_per_phase, 2,
        o_jobs_per_worker, 2 * rank_tiles, 2, o_jobs_per_worker, 2,
        o_jobs_per_worker, o_jobs_per_worker,
    )
    if input_rmsnorm:
        aliased = tuple(max(qkv, o) for qkv, o in zip(qkv_caps, o_caps))
        capacities = aliased + (
            padded_hidden_tiles, padded_hidden_tiles,
            2, 2, 2, 8, 1, padded_hidden_tiles,
            2, 2, 2, padded_hidden_tiles, 8,
            1, 1, 1, 1,
        )
        if post_attention_block:
            capacities_list = list(capacities)
            capacities_list[13] = max(
                capacities_list[13], padded_hidden_tiles
            )
            capacities = tuple(capacities_list)
        fp32 = {2, 7, 18, 19, 20, 21, 23, 24, 25, 26, 27}
    else:
        capacities = qkv_caps + o_caps
        fp32 = {2, 7, 17, 22}
    operation.projection_cb_bytes = sum(
        blocks * (4096 if index in fp32 else 2048)
        for index, blocks in enumerate(capacities)
    )
    if dedicated_o_rank_producer:
        operation.projection_cb_bytes += (
            hidden_tiles * 2048 * 2 + 2 * 4096 + 2048
        )
    operation.dedicated_o_rank_producer = bool(
        dedicated_o_rank_producer
    )
    operation.projection_workers = projection_workers
    operation.input_rmsnorm_resident = input_rmsnorm
    operation.rebase_projection_cbs = bool(rebase_projection_cbs)
    operation.separate_qkv_lora = bool(separate_qkv_lora)
    operation.publishes_raw_score_checkpoint = bool(
        publish_raw_score_checkpoint
    )
    operation.o_projection_workers = o_projection_workers
    operation.o_jobs_per_worker = o_jobs_per_worker
    operation.input_rmsnorm_epsilon = input_rmsnorm_epsilon
    operation.post_attention_rmsnorm_resident = post_attention_block
    operation.post_attention_rmsnorm_epsilon = post_attention_rmsnorm_epsilon
    operation.resident_gate_up = resident_gate_up
    operation.resident_full_mlp_forward = full_mlp_forward
    operation.input_rmsnorm_debug_page_offset = (
        input_rmsnorm_debug_base if debug_input_rmsnorm_checkpoint else None
    )
    operation.resident_gate_up_intermediate_features = (
        resident_gate_up_intermediate_features
    )
    operation.mlp_gate_page_offset = gate_base if resident_gate_up else None
    operation.mlp_up_page_offset = up_base if resident_gate_up else None
    operation.mlp_gate_h_page_offset = (
        gate_h_base if resident_gate_up else None
    )
    operation.mlp_up_h_page_offset = up_h_base if resident_gate_up else None
    operation.layer_output_page_offset = (
        layer_output_base if full_mlp_forward else None
    )
    operation.hidden_page_base = hidden_page_base
    operation.publishes_layer_output_at_page_zero = bool(
        publish_layer_output_at_page_zero
    )
    operation.layer_used_pages = layer_used_pages
    operation.layer_allocated_pages = layer_allocated_pages
    operation.mlp_parameter_pack_offsets = {
        "gate_a": gate_a_pack_base,
        "gate_b": gate_b_pack_base,
        "up_a": up_a_pack_base,
        "up_b": up_b_pack_base,
        "down_a": down_a_pack_base,
        "down_b": down_b_pack_base,
    }
    operation.experimental_backend = (
        "transformer_layer_forward_full_mlp_resident_v2"
        if full_mlp_forward else (
            "transformer_layer_forward_gate_up_resident_v1"
            if resident_gate_up else (
            "attention_block_forward_input_norm_resident_v1"
            if input_rmsnorm else "attention_forward_full_resident_v1"
            )
        )
    )
    # Artifact-v4 schedules compile this already-qualified physical owner as a
    # compact child.  Exposing the private TTL entry does not alter the normal
    # one-call path; it only lets the sequence composer preserve the accepted
    # source, CB map and numerical schedule without rebuilding them.
    operation._ttl_program = _program_paired_v23
    return operation
