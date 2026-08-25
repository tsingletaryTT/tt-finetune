"""TTNN-schedule large-vocabulary log-softmax for the resident LM edge."""

from __future__ import annotations

import hashlib
import inspect
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID
from .tt_lang_lora import _FP32_DTYPE_ONLY_TENSOR


ttl = None
tt_lang_operation = None


def _ttnn_large_logsoftmax_compute_source(
    *, row_tiles: int, vocabulary_tiles: int, diagnostic_stage: str,
) -> str:
    metal_home = os.environ.get("TT_METAL_HOME")
    if not metal_home:
        raise RuntimeError("large log-softmax requires TT_METAL_HOME")
    path = (
        Path(metal_home)
        / "ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/"
        "moreh_softmax_w_large.cpp"
    )
    if not path.is_file():
        raise RuntimeError(f"TTNN large log-softmax source not found: {path}")
    source = path.read_text(encoding="utf-8")
    source = "#define SOFTMAX 1\n#define LOG 1\n#define FP32_DEST_ACC_EN 1\n" + source
    source = source.replace(
        "    uint32_t N = get_compile_time_arg_val(0);\n"
        "    uint32_t Wt = get_compile_time_arg_val(1);\n",
        "    constexpr uint32_t N = 1;\n"
        f"    constexpr uint32_t Wt = {vocabulary_tiles};\n",
        1,
    )
    marker = "void kernel_main() {\n"
    guard = (
        marker
        + "    const uint32_t tt_finetune_node =\n"
        + "        get_absolute_logical_y() * 11 + get_absolute_logical_x();\n"
        + f"    if (tt_finetune_node >= {row_tiles}) return;\n"
    )
    if source.count(marker) != 1:
        raise RuntimeError("TTNN large log-softmax entry point changed")
    source = source.replace(marker, guard, 1)
    # StableHLO explicitly converts BF16 logits to FP32 before every following
    # arithmetic operation.  Feeding c0 directly to a mixed BF16/FP32
    # broadcast-sub changes the FPU input path even when its destination is
    # FP32.  Repack each streamed tile through c24 first; c24 is empty again
    # before the exp result is produced and before the final output pass.
    original_sub = (
        "sub_tiles_bcast_cols_to_cb("
        "dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, "
        "0, 0, /*pop0=*/1, /*pop1=*/0);"
    )
    converted_sub = (
        "copy_tile_to_cb(dfb_in0_obj, dfb_exps_obj);\n"
        "                sub_tiles_bcast_cols_to_cb("
        "dfb_exps_obj, dfb_max_obj, dfb_tmp_obj, "
        "0, 0, /*pop0=*/1, /*pop1=*/0);"
    )
    if source.count(original_sub) != 6:
        raise RuntimeError("TTNN log-softmax input-subtraction schedule changed")
    source = source.replace(original_sub, converted_sub)
    max_start_marker = "        // find max\n"
    max_end_marker = "        // step 1\n"
    max_start = source.find(max_start_marker)
    max_end = source.find(max_end_marker, max_start)
    if max_start < 0 or max_end < 0:
        raise RuntimeError("TTNN log-softmax max-reduction schedule changed")
    exact_aligned_max = (
        max_start_marker
        + "        // Every supported causal-LM vocabulary is tile aligned;\n"
        + "        // reduce all full tiles in the same contiguous epoch as\n"
        + "        // StableHLO/TTNN max and do not run a masked tail phase.\n"
        + "        compute_kernel_lib::reduce<\n"
        + "            PoolType::MAX, ReduceDim::REDUCE_ROW,\n"
        + "            cb_in0, cb_max_scaler, cb_max>(\n"
        + "            compute_kernel_lib::ReduceInputBlockShape::row(Wt));\n\n"
    )
    source = source[:max_start] + exact_aligned_max + source[max_end:]
    if diagnostic_stage == "maximum":
        diagnostic = (
            "        // tt-finetune exact-schedule diagnostic boundary.\n"
            "        for (uint32_t w = 0; w < Wt; ++w) {\n"
            "            copy_tile_to_cb(\n"
            "                dfb_max_obj, dfb_out0_obj, 0, /*pop=*/0);\n"
            "        }\n"
            "        dfb_max_obj.pop_front(onetile);\n"
            "        return;\n\n"
        )
        marker = "        // step 1\n"
        if source.count(marker) != 1:
            raise RuntimeError("TTNN log-softmax step-one marker changed")
        source = source.replace(marker, diagnostic + marker, 1)
    elif diagnostic_stage in ("shifted", "exponentials"):
        stage_body = (
            "            copy_tile_to_cb(dfb_in0_obj, dfb_exps_obj);\n"
            "            sub_tiles_bcast_cols_to_cb(\n"
            "                dfb_exps_obj, dfb_max_obj, dfb_out0_obj,\n"
            "                0, 0, /*pop0=*/1, /*pop1=*/0);\n"
            if diagnostic_stage == "shifted"
            else
            "            copy_tile_to_cb(dfb_in0_obj, dfb_exps_obj);\n"
            "            sub_tiles_bcast_cols_to_cb(\n"
            "                dfb_exps_obj, dfb_max_obj, dfb_tmp_obj,\n"
            "                0, 0, /*pop0=*/1, /*pop1=*/0);\n"
            "            exp_tile_to_cb(dfb_tmp_obj, dfb_out0_obj);\n"
        )
        diagnostic = (
            "        // tt-finetune exact-schedule diagnostic boundary.\n"
            "        for (uint32_t w = 0; w < Wt; ++w) {\n"
            + stage_body
            + "        }\n"
            "        dfb_max_obj.pop_front(onetile);\n"
            "        return;\n\n"
        )
        marker = "        // step 1\n"
        if source.count(marker) != 1:
            raise RuntimeError("TTNN log-softmax step-one marker changed")
        source = source.replace(marker, diagnostic + marker, 1)
    elif diagnostic_stage == "logsum":
        diagnostic = (
            "        // tt-finetune exact-schedule diagnostic boundary.\n"
            "        for (uint32_t w = 0; w < Wt; ++w) {\n"
            "            copy_tile_to_cb(\n"
            "                dfb_recipsumexps_obj, dfb_out0_obj,\n"
            "                0, /*pop=*/0);\n"
            "        }\n"
            "        dfb_recipsumexps_obj.pop_front(onetile);\n"
            "        dfb_max_obj.pop_front(onetile);\n"
            "        return;\n\n"
        )
        marker = "        // step 3, compute final result\n"
        if source.count(marker) != 1:
            raise RuntimeError("TTNN log-softmax step-three marker changed")
        source = source.replace(marker, diagnostic + marker, 1)
    elif diagnostic_stage != "final":
        raise ValueError(f"unknown log-softmax diagnostic stage: {diagnostic_stage}")
    tmp_declaration = "    DataflowBuffer dfb_tmp_obj(cb_tmp);\n"
    if source.count(tmp_declaration) != 1:
        raise RuntimeError("TTNN log-softmax temporary declaration changed")
    source = source.replace(
        tmp_declaration,
        tmp_declaration
        + "    DataflowBuffer dfb_shifted_obj(tt::CBIndex::c_22);\n",
        1,
    )
    # c28 remains the max-reduction mask scratch with the generic reduction's
    # normal unpack mode.  c22 owns every post-max FP32 shifted tile and can
    # therefore use UnpackToDestFp32 for the accurate unary exponential.
    source = source.replace(
        "dfb_max_obj, dfb_tmp_obj", "dfb_max_obj, dfb_shifted_obj"
    )
    source = source.replace(
        "dfb_tmp_obj, dfb_recipsumexps_obj",
        "dfb_shifted_obj, dfb_recipsumexps_obj",
    )
    source = source.replace(
        "exp_tile_and_mask_tile_to_cb(\n                    dfb_tmp_obj",
        "exp_tile_and_mask_tile_to_cb(\n                    dfb_shifted_obj",
    )
    source = source.replace("exp_tile_to_cb(dfb_tmp_obj", "exp_tile_to_cb(dfb_shifted_obj")
    source = source.replace("rexp_tile_to_cb(dfb_tmp_obj", "rexp_tile_to_cb(dfb_shifted_obj")
    source = source.replace(
        "sub_tiles_bcast_cols_to_cb(",
        "tt_finetune_sfpu_sub_tiles_bcast_cols_to_cb(",
    )
    sfpu_sub = r'''
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"

ALWI void tt_finetune_sfpu_sub_tiles_bcast_cols_to_cb(
    DataflowBuffer lhs,
    DataflowBuffer reduced_rhs,
    DataflowBuffer output,
    uint32_t lhs_tile = 0,
    uint32_t rhs_tile = 0,
    uint32_t pop_lhs = 1,
    uint32_t pop_rhs = 1) {
    constexpr uint32_t one_tile = 1;
    constexpr uint32_t lhs_dst = 0;
    constexpr uint32_t rhs_dst = 1;
    DataflowBuffer expanded_rhs(tt::CBIndex::c_23);

    expanded_rhs.reserve_back(one_tile);
    reduced_rhs.wait_front(rhs_tile + 1);
    pack_reconfig_data_format(output.get_id(), expanded_rhs.get_id());
    unary_bcast_init<BroadcastType::COL>(
        reduced_rhs.get_id(), expanded_rhs.get_id());
    tile_regs_acquire();
    unary_bcast<BroadcastType::COL>(reduced_rhs.get_id(), rhs_tile, rhs_dst);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(rhs_dst, expanded_rhs.get_id());
    expanded_rhs.push_back(one_tile);
    tile_regs_release();

    lhs.wait_front(lhs_tile + 1);
    expanded_rhs.wait_front(one_tile);
    output.reserve_back(one_tile);
    pack_reconfig_data_format(expanded_rhs.get_id(), output.get_id());
#ifdef ARCH_BLACKHOLE
    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(output.get_id())));
#endif
    sub_binary_tile_init();
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(expanded_rhs.get_id(), lhs.get_id());
    copy_tile(lhs.get_id(), lhs_tile, lhs_dst);
    copy_tile_to_dst_init_short_with_dt(lhs.get_id(), expanded_rhs.get_id());
    copy_tile(expanded_rhs.get_id(), 0, rhs_dst);
    sub_binary_tile(lhs_dst, rhs_dst, lhs_dst);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(lhs_dst, output.get_id());
    tile_regs_release();
    output.push_back(one_tile);

    if (pop_lhs) lhs.pop_front(pop_lhs);
    expanded_rhs.pop_front(one_tile);
    if (pop_rhs) reduced_rhs.pop_front(pop_rhs);
}
'''
    helper_marker = "void kernel_main() {\n"
    if source.count(helper_marker) != 1:
        raise RuntimeError("TTNN log-softmax helper insertion point changed")
    source = source.replace(helper_marker, sfpu_sub + "\n" + helper_marker, 1)
    if "get_compile_time_arg_val" in source:
        raise RuntimeError("unresolved TTNN log-softmax compile-time argument")
    return source


@lru_cache(maxsize=8)
def build_tt_lang_causal_lm_logsoftmax(
    *, tokens: int, vocabulary_size: int, diagnostic_stage: str = "final",
) -> Callable:
    if tokens <= 0 or tokens % 32:
        raise ValueError("log-softmax tokens must be positive /32")
    if vocabulary_size <= 0 or vocabulary_size % 32:
        raise ValueError("log-softmax vocabulary must be positive /32")
    row_tiles = tokens // 32
    vocabulary_tiles = vocabulary_size // 32
    if row_tiles > 110:
        raise ValueError("log-softmax row tiles exceed the Blackhole grid")
    if diagnostic_stage not in (
        "maximum", "shifted", "exponentials", "logsum", "final",
    ):
        raise ValueError("unsupported log-softmax diagnostic stage")
    reader_passes = {
        "maximum": 1,
        "shifted": 2,
        "exponentials": 2,
        "logsum": 2,
        "final": 3,
    }[diagnostic_stage]
    # The causal-LM contract upcasts logits before cross-entropy, so every
    # diagnostic boundary and the standalone final log-probability view are
    # FP32.  The composed loss owner will consume this row locally rather than
    # retaining a full FP32 vocabulary tensor.
    output_tile_bytes = 4096

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("native log-softmax requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    options: dict[str, object] = {
        "grid": BLACKHOLE_P150A_GRID,
        "fp32_dest_acc_en": True,
    }
    if "math_approx_mode" in inspect.signature(ttl.operation).parameters:
        # StableHLO's decomposed exponential/log operations lower through the
        # accurate TTNN unary spelling.  The Moreh softmax factory normally
        # opts into approximate SFPU math, which already diverges for W=32.
        options["math_approx_mode"] = False

    @ttl.operation(**options)
    def _program(logits, output):
        def dfb(tensor, blocks=1):
            return ttl.make_dataflow_buffer_like(
                tensor, shape=(1, 1), block_count=blocks
            )

        c0 = dfb(logits, 2)
        c1, c2, c3 = dfb(logits), dfb(logits), dfb(logits)
        c4, c5, c6, c7 = dfb(logits), dfb(logits), dfb(logits), dfb(logits)
        c8, c9, c10, c11 = dfb(logits), dfb(logits), dfb(logits), dfb(logits)
        c12, c13, c14, c15 = dfb(logits), dfb(logits), dfb(logits), dfb(logits)
        c16 = dfb(output, 2)
        c17, c18, c19, c20 = dfb(output), dfb(output), dfb(output), dfb(output)
        c21, c22, c23 = dfb(output), dfb(output), dfb(output)
        c24 = dfb(output, 2)
        c25, c26 = dfb(output), dfb(output, 2)
        c27, c28 = dfb(output), dfb(output)

        @ttl.datamovement()
        def read():
            with c0.reserve() as tile:
                transaction = ttl.copy(logits[0:1, 0:1], tile)
                transaction.wait()

        @ttl.compute()
        def compute():
            with c0.wait() as value:
                with c16.reserve() as result:
                    result.store(ttl.math.typecast(value, torch.float32))
            node_x, node_y = ttl.node(dims=2)
            if node_x + node_y < 0:
                with c1.wait() as _: pass
                with c2.wait() as _: pass
                with c3.wait() as _: pass
                with c4.wait() as _: pass
                with c5.wait() as _: pass
                with c6.wait() as _: pass
                with c7.wait() as _: pass
                with c8.wait() as _: pass
                with c9.wait() as _: pass
                with c10.wait() as _: pass
                with c11.wait() as _: pass
                with c12.wait() as _: pass
                with c13.wait() as _: pass
                with c14.wait() as _: pass
                with c15.wait() as _: pass
                with c17.wait() as _: pass
                with c18.wait() as _: pass
                with c19.wait() as _: pass
                with c20.wait() as _: pass
                with c21.wait() as _: pass
                with c22.wait() as _: pass
                with c23.wait() as _: pass
                with c24.wait() as _: pass
                with c25.wait() as _: pass
                with c26.wait() as _: pass
                with c27.wait() as _: pass
                with c28.wait() as _: pass

        @ttl.datamovement()
        def write():
            with c16.wait() as value:
                transaction = ttl.copy(value, output[0:1, 0:1])
                transaction.wait()

    variant = hashlib.sha256(
        (
            "causal-lm-logsoftmax-v11:"
            f"{tokens}:{vocabulary_size}:{diagnostic_stage}"
        ).encode()
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.causal_lm_logsoftmax.{variant}",
        arg_roles=("in", "out"),
        version_tag=f"causal-lm-logsoftmax-{variant}",
    )
    def operation(logits, output):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(logits, output)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 29:
                    raise RuntimeError(
                        f"causal-LM log-softmax expected 29 CBs, got {len(compiled.cb_configs)}"
                    )
                for index in range(22, 29):
                    compiled.cb_configs[index].tensor = _FP32_DTYPE_ONLY_TENSOR
                reader_indices = [
                    compiled.kernel_tensor_indices[index]
                    for index, (path, thread) in enumerate(compiled.kernel_paths)
                    if thread != "compute" and "read" in Path(path).name
                ]
                writer_indices = [
                    compiled.kernel_tensor_indices[index]
                    for index, (path, thread) in enumerate(compiled.kernel_paths)
                    if thread != "compute" and "write" in Path(path).name
                ]
                if reader_indices != [[0]] or writer_indices != [[1]]:
                    raise RuntimeError(
                        "causal-LM log-softmax tensor ABI changed: "
                        f"reader={reader_indices}, writer={writer_indices}"
                    )
                cpp = Path(__file__).with_name("cpp")
                sources = {
                    "read": (cpp / "causal_lm_logsoftmax_reader.cpp").read_text(),
                    "compute": _ttnn_large_logsoftmax_compute_source(
                        row_tiles=row_tiles,
                        vocabulary_tiles=vocabulary_tiles,
                        diagnostic_stage=diagnostic_stage,
                    ),
                    "write": (cpp / "causal_lm_logsoftmax_writer.cpp").read_text(),
                }
                for kind in ("read", "write"):
                    sources[kind] = (
                        sources[kind]
                        .replace("@VOCAB_TILES@", str(vocabulary_tiles))
                        .replace("@ROW_TILES@", str(row_tiles))
                        .replace("@PASSES@", str(reader_passes))
                        .replace("@OUTPUT_TILE_BYTES@", str(output_tile_bytes))
                    )
                for path, thread in compiled.kernel_paths:
                    kind = "compute" if thread == "compute" else (
                        "read" if "read" in Path(path).name else "write"
                    )
                    Path(path).write_text(sources[kind], encoding="utf-8")
                for config in compiled.kernel_configs:
                    if hasattr(config, "math_fidelity"):
                        enum_type = type(config.math_fidelity)
                        config.math_fidelity = getattr(enum_type, "HiFi4")
                    if hasattr(config, "math_approx_mode"):
                        config.math_approx_mode = False
                    if hasattr(config, "fp32_dest_acc_en"):
                        config.fp32_dest_acc_en = True
                    if hasattr(config, "unpack_to_dest_mode"):
                        import ttnn
                        while len(config.unpack_to_dest_mode) < 64:
                            config.unpack_to_dest_mode.append(
                                ttnn.UnpackToDestMode.Default
                            )
                        for index in (22, 23, 24, 25, 27):
                            config.unpack_to_dest_mode[index] = (
                                ttnn.UnpackToDestMode.UnpackToDestFp32
                            )
            break
        return result

    operation.output_shape = (tokens, vocabulary_size)
    operation.physical_programs = 1
    operation.safe_default = False
    operation.experimental_backend = "causal_lm_logsoftmax_exact_candidate_v1"
    operation._ttl_program = _program
    return operation


__all__ = ["build_tt_lang_causal_lm_logsoftmax"]
