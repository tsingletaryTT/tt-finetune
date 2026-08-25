"""TTNN-bit-exact native RMSNorm forward ownership primitive."""

from __future__ import annotations

import hashlib
import os
import re
import struct
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID


ttl = None
tt_lang_operation = None


def _float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _ttnn_fused_rmsnorm_compute_source(
    *, sequence_length: int, hidden_size: int,
    cb_ids: dict[str, int] | None = None,
    logical_grid_width: int = 11,
    function_signature: str = "void kernel_main()",
    balance_resident_inputs: bool = False,
    publish_rstd: bool = False,
) -> str:
    """Specialize TTNN's exact non-sharded RMSNorm compute kernel.

    The optional ABI controls keep the arithmetic body byte-for-byte aligned
    with TTNN while allowing the program to live inside a larger resident
    projection island.  Only CB identities, the active-core guard and
    producer/consumer lifecycle are changed; the reduction loop and pack
    boundaries are not rewritten.
    """

    metal_home = os.environ.get("TT_METAL_HOME")
    if not metal_home:
        raise RuntimeError(
            "TTNN fused RMSNorm qualification requires TT_METAL_HOME"
        )
    path = (
        Path(metal_home)
        / "ttnn/cpp/ttnn/operations/normalization/layernorm/device/"
        "kernels/compute/layernorm.cpp"
    )
    if not path.is_file():
        raise RuntimeError(f"TTNN RMSNorm compute kernel not found: {path}")
    source = path.read_text(encoding="utf-8")
    compute_utils = path.with_name("layernorm_compute_utils.h")
    if not compute_utils.is_file():
        raise RuntimeError(
            f"TTNN RMSNorm compute utility header not found: {compute_utils}"
        )
    source = source.replace(
        '#include "layernorm_compute_utils.h"',
        f'#include "{compute_utils}"',
        1,
    )
    token_tiles = sequence_length // 32
    hidden_tiles = hidden_size // 32
    expected_runtime = "    uint32_t NCHt = get_arg_val<uint32_t>(0);\n"
    replacement_runtime = (
        "    const uint32_t node = get_absolute_logical_y() * "
        f"{logical_grid_width} + "
        "get_absolute_logical_x();\n"
        f"    if (node >= {token_tiles}) return;\n"
        "    constexpr uint32_t NCHt = 1;\n"
    )
    if expected_runtime not in source:
        raise RuntimeError("TTNN RMSNorm runtime-row ABI changed")
    source = source.replace(expected_runtime, replacement_runtime, 1)
    compile_args = {
        0: hidden_tiles,
        1: 4,
        2: 1,
        3: 0,
        4: 1,
        5: 1,
        6: 0,
        7: hidden_size,
        8: 32,
    }
    source = re.sub(
        r"get_compile_time_arg_val\((\d+)\)",
        lambda match: str(compile_args[int(match.group(1))]),
        source,
    )
    default_cb_ids = {
        "cb_in": 0,
        "cb_inb": 1,
        "cb_scaler": 2,
        "cb_eps": 3,
        "cb_gamma": 5,
        "cb_beta": 6,
        "cb_out": 16,
        "cb_ex": 18,
        "cb_ex2": 19,
        "cb_xmm2": 20,
        "cb_ex2pe": 21,
        "cb_fusion": 22,
        "cb_x": 23,
        "cb_xmm": 24,
        "cb_in_rm": 0,
        "cb_out_rm": 16,
    }
    if cb_ids is None:
        cb_ids = default_cb_ids
    else:
        missing = set(default_cb_ids) - set(cb_ids)
        if missing:
            raise ValueError(
                "resident TTNN RMSNorm CB map is incomplete: "
                f"{sorted(missing)}"
            )
    source = re.sub(
        r'get_named_compile_time_arg_val\("([A-Za-z0-9_]+)"\)',
        lambda match: str(cb_ids[match.group(1)]),
        source,
    )
    include_marker = '#include "api/compute/compute_kernel_api.h"\n'
    if include_marker not in source:
        raise RuntimeError("TTNN RMSNorm compute include ABI changed")
    source = source.replace(
        include_marker,
        "#define RMSNORM 1\n" + include_marker,
        1,
    )
    if function_signature != "void kernel_main()":
        if "void kernel_main()" not in source:
            raise RuntimeError("TTNN RMSNorm entry point changed")
        source = source.replace(
            "void kernel_main()", function_signature, 1
        )
    if publish_rstd:
        marker = "        cb_ex2pe.push_back(1);\n"
        if marker not in source:
            raise RuntimeError("TTNN RMSNorm rstd publication point changed")
        source = source.replace(
            marker,
            marker
            + "        if (publish_post_rstd) {\n"
            + "            CircularBuffer post_rstd_ready(29);\n"
            + "            CircularBuffer post_rstd_copied(28);\n"
            + "            post_rstd_ready.reserve_back(1);\n"
            + "            post_rstd_ready.push_back(1);\n"
            + "            post_rstd_copied.wait_front(1);\n"
            + "            post_rstd_copied.pop_front(1);\n"
            + "        }\n",
            1,
        )
    if balance_resident_inputs:
        marker = (
            "    CircularBuffer(cb_scaler).pop_front(num_scaler_tiles);\n"
            "}"
        )
        if marker not in source:
            raise RuntimeError("TTNN RMSNorm final lifecycle marker changed")
        source = source.replace(
            marker,
            "    CircularBuffer(cb_scaler).pop_front(num_scaler_tiles);\n"
            "    cb_eps.pop_front(1);\n"
            "    if constexpr (do_gamma) {\n"
            "        cb_gamma.pop_front(total_buffer_size);\n"
            "    }\n"
            "}",
            1,
        )
    if (
        "get_compile_time_arg_val" in source
        or "get_named_compile_time_arg_val" in source
    ):
        raise RuntimeError("unresolved TTNN RMSNorm compile-time argument")
    return source


@lru_cache(maxsize=8)
def build_tt_lang_rmsnorm_forward(
    *, sequence_length: int, hidden_size: int, epsilon: float,
    stage: str = "final",
) -> Callable:
    if sequence_length <= 0 or sequence_length % 32:
        raise ValueError(
            "RMSNorm forward requires a positive 32-aligned sequence length"
        )
    if hidden_size <= 0 or hidden_size % 32:
        raise ValueError(
            "RMSNorm forward requires a positive 32-aligned hidden size"
        )
    if epsilon <= 0.0:
        raise ValueError("RMSNorm forward epsilon must be positive")
    if stage not in {
        "final", "normalized", "normalized_fp32", "weight",
        "square_fp32", "sum_fp32", "mean_fp32", "rstd_fp32",
        "rstd_expanded_fp32", "ttnn_fused_final",
    }:
        raise ValueError("invalid RMSNorm forward diagnostic stage")
    token_tiles = sequence_length // 32
    hidden_tiles = hidden_size // 32
    if token_tiles > 110:
        raise ValueError("RMSNorm forward exceeds the 110-worker Blackhole grid")
    block_size = 4
    padded_tiles = ((hidden_tiles + block_size - 1) // block_size) * block_size

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        import ttl as ttl_module
        from tt_torch.tt_lang import tt_lang_operation as decorator

        ttl = ttl_module
        tt_lang_operation = decorator

    @ttl.operation(grid=BLACKHOLE_P150A_GRID, fp32_dest_acc_en=True)
    def _program(x, weight, seed, fp32_seed, output):
        def dfb(tensor, blocks):
            return ttl.make_dataflow_buffer_like(
                tensor, shape=(1, 1), block_count=max(1, blocks)
            )

        # Declare the actual physical c0..c22 map up front.  Besides making
        # source CB identities stable, this lets TT-Lang select per-CB
        # UnpackToDestFp32 modes before the generic-op descriptor is
        # serialized.  Replacing a compact 10-CB descriptor afterwards is too
        # late for compute-kernel configuration.
        x_cb = dfb(x, padded_tiles)             # c0 BF16
        output_seed = dfb(seed, 1)              # c1 output-typed scaffold
        scaler = dfb(fp32_seed, 2)              # c2 FP32
        eps = dfb(fp32_seed, 2)                 # c3 FP32
        inverse_hidden = dfb(fp32_seed, 2)       # c4 FP32
        weight_cb = dfb(weight, padded_tiles)   # c5 BF16
        fp32_marker = dfb(x, 1)                 # c6 BF16 sink
        normalized_fp32 = dfb(fp32_seed, 8)      # c7 FP32
        rstd_expanded = dfb(fp32_seed, 1)         # c8 FP32
        _hole9 = dfb(x, 1)
        _hole10 = dfb(x, 1)
        _hole11 = dfb(x, 1)
        _hole12 = dfb(x, 1)
        _hole13 = dfb(x, 1)
        _hole14 = dfb(x, 1)
        _hole15 = dfb(x, 1)
        result = dfb(output, 8)                  # c16
        mean = dfb(fp32_seed, 2)                # c17 FP32
        variance_eps = dfb(fp32_seed, 2)        # c18 FP32
        variance = dfb(fp32_seed, 2)            # c19 FP32 sum
        square = dfb(fp32_seed, padded_tiles)    # c20 FP32
        rstd = dfb(fp32_seed, 8)                # c21 FP32
        normalized = dfb(x, 8)                  # c22 BF16

        @ttl.datamovement()
        def read():
            with x_cb.reserve() as block:
                tx = ttl.copy(x[0:1, 0:1], block)
                tx.wait()
            with weight_cb.reserve() as block:
                tx = ttl.copy(weight[0:1, 0:1], block)
                tx.wait()
            with output_seed.reserve() as block:
                tx = ttl.copy(seed[0:1, 0:1], block)
                tx.wait()

        @ttl.compute()
        def compute():
            with output_seed.wait() as item:
                with result.reserve() as target:
                    target.store(item)
            # These compile-only typecasts cause TT-Lang to emit full-FP32
            # unpack mode for every explicit post-reduction FP32 source.
            # They are unconditional in IR so the precision-selection pass
            # cannot remove them; native C++ replaces this entire compute
            # body before device execution. Square is intentionally excluded
            # because TTNN's sum reads it via TF32.
            with scaler.wait() as value:
                with fp32_marker.reserve() as target:
                    target.store(ttl.math.typecast(value, torch.bfloat16))
            with eps.wait() as value:
                with fp32_marker.reserve() as target:
                    target.store(ttl.math.typecast(value, torch.bfloat16))
            with inverse_hidden.wait() as value:
                with fp32_marker.reserve() as target:
                    target.store(ttl.math.typecast(value, torch.bfloat16))
            with mean.wait() as value:
                with fp32_marker.reserve() as target:
                    target.store(ttl.math.typecast(value, torch.bfloat16))
            with variance_eps.wait() as value:
                with fp32_marker.reserve() as target:
                    target.store(ttl.math.typecast(value, torch.bfloat16))
            with variance.wait() as value:
                with fp32_marker.reserve() as target:
                    target.store(ttl.math.typecast(value, torch.bfloat16))
            with rstd.wait() as value:
                # Give the dedicated suffix FIFO an explicit producer and
                # consumer in TT-Lang IR.  A consumer-only trailing DFB is
                # currently removed after `_compile_kernel` returns even
                # though it was present in the wrapper's cached descriptor.
                with normalized_fp32.reserve() as target:
                    target.store(value)
                with fp32_marker.reserve() as target:
                    target.store(ttl.math.typecast(value, torch.bfloat16))
            with normalized_fp32.wait() as value:
                with fp32_marker.reserve() as target:
                    target.store(ttl.math.typecast(value, torch.bfloat16))

            # Retain the remaining CBs in the compiler ABI. Native C++
            # replaces this unreachable scaffold after indices are assigned.
            cx, cy = ttl.node(dims=2)
            if cy * 11 + cx < 0:
                with x_cb.wait() as _: pass
                with weight_cb.wait() as _: pass
                with square.wait() as _: pass
                with normalized.wait() as _: pass

        @ttl.datamovement()
        def write():
            with result.wait() as item:
                tx = ttl.copy(item, output[0:1, 0:1])
                tx.wait()

    variant = hashlib.sha256(
        f"rmsnorm-forward-v60:{sequence_length}:{hidden_size}:{epsilon}:{stage}".encode()
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.rmsnorm_forward.{variant}",
        arg_roles=("in", "in", "in", "in", "out"),
        version_tag=f"rmsnorm-forward-v1-{variant}",
    )
    def operation(x, weight, seed, fp32_seed, output):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(x, weight, seed, fp32_seed, output)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 23:
                    raise RuntimeError(
                        f"RMSNorm forward expected 23 CBs, got "
                        f"{len(compiled.cb_configs)}"
                    )
                if compiled.cb_configs[7].tensor.dtype != torch.float32:
                    raise RuntimeError(
                        "RMSNorm forward c7 must be the FP32 suffix FIFO, "
                        f"got {compiled.cb_configs[7].tensor.dtype}"
                    )
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
                # Diagnostic FP32-output stages may deliberately pass one
                # FP32 scaffold for both the otherwise-dead output-seed and
                # fp32-seed operands. TT-Lang then canonicalizes local input
                # 2 to local input 3. Native C++ consumes only X and gamma;
                # accept either identity for that compile-only scaffold.
                if (
                    len(reader_indices) != 1
                    or len(reader_indices[0]) != 3
                    or set(reader_indices[0][:]) not in (
                        {0, 1, 2}, {0, 1, 3}
                    )
                ):
                    raise RuntimeError(
                        f"RMSNorm forward reader ABI changed: {reader_indices!r}"
                    )
                if writer_indices != [[4]]:
                    raise RuntimeError(
                        f"RMSNorm forward writer ABI changed: {writer_indices!r}"
                    )
                positions = {
                    tensor: position
                    for position, tensor in enumerate(reader_indices[0])
                }
                replacements = {
                    "@HIDDEN_TILES@": str(hidden_tiles),
                    "@PADDED_TILES@": str(padded_tiles),
                    "@TOKEN_TILES@": str(token_tiles),
                    "@HIDDEN_SIZE@": str(hidden_size),
                    "@EPSILON_BITS@": str(_float_bits(epsilon)),
                    "@INVERSE_HIDDEN_BITS@": str(
                        _float_bits(1.0 / hidden_size)
                    ),
                    "@OUTPUT_STAGE@": {
                        "final": "0", "normalized": "1",
                        "normalized_fp32": "2", "weight": "3",
                        "square_fp32": "4", "sum_fp32": "5",
                        "mean_fp32": "6", "rstd_fp32": "7",
                        "rstd_expanded_fp32": "9",
                        "ttnn_fused_final": "11",
                    }[stage],
                    "@OUTPUT_TILE_BYTES@": (
                        "4096" if stage.endswith("_fp32") else "2048"
                    ),
                    "@X_POSITION@": str(positions[0]),
                    "@WEIGHT_POSITION@": str(positions[1]),
                    "@OUTPUT_POSITION@": "0",
                    "@OUTPUT_TENSOR_INDEX@": "4",
                    "@WRITER_TILES@": str(
                        1 if stage in {
                            "sum_fp32", "mean_fp32", "rstd_fp32"
                        } else hidden_tiles
                    ),
                    "@WRITER_BLOCK_TILES@": str(
                        1 if stage in {
                            "sum_fp32", "mean_fp32", "rstd_fp32",
                            "rstd_expanded_fp32",
                        } else 2 if stage in {
                            "final", "normalized", "normalized_fp32"
                        } else block_size
                    ),
                    "@WRITER_CB_ID@": (
                        "5" if stage == "weight" else
                        "20" if stage == "square_fp32" else
                        "22" if stage == "normalized" else "16"
                        if stage not in {
                            "sum_fp32", "mean_fp32", "rstd_fp32",
                            "rstd_expanded_fp32",
                        } else {
                            "sum_fp32": "19",
                            "mean_fp32": "17",
                            "rstd_fp32": "21",
                            "rstd_expanded_fp32": "8",
                        }[stage]
                    ),
                    "@REPLICATE_SCALAR@": (
                        "true" if stage == "rstd_expanded_fp32" else "false"
                    ),
                }
                cpp = Path(__file__).with_name("cpp")
                sources = {
                    "read": (cpp / "rmsnorm_forward_reader.cpp").read_text(),
                    "compute": (cpp / "rmsnorm_forward_compute.cpp").read_text(),
                    "write": (cpp / "rmsnorm_forward_writer.cpp").read_text(),
                }
                if stage == "ttnn_fused_final":
                    sources["compute"] = _ttnn_fused_rmsnorm_compute_source(
                        sequence_length=sequence_length,
                        hidden_size=hidden_size,
                    )
                for kind, source in tuple(sources.items()):
                    for placeholder, replacement in replacements.items():
                        source = source.replace(placeholder, replacement)
                    if re.search(r"@[A-Z0-9_]+@", source):
                        raise RuntimeError(
                            f"unexpanded RMSNorm forward {kind} placeholder"
                        )
                    sources[kind] = source
                for path, thread in compiled.kernel_paths:
                    kind = "compute" if thread == "compute" else (
                        "read" if "read" in Path(path).name else "write"
                    )
                    Path(path).write_text(sources[kind], encoding="utf-8")
                from ttl.dataflow_buffer import CompilerAllocatedDFBConfig

                # TT-Lang normally compacts DFBs to 0..N.  Exact layernorm
                # instead needs TTNN's real physical CB identities, so retain
                # the holes and allocate the complete c0..c22 index space.
                capacities = {index: 1 for index in range(23)}
                capacities.update({
                    0: padded_tiles,
                    2: 2,
                    3: 2,
                    4: 2,
                    5: padded_tiles,
                    7: 8,
                    8: 1,
                    16: 8,
                    17: 2,
                    18: 2,
                    19: 2,
                    20: padded_tiles,
                    21: 8,
                    22: 8,
                })
                # CB22 is deliberately BF16.  The actual Transformers graph
                # converts x*rstd to BF16 before its BF16 gamma multiply; the
                # fused TTNN RMSNorm kernel keeps this value in FP32 and is a
                # different numerical contract for learned gamma.
                fp32_indices = {4, 7, 8, 17, 18, 19, 20, 21}
                if stage == "ttnn_fused_final":
                    fp32_indices.add(22)
                if stage.endswith("_fp32") and stage != "square_fp32":
                    fp32_indices.add(16)
                physical_configs = []
                for cb_index in range(23):
                    config = CompilerAllocatedDFBConfig(
                        dfb_index=cb_index,
                        num_tiles=1,
                        data_format=(
                            "float32" if cb_index in fp32_indices
                            else "bfloat16"
                        ),
                        block_count=capacities[cb_index],
                    )
                    config._cb_index = cb_index
                    physical_configs.append(config)
                compiled.cb_configs = physical_configs
                for config in compiled.kernel_configs:
                    if hasattr(config, "math_fidelity"):
                        enum_type = type(config.math_fidelity)
                        config.math_fidelity = getattr(enum_type, "HiFi4")
                    if hasattr(config, "math_approx_mode"):
                        config.math_approx_mode = False
                    if hasattr(config, "fp32_dest_acc_en"):
                        config.fp32_dest_acc_en = True
                    if hasattr(config, "packer_l1_acc"):
                        config.packer_l1_acc = False
                    if hasattr(config, "dst_full_sync_en"):
                        config.dst_full_sync_en = False
                    if hasattr(config, "unpack_to_dest_mode"):
                        import ttnn

                        # nanobind exposes this as a mutable C++ vector.  Do
                        # not clear and rebuild it: on current PJRT that can
                        # mutate a temporary wrapper instead of the retained
                        # descriptor.  Grow it in place, then assign the
                        # relevant physical CB entries directly.
                        while len(config.unpack_to_dest_mode) < 64:
                            config.unpack_to_dest_mode.append(
                                ttnn.UnpackToDestMode.Default
                            )
                        # The production mean uses the accurate FP32 SFPU
                        # path.  Its square input (CB20) is copied directly
                        # into DEST and therefore must use full-FP32 unpack;
                        # leaving it in the former TF32/FPU sum mode changes
                        # every row statistic.  CB2 is still only a reduction
                        # protocol token and CB21 later feeds an FPU column
                        # broadcast, so neither may use direct-to-DEST mode.
                        full_fp32_cbs = (
                            set()
                            if stage in {
                                "final", "ttnn_fused_final",
                                "rstd_expanded_fp32",
                            }
                            else {4, 7, 8, 17, 18, 19, 20, 21}
                        )
                        for cb_index in range(64):
                            config.unpack_to_dest_mode[cb_index] = (
                                ttnn.UnpackToDestMode.UnpackToDestFp32
                                if cb_index in full_fp32_cbs
                                else ttnn.UnpackToDestMode.Default
                            )
            break
        return result

    operation.sequence_length = sequence_length
    operation.hidden_size = hidden_size
    operation.epsilon = float(epsilon)
    operation.stage = stage
    # Artifact-v4 owners compile this accepted numerical primitive as a
    # compact physical child while retaining one public StableHLO call.
    operation._ttl_program = _program
    return operation
