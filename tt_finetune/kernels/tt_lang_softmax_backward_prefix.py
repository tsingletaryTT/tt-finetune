"""Small exactness probe for the FP32 softmax-backward prefix."""

from __future__ import annotations

import hashlib
import re
import struct
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID


ttl = None
tt_lang_operation = None


@lru_cache(maxsize=8)
def build_tt_lang_softmax_backward_prefix(
    *, sequence_length: int = 128, heads: int = 1,
    stage: str = "product", sub_sfpu: bool = False, scale: float = 0.125,
) -> Callable:
    if sequence_length != 128:
        raise ValueError("softmax prefix probe currently requires S=128")
    if stage not in {"product", "rowsum", "centered", "ds", "scaled_ds"}:
        raise ValueError("stage must be product, rowsum, centered, ds, or scaled_ds")
    seq_tiles = sequence_length // 32
    if heads * seq_tiles > 110:
        raise ValueError("softmax prefix probe exceeds the 110-worker grid")
    stage_id = {
        "product": 0, "centered": 1, "ds": 2,
        "rowsum": 3, "scaled_ds": 4,
    }[stage]
    scale_bits = struct.unpack("<I", struct.pack("<f", float(scale)))[0]

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        import ttl as ttl_module
        from tt_torch.tt_lang import tt_lang_operation as decorator
        ttl = ttl_module
        tt_lang_operation = decorator

    @ttl.operation(
        grid=BLACKHOLE_P150A_GRID,
        fp32_dest_acc_en=True,
    )
    def _program(probability, dp, arena):
        def dfb(tensor, blocks):
            return ttl.make_dataflow_buffer_like(
                tensor, shape=(1, 1), block_count=max(1, blocks)
            )

        p_cb = dfb(probability, seq_tiles)
        dp_cb = dfb(dp, seq_tiles)
        scaler = dfb(arena, 1)
        dp_fp32 = dfb(probability, seq_tiles)
        product = dfb(probability, seq_tiles)
        rowsum = dfb(probability, 1)
        centered = dfb(probability, seq_tiles)
        ds = dfb(probability, seq_tiles)
        out = dfb(arena, seq_tiles)

        @ttl.datamovement()
        def read():
            with p_cb.reserve() as block:
                tx = ttl.copy(probability[0:1, 0:1], block); tx.wait()
            with dp_cb.reserve() as block:
                tx = ttl.copy(dp[0:1, 0:1], block); tx.wait()

        @ttl.compute()
        def compute():
            with p_cb.wait() as item:
                with product.reserve() as result:
                    result.store(item)
            with dp_cb.wait() as item:
                with out.reserve() as result:
                    result.store(item)
            x, y = ttl.node(dims=2)
            if y * 11 + x < 0:
                with scaler.wait() as _: pass
                with dp_fp32.wait() as _: pass
                with product.wait() as _: pass
                with rowsum.wait() as _: pass
                with centered.wait() as _: pass
                with ds.wait() as _: pass

        @ttl.datamovement()
        def write():
            for _ in range(2):
                with out.wait() as item:
                    tx = ttl.copy(item, arena[0:1, 0:1]); tx.wait()

    variant = hashlib.sha256(
        f"softmax-prefix-v2:{sequence_length}:{heads}:{stage}:{int(sub_sfpu)}:{scale:.17g}".encode()
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.softmax_backward_prefix.{variant}",
        arg_roles=("in", "in", "out"),
        version_tag=f"softmax-backward-prefix-v1-{variant}",
    )
    def operation(probability, dp, arena):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(probability, dp, arena)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 9:
                    raise RuntimeError(
                        f"softmax prefix expected 9 CBs, got {len(compiled.cb_configs)}"
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
                if len(reader_indices) != 1 or sorted(reader_indices[0]) != [0, 1]:
                    raise RuntimeError(f"softmax prefix reader ABI changed: {reader_indices!r}")
                if writer_indices != [[2]]:
                    raise RuntimeError(f"softmax prefix writer ABI changed: {writer_indices!r}")
                positions = {tensor: pos for pos, tensor in enumerate(reader_indices[0])}
                replacements = {
                    "@SEQ_TILES@": str(seq_tiles),
                    "@HEADS@": str(heads),
                    "@STAGE@": str(stage_id),
                    "@SUB_SFPU@": str(int(sub_sfpu)),
                    "@SCALE_BITS@": str(scale_bits),
                }
                cpp = Path(__file__).with_name("cpp")
                sources = {
                    "read": (cpp / "softmax_backward_prefix_reader.cpp").read_text(),
                    "compute": (cpp / "softmax_backward_prefix_compute.cpp").read_text(),
                    "write": (cpp / "softmax_backward_prefix_writer.cpp").read_text(),
                }
                sources["read"] = re.sub(
                    r"get_common_arg_val<uint32_t>\((\d+)\)",
                    lambda match: f"get_common_arg_val<uint32_t>({positions[int(match.group(1))]})",
                    sources["read"],
                )
                sources["read"] = re.sub(
                    r"get_tensor_accessor_args_cta_offset<(\d+), 9>\(\),\s*(\d+)",
                    lambda match: (
                        f"get_tensor_accessor_args_cta_offset<{match.group(1)}, 9>(), "
                        f"{positions[int(match.group(1))]}"
                    ),
                    sources["read"],
                )
                for kind, source in tuple(sources.items()):
                    for placeholder, value in replacements.items():
                        source = source.replace(placeholder, value)
                    if re.search(r"@[A-Z0-9_]+@", source):
                        raise RuntimeError(f"unexpanded softmax prefix {kind} placeholder")
                    sources[kind] = source
                for path, thread in compiled.kernel_paths:
                    kind = "compute" if thread == "compute" else (
                        "read" if "read" in Path(path).name else "write"
                    )
                    Path(path).write_text(sources[kind], encoding="utf-8")

                from ttl.dataflow_buffer import CompilerAllocatedDFBConfig
                for cb_index, blocks in ((3, seq_tiles), (4, seq_tiles), (5, 1),
                                         (6, seq_tiles), (7, seq_tiles)):
                    config = CompilerAllocatedDFBConfig(
                        dfb_index=cb_index, num_tiles=1,
                        data_format="float32", block_count=blocks,
                    )
                    config._cb_index = cb_index
                    compiled.cb_configs[cb_index] = config
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
                        if len(config.unpack_to_dest_mode) == 0:
                            for cb_index in range(64):
                                config.unpack_to_dest_mode.append(
                                    ttnn.UnpackToDestMode.UnpackToDestFp32
                                    # CB4 is the physical FP32 product, but
                                    # TTNN's REDUCE_ROW deliberately unpacks
                                    # that input through its normal TF32/FPU
                                    # reduction path.  Elementwise operands
                                    # retain full FP32 unpacking.
                                    if cb_index in {0, 3, 5, 6, 7}
                                    else ttnn.UnpackToDestMode.Default
                                )
            break
        return result

    operation.stage = stage
    operation.sub_sfpu = bool(sub_sfpu)
    return operation
