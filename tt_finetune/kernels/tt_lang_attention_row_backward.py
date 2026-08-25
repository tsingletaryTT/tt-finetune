"""Query-row-sharded exact-schedule attention backward diagnostic."""

from __future__ import annotations

import hashlib
import re
import struct
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .attention_block_planner import plan_attention_block
from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID


ttl = None
tt_lang_operation = None


@lru_cache(maxsize=16)
def build_tt_lang_attention_row_backward(
    *,
    sequence_length: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    scale: float,
    dp_phase_tiles: int = 2,
    dp_n_subblock_tiles: int = 4,
    diagnostic_stage: str = "scaled_ds",
    math_fidelity: str = "HiFi4",
    fp32_dest_acc_en: bool = True,
    math_approx_mode: bool = True,
    value_pretransposed: bool = False,
) -> Callable:
    """Build dV/dP/dS/dQ under the TTNN query-row ownership map."""

    plan = plan_attention_block(
        batch_size=1,
        sequence_length=sequence_length,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )
    if plan.query_chunks != 1:
        raise ValueError("reference attention row phase requires sequence 128")
    if plan.reference_dpdqdv_workers > 110:
        raise ValueError("reference attention row phase exceeds 110 workers")
    seq_tiles = sequence_length // 32
    dim_tiles = head_dim // 32
    if dp_phase_tiles not in {1, 2} or dim_tiles % dp_phase_tiles:
        raise ValueError("dp_phase_tiles must divide head-dimension tiles")
    if dp_n_subblock_tiles not in {1, 2, 4} or seq_tiles % dp_n_subblock_tiles:
        raise ValueError("dp_n_subblock_tiles must divide sequence tiles")
    if diagnostic_stage not in {
        "scaled_ds", "dp", "ds", "product", "centered", "rowsum"
    }:
        raise ValueError(
            "diagnostic_stage must be scaled_ds, dp, ds, product, centered, or rowsum"
        )
    diagnostic_stage_id = {
        "scaled_ds": 0, "dp": 1, "ds": 2, "product": 3,
        "centered": 4, "rowsum": 5,
    }[diagnostic_stage]
    if math_fidelity not in {"HiFi2", "HiFi3", "HiFi4"}:
        raise ValueError("math_fidelity must be HiFi2, HiFi3, or HiFi4")
    scale_bits = struct.unpack("<I", struct.pack("<f", float(scale)))[0]

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import (
                tt_lang_operation as tt_lang_operation_decorator,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("native attention requires TT-XLA") from exc
        ttl = ttl_module
        tt_lang_operation = tt_lang_operation_decorator

    score_tiles = seq_tiles * seq_tiles
    vector_tiles = seq_tiles * dim_tiles

    @ttl.operation(
        grid=BLACKHOLE_P150A_GRID,
        fp32_dest_acc_en=fp32_dest_acc_en,
        options="--ttl-matmul-full-fp32",
    )
    def _program(probability, output_gradient, expanded_key, expanded_value, arena):
        def dfb(tensor, blocks):
            return ttl.make_dataflow_buffer_like(
                tensor, shape=(1, 1), block_count=blocks
            )

        p_row = dfb(probability, seq_tiles)
        p_column = dfb(probability, seq_tiles)
        dout_row = dfb(output_gradient, dim_tiles)
        dout_full = dfb(output_gradient, vector_tiles)
        k = dfb(expanded_key, vector_tiles)
        v = dfb(expanded_value, vector_tiles)
        scaler = dfb(arena, 1)
        pt = dfb(arena, seq_tiles)
        dv = dfb(arena, dim_tiles)
        vt = dfb(arena, vector_tiles)
        dp = dfb(arena, seq_tiles)
        product = dfb(arena, seq_tiles)
        rowsum = dfb(arena, 1)
        centered = dfb(arena, seq_tiles)
        ds = dfb(arena, seq_tiles)
        ds_scaled = dfb(arena, seq_tiles)
        dq = dfb(arena, dim_tiles)
        partial = dfb(arena, seq_tiles)
        dp_snapshot = dfb(arena, seq_tiles)

        @ttl.datamovement()
        def read():
            with p_row.reserve() as block:
                tx = ttl.copy(probability[0:1, 0:1], block)
                tx.wait()
            with p_row.reserve() as block:
                tx = ttl.copy(output_gradient[0:1, 0:1], block)
                tx.wait()
            with p_row.reserve() as block:
                tx = ttl.copy(expanded_key[0:1, 0:1], block)
                tx.wait()
            with p_row.reserve() as block:
                tx = ttl.copy(expanded_value[0:1, 0:1], block)
                tx.wait()

        @ttl.compute()
        def compute():
            for _ in range(4):
                with p_row.wait() as item:
                    with dq.reserve() as result:
                        result.store(item)
            node_col, node_row = ttl.node(dims=2)
            if node_row * 11 + node_col < 0:
                with p_column.wait() as _unused: pass
                with dout_row.wait() as _unused: pass
                with dout_full.wait() as _unused: pass
                with k.wait() as _unused: pass
                with v.wait() as _unused: pass
                with scaler.wait() as _unused: pass
                with pt.wait() as _unused: pass
                with dv.wait() as _unused: pass
                with vt.wait() as _unused: pass
                with dp.wait() as _unused: pass
                with product.wait() as _unused: pass
                with rowsum.wait() as _unused: pass
                with centered.wait() as _unused: pass
                with ds.wait() as _unused: pass
                with ds_scaled.wait() as _unused: pass
                with partial.wait() as _unused: pass
                with dp_snapshot.wait() as _unused: pass

        @ttl.datamovement()
        def write():
            for _ in range(4):
                with dq.wait() as item:
                    tx = ttl.copy(item, arena[0:1, 0:1])
                    tx.wait()

    variant = hashlib.sha256(
        (
            f"attention-row-bwd-v1:{sequence_length}:{query_heads}:{kv_heads}:"
            f"{head_dim}:{scale:.17g}:dp-phase={dp_phase_tiles}:"
            f"dp-n-subblock={dp_n_subblock_tiles}:"
            f"diagnostic={diagnostic_stage}"
            f":fidelity={math_fidelity}"
            f":fp32dest={int(fp32_dest_acc_en)}"
            f":approx={int(math_approx_mode)}"
            f":value-pretransposed={int(value_pretransposed)}"
        ).encode("ascii")
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.attention_row_backward.{variant}",
        arg_roles=("in", "in", "in", "in", "out"),
        version_tag=f"attention-row-backward-{variant}",
    )
    def operation(probability, output_gradient, expanded_key, expanded_value, arena):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(
            probability, output_gradient, expanded_key, expanded_value, arena
        )
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 19:
                    raise RuntimeError(
                        "attention row backward expected 19 CBs, got "
                        f"{len(compiled.cb_configs)}"
                    )
                reader_indices = [
                    compiled.kernel_tensor_indices[index]
                    for index, (path, thread_type) in enumerate(compiled.kernel_paths)
                    if thread_type != "compute" and "read" in Path(path).name
                ]
                writer_indices = [
                    compiled.kernel_tensor_indices[index]
                    for index, (path, thread_type) in enumerate(compiled.kernel_paths)
                    if thread_type != "compute" and "write" in Path(path).name
                ]
                if len(reader_indices) != 1 or sorted(reader_indices[0]) != [0, 1, 2, 3]:
                    raise RuntimeError(f"attention row reader ABI changed: {reader_indices!r}")
                if writer_indices != [[4]]:
                    raise RuntimeError(f"attention row writer ABI changed: {writer_indices!r}")
                reader_position = {
                    tensor_index: position
                    for position, tensor_index in enumerate(reader_indices[0])
                }
                replacements = {
                    "@SEQ_TILES@": str(seq_tiles),
                    "@DIM_TILES@": str(dim_tiles),
                    "@QUERY_HEADS@": str(query_heads),
                    "@SCALE_BITS@": str(scale_bits),
                    "@DP_PHASE_TILES@": str(dp_phase_tiles),
                    "@DP_N_SUBBLOCK_TILES@": str(dp_n_subblock_tiles),
                    "@DIAGNOSTIC_STAGE@": str(diagnostic_stage_id),
                    "@VALUE_PRETRANSPOSED@": str(int(value_pretransposed)),
                    "@FP32_DEST_ACC_EN@": str(int(fp32_dest_acc_en)),
                }
                cpp_dir = Path(__file__).with_name("cpp")
                sources = {
                    "read": (cpp_dir / "attention_row_backward_reader.cpp").read_text(encoding="utf-8"),
                    "compute": (cpp_dir / "attention_row_backward_compute.cpp").read_text(encoding="utf-8"),
                    "write": (cpp_dir / "attention_row_backward_writer.cpp").read_text(encoding="utf-8"),
                }
                sources["read"] = re.sub(
                    r"get_common_arg_val<uint32_t>\((\d+)\)",
                    lambda match: (
                        "get_common_arg_val<uint32_t>("
                        f"{reader_position[int(match.group(1))]})"
                    ),
                    sources["read"],
                )
                sources["read"] = re.sub(
                    r"get_tensor_accessor_args_cta_offset<(\d+), 19>\(\),\s*(\d+)",
                    lambda match: (
                        "get_tensor_accessor_args_cta_offset<"
                        f"{match.group(1)}, 19>(), "
                        f"{reader_position[int(match.group(1))]}"
                    ),
                    sources["read"],
                )
                for kind, source in tuple(sources.items()):
                    for placeholder, value in replacements.items():
                        source = source.replace(placeholder, value)
                    if re.search(r"@[A-Z0-9_]+@", source):
                        raise RuntimeError(f"unexpanded attention row {kind} placeholder")
                    sources[kind] = source
                for path, thread_type in compiled.kernel_paths:
                    kind = "compute" if thread_type == "compute" else (
                        "read" if "read" in Path(path).name else "write"
                    )
                    Path(path).write_text(sources[kind], encoding="utf-8")
                for config in compiled.kernel_configs:
                    if hasattr(config, "math_fidelity"):
                        enum_type = type(config.math_fidelity)
                        config.math_fidelity = getattr(enum_type, math_fidelity)
                    if hasattr(config, "math_approx_mode"):
                        config.math_approx_mode = bool(math_approx_mode)
                    if hasattr(config, "fp32_dest_acc_en"):
                        config.fp32_dest_acc_en = bool(fp32_dest_acc_en)
                    if hasattr(config, "unpack_to_dest_mode"):
                        # TTNN's decomposed softmax backward runs its FP32
                        # elementwise/reduction chain through unpack-to-dest
                        # FP32.  Matmul operands (P-column, dO, K and V) keep
                        # their normal TF32 source-unpack contract.  Mirroring
                        # this per-CB distinction is required for bit parity;
                        # a blanket FP32 mode would change the matmuls too.
                        # The generic reduction intentionally consumes the
                        # SFPU product through TTNN's normal TF32 reduction
                        # input path; only the elementwise operands use the
                        # full-FP32 unpack mode.
                        softmax_fp32_cbs = {0, 10, 12, 13, 14}
                        import ttnn
                        if len(config.unpack_to_dest_mode) == 0:
                            for cb_index in range(64):
                                config.unpack_to_dest_mode.append(
                                    ttnn.UnpackToDestMode.UnpackToDestFp32
                                    if cb_index in softmax_fp32_cbs
                                    else ttnn.UnpackToDestMode.Default
                                )
            break
        return result

    operation.plan = plan
    operation.dp_phase_tiles = dp_phase_tiles
    operation.dp_n_subblock_tiles = dp_n_subblock_tiles
    operation.diagnostic_stage = diagnostic_stage
    operation.math_fidelity = math_fidelity
    operation.fp32_dest_acc_en = bool(fp32_dest_acc_en)
    operation.math_approx_mode = bool(math_approx_mode)
    operation.value_pretransposed = bool(value_pretransposed)
    operation.output_tile_shape = (
        3 * query_heads * sequence_length,
        sequence_length,
    )
    return operation
