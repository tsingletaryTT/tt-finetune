"""TT-MLIR-compatible multicore FP32 attention dV diagnostic."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .attention_block_planner import plan_attention_block
from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID


ttl = None
tt_lang_operation = None


@lru_cache(maxsize=16)
def build_tt_lang_attention_dv_reference(
    *,
    sequence_length: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    phase_tiles: int = 4,
    dtype_name: str = "float32",
    fp32_dest_acc_en: bool = True,
    matmul_full_fp32: bool = True,
    out_subblock_tiles: int | None = None,
    rows_per_worker: int = 1,
) -> Callable:
    """Build dV-heads with the exact reference query-head row sharding."""

    plan = plan_attention_block(
        batch_size=1,
        sequence_length=sequence_length,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )
    if plan.query_chunks != 1:
        raise ValueError("reference dV currently requires sequence 128")
    seq_tiles = sequence_length // 32
    dim_tiles = head_dim // 32
    if phase_tiles not in {1, 2, 4} or seq_tiles % phase_tiles:
        raise ValueError("phase_tiles must divide sequence tiles and be 1, 2, or 4")
    if out_subblock_tiles is None:
        out_subblock_tiles = dim_tiles
    if out_subblock_tiles not in {1, 2, 4} or dim_tiles % out_subblock_tiles:
        raise ValueError("out_subblock_tiles must divide head-dimension tiles")
    if rows_per_worker not in {1, 2, 4} or seq_tiles % rows_per_worker:
        raise ValueError("rows_per_worker must divide sequence tiles")
    if plan.reference_dpdqdv_workers > 110:
        raise ValueError("reference dV schedule exceeds the 110-worker grid")
    if dtype_name not in {"float32", "bfloat16"}:
        raise ValueError("reference dV dtype must be float32 or bfloat16")
    tile_bytes = 4096 if dtype_name == "float32" else 2048

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

    @ttl.operation(
        grid=BLACKHOLE_P150A_GRID,
        fp32_dest_acc_en=fp32_dest_acc_en,
        options="--ttl-matmul-full-fp32" if matmul_full_fp32 else "",
    )
    def _program(probability, output_gradient, output):
        def dfb(tensor, blocks):
            return ttl.make_dataflow_buffer_like(
                tensor, shape=(1, 1), block_count=blocks
            )

        p_column = dfb(probability, rows_per_worker * seq_tiles)
        dout = dfb(output_gradient, seq_tiles * dim_tiles)
        pt_row = dfb(output, rows_per_worker * seq_tiles)
        dv = dfb(output, rows_per_worker * dim_tiles)
        partial = dfb(output, rows_per_worker * dim_tiles)

        @ttl.datamovement()
        def read():
            with p_column.reserve() as block:
                tx = ttl.copy(probability[0:1, 0:1], block)
                tx.wait()
            with p_column.reserve() as block:
                tx = ttl.copy(output_gradient[0:1, 0:1], block)
                tx.wait()

        @ttl.compute()
        def compute():
            for _ in range(2):
                with p_column.wait() as item:
                    with dv.reserve() as result:
                        result.store(item)
            node_col, node_row = ttl.node(dims=2)
            if node_row * 11 + node_col < 0:
                with dout.wait() as _unused: pass
                with pt_row.wait() as _unused: pass
                with partial.wait() as _unused: pass

        @ttl.datamovement()
        def write():
            for _ in range(2):
                with dv.wait() as item:
                    tx = ttl.copy(item, output[0:1, 0:1])
                    tx.wait()

    variant = hashlib.sha256(
        (
            f"attention-dv-reference-v1:{sequence_length}:{query_heads}:"
            f"{kv_heads}:{head_dim}:phase={phase_tiles}:{dtype_name}:"
            f"fp32-dest={int(fp32_dest_acc_en)}:full-fp32={int(matmul_full_fp32)}:"
            f"out-subblock={out_subblock_tiles}:rows={rows_per_worker}"
        ).encode("ascii")
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.attention_dv_reference.{variant}",
        arg_roles=("in", "in", "out"),
        version_tag=f"attention-dv-reference-{variant}",
    )
    def operation(probability, output_gradient, output):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(probability, output_gradient, output)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 5:
                    raise RuntimeError(
                        "reference dV expected five CBs, got "
                        f"{len(compiled.cb_configs)}"
                    )
                reader_indices = [
                    compiled.kernel_tensor_indices[index]
                    for index, (path, thread_type) in enumerate(
                        compiled.kernel_paths
                    )
                    if thread_type != "compute" and "read" in Path(path).name
                ]
                writer_indices = [
                    compiled.kernel_tensor_indices[index]
                    for index, (path, thread_type) in enumerate(
                        compiled.kernel_paths
                    )
                    if thread_type != "compute" and "write" in Path(path).name
                ]
                if len(reader_indices) != 1 or sorted(reader_indices[0]) != [0, 1]:
                    raise RuntimeError(f"reference dV reader ABI changed: {reader_indices!r}")
                if writer_indices != [[2]]:
                    raise RuntimeError(f"reference dV writer ABI changed: {writer_indices!r}")
                replacements = {
                    "@SEQ_TILES@": str(seq_tiles),
                    "@DIM_TILES@": str(dim_tiles),
                    "@QUERY_HEADS@": str(query_heads),
                    "@PHASE_TILES@": str(phase_tiles),
                    "@FP32_DEST@": str(int(fp32_dest_acc_en)),
                    "@TILE_BYTES@": str(tile_bytes),
                    "@OUT_SUBBLOCK_TILES@": str(out_subblock_tiles),
                    "@ROWS_PER_WORKER@": str(rows_per_worker),
                }
                cpp_dir = Path(__file__).with_name("cpp")
                sources = {
                    "read": (cpp_dir / "attention_dv_reference_reader.cpp").read_text(encoding="utf-8"),
                    "compute": (cpp_dir / "attention_dv_reference_compute.cpp").read_text(encoding="utf-8"),
                    "write": (cpp_dir / "attention_dv_reference_writer.cpp").read_text(encoding="utf-8"),
                }
                reader_position = {
                    tensor_index: position
                    for position, tensor_index in enumerate(reader_indices[0])
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
                    r"get_tensor_accessor_args_cta_offset<(\d+), 5>\(\),\s*(\d+)",
                    lambda match: (
                        "get_tensor_accessor_args_cta_offset<"
                        f"{match.group(1)}, 5>(), "
                        f"{reader_position[int(match.group(1))]}"
                    ),
                    sources["read"],
                )
                for kind, source in tuple(sources.items()):
                    for placeholder, value in replacements.items():
                        source = source.replace(placeholder, value)
                    if re.search(r"@[A-Z0-9_]+@", source):
                        raise RuntimeError(f"unexpanded reference dV {kind} placeholder")
                    sources[kind] = source
                for path, thread_type in compiled.kernel_paths:
                    kind = "compute" if thread_type == "compute" else (
                        "read" if "read" in Path(path).name else "write"
                    )
                    Path(path).write_text(sources[kind], encoding="utf-8")
                if dtype_name == "bfloat16" and fp32_dest_acc_en:
                    # TTNN's BF16 BMM always configures compute against a
                    # Float32 mm-partials CB, even when K fits in one block and
                    # the final output is packed directly to BF16.  The CB
                    # format changes the observable low-level matmul setup.
                    from ttl.dataflow_buffer import CompilerAllocatedDFBConfig

                    partial_config = CompilerAllocatedDFBConfig(
                        dfb_index=4,
                        num_tiles=1,
                        data_format="float32",
                        block_count=rows_per_worker * dim_tiles,
                    )
                    partial_config._cb_index = 4
                    compiled.cb_configs[4] = partial_config
                for config in compiled.kernel_configs:
                    if hasattr(config, "math_fidelity"):
                        enum_type = type(config.math_fidelity)
                        config.math_fidelity = getattr(enum_type, "HiFi4")
                    if hasattr(config, "math_approx_mode"):
                        # TT-MLIR/TTNN's default Blackhole BF16 BMM is HiFi4
                        # with approximation mode enabled.  This is observable
                        # at the final BF16 boundary even for plain matmul.
                        config.math_approx_mode = True
                    if hasattr(config, "fp32_dest_acc_en"):
                        config.fp32_dest_acc_en = fp32_dest_acc_en
            break
        return result

    operation.plan = plan
    operation.phase_tiles = phase_tiles
    operation.dtype_name = dtype_name
    operation.fp32_dest_acc_en = fp32_dest_acc_en
    operation.matmul_full_fp32 = matmul_full_fp32
    operation.out_subblock_tiles = out_subblock_tiles
    operation.rows_per_worker = rows_per_worker
    return operation
