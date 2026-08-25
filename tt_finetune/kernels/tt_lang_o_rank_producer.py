"""Minimal exact O-LoRA rank producer for the resident resident runtime forward."""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache


@dataclass(frozen=True)
class _DTypeOnlyTensor:
    dtype: torch.dtype


_FP32_DTYPE_ONLY_TENSOR = _DTypeOnlyTensor(torch.float32)
ttl = None
tt_lang_operation = None


@lru_cache(maxsize=8)
def build_tt_lang_o_rank_producer(
    *,
    tokens: int,
    input_features: int,
    worker_grid: tuple[int, int] = (1, 4),
) -> Callable:
    """Build four independent token-tile H producers with FP32 partials."""

    if tokens <= 0 or tokens % 32 or input_features <= 0 or input_features % 32:
        raise ValueError("O-rank dimensions must be positive multiples of 32")
    m_tiles = tokens // 32
    k_tiles = input_features // 32
    grid_cols = int(worker_grid[0])
    if worker_grid[0] * worker_grid[1] < m_tiles:
        raise ValueError("O-rank grid must provide one core per token tile")

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("O-rank producer requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    options = {
        "grid": worker_grid,
        "fp32_dest_acc_en": True,
        "options": "--ttl-matmul-full-fp32",
    }
    if "math_approx_mode" in inspect.signature(ttl.operation).parameters:
        options["math_approx_mode"] = False

    @ttl.operation(**options)
    def _program(inputs, lora_a_t, arena):
        x_dfb = ttl.make_dataflow_buffer_like(
            inputs, shape=(1, 1), block_count=2
        )
        a_dfb = ttl.make_dataflow_buffer_like(
            lora_a_t, shape=(1, 1), block_count=2
        )
        partial_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=2
        )
        output_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=2
        )

        @ttl.datamovement()
        def read():
            col, row = ttl.node(dims=2)
            job = row * grid_cols + col
            if job < m_tiles:
                for k in range(k_tiles):
                    with (
                        x_dfb.reserve() as x_tile,
                        a_dfb.reserve() as a_tile,
                    ):
                        tx = ttl.copy(
                            inputs[job:job + 1, k:k + 1], x_tile
                        )
                        ta = ttl.copy(lora_a_t[k:k + 1, 0:1], a_tile)
                        tx.wait()
                        ta.wait()

        @ttl.compute()
        def compute():
            col, row = ttl.node(dims=2)
            job = row * grid_cols + col
            if job < m_tiles:
                with partial_dfb.reserve() as partial:
                    partial.store(ttl.block.fill(0, shape=partial.shape))
                for _ in range(k_tiles):
                    with (
                        x_dfb.wait() as x_tile,
                        a_dfb.wait() as a_tile,
                        partial_dfb.wait() as previous,
                    ):
                        with partial_dfb.reserve() as next_value:
                            next_value.store(
                                previous
                                + ttl.matmul(
                                    x_tile, a_tile
                                )
                            )
                with partial_dfb.wait() as partial:
                    with output_dfb.reserve() as output:
                        output.store(partial)

        @ttl.datamovement()
        def write():
            col, row = ttl.node(dims=2)
            job = row * grid_cols + col
            if job < m_tiles:
                with output_dfb.wait() as output:
                    tx = ttl.copy(output, arena[job:job + 1, 0:1])
                    tx.wait()

    variant = hashlib.sha256(
        f"{tokens}:{input_features}:{worker_grid}:o-rank-v1".encode()
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.o_rank_producer.{variant}",
        arg_roles=("in", "in", "out"),
        version_tag=f"o-rank-producer-{variant}",
    )
    def operation(inputs, lora_a_t, arena):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(inputs, lora_a_t, arena)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 4:
                    raise RuntimeError(
                        "O-rank producer expected four CBs, got "
                        f"{len(compiled.cb_configs)}"
                    )
                compiled.cb_configs[2].tensor = _FP32_DTYPE_ONLY_TENSOR
                compute_paths = [
                    Path(path)
                    for path, kind in compiled.kernel_paths
                    if kind == "compute"
                ]
                if len(compute_paths) != 1:
                    raise RuntimeError("O-rank producer expected one compute kernel")
                source = (
                    Path(__file__).with_name("cpp")
                    / "o_rank_ttnn_compute.cpp"
                ).read_text()
                source = source.replace("@K_TILES@", str(k_tiles))
                source = source.replace("@M_TILES@", str(m_tiles))
                source = source.replace("@GRID_COLS@", str(grid_cols))
                compute_paths[0].write_text(source)
                # Match the untouched TTNN matmul that produced the captured
                # full-model reference.  On Blackhole, APPROX participates in
                # the internal matmul schedule even with HiFi4 and FP32 dest
                # accumulation, so it is part of the exact numerical ABI.
                for config in compiled.kernel_configs:
                    if hasattr(config, "math_fidelity"):
                        enum_type = type(config.math_fidelity)
                        config.math_fidelity = getattr(enum_type, "HiFi4")
                    if hasattr(config, "math_approx_mode"):
                        config.math_approx_mode = True
                    if hasattr(config, "fp32_dest_acc_en"):
                        config.fp32_dest_acc_en = True
            break
        return result

    operation.output_shape = (tokens, 32)
    operation.physical_programs = 1
    return operation
