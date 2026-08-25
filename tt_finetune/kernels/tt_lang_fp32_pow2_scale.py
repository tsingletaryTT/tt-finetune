"""Exact resident FP32 power-of-two scale without an L1 TTNN bcast path."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID


ttl = None
tt_lang_operation = None


def _load_runtime() -> None:
    global ttl, tt_lang_operation
    if ttl is not None and tt_lang_operation is not None:
        return
    try:
        import ttl as ttl_module
        from tt_torch.tt_lang import tt_lang_operation as decorator
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("FP32 power-of-two scale requires TT-XLA") from exc
    ttl = ttl_module
    tt_lang_operation = decorator


def _replace_sources(
    program, *, jobs: int, shift: int, canonical_ttnn_multiply: bool,
) -> None:
    closure = program.__closure__ or ()
    for name, cell in zip(program.__code__.co_freevars, closure):
        if name != "cache" or not isinstance(cell.cell_contents, dict):
            continue
        for compiled in cell.cell_contents.values():
            if len(compiled.cb_configs) != 2:
                raise RuntimeError(
                    f"FP32 scale expected two CBs, got {len(compiled.cb_configs)}"
                )
            compiled.num_pipe_sync_semaphores = 0
            jobs_per_core = -(-jobs // 110)
            cpp = Path(__file__).with_name("cpp")
            replacements = {
                "@JOBS@": str(jobs),
                "@JOBS_PER_CORE@": str(jobs_per_core),
                "@SHIFT@": str(shift),
                "@CANONICAL_TTNN_MULTIPLY@": (
                    "1" if canonical_ttnn_multiply else "0"
                ),
            }
            for path, thread in compiled.kernel_paths:
                target = Path(path)
                if thread == "compute":
                    target.write_text("void kernel_main() {}\n", encoding="utf-8")
                    continue
                kind = "reader" if "read" in target.name else "writer"
                source = (cpp / f"fp32_pow2_scale_{kind}.cpp").read_text()
                for marker, value in replacements.items():
                    source = source.replace(marker, value)
                target.write_text(source, encoding="utf-8")
        break


@lru_cache(maxsize=8)
def build_fp32_pow2_scale(
    *, rows: int, columns: int, shift: int,
    canonical_ttnn_multiply: bool = False,
) -> Callable:
    if rows <= 0 or columns <= 0 or rows % 32 or columns % 32:
        raise ValueError("FP32 scale dimensions must be positive and tile aligned")
    if shift < 0 or shift > 31:
        raise ValueError("FP32 scale shift must be in [0, 31]")
    _load_runtime()
    jobs = (rows // 32) * (columns // 32)

    @ttl.operation(grid=BLACKHOLE_P150A_GRID)
    def _program(source, output):
        incoming = ttl.make_dataflow_buffer_like(
            source, shape=(1, 1), block_count=1
        )
        scaled = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=1
        )

        @ttl.datamovement()
        def read():
            with incoming.reserve() as tile:
                tx = ttl.copy(source[0, 0, 0:1, 0:1], tile); tx.wait()

        @ttl.compute()
        def compute():
            with incoming.wait() as value:
                with scaled.reserve() as result:
                    result.store(ttl.math.typecast(value, torch.float32))

        @ttl.datamovement()
        def write():
            with scaled.wait() as tile:
                tx = ttl.copy(tile, output[0, 0, 0:1, 0:1]); tx.wait()

    variant = hashlib.sha256(
        f"fp32-pow2:{rows}:{columns}:{shift}:"
        f"ttnnmul{int(canonical_ttnn_multiply)}:v4".encode()
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.fp32_pow2_scale.{variant}",
        arg_roles=("in", "out"),
        version_tag=f"fp32-pow2-scale-{variant}-v4",
    )
    def operation(source, output):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(source, output)
        _replace_sources(
            _program,
            jobs=jobs,
            shift=shift,
            canonical_ttnn_multiply=canonical_ttnn_multiply,
        )
        return result

    operation.output_shape = (rows, columns)
    operation.raw_tiles = jobs
    operation.shift = shift
    operation.canonical_ttnn_multiply = bool(canonical_ttnn_multiply)
    operation._ttl_program = _program
    return operation


__all__ = ["build_fp32_pow2_scale"]


@lru_cache(maxsize=8)
def build_raw_arena_fp32_pow2_scale(
    *, rows: int, columns: int, arena_rows: int, arena_columns: int,
    page_offset: int, shift: int,
) -> Callable:
    """Load split FP32 arena pages and scale before any reusable L1 tensor."""
    if rows <= 0 or columns <= 0 or rows % 32 or columns % 32:
        raise ValueError("FP32 scale dimensions must be positive and tile aligned")
    if arena_rows <= 0 or arena_columns <= 0 or arena_rows % 32 or arena_columns % 32:
        raise ValueError("FP32 arena dimensions must be positive and tile aligned")
    if shift < 0 or shift > 31:
        raise ValueError("FP32 scale shift must be in [0, 31]")
    jobs = (rows // 32) * (columns // 32)
    arena_pages = (arena_rows // 32) * (arena_columns // 32)
    if page_offset < 0 or page_offset + 2 * jobs > arena_pages:
        raise ValueError("FP32 raw scale region exceeds arena")
    _load_runtime()

    @ttl.operation(grid=BLACKHOLE_P150A_GRID)
    def _program(arena, output):
        incoming = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=2
        )
        scaled = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=1
        )

        @ttl.datamovement()
        def read():
            with incoming.reserve() as tile:
                tx = ttl.copy(arena[0:1, 0:1], tile); tx.wait()

        @ttl.compute()
        def compute():
            with incoming.wait() as value:
                with scaled.reserve() as result:
                    result.store(ttl.math.typecast(value, torch.float32))

        @ttl.datamovement()
        def write():
            with scaled.wait() as tile:
                tx = ttl.copy(tile, output[0, 0, 0:1, 0:1]); tx.wait()

    variant = hashlib.sha256(
        f"raw-arena-fp32-pow2:{rows}:{columns}:{arena_rows}:"
        f"{arena_columns}:{page_offset}:{shift}:v2".encode()
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.raw_arena_fp32_pow2_scale.{variant}",
        arg_roles=("in", "out"),
        version_tag=f"raw-arena-fp32-pow2-scale-{variant}",
    )
    def operation(arena, output):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(arena, output)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                compiled.num_pipe_sync_semaphores = 0
                jobs_per_core = -(-jobs // 110)
                cpp = Path(__file__).with_name("cpp")
                replacements = {
                    "@JOBS@": str(jobs),
                    "@JOBS_PER_CORE@": str(jobs_per_core),
                    "@PAGE_OFFSET@": str(page_offset),
                    "@SHIFT@": str(shift),
                }
                for path, thread in compiled.kernel_paths:
                    target = Path(path)
                    if thread == "compute":
                        target.write_text("void kernel_main() {}\n", encoding="utf-8")
                        continue
                    filename = (
                        "raw_fp32_checkpoint_scale_reader.cpp"
                        if "read" in target.name else
                        "fp32_pow2_scale_writer.cpp"
                    )
                    source = (cpp / filename).read_text()
                    for marker, value in replacements.items():
                        source = source.replace(marker, value)
                    target.write_text(source, encoding="utf-8")
            break
        return result

    operation.output_shape = (rows, columns)
    operation.raw_tiles = jobs
    operation.shift = shift
    operation._ttl_program = _program
    return operation


__all__.append("build_raw_arena_fp32_pow2_scale")
