"""Opaque FP32 checkpoint storage inside one BF16 TT-Lang output arena.

This small qualified primitive establishes the physical representation used
by the full attention-forward owner.  It performs no numerical conversion:
one FP32 tile is split over two BF16-sized pages and reconstructed verbatim.
"""

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
        raise RuntimeError("raw FP32 checkpoint requires TT-XLA") from exc
    ttl = ttl_module
    tt_lang_operation = decorator


def raw_fp32_arena_shape(*, rows: int, columns: int) -> tuple[int, int]:
    if rows <= 0 or columns <= 0 or rows % 32 or columns % 32:
        raise ValueError("raw checkpoint dimensions must be positive /32")
    return rows * 2, columns


def _replace_sources(
    program, *, jobs: int, mode: str, raw_cb_index: int,
    page_offset: int = 0,
) -> None:
    closure = program.__closure__ or ()
    for name, cell in zip(program.__code__.co_freevars, closure):
        if name != "cache" or not isinstance(cell.cell_contents, dict):
            continue
        for compiled in cell.cell_contents.values():
            if len(compiled.cb_configs) != 2:
                raise RuntimeError(
                    f"raw checkpoint expected two CBs, got {len(compiled.cb_configs)}"
                )
            compiled.num_pipe_sync_semaphores = 0
            jobs_per_core = -(-jobs // 110)
            cpp = Path(__file__).with_name("cpp")
            sources = {
                "read": (cpp / f"raw_fp32_checkpoint_{mode}_reader.cpp").read_text(),
                "write": (cpp / f"raw_fp32_checkpoint_{mode}_writer.cpp").read_text(),
            }
            for kind in sources:
                sources[kind] = (
                    sources[kind]
                    .replace("@JOBS@", str(jobs))
                    .replace("@JOBS_PER_CORE@", str(jobs_per_core))
                    .replace("@RAW_CB_INDEX@", str(raw_cb_index))
                    .replace("@PAGE_OFFSET@", str(page_offset))
                )
            for path, thread in compiled.kernel_paths:
                if thread == "compute":
                    Path(path).write_text("void kernel_main() {}\n", encoding="utf-8")
                else:
                    kind = "read" if "read" in Path(path).name else "write"
                    Path(path).write_text(sources[kind], encoding="utf-8")
        break


@lru_cache(maxsize=16)
def build_raw_fp32_checkpoint_pack(
    *, rows: int, columns: int,
    arena_rows: int | None = None,
    arena_columns: int | None = None,
    page_offset: int = 0,
) -> Callable:
    _load_runtime()
    default_arena_rows, default_arena_columns = raw_fp32_arena_shape(
        rows=rows, columns=columns
    )
    arena_rows = default_arena_rows if arena_rows is None else arena_rows
    arena_columns = (
        default_arena_columns if arena_columns is None else arena_columns
    )
    if (
        arena_rows <= 0 or arena_columns <= 0
        or arena_rows % 32 or arena_columns % 32
        or page_offset < 0
    ):
        raise ValueError("physical raw arena must be positive /32")
    jobs = (rows // 32) * (columns // 32)
    arena_pages = (arena_rows // 32) * (arena_columns // 32)
    if page_offset + 2 * jobs > arena_pages:
        raise ValueError("raw checkpoint region exceeds physical arena")

    @ttl.operation(grid=BLACKHOLE_P150A_GRID)
    def _program(source, arena):
        raw = ttl.make_dataflow_buffer_like(source, shape=(1, 1), block_count=2)
        typed_output = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=2
        )

        @ttl.datamovement()
        def read():
            with raw.reserve() as block:
                tx = ttl.copy(source[0:1, 0:1], block); tx.wait()

        @ttl.compute()
        def compute():
            with raw.wait() as value:
                with typed_output.reserve() as result:
                    result.store(ttl.math.typecast(value, torch.bfloat16))

        @ttl.datamovement()
        def write():
            with typed_output.wait() as block:
                tx = ttl.copy(block, arena[0:1, 0:1]); tx.wait()

    variant = hashlib.sha256(
        f"pack:{rows}:{columns}:{arena_rows}:{arena_columns}:"
        f"{page_offset}:v2".encode()
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.raw_fp32_pack.{variant}",
        arg_roles=("in", "out"),
        version_tag=f"raw-fp32-pack-{variant}",
    )
    def operation(source, arena):
        if tuple(arena.shape) != (arena_rows, arena_columns):
            raise ValueError(
                f"raw arena must be {(arena_rows, arena_columns)}, "
                f"got {tuple(arena.shape)}"
            )
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(source, arena)
        _replace_sources(
            _program, jobs=jobs, mode="pack", raw_cb_index=0,
            page_offset=page_offset,
        )
        return result

    operation.output_shape = (arena_rows, arena_columns)
    operation.raw_tiles = jobs
    operation._ttl_program = _program
    return operation


@lru_cache(maxsize=8)
def build_raw_fp32_checkpoint_unpack(
    *, rows: int, columns: int,
    arena_rows: int | None = None,
    arena_columns: int | None = None,
    page_offset: int = 0,
) -> Callable:
    _load_runtime()
    default_arena_rows, default_arena_columns = raw_fp32_arena_shape(
        rows=rows, columns=columns
    )
    arena_rows = default_arena_rows if arena_rows is None else arena_rows
    arena_columns = (
        default_arena_columns if arena_columns is None else arena_columns
    )
    if (
        arena_rows <= 0 or arena_columns <= 0
        or arena_rows % 32 or arena_columns % 32
        or page_offset < 0
    ):
        raise ValueError("physical raw arena must be positive /32")
    jobs = (rows // 32) * (columns // 32)
    arena_pages = (arena_rows // 32) * (arena_columns // 32)
    if page_offset + 2 * jobs > arena_pages:
        raise ValueError("raw checkpoint region exceeds physical arena")

    @ttl.operation(grid=BLACKHOLE_P150A_GRID)
    def _program(arena, output):
        typed_input = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=2
        )
        raw = ttl.make_dataflow_buffer_like(output, shape=(1, 1), block_count=2)

        @ttl.datamovement()
        def read():
            with typed_input.reserve() as block:
                tx = ttl.copy(arena[0:1, 0:1], block); tx.wait()

        @ttl.compute()
        def compute():
            with typed_input.wait() as value:
                with raw.reserve() as result:
                    result.store(ttl.math.typecast(value, torch.float32))

        @ttl.datamovement()
        def write():
            with raw.wait() as block:
                tx = ttl.copy(block, output[0:1, 0:1]); tx.wait()

    variant = hashlib.sha256(
        f"unpack:{rows}:{columns}:{arena_rows}:{arena_columns}:{page_offset}:v2".encode()
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.raw_fp32_unpack.{variant}",
        arg_roles=("in", "out"),
        version_tag=f"raw-fp32-unpack-{variant}",
    )
    def operation(arena, output):
        if tuple(arena.shape) != (arena_rows, arena_columns):
            raise ValueError(
                f"raw arena must be {(arena_rows, arena_columns)}, got {tuple(arena.shape)}"
            )
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(arena, output)
        _replace_sources(
            _program, jobs=jobs, mode="unpack", raw_cb_index=1,
            page_offset=page_offset,
        )
        return result

    operation.output_shape = (rows, columns)
    operation.raw_tiles = jobs
    operation._ttl_program = _program
    return operation
