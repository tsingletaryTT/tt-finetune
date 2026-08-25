"""Row-cached grouped gate/up forward for layer composition.

Unlike the earlier generic grouped projection, this schedule caches each
normalized token row once per participating core and computes each logical
gate/up rank projection once per row. Rank-H is then multicast through L1/NoC
to the output owners. The standalone operation qualifies the arithmetic and
CB lifecycle.  The 50-core form matches the attention projection island; the
qualified 110-core form rejoins both attention islands after the device-side
post-normalization barrier and is embedded by the Transformer-layer owner.
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

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_attention_forward_resident import (
    _ALL_WORKERS_RANGE,
    _PROJECTION_RANGE,
)
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID
from .tt_lang_lora import TILE_SIZE


ttl = None
tt_lang_operation = None
_CBS = 15
_WORKERS = 50


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _maximum_cached_rows(
    *, sequence_tiles: int, intermediate_tiles: int, jobs_per_core: int,
    workers: int,
) -> int:
    total_jobs = sequence_tiles * 2 * intermediate_tiles
    maximum = 0
    for core in range(workers):
        first = core * jobs_per_core
        stop = min(first + jobs_per_core, total_jobs)
        if stop <= first:
            continue
        maximum = max(
            maximum,
            (stop - 1) // (2 * intermediate_tiles)
            - first // (2 * intermediate_tiles)
            + 1,
        )
    return maximum


def _maximum_local_groups(
    *, sequence_tiles: int, intermediate_tiles: int, jobs_per_core: int,
    workers: int,
) -> int:
    total_jobs = sequence_tiles * 2 * intermediate_tiles
    maximum = 0
    for core in range(workers):
        first = core * jobs_per_core
        stop = min(first + jobs_per_core, total_jobs)
        if stop <= first:
            continue
        maximum = max(
            maximum,
            (stop - 1) // intermediate_tiles
            - first // intermediate_tiles
            + 1,
        )
    return maximum


def resident_grouped_gate_up_output_shape(
    *, tokens: int, intermediate_features: int,
) -> tuple[int, int]:
    return tokens, 2 * intermediate_features + 2 * TILE_SIZE


@lru_cache(maxsize=16)
def build_tt_lang_resident_grouped_gate_up_forward(
    *, scale: float, tokens: int, hidden_features: int,
    intermediate_features: int, base_k_tiles_per_phase: int,
    worker_count: int = _WORKERS,
) -> Callable:
    dimensions = (tokens, hidden_features, intermediate_features)
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        raise ValueError("resident grouped gate/up dimensions must be positive /32")
    sequence_tiles = tokens // TILE_SIZE
    hidden_tiles = hidden_features // TILE_SIZE
    intermediate_tiles = intermediate_features // TILE_SIZE
    if hidden_tiles % 2:
        raise ValueError("resident grouped gate/up requires an even hidden tile count")
    if (
        base_k_tiles_per_phase <= 0
        or hidden_tiles % base_k_tiles_per_phase
    ):
        raise ValueError("invalid grouped gate/up frozen-base K phase")
    if worker_count not in (50, 110):
        raise ValueError("resident grouped gate/up requires 50 or 110 workers")
    grid_columns = 5 if worker_count == 50 else 11
    core_range = _PROJECTION_RANGE if worker_count == 50 else _ALL_WORKERS_RANGE
    total_jobs = sequence_tiles * 2 * intermediate_tiles
    jobs_per_core = _ceil_div(total_jobs, worker_count)
    if jobs_per_core > 32:
        raise ValueError("resident grouped gate/up exceeds 32 jobs per core")
    cached_rows = _maximum_cached_rows(
        sequence_tiles=sequence_tiles,
        intermediate_tiles=intermediate_tiles,
        jobs_per_core=jobs_per_core,
        workers=worker_count,
    )
    local_groups = _maximum_local_groups(
        sequence_tiles=sequence_tiles,
        intermediate_tiles=intermediate_tiles,
        jobs_per_core=jobs_per_core,
        workers=worker_count,
    )

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("resident grouped gate/up requires TT-XLA") from exc
        ttl = ttl_module
        tt_lang_operation = decorator

    options: dict[str, object] = {
        "grid": BLACKHOLE_P150A_GRID,
        "fp32_dest_acc_en": True,
        "options": "--ttl-matmul-full-fp32",
    }
    if "math_approx_mode" in inspect.signature(ttl.operation).parameters:
        options["math_approx_mode"] = False

    @ttl.operation(**options)
    def _program(inputs, frozen_weight_t, a0, b0, a1, b1, arena):
        b_0 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b_1 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b_2 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b_3 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b_4 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b_5 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b_6 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b_7 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b_8 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b_9 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b_10 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b_11 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b_12 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b_13 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=1)
        b_14 = ttl.make_dataflow_buffer_like(arena, shape=(1, 1), block_count=8)

        @ttl.datamovement()
        def read():
            with b_0.reserve() as block:
                tx = ttl.copy(inputs[0:1, 0:1], block); tx.wait()
            with b_0.reserve() as block:
                tx = ttl.copy(frozen_weight_t[0:1, 0:1], block); tx.wait()
            with b_0.reserve() as block:
                tx = ttl.copy(a0[0:1, 0:1], block); tx.wait()
            with b_0.reserve() as block:
                tx = ttl.copy(b0[0:1, 0:1], block); tx.wait()
            with b_0.reserve() as block:
                tx = ttl.copy(a1[0:1, 0:1], block); tx.wait()
            with b_0.reserve() as block:
                tx = ttl.copy(b1[0:1, 0:1], block); tx.wait()

        @ttl.compute()
        def compute():
            for _ in range(6):
                with b_0.wait() as value:
                    with b_14.reserve() as result:
                        result.store(value)
            x, y = ttl.node(dims=2)
            if x + y < 0:
                with b_1.wait() as _: pass
                with b_2.wait() as _: pass
                with b_3.wait() as _: pass
                with b_4.wait() as _: pass
                with b_5.wait() as _: pass
                with b_6.wait() as _: pass
                with b_7.wait() as _: pass
                with b_8.wait() as _: pass
                with b_9.wait() as _: pass
                with b_10.wait() as _: pass
                with b_11.wait() as _: pass
                with b_12.wait() as _: pass
                with b_13.wait() as _: pass

        @ttl.datamovement()
        def write():
            for _ in range(6):
                with b_14.wait() as value:
                    tx = ttl.copy(value, arena[0:1, 0:1])
                    tx.wait()

    scale_bf16 = float(torch.tensor(scale, dtype=torch.bfloat16).float().item())
    scale_bits = struct.unpack("<I", struct.pack("<f", scale_bf16))[0]
    variant = hashlib.sha256(
        (
            f"resident-grouped-gate-up-v1:{tokens}:{hidden_features}:"
            f"{intermediate_features}:{base_k_tiles_per_phase}:"
            f"{scale_bf16:.17g}:{worker_count}"
        ).encode("ascii")
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.resident_grouped_gate_up.{variant}",
        arg_roles=("in",) * 6 + ("out",),
        version_tag=f"resident-grouped-gate-up-{variant}",
    )
    def operation(inputs, frozen_weight_t, a0, b0, a1, b1, arena):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(inputs, frozen_weight_t, a0, b0, a1, b1, arena)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != _CBS:
                    raise RuntimeError(
                        f"resident grouped gate/up expected {_CBS} CBs, "
                        f"got {len(compiled.cb_configs)}"
                    )
                replacements = {
                    "@SEQ_TILES@": str(sequence_tiles),
                    "@HIDDEN_TILES@": str(hidden_tiles),
                    "@INTERMEDIATE_TILES@": str(intermediate_tiles),
                    "@BASE_PHASE@": str(base_k_tiles_per_phase),
                    "@SCALE_BITS@": str(scale_bits),
                    "@WORKERS@": str(worker_count),
                    "@GRID_COLUMNS@": str(grid_columns),
                }
                cpp = Path(__file__).with_name("cpp")
                source_specs = (
                    ("resident_grouped_gate_up_reader.cpp", "noc"),
                    ("resident_grouped_gate_up_compute.cpp", "compute"),
                    ("resident_grouped_gate_up_writer.cpp", "noc"),
                )
                generated = Path(compiled.kernel_paths[0][0]).parent
                paths: list[tuple[str, str]] = []
                for filename, thread in source_specs:
                    source = (cpp / filename).read_text(encoding="utf-8")
                    for marker, value in replacements.items():
                        source = source.replace(marker, value)
                    if re.search(r"@[A-Z0-9_]+@", source):
                        raise RuntimeError(
                            f"unexpanded resident grouped gate/up marker in {filename}"
                        )
                    path = generated / filename
                    path.write_text(source, encoding="utf-8")
                    paths.append((str(path), thread))

                from ttl.dataflow_buffer import CompilerAllocatedDFBConfig

                product_per_core = _ceil_div(
                    sequence_tiles * intermediate_tiles, worker_count
                )
                capacities = (
                    cached_rows * hidden_tiles,
                    hidden_tiles,
                    2,
                    1,
                    1,
                    local_groups,
                    2 * base_k_tiles_per_phase,
                    2,
                    max(2, jobs_per_core),
                    2,
                    2,
                    max(2, jobs_per_core),
                    2,
                    max(1, product_per_core),
                    max(1, product_per_core),
                )
                for index, blocks in enumerate(capacities):
                    config = CompilerAllocatedDFBConfig(
                        dfb_index=index,
                        num_tiles=1,
                        data_format=(
                            "float32" if index in (2, 7) else "bfloat16"
                        ),
                        block_count=max(1, blocks),
                    )
                    config._cb_index = index
                    compiled.cb_configs[index] = config
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
                ]
                compiled.kernel_tensor_indices = [
                    [0, 1, 2, 3, 4, 5], [], [6],
                ]
                compiled.kernel_core_ranges = [
                    core_range, core_range, core_range,
                ]
                compiled.kernel_noc_roles = ["reader", None, "writer"]
                compiled.cb_core_ranges = [core_range] * _CBS
                compiled.semaphore_configs = [
                    {
                        "id": semaphore,
                        "core_type": "worker",
                        "core_range": core_range,
                        "initial_value": 0,
                    }
                    for semaphore in range(3)
                ]
                compiled.num_pipe_sync_semaphores = 3
                compiled.kernel_arg_specs = [[], [], []]
            break
        return result

    operation.output_shape = resident_grouped_gate_up_output_shape(
        tokens=tokens, intermediate_features=intermediate_features
    )
    operation.jobs_per_core = jobs_per_core
    operation.maximum_cached_rows = cached_rows
    operation.maximum_local_groups = local_groups
    operation.rank_reductions = 2 * sequence_tiles
    operation.worker_count = worker_count
    operation.intermediate_dram_bytes = 0
    operation.experimental_backend = "resident_grouped_gate_up_row_cached_v1"
    operation._ttl_program = _program
    return operation
