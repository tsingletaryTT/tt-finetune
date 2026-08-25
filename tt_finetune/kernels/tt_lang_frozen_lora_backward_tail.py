"""Exact native second phase for frozen Linear + LoRA backward.

The first native phase already materializes the required BF16 ``dH`` boundary
while producing frozen-base dX and dB.  This operation consumes base dX and dH
in one custom call, returning both the final combined dX and exact dA:

    rows [0:M],   cols [0:K] = final dX
    rows [M:M+R], cols [0:K] = dA
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import (
    BLACKHOLE_P150A_GRID,
)
from .tt_lang_lora import TILE_SIZE, _FP32_DTYPE_ONLY_TENSOR


ttl = None
tt_lang_operation = None


def _shift_compile_time_cb_indices(source: str, offset: int) -> str:
    import re

    pattern = re.compile(r"get_compile_time_arg_val\((\d+)\)")
    return pattern.sub(
        lambda match: (
            "get_compile_time_arg_val("
            f"{int(match.group(1)) + offset})"
        ),
        source,
    )


@lru_cache(maxsize=32)
def build_tt_lang_frozen_lora_backward_tail(
    *,
    tokens: int,
    input_features: int,
    physical_rank: int,
) -> Callable:
    dimensions = (tokens, input_features, physical_rank)
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        raise ValueError("all backward-tail dimensions must be positive /32")
    if physical_rank != TILE_SIZE:
        raise ValueError("exact backward tail requires physical rank 32")
    m_tiles = tokens // TILE_SIZE
    k_tiles = input_features // TILE_SIZE
    if m_tiles % 2:
        raise ValueError("exact dA requires an even token-tile count")
    if k_tiles > 32:
        raise ValueError("backward-tail staging currently supports K <= 32")

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import (
                tt_lang_operation as tt_lang_operation_decorator,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The native backward tail requires TT-XLA"
            ) from exc
        ttl = ttl_module
        tt_lang_operation = tt_lang_operation_decorator

    variant = hashlib.sha256(
        (
            f"{tokens}:{input_features}:{physical_rank}:"
            "frozen-lora-backward-tail-v2-full-grid-dx"
        ).encode("ascii")
    ).hexdigest()[:16]

    @ttl.operation(
        grid=BLACKHOLE_P150A_GRID,
        fp32_dest_acc_en=True,
        options="--ttl-matmul-full-fp32",
    )
    def _tail_program(
        base_input_gradient,
        hidden_gradient,
        lora_a,
        inputs,
        arena,
    ):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        dx_jobs = m_tiles * k_tiles
        dx_jobs_per_node = -(-dx_jobs // total_nodes)
        k_tiles_per_node = -(-k_tiles // total_nodes)
        staged_dx_tiles = max(2, dx_jobs_per_node)

        base_dx_dfb = ttl.make_dataflow_buffer_like(
            base_input_gradient,
            shape=(1, 1),
            block_count=staged_dx_tiles,
        )
        dh_dfb = ttl.make_dataflow_buffer_like(
            hidden_gradient, shape=(1, 1), block_count=2
        )
        a_dfb = ttl.make_dataflow_buffer_like(
            lora_a, shape=(1, 1), block_count=2
        )
        lora_dx_dfb = ttl.make_dataflow_buffer_like(
            arena,
            shape=(1, 1),
            block_count=staged_dx_tiles,
        )
        combined_dx_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=2
        )

        da_dh_dfb = ttl.make_dataflow_buffer_like(
            hidden_gradient, shape=(m_tiles, 1), block_count=1
        )
        da_x_dfb = ttl.make_dataflow_buffer_like(
            inputs, shape=(m_tiles, 1), block_count=1
        )
        da_transposed_x_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, m_tiles), block_count=1
        )
        da_partial_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=1
        )
        da_pretranspose_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=1
        )
        da_output_dfb = ttl.make_dataflow_buffer_like(
            arena, shape=(1, 1), block_count=1
        )

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(dx_jobs_per_node):
                job = node_index * dx_jobs_per_node + local_job
                if job < dx_jobs:
                    m_tile = job // k_tiles
                    k_tile = job - m_tile * k_tiles
                    with (
                        base_dx_dfb.reserve() as base,
                        dh_dfb.reserve() as dh,
                        a_dfb.reserve() as a,
                    ):
                        tx_base = ttl.copy(
                            base_input_gradient[
                                m_tile : m_tile + 1,
                                k_tile : k_tile + 1,
                            ],
                            base,
                        )
                        tx_dh = ttl.copy(
                            hidden_gradient[
                                m_tile : m_tile + 1, 0:1
                            ],
                            dh,
                        )
                        tx_a = ttl.copy(
                            lora_a[0:1, k_tile : k_tile + 1],
                            a,
                        )
                        tx_base.wait()
                        tx_dh.wait()
                        tx_a.wait()
            for local_k in range(k_tiles_per_node):
                k_tile = node_index * k_tiles_per_node + local_k
                if k_tile < k_tiles:
                    with (
                        da_dh_dfb.reserve() as dh,
                        da_x_dfb.reserve() as x,
                    ):
                        tx_dh = ttl.copy(
                            hidden_gradient[0:m_tiles, 0:1], dh
                        )
                        tx_x = ttl.copy(
                            inputs[
                                0:m_tiles, k_tile : k_tile + 1
                            ],
                            x,
                        )
                        tx_dh.wait()
                        tx_x.wait()

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(dx_jobs_per_node):
                job = node_index * dx_jobs_per_node + local_job
                if job < dx_jobs:
                    with (
                        dh_dfb.wait() as dh,
                        a_dfb.wait() as a,
                    ):
                        with lora_dx_dfb.reserve() as lora_dx:
                            lora_dx.store(ttl.matmul(dh, a))
                    with (
                        base_dx_dfb.wait() as base,
                        lora_dx_dfb.wait() as lora_dx,
                    ):
                        with combined_dx_dfb.reserve() as combined:
                            combined.store(base + lora_dx)
            for local_k in range(k_tiles_per_node):
                k_tile = node_index * k_tiles_per_node + local_k
                if k_tile < k_tiles:
                    with da_x_dfb.wait() as x:
                        with da_transposed_x_dfb.reserve() as transposed:
                            transposed.store(ttl.transpose(x))
                    with (
                        da_transposed_x_dfb.wait() as transposed,
                        da_dh_dfb.wait() as dh,
                    ):
                        with da_partial_dfb.reserve() as partial:
                            partial.store(ttl.matmul(transposed, dh))
                    with da_partial_dfb.wait() as partial:
                        with da_pretranspose_dfb.reserve() as before:
                            before.store(partial)
                    with da_pretranspose_dfb.wait() as before:
                        with da_output_dfb.reserve() as output:
                            output.store(ttl.transpose(before))

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(dx_jobs_per_node):
                job = node_index * dx_jobs_per_node + local_job
                if job < dx_jobs:
                    m_tile = job // k_tiles
                    k_tile = job - m_tile * k_tiles
                    with combined_dx_dfb.wait() as combined:
                        tx = ttl.copy(
                            combined,
                            arena[
                                m_tile : m_tile + 1,
                                k_tile : k_tile + 1,
                            ],
                        )
                        tx.wait()
            for local_k in range(k_tiles_per_node):
                k_tile = node_index * k_tiles_per_node + local_k
                if k_tile < k_tiles:
                    with da_output_dfb.wait() as da:
                        tx = ttl.copy(
                            da,
                            arena[
                                m_tiles : m_tiles + 1,
                                k_tile : k_tile + 1,
                            ],
                        )
                        tx.wait()

    @tt_lang_operation(
        operation_id=f"tt_finetune.frozen_lora_backward_tail.{variant}",
        arg_roles=("in", "in", "in", "in", "out"),
        version_tag=f"frozen-lora-backward-tail-{variant}",
    )
    def backward_tail(
        base_input_gradient,
        hidden_gradient,
        lora_a,
        inputs,
        arena,
    ):
        _clear_ttl_compile_wrapper_cache(_tail_program)
        result = _tail_program(
            base_input_gradient,
            hidden_gradient,
            lora_a,
            inputs,
            arena,
        )
        closure = _tail_program.__closure__ or ()
        for name, cell in zip(
            _tail_program.__code__.co_freevars, closure
        ):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 11:
                    raise RuntimeError(
                        "backward tail expected 11 CBs, got "
                        f"{len(compiled.cb_configs)}"
                    )
                compiled.cb_configs[8].tensor = _FP32_DTYPE_ONLY_TENSOR
                compute_paths = [
                    Path(path)
                    for path, thread_type in compiled.kernel_paths
                    if thread_type == "compute"
                ]
                if len(compute_paths) != 1:
                    raise RuntimeError(
                        "backward tail expected one compute kernel"
                    )
                grid = compiled.core_ranges.bounding_box().grid_size()
                total_nodes = int(grid.x) * int(grid.y)
                replacements = {
                    "@K_TILES@": str(k_tiles),
                    "@DX_JOBS_PER_NODE@": str(
                        -(-(m_tiles * k_tiles) // total_nodes)
                    ),
                    "@TOTAL_DX_JOBS@": str(m_tiles * k_tiles),
                    "@K_TILES_PER_NODE@": str(
                        -(-k_tiles // total_nodes)
                    ),
                    "@TOTAL_K_TILES@": str(k_tiles),
                    "@REDUCTION_TILES@": str(m_tiles),
                    "@RANK_TILES@": "1",
                    "@GRID_COLS@": str(int(grid.x)),
                }
                da_source = (
                    Path(__file__).with_name("cpp")
                    / "lora_backward_da_exact_compute.cpp"
                ).read_text(encoding="utf-8")
                da_source = _shift_compile_time_cb_indices(
                    da_source, 5
                ).replace(
                    "void kernel_main()",
                    "FORCE_INLINE void da_phase()",
                    1,
                )
                tail_source = (
                    Path(__file__).with_name("cpp")
                    / "frozen_lora_backward_tail_exact_compute.cpp"
                ).read_text(encoding="utf-8")
                source = da_source + "\n" + tail_source
                for placeholder, value in replacements.items():
                    source = source.replace(placeholder, value)
                if any(key in source for key in replacements):
                    raise RuntimeError(
                        "unexpanded backward-tail placeholder"
                    )
                compute_paths[0].write_text(source, encoding="utf-8")
            break
        return result

    return backward_tail
