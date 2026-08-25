"""Exact native forward for a frozen biasless TT Linear on Blackhole P150a.

This is the base-only counterpart of the accepted frozen Linear backward.
It exists primarily for the causal-LM vocabulary projection, where lowering a
49k--152k-wide generic ``F.linear`` graph is compile-prohibitive.  The public
contract is ``Y[M,N] = X[M,K] @ W_t[K,N]``; weights are transposed once when
the frozen model edge is packed, never inside a training step.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import (
    BLACKHOLE_P150A_CORE_COUNT,
    BLACKHOLE_P150A_GRID,
)
from .tt_lang_lora import BF16_TILE_BYTES, TILE_SIZE, _FP32_DTYPE_ONLY_TENSOR


ttl = None
tt_lang_operation = None


def estimate_frozen_linear_forward_l1_bytes(
    *, reduction_tiles_per_phase: int = 1,
) -> int:
    """Conservative double-buffered exact long-K footprint."""

    if reduction_tiles_per_phase <= 0:
        raise ValueError("reduction phase must be positive")
    # Double-buffered X/W, double-buffered FP32 partial, BF16 result.
    bf16_equivalent_tiles = 4 * reduction_tiles_per_phase + 6
    return bf16_equivalent_tiles * BF16_TILE_BYTES


@lru_cache(maxsize=24)
def build_tt_lang_frozen_linear_forward(
    *,
    tokens: int,
    input_features: int,
    output_features: int,
    reduction_tiles_per_phase: int = 1,
) -> Callable:
    """Build exact ``X[M,K] @ W_t[K,N]`` without bias or parameter grads."""

    dimensions = (tokens, input_features, output_features)
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        raise ValueError("all frozen forward dimensions must be positive /32")
    m_tiles = tokens // TILE_SIZE
    k_tiles = input_features // TILE_SIZE
    n_tiles = output_features // TILE_SIZE
    if reduction_tiles_per_phase <= 0 or k_tiles % reduction_tiles_per_phase:
        raise ValueError("forward reduction phase must divide input-feature tiles")
    reduction_blocks = k_tiles // reduction_tiles_per_phase
    total_jobs = m_tiles * n_tiles
    jobs_per_node = -(-total_jobs // BLACKHOLE_P150A_CORE_COUNT)
    l1_bytes = estimate_frozen_linear_forward_l1_bytes(
        reduction_tiles_per_phase=reduction_tiles_per_phase
    )
    l1_budget = int(1.46 * 1024 * 1024 * 0.75)
    if l1_bytes > l1_budget:
        raise ValueError(
            "frozen forward exceeds the 75% worker L1 budget: "
            f"{l1_bytes} > {l1_budget}"
        )

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import (
                tt_lang_operation as tt_lang_operation_decorator,
            )
        except ImportError as exc:  # pragma: no cover - silicon-only path
            raise RuntimeError("native frozen Linear forward requires TT-XLA") from exc
        ttl = ttl_module
        tt_lang_operation = tt_lang_operation_decorator

    variant = hashlib.sha256(
        (
            f"{tokens}:{input_features}:{output_features}:"
            f"{reduction_tiles_per_phase}:"
            "frozen-linear-forward-v1-phased-p150a"
        ).encode("ascii")
    ).hexdigest()[:16]

    @ttl.operation(
        grid=BLACKHOLE_P150A_GRID,
        fp32_dest_acc_en=True,
        options="--ttl-matmul-full-fp32",
    )
    def _forward_program(inputs, frozen_weight_t, output):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        local_jobs = -(-total_jobs // total_nodes)

        x_dfb = ttl.make_dataflow_buffer_like(
            inputs, shape=(1, reduction_tiles_per_phase), block_count=2
        )
        weight_dfb = ttl.make_dataflow_buffer_like(
            frozen_weight_t,
            shape=(reduction_tiles_per_phase, 1),
            block_count=2,
        )
        partial_dfb = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=2
        )
        result_dfb = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=2
        )

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(local_jobs):
                job = node_index * local_jobs + local_job
                if job < total_jobs:
                    m_tile = job // n_tiles
                    n_tile = job - m_tile * n_tiles
                    for block in range(reduction_blocks):
                        k_start = block * reduction_tiles_per_phase
                        with x_dfb.reserve() as x_block, weight_dfb.reserve() as weight:
                            tx_x = ttl.copy(
                                inputs[
                                    m_tile : m_tile + 1,
                                    k_start : k_start + reduction_tiles_per_phase,
                                ],
                                x_block,
                            )
                            tx_w = ttl.copy(
                                frozen_weight_t[
                                    k_start : k_start + reduction_tiles_per_phase,
                                    n_tile : n_tile + 1,
                                ],
                                weight,
                            )
                            tx_x.wait()
                            tx_w.wait()

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(local_jobs):
                job = node_index * local_jobs + local_job
                if job < total_jobs:
                    with partial_dfb.reserve() as partial:
                        partial.store(ttl.block.fill(0, shape=partial.shape))
                    for _ in range(reduction_blocks):
                        with (
                            x_dfb.wait() as x_block,
                            weight_dfb.wait() as weight,
                            partial_dfb.wait() as previous,
                        ):
                            with partial_dfb.reserve() as next_value:
                                next_value.store(previous + ttl.matmul(x_block, weight))
                    with partial_dfb.wait() as partial:
                        with result_dfb.reserve() as result:
                            result.store(partial)

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(local_jobs):
                job = node_index * local_jobs + local_job
                if job < total_jobs:
                    m_tile = job // n_tiles
                    n_tile = job - m_tile * n_tiles
                    with result_dfb.wait() as result:
                        tx = ttl.copy(
                            result,
                            output[m_tile : m_tile + 1, n_tile : n_tile + 1],
                        )
                        tx.wait()

    @tt_lang_operation(
        operation_id=f"tt_finetune.frozen_linear_forward.{variant}",
        arg_roles=("in", "in", "out"),
        version_tag=f"frozen-linear-forward-{variant}",
    )
    def frozen_forward(inputs, frozen_weight_t, output):
        _clear_ttl_compile_wrapper_cache(_forward_program)
        result = _forward_program(inputs, frozen_weight_t, output)
        closure = _forward_program.__closure__ or ()
        for name, cell in zip(_forward_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 4:
                    raise RuntimeError(
                        "exact frozen forward expected 4 CBs, got "
                        f"{len(compiled.cb_configs)}"
                    )
                compiled.cb_configs[2].tensor = _FP32_DTYPE_ONLY_TENSOR
                compute_paths = [
                    Path(path)
                    for path, thread_type in compiled.kernel_paths
                    if thread_type == "compute"
                ]
                if len(compute_paths) != 1:
                    raise RuntimeError("exact frozen forward expected one compute kernel")
                grid = compiled.core_ranges.bounding_box().grid_size()
                total_nodes = int(grid.x) * int(grid.y)
                replacements = {
                    "@JOBS_PER_NODE@": str(-(-total_jobs // total_nodes)),
                    "@TOTAL_JOBS@": str(total_jobs),
                    # The shared exact source names the output width K and the
                    # reduction width N. Only N participates in arithmetic.
                    "@K_TILES@": str(n_tiles),
                    "@N_TILES@": str(k_tiles),
                    "@REDUCTION_TILES_PER_PHASE@": str(
                        reduction_tiles_per_phase
                    ),
                    "@GRID_COLS@": str(int(grid.x)),
                }
                source = (
                    Path(__file__).with_name("cpp")
                    / "frozen_linear_backward_dx_exact_compute.cpp"
                ).read_text(encoding="utf-8")
                for placeholder, value in replacements.items():
                    source = source.replace(placeholder, value)
                if any(key in source for key in replacements):
                    raise RuntimeError("unexpanded frozen forward placeholder")
                compute_paths[0].write_text(source, encoding="utf-8")
            break
        return result

    frozen_forward.output_shape = (tokens, output_features)
    frozen_forward.total_jobs = total_jobs
    frozen_forward.jobs_per_node = jobs_per_node
    frozen_forward.reduction_tiles_per_phase = reduction_tiles_per_phase
    frozen_forward.estimated_l1_bytes = l1_bytes
    frozen_forward.safe_default = False
    frozen_forward.experimental_backend = "frozen_linear_forward_exact_v1"
    frozen_forward._ttl_program = _forward_program
    return frozen_forward


__all__ = [
    "build_tt_lang_frozen_linear_forward",
    "estimate_frozen_linear_forward_l1_bytes",
]
