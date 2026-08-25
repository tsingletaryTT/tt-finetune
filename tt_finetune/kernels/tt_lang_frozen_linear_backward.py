"""Exact native input gradient for a frozen TT Linear on Blackhole P150a.

For ``Y = X @ W`` this operation computes only ``dX = dY @ W.T`` (the
orientation passed to this builder is already the required ``[N, K]``
matrix).  It deliberately has no dW output: base weights are frozen during
LoRA fine-tuning.
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
from .tt_lang_lora import (
    BF16_TILE_BYTES,
    TILE_SIZE,
    _FP32_DTYPE_ONLY_TENSOR,
)


ttl = None
tt_lang_operation = None


def estimate_frozen_linear_backward_l1_bytes(
    *,
    staged_jobs_per_node: int,
    reduction_tiles_per_phase: int = 1,
) -> int:
    """Conservative CB footprint for an exact phased K reduction."""

    if staged_jobs_per_node <= 0 or reduction_tiles_per_phase <= 0:
        raise ValueError("staging and reduction phase must be positive")
    # Double-buffered dY/W, double-buffered FP32 partials, staged BF16 dX.
    bf16_equivalent_tiles = (
        2 * reduction_tiles_per_phase
        + 2 * reduction_tiles_per_phase
        + 4
        + staged_jobs_per_node
    )
    return bf16_equivalent_tiles * BF16_TILE_BYTES


@lru_cache(maxsize=32)
def build_tt_lang_frozen_linear_backward_dx(
    *,
    tokens: int,
    input_features: int,
    output_features: int,
    reduction_tiles_per_phase: int = 1,
    retain_destination_across_reduction: bool = False,
    packer_l1_accumulate_across_reduction: bool = False,
) -> Callable:
    """Build exact ``dX = dY[M,N] @ W[N,K]`` without computing dW."""

    dimensions = (tokens, input_features, output_features)
    if any(value <= 0 or value % TILE_SIZE for value in dimensions):
        raise ValueError("all frozen backward dimensions must be positive /32")
    m_tiles = tokens // TILE_SIZE
    k_tiles = input_features // TILE_SIZE
    n_tiles = output_features // TILE_SIZE
    if (
        reduction_tiles_per_phase <= 0
        or n_tiles % reduction_tiles_per_phase
    ):
        raise ValueError(
            "backward reduction phase must divide output-feature tiles"
        )
    if (
        retain_destination_across_reduction
        and packer_l1_accumulate_across_reduction
    ):
        raise ValueError(
            "destination-resident and packer-L1 accumulation modes are "
            "mutually exclusive"
        )
    reduction_blocks = n_tiles // reduction_tiles_per_phase
    if packer_l1_accumulate_across_reduction and reduction_blocks < 2:
        raise ValueError(
            "packer-L1 accumulation requires at least two reduction blocks"
        )
    total_jobs = m_tiles * k_tiles
    jobs_per_node = -(-total_jobs // BLACKHOLE_P150A_CORE_COUNT)
    l1_bytes = estimate_frozen_linear_backward_l1_bytes(
        staged_jobs_per_node=jobs_per_node,
        reduction_tiles_per_phase=reduction_tiles_per_phase,
    )
    l1_budget = int(1.46 * 1024 * 1024 * 0.75)
    if l1_bytes > l1_budget:
        raise ValueError(
            "frozen backward exceeds the 75% worker L1 budget: "
            f"{l1_bytes} > {l1_budget}"
        )

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import (
                tt_lang_operation as tt_lang_operation_decorator,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The native frozen Linear backward requires TT-XLA"
            ) from exc
        ttl = ttl_module
        tt_lang_operation = tt_lang_operation_decorator

    variant = hashlib.sha256(
        (
            f"{tokens}:{input_features}:{output_features}:"
            f"{reduction_tiles_per_phase}:"
            f"{int(retain_destination_across_reduction)}:"
            f"{int(packer_l1_accumulate_across_reduction)}:"
            "frozen-linear-backward-dx-v8-packer-l1-diagnostic-p150a"
        ).encode("ascii")
    ).hexdigest()[:16]
    compile_evidence: dict[str, str] = {}

    @ttl.operation(
        grid=BLACKHOLE_P150A_GRID,
        fp32_dest_acc_en=True,
        options="--ttl-matmul-full-fp32",
    )
    def _backward_program(output_gradient, frozen_weight, input_gradient):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        local_jobs = -(-total_jobs // total_nodes)

        dy_dfb = ttl.make_dataflow_buffer_like(
            output_gradient,
            shape=(1, reduction_tiles_per_phase),
            block_count=2,
        )
        weight_dfb = ttl.make_dataflow_buffer_like(
            frozen_weight,
            shape=(reduction_tiles_per_phase, 1),
            block_count=2,
        )
        partial_dfb = ttl.make_dataflow_buffer_like(
            input_gradient,
            shape=(1, 1),
            block_count=(
                1 if packer_l1_accumulate_across_reduction else 2
            ),
        )
        result_dfb = ttl.make_dataflow_buffer_like(
            input_gradient,
            shape=(1, 1),
            block_count=max(2, jobs_per_node),
        )

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(local_jobs):
                job = node_index * local_jobs + local_job
                if job < total_jobs:
                    m_tile = job // k_tiles
                    k_tile = job - m_tile * k_tiles
                    for block in range(reduction_blocks):
                        n_start = block * reduction_tiles_per_phase
                        with (
                            dy_dfb.reserve() as dy,
                            weight_dfb.reserve() as weight,
                        ):
                            tx_dy = ttl.copy(
                                output_gradient[
                                    m_tile : m_tile + 1,
                                    n_start : n_start
                                    + reduction_tiles_per_phase,
                                ],
                                dy,
                            )
                            tx_weight = ttl.copy(
                                frozen_weight[
                                    n_start : n_start
                                    + reduction_tiles_per_phase,
                                    k_tile : k_tile + 1,
                                ],
                                weight,
                            )
                            tx_dy.wait()
                            tx_weight.wait()

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(local_jobs):
                job = node_index * local_jobs + local_job
                if job < total_jobs:
                    with partial_dfb.reserve() as partial:
                        partial.store(
                            ttl.block.fill(0, shape=partial.shape)
                        )
                    for _ in range(reduction_blocks):
                        with (
                            dy_dfb.wait() as dy,
                            weight_dfb.wait() as weight,
                            partial_dfb.wait() as previous,
                        ):
                            with partial_dfb.reserve() as next_value:
                                next_value.store(
                                    previous + ttl.matmul(dy, weight)
                                )
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
                    m_tile = job // k_tiles
                    k_tile = job - m_tile * k_tiles
                    with result_dfb.wait() as result:
                        tx = ttl.copy(
                            result,
                            input_gradient[
                                m_tile : m_tile + 1,
                                k_tile : k_tile + 1,
                            ],
                        )
                        tx.wait()

    @tt_lang_operation(
        operation_id=f"tt_finetune.frozen_linear_backward_dx.{variant}",
        arg_roles=("in", "in", "out"),
        version_tag=f"frozen-linear-backward-dx-{variant}",
    )
    def frozen_backward(output_gradient, frozen_weight, input_gradient):
        _clear_ttl_compile_wrapper_cache(_backward_program)
        result = _backward_program(
            output_gradient, frozen_weight, input_gradient
        )
        closure = _backward_program.__closure__ or ()
        for name, cell in zip(
            _backward_program.__code__.co_freevars, closure
        ):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 4:
                    raise RuntimeError(
                        "exact frozen backward expected 4 CBs, got "
                        f"{len(compiled.cb_configs)}"
                    )
                compiled.cb_configs[2].tensor = _FP32_DTYPE_ONLY_TENSOR
                compute_paths = [
                    Path(path)
                    for path, thread_type in compiled.kernel_paths
                    if thread_type == "compute"
                ]
                if len(compute_paths) != 1:
                    raise RuntimeError(
                        "exact frozen backward expected one compute kernel"
                    )
                for (_, thread_type), config in zip(
                    compiled.kernel_paths, compiled.kernel_configs
                ):
                    if thread_type != "compute":
                        continue
                    # TTNN's bmm_large 1D-systolic path is compiled with
                    # approximation mode enabled even at HiFi4.  This flag is
                    # part of the numerical contract we are reproducing for
                    # wide frozen-head reductions.
                    if packer_l1_accumulate_across_reduction:
                        config.math_approx_mode = True
                    for field in (
                        "math_fidelity",
                        "math_approx_mode",
                        "fp32_dest_acc_en",
                        "dst_full_sync_en",
                    ):
                        compile_evidence[field] = repr(
                            getattr(config, field, "unavailable")
                        )
                grid = compiled.core_ranges.bounding_box().grid_size()
                total_nodes = int(grid.x) * int(grid.y)
                replacements = {
                    "@JOBS_PER_NODE@": str(
                        -(-total_jobs // total_nodes)
                    ),
                    "@TOTAL_JOBS@": str(total_jobs),
                    "@K_TILES@": str(k_tiles),
                    "@N_TILES@": str(n_tiles),
                    "@REDUCTION_TILES_PER_PHASE@": str(
                        reduction_tiles_per_phase
                    ),
                    "@RETAIN_DESTINATION_ACROSS_REDUCTION@": (
                        "true"
                        if retain_destination_across_reduction
                        else "false"
                    ),
                    "@PACKER_L1_ACCUMULATE_ACROSS_REDUCTION@": (
                        "true"
                        if packer_l1_accumulate_across_reduction
                        else "false"
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
                    raise RuntimeError(
                        "unexpanded frozen backward placeholder"
                    )
                compute_paths[0].write_text(source, encoding="utf-8")
            break
        return result

    frozen_backward.compile_evidence = compile_evidence
    return frozen_backward
