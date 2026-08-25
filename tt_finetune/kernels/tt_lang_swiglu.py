"""Tile-native SwiGLU activation with an explicit BF16 SiLU boundary."""

from __future__ import annotations

import hashlib
import inspect
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import (
    BLACKHOLE_P150A_CORE_COUNT,
    BLACKHOLE_P150A_GRID,
)
from .tt_lang_lora import BF16_TILE_BYTES, TILE_SIZE


ttl = None
tt_lang_operation = None


def estimate_swiglu_forward_l1_bytes(*, jobs_per_core: int) -> int:
    """Peak local CB allocation for the two-phase exact program."""

    if jobs_per_core <= 0:
        raise ValueError("jobs_per_core must be positive")
    # Generic BF16 semantics are logistic -> gate multiply -> up multiply.
    # The first gate stream is consumed by logistic; a second local gate copy,
    # up, packed sigmoid, and packed SiLU each span the phase boundary.
    return (4 + 4 * jobs_per_core) * BF16_TILE_BYTES


@lru_cache(maxsize=32)
def build_tt_lang_swiglu_forward(
    *,
    tokens: int,
    intermediate_features: int,
    diagnostic_stage: str = "full_builtin",
    fp32_dest_acc_en: bool = False,
    packed_gate_up_arena: bool = False,
) -> Callable:
    """Build generic-exact SwiGLU arithmetic as one custom call.

    The accepted production stage is ``BF16(SiLU(gate))`` using Blackhole's
    fused SFPU instruction, followed by TT-NN's exact SFPU BF16 multiply with
    ``up``. This reproduces the generic graph bit-for-bit without a PJRT result
    or DRAM round-trip between the two arithmetic phases. The other stage names
    are retained only for first-divergence diagnostics.
    """

    if (
        tokens <= 0
        or intermediate_features <= 0
        or tokens % TILE_SIZE
        or intermediate_features % TILE_SIZE
    ):
        raise ValueError("SwiGLU dimensions must be positive /32")
    if diagnostic_stage not in {
        "multiply",
        "sigmoid",
        "silu",
        "full",
        "silu_builtin",
        "full_builtin",
    }:
        raise ValueError(
            "diagnostic_stage must be multiply, sigmoid, silu, or full"
        )
    m_tiles = tokens // TILE_SIZE
    n_tiles = intermediate_features // TILE_SIZE
    total_jobs = m_tiles * n_tiles
    jobs_per_core = -(-total_jobs // BLACKHOLE_P150A_CORE_COUNT)
    if jobs_per_core > 32:
        raise ValueError("SwiGLU local staging exceeds 32 CB blocks")
    l1_bytes = estimate_swiglu_forward_l1_bytes(
        jobs_per_core=jobs_per_core
    )
    l1_budget = int(1.46 * 1024 * 1024 * 0.75)
    if l1_bytes > l1_budget:
        raise ValueError(
            f"SwiGLU forward exceeds 75% worker L1: {l1_bytes} > {l1_budget}"
        )
    stage_id = {
        "multiply": 0,
        "sigmoid": 1,
        "silu": 2,
        "full": 3,
        "silu_builtin": 4,
        "full_builtin": 5,
    }[diagnostic_stage]
    up_column_offset = (
        intermediate_features // TILE_SIZE if packed_gate_up_arena else 0
    )

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import (
                tt_lang_operation as tt_lang_operation_decorator,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("native SwiGLU requires TT-XLA") from exc
        ttl = ttl_module
        tt_lang_operation = tt_lang_operation_decorator

    # StableHLO's BF16 logistic uses the non-FP32 destination SFPU schedule on
    # Blackhole. This selection is numerically significant: enabling FP32
    # destination mode changes the exp/reciprocal implementation and is not
    # bit-exact with the production graph.
    options: dict[str, object] = {
        "grid": BLACKHOLE_P150A_GRID,
        "fp32_dest_acc_en": fp32_dest_acc_en,
    }
    if "math_approx_mode" in inspect.signature(ttl.operation).parameters:
        options["math_approx_mode"] = False

    @ttl.operation(**options)
    def _program(gate, up, output):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        local_jobs = -(-total_jobs // total_nodes)
        gate_dfb = ttl.make_dataflow_buffer_like(
            gate, shape=(1, 1), block_count=2
        )
        gate_saved_dfb = ttl.make_dataflow_buffer_like(
            gate, shape=(1, 1), block_count=max(2, local_jobs)
        )
        up_dfb = ttl.make_dataflow_buffer_like(
            up, shape=(1, 1), block_count=max(2, local_jobs)
        )
        sigmoid_dfb = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=max(2, local_jobs)
        )
        silu_dfb = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=max(2, local_jobs)
        )
        output_dfb = ttl.make_dataflow_buffer_like(
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
                    with (
                        gate_dfb.reserve() as gate_block,
                        gate_saved_dfb.reserve() as gate_saved_block,
                        up_dfb.reserve() as up_block,
                    ):
                        gate_tx = ttl.copy(
                            gate[
                                m_tile : m_tile + 1,
                                n_tile : n_tile + 1,
                            ],
                            gate_block,
                        )
                        up_tx = ttl.copy(
                            up[
                                m_tile : m_tile + 1,
                                n_tile + up_column_offset :
                                n_tile + up_column_offset + 1,
                            ],
                            up_block,
                        )
                        gate_saved_tx = ttl.copy(
                            gate[
                                m_tile : m_tile + 1,
                                n_tile : n_tile + 1,
                            ],
                            gate_saved_block,
                        )
                        gate_tx.wait()
                        up_tx.wait()
                        gate_saved_tx.wait()

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            if stage_id == 0:
                for local_job in range(local_jobs):
                    job = node_index * local_jobs + local_job
                    if job < total_jobs:
                        with (
                            gate_dfb.wait() as gate_tile,
                            up_dfb.wait() as up_tile,
                            output_dfb.reserve() as result,
                        ):
                            result.store(gate_tile * up_tile)
            # Phase one reproduces the generic BF16 logistic result.
            for local_job in range(local_jobs):
                job = node_index * local_jobs + local_job
                if job < total_jobs and stage_id != 0:
                    if stage_id == 1:
                        with (
                            gate_dfb.wait() as gate_tile,
                            output_dfb.reserve() as result,
                        ):
                            result.store(ttl.sigmoid(gate_tile))
                    else:
                        with (
                            gate_dfb.wait() as gate_tile,
                            sigmoid_dfb.reserve() as sigmoid_tile,
                        ):
                            sigmoid_tile.store(ttl.sigmoid(gate_tile))
            # Phase two packs gate*sigmoid to BF16, matching torch SiLU's
            # visible BF16 output before the final gated multiply.
            if stage_id > 1:
                for local_job in range(local_jobs):
                    job = node_index * local_jobs + local_job
                    if job < total_jobs:
                        if stage_id == 2:
                            with (
                                gate_saved_dfb.wait() as gate_tile,
                                sigmoid_dfb.wait() as sigmoid_tile,
                                output_dfb.reserve() as result,
                            ):
                                result.store(gate_tile * sigmoid_tile)
                        else:
                            with (
                                gate_saved_dfb.wait() as gate_tile,
                                sigmoid_dfb.wait() as sigmoid_tile,
                                silu_dfb.reserve() as activated,
                            ):
                                activated.store(gate_tile * sigmoid_tile)
            # Phase three applies the second generic BF16 multiply.
            if stage_id == 3:
                for local_job in range(local_jobs):
                    job = node_index * local_jobs + local_job
                    if job < total_jobs:
                        with (
                            silu_dfb.wait() as activated,
                            up_dfb.wait() as up_tile,
                            output_dfb.reserve() as result,
                        ):
                            result.store(activated * up_tile)

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_job in range(local_jobs):
                job = node_index * local_jobs + local_job
                if job < total_jobs:
                    m_tile = job // n_tiles
                    n_tile = job - m_tile * n_tiles
                    with output_dfb.wait() as result:
                        tx = ttl.copy(
                            result,
                            output[
                                m_tile : m_tile + 1,
                                n_tile : n_tile + 1,
                            ],
                        )
                        tx.wait()

    variant = hashlib.sha256(
        (
            f"{tokens}:{intermediate_features}:{diagnostic_stage}:"
            f"fp32={fp32_dest_acc_en}:"
            f"packed={packed_gate_up_arena}:"
            "swiglu-forward-v10-packed-offset-in-tiles"
        ).encode(
            "ascii"
        )
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.swiglu_forward.{variant}",
        arg_roles=("in", "in", "out"),
        version_tag=f"swiglu-forward-{variant}",
    )
    def swiglu_forward(gate, up, output):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(gate, up, output)
        # Make the binary FPU contract explicit. TT-Lang does not expose
        # fidelity on @ttl.operation, and relying on its evolving descriptor
        # default already caused an avoidable numerical ambiguity here.
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                for config in compiled.kernel_configs:
                    if not hasattr(config, "math_fidelity"):
                        continue
                    enum_type = type(config.math_fidelity)
                    config.math_fidelity = getattr(enum_type, "HiFi4")
                compute_paths = [
                    Path(path)
                    for path, thread_type in compiled.kernel_paths
                    if thread_type == "compute"
                ]
                if len(compute_paths) != 1:
                    raise RuntimeError(
                        "native SwiGLU expected exactly one compute kernel"
                    )
                if len(compiled.cb_configs) != 6:
                    raise RuntimeError(
                        "native SwiGLU expected six circular buffers, got "
                        f"{len(compiled.cb_configs)}"
                    )
                template = (
                    Path(__file__).with_name("cpp")
                    / "swiglu_exact_compute.cpp"
                )
                source = template.read_text(encoding="utf-8")
                replacements = {
                    "@TOTAL_JOBS@": str(total_jobs),
                    "@JOBS_PER_NODE@": str(jobs_per_core),
                    "@GRID_COLS@": str(BLACKHOLE_P150A_GRID[0]),
                    "@STAGE_ID@": str(stage_id),
                    "@STARTS_FP32@": "true" if fp32_dest_acc_en else "false",
                }
                for placeholder, value in replacements.items():
                    source = source.replace(placeholder, value)
                if any(key in source for key in replacements):
                    raise RuntimeError(
                        "unexpanded placeholder in native SwiGLU compute"
                    )
                compute_paths[0].write_text(source, encoding="utf-8")
            break
        return result

    return swiglu_forward


@lru_cache(maxsize=16)
def build_tt_lang_swiglu_backward(
    *,
    tokens: int,
    intermediate_features: int,
    diagnostic_stage: str = "dgate",
) -> Callable:
    """Build the exact two-output SwiGLU activation backward prototype.

    The program reproduces every BF16 edge visible in the StableHLO lowering
    of ``aten.silu_backward(dproduct * up, gate)`` and
    ``dproduct * silu(gate)``.  It is intentionally isolated first; once its
    numerical contract is accepted on silicon, the same compute body can feed
    the resident gate/up backward CBs without materializing dGate/dUp.
    """

    if (
        tokens <= 0
        or intermediate_features <= 0
        or tokens % TILE_SIZE
        or intermediate_features % TILE_SIZE
    ):
        raise ValueError("SwiGLU backward dimensions must be positive /32")
    m_tiles = tokens // TILE_SIZE
    n_tiles = intermediate_features // TILE_SIZE
    total_jobs = m_tiles * n_tiles
    jobs_per_core = -(-total_jobs // BLACKHOLE_P150A_CORE_COUNT)
    if jobs_per_core > 32:
        raise ValueError("SwiGLU backward local staging exceeds 32 CB blocks")
    stage_names = (
        "pregradient",
        "sigmoid",
        "one_minus_sigmoid",
        "gate_times_one_minus",
        "derivative_inner",
        "derivative",
        "dgate",
    )
    if diagnostic_stage not in stage_names:
        raise ValueError(
            "SwiGLU backward diagnostic stage must be one of "
            f"{stage_names}"
        )
    stage_id = stage_names.index(diagnostic_stage)
    # The compute RISC is the sole consumer, so dProduct, gate, and sigmoid
    # stay at the CB front across their dependent uses. Only three input
    # streams, seven one-tile intermediates, two drain buffers, and one
    # explicit BF16-one tile are needed.
    l1_bytes = (
        3 * max(2, jobs_per_core) + 10 * 2
    ) * BF16_TILE_BYTES
    l1_budget = int(1.46 * 1024 * 1024 * 0.75)
    if l1_bytes > l1_budget:
        raise ValueError(
            "SwiGLU backward exceeds 75% worker L1: "
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
            raise RuntimeError("native SwiGLU backward requires TT-XLA") from exc
        ttl = ttl_module
        tt_lang_operation = tt_lang_operation_decorator

    options: dict[str, object] = {
        "grid": BLACKHOLE_P150A_GRID,
        "fp32_dest_acc_en": False,
    }
    if "math_approx_mode" in inspect.signature(ttl.operation).parameters:
        options["math_approx_mode"] = False

    @ttl.operation(**options)
    def _program(product_gradient, gate, up, output):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        local_jobs = -(-total_jobs // total_nodes)
        block_count = max(2, local_jobs)

        dproduct = ttl.make_dataflow_buffer_like(
            product_gradient, shape=(1, 1), block_count=block_count
        )
        gate_saved = ttl.make_dataflow_buffer_like(
            gate, shape=(1, 1), block_count=block_count
        )
        up_saved = ttl.make_dataflow_buffer_like(
            up, shape=(1, 1), block_count=block_count
        )
        pregradient = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=2
        )
        sigmoid_saved = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=2
        )
        one_minus_sigmoid = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=2
        )
        gate_times_one_minus = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=2
        )
        derivative_inner = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=2
        )
        derivative = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=2
        )
        silu = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=2
        )
        dgate = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=2
        )
        dup = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=2
        )
        one = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=2
        )

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            node = node_row * grid_cols + node_col
            for local_job in range(local_jobs):
                job = node * local_jobs + local_job
                if job < total_jobs:
                    m = job // n_tiles
                    n = job - m * n_tiles
                    with (
                        dproduct.reserve() as pg,
                        gate_saved.reserve() as g,
                        up_saved.reserve() as u,
                    ):
                        tx0 = ttl.copy(
                            product_gradient[m : m + 1, n : n + 1], pg
                        )
                        tx1 = ttl.copy(gate[m : m + 1, n : n + 1], g)
                        tx2 = ttl.copy(up[m : m + 1, n : n + 1], u)
                        tx0.wait()
                        tx1.wait()
                        tx2.wait()

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            node = node_row * grid_cols + node_col
            for local_job in range(local_jobs):
                if node * local_jobs + local_job < total_jobs:
                    with (
                        dproduct.wait() as pg,
                        gate_saved.wait() as g,
                        up_saved.wait() as u,
                        dgate.reserve() as dg,
                        dup.reserve() as du,
                    ):
                        # Valid lowering-only pipeline. The reviewed C++ body
                        # replaces this before the first silicon launch.
                        dg.store(pg * u)
                        du.store(pg * g)
            impossible = node < 0
            if impossible:
                with pregradient.wait() as _value:
                    pass
                with sigmoid_saved.wait() as _value:
                    pass
                with one_minus_sigmoid.wait() as _value:
                    pass
                with gate_times_one_minus.wait() as _value:
                    pass
                with derivative_inner.wait() as _value:
                    pass
                with derivative.wait() as _value:
                    pass
                with silu.wait() as _value:
                    pass
                with one.wait() as _value:
                    pass

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node = node_row * grid_cols + node_col
            for local_job in range(local_jobs):
                job = node * local_jobs + local_job
                if job < total_jobs:
                    m = job // n_tiles
                    n = job - m * n_tiles
                    with dgate.wait() as dg:
                        tx0 = ttl.copy(dg, output[m : m + 1, n : n + 1])
                        tx0.wait()
                    with dup.wait() as du:
                        tx1 = ttl.copy(
                            du,
                            output[
                                m_tiles + m : m_tiles + m + 1,
                                n : n + 1,
                            ],
                        )
                        tx1.wait()

    variant = hashlib.sha256(
        (
            f"{tokens}:{intermediate_features}:{diagnostic_stage}:"
            "swiglu-backward-v2-resident-inputs"
        ).encode("ascii")
    ).hexdigest()[:16]

    @tt_lang_operation(
        operation_id=f"tt_finetune.swiglu_backward.{variant}",
        arg_roles=("in", "in", "in", "out"),
        version_tag=f"swiglu-backward-{variant}",
    )
    def swiglu_backward(product_gradient, gate, up, output):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(product_gradient, gate, up, output)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                if len(compiled.cb_configs) != 13:
                    raise RuntimeError(
                        "native SwiGLU backward expected 13 CBs, got "
                        f"{len(compiled.cb_configs)}"
                    )
                for config in compiled.kernel_configs:
                    if not hasattr(config, "math_fidelity"):
                        continue
                    enum_type = type(config.math_fidelity)
                    config.math_fidelity = getattr(enum_type, "HiFi4")
                compute_paths = [
                    Path(path)
                    for path, thread_type in compiled.kernel_paths
                    if thread_type == "compute"
                ]
                if len(compute_paths) != 1:
                    raise RuntimeError(
                        "native SwiGLU backward expected one compute kernel"
                    )
                source = (
                    Path(__file__).with_name("cpp")
                    / "swiglu_backward_exact_compute.cpp"
                ).read_text(encoding="utf-8")
                replacements = {
                    "@TOTAL_JOBS@": str(total_jobs),
                    "@JOBS_PER_NODE@": str(jobs_per_core),
                    "@GRID_COLS@": str(BLACKHOLE_P150A_GRID[0]),
                    "@DIAGNOSTIC_STAGE@": str(stage_id),
                }
                for placeholder, value in replacements.items():
                    source = source.replace(placeholder, value)
                if any(key in source for key in replacements):
                    raise RuntimeError(
                        "unexpanded placeholder in native SwiGLU backward"
                    )
                compute_paths[0].write_text(source, encoding="utf-8")
            break
        return result

    swiglu_backward.output_tile_shape = (2 * m_tiles, n_tiles)
    return swiglu_backward


class NativeExactSwiGLUFunction(torch.autograd.Function):
    """Autograd boundary for the exact native forward prototype.

    Backward intentionally remains the framework reference while the resident
    activation-backward phase is being built. This makes forward-only impact
    measurable without silently changing training gradients.
    """

    @staticmethod
    def forward(ctx, gate, up, output_arena, diagnostic_stage):
        if gate.shape != up.shape or gate.shape != output_arena.shape:
            raise ValueError("native SwiGLU tensors must have equal shapes")
        if gate.ndim != 2 or gate.dtype != torch.bfloat16:
            raise ValueError("native SwiGLU requires rank-2 bfloat16 tensors")
        operation = build_tt_lang_swiglu_forward(
            tokens=gate.shape[0],
            intermediate_features=gate.shape[1],
            diagnostic_stage=str(diagnostic_stage),
        )
        output = operation(gate, up, output_arena)
        ctx.save_for_backward(gate, up)
        return output

    @staticmethod
    def backward(ctx, output_gradient):
        gate, up = ctx.saved_tensors
        # Match the ordinary autograd edge ordering: product backward first,
        # then the fused SiLU backward for the gate operand.
        gate_gradient = torch.ops.aten.silu_backward(
            output_gradient * up, gate
        )
        up_gradient = output_gradient * F.silu(gate)
        return gate_gradient, up_gradient, None, None


def native_exact_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    output_arena: torch.Tensor,
    *,
    diagnostic_stage: str = "full_builtin",
) -> torch.Tensor:
    return NativeExactSwiGLUFunction.apply(
        gate, up, output_arena, diagnostic_stage
    )
