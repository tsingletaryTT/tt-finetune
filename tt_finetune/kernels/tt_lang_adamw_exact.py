"""Register-safe, exact-order capturable AdamW kernels for TT-Lang.

This module deliberately stages the update at the BF16/F32 rounding
boundaries emitted by Torch/XLA.  Blackhole exposes four FP32 destination
tiles; the monolithic mixed-precision expression exceeds that capacity.
Four custom calls keep each phase within the register budget while retaining
double-buffered reader/compute/writer pipelines.

The staged implementation is experimental until silicon parity and performance
gates pass.  The public trainer continues to use ``TilePackedAdamW``.
"""

from __future__ import annotations

import hashlib
import struct
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache


TILE_SIZE = 32
ttl = None
tt_lang_operation = None


def _float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _install_native_compute(
    program,
    *,
    source_name: str,
    expected_cbs: int,
    replacements: dict[str, str],
) -> None:
    """Replace a compile-safe TT-Lang compute shell with native TT-Metal.

    TT-Lang 1.1.5 rejects a mixed BF16/FP32 expression when its liveness
    model conservatively applies the four-tile FP32 DST limit to every tile
    value in the expression.  The hardware supports the required explicitly
    phased schedule.  Keep TT-Lang responsible for CB allocation, runtime
    arguments, readers and writers, and replace only the compute source after
    those ABIs have been materialized.
    """

    template = (
        Path(__file__).with_name("cpp").joinpath(source_name).read_text()
    )
    for key, value in replacements.items():
        template = template.replace(key, value)
    unresolved = [
        token for token in ("@TOTAL_BLOCKS@", "@BLOCKS_PER_NODE@",
                            "@ROWS_PER_BLOCK@")
        if token in template
    ]
    if unresolved:
        raise RuntimeError(
            f"unresolved native AdamW placeholders: {unresolved}"
        )

    closure = program.__closure__ or ()
    for name, cell in zip(program.__code__.co_freevars, closure):
        if name != "cache" or not isinstance(cell.cell_contents, dict):
            continue
        for compiled in cell.cell_contents.values():
            if len(compiled.cb_configs) != expected_cbs:
                raise RuntimeError(
                    f"native AdamW {source_name} expected {expected_cbs} "
                    f"CBs, got {len(compiled.cb_configs)}"
                )
            compute_paths = [
                Path(path) for path, thread in compiled.kernel_paths
                if thread == "compute"
            ]
            if len(compute_paths) != 1:
                raise RuntimeError(
                    f"native AdamW expected one compute kernel, got "
                    f"{len(compute_paths)}"
                )
            compute_paths[0].write_text(template, encoding="utf-8")
        return
    raise RuntimeError("TT-Lang AdamW compile cache was not materialized")


def _set_native_unpack_fp32(program, cb_indices: set[int]) -> None:
    """Match TTNN's full-FP32 unpack mode for native SFPU operands."""

    import ttnn

    closure = program.__closure__ or ()
    for name, cell in zip(program.__code__.co_freevars, closure):
        if name != "cache" or not isinstance(cell.cell_contents, dict):
            continue
        for compiled in cell.cell_contents.values():
            for config in compiled.kernel_configs:
                if not hasattr(config, "unpack_to_dest_mode"):
                    continue
                while len(config.unpack_to_dest_mode) < 64:
                    config.unpack_to_dest_mode.append(
                        ttnn.UnpackToDestMode.Default
                    )
                for index in range(64):
                    config.unpack_to_dest_mode[index] = (
                        ttnn.UnpackToDestMode.UnpackToDestFp32
                        if index in cb_indices
                        else ttnn.UnpackToDestMode.Default
                    )
        return
    raise RuntimeError("TT-Lang AdamW compile cache was not materialized")


def _operation_id(
    phase: str,
    beta1: float,
    beta2: float,
    granularity: int,
    gradient_page_offset: int = 0,
    gradient_pages: int | None = None,
) -> str:
    payload = (
        f"{phase}:{beta1:.17g}:{beta2:.17g}:{granularity}:"
        f"off{gradient_page_offset}:pages{gradient_pages}:v11"
    )
    digest = hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]
    return f"tt_finetune.adamw_exact.{phase}.{digest}"


def _load_tt_lang():
    global ttl, tt_lang_operation
    if ttl is not None and tt_lang_operation is not None:
        return
    try:
        import ttl as ttl_module
        from tt_torch.tt_lang import (
            tt_lang_operation as tt_lang_operation_decorator,
        )
    except ImportError as exc:  # pragma: no cover - host-only environment
        raise RuntimeError(
            "The staged TT-Lang AdamW kernel requires the TT-XLA worker "
            "environment"
        ) from exc
    ttl = ttl_module
    tt_lang_operation = tt_lang_operation_decorator


@lru_cache(maxsize=16)
def build_tt_lang_adamw_exact(
    beta1: float = 0.9,
    beta2: float = 0.999,
    granularity: int = 1,
    gradient_page_offset: int = 0,
    gradient_pages: int | None = None,
    parameter_backend: str = "native",
) -> Callable:
    """Build the staged capturable-AdamW callable.

    The callable has the same external signature as the monolithic kernel:

    ``(state, grad, decay_factor, denominator_scale, scaled_epsilon,``
    ``compatibility_tile, state_out)``.

    ``state`` and ``state_out`` contain three equal BF16 row regions:
    ``[parameter | first moment | second moment]``.  Denominator scale and
    scaled epsilon are FP32 tiles.  The compatibility tile is currently
    unused, preserving a common harness signature.
    """

    if not 0.0 <= beta1 < 1.0:
        raise ValueError("beta1 must be in [0, 1)")
    if not 0.0 <= beta2 < 1.0:
        raise ValueError("beta2 must be in [0, 1)")
    if granularity <= 0:
        raise ValueError("granularity must be positive")
    if gradient_page_offset < 0:
        raise ValueError("gradient_page_offset must be non-negative")
    if gradient_pages is not None:
        if gradient_pages <= 0:
            raise ValueError("gradient_pages must be positive")
        if gradient_pages % granularity:
            raise ValueError("gradient_pages must be divisible by granularity")
    if parameter_backend not in {"native", "native_split", "xla_oracle"}:
        raise ValueError(
            "parameter_backend must be 'native', 'native_split' or "
            "'xla_oracle'"
        )

    _load_tt_lang()
    one_minus_beta1 = float(
        torch.tensor(1.0 - beta1, dtype=torch.bfloat16).float().item()
    )
    beta2_bf16 = float(
        torch.tensor(beta2, dtype=torch.bfloat16).float().item()
    )
    one_minus_beta2 = float(
        torch.tensor(1.0 - beta2, dtype=torch.bfloat16).float().item()
    )

    @ttl.operation(grid=(8, 8))
    def moments_program(state_in, grad_in, state_out):
        rows_per_block = granularity
        state_region_rows = (
            grad_in.shape[0] // TILE_SIZE
            if gradient_pages is None else gradient_pages
        )
        total_blocks = state_region_rows // rows_per_block
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        blocks_per_node = -(-total_blocks // total_nodes)
        block_shape = (rows_per_block, 1)

        param_dfb = ttl.make_dataflow_buffer_like(
            state_in, shape=block_shape, block_count=2
        )
        grad_dfb = ttl.make_dataflow_buffer_like(
            grad_in, shape=block_shape, block_count=2
        )
        exp_avg_dfb = ttl.make_dataflow_buffer_like(
            state_in, shape=block_shape, block_count=2
        )
        exp_avg_sq_dfb = ttl.make_dataflow_buffer_like(
            state_in, shape=block_shape, block_count=2
        )
        param_out_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=block_shape, block_count=2
        )
        exp_avg_out_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=block_shape, block_count=2
        )
        exp_avg_sq_out_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=block_shape, block_count=2
        )
        sqrt_exp_avg_sq_out_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=block_shape, block_count=2
        )
        # Private BF16 FIFOs are numerical checkpoints, not merely storage.
        # Each one matches a StableHLO BF16 node and therefore forces the
        # same RNE boundary while keeping all intermediates in per-core L1.
        exp_avg_delta_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=block_shape, block_count=2
        )
        exp_avg_scaled_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=block_shape, block_count=2
        )
        grad_square_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=block_shape, block_count=2
        )
        grad_square_scaled_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=block_shape, block_count=2
        )
        exp_avg_sq_scaled_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=block_shape, block_count=2
        )
        next_exp_avg_sq_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=block_shape, block_count=2
        )
        one_minus_beta1_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=(1, 1), block_count=1
        )
        beta2_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=(1, 1), block_count=1
        )
        one_minus_beta2_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=(1, 1), block_count=1
        )

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node_index * blocks_per_node + local_block
                if block < total_blocks:
                    with (
                        param_dfb.wait() as param,
                        grad_dfb.wait() as grad,
                        exp_avg_dfb.wait() as exp_avg,
                        exp_avg_sq_dfb.wait() as exp_avg_sq,
                    ):
                        with param_out_dfb.reserve() as param_out:
                            param_out.store(param)

                        with exp_avg_delta_dfb.reserve() as delta:
                            delta.store(grad - exp_avg)
                        with (
                            exp_avg_delta_dfb.wait() as delta,
                            exp_avg_scaled_dfb.reserve() as scaled,
                        ):
                            scaled.store(delta * one_minus_beta1)
                        with (
                            exp_avg_scaled_dfb.wait() as scaled,
                            exp_avg_out_dfb.reserve() as exp_avg_out,
                        ):
                            exp_avg_out.store(exp_avg + scaled)

                        with grad_square_dfb.reserve() as square:
                            square.store(grad * grad)
                        with (
                            grad_square_dfb.wait() as square,
                            grad_square_scaled_dfb.reserve() as scaled,
                        ):
                            scaled.store(square * one_minus_beta2)
                        with exp_avg_sq_scaled_dfb.reserve() as scaled:
                            scaled.store(exp_avg_sq * beta2_bf16)
                        with (
                            grad_square_scaled_dfb.wait() as grad_scaled,
                            exp_avg_sq_scaled_dfb.wait() as moment_scaled,
                            next_exp_avg_sq_dfb.reserve() as next_moment,
                        ):
                            next_moment.store(moment_scaled + grad_scaled)

                        next_moment = next_exp_avg_sq_dfb.wait()
                        with (
                            exp_avg_sq_out_dfb.reserve() as exp_avg_sq_out,
                            sqrt_exp_avg_sq_out_dfb.reserve()
                            as sqrt_exp_avg_sq_out,
                        ):
                            exp_avg_sq_out.store(next_moment)
                            sqrt_exp_avg_sq_out.store(ttl.sqrt(next_moment))
                        next_moment.pop()

                        # Native compute creates exact full BF16 constant
                        # tiles in these FIFOs. Keep their ABI slots alive in
                        # the compile-safe TT-Lang shell.
                        if node_index < 0:
                            with (
                                one_minus_beta1_dfb.reserve() as c0,
                                beta2_dfb.reserve() as c1,
                                one_minus_beta2_dfb.reserve() as c2,
                            ):
                                c0.store(param)
                                c1.store(param)
                                c2.store(param)

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node_index * blocks_per_node + local_block
                if block < total_blocks:
                    row_start = block * rows_per_block
                    row_stop = row_start + rows_per_block
                    with (
                        param_dfb.reserve() as param,
                        grad_dfb.reserve() as grad,
                        exp_avg_dfb.reserve() as exp_avg,
                        exp_avg_sq_dfb.reserve() as exp_avg_sq,
                    ):
                        tx_param = ttl.copy(
                            state_in[row_start:row_stop, 0:1], param
                        )
                        tx_grad = ttl.copy(
                            grad_in[
                                row_start + gradient_page_offset :
                                row_stop + gradient_page_offset,
                                0:1,
                            ], grad
                        )
                        tx_exp_avg = ttl.copy(
                            state_in[
                                row_start
                                + state_region_rows : row_stop
                                + state_region_rows,
                                0:1,
                            ],
                            exp_avg,
                        )
                        tx_exp_avg_sq = ttl.copy(
                            state_in[
                                row_start
                                + 2 * state_region_rows : row_stop
                                + 2 * state_region_rows,
                                0:1,
                            ],
                            exp_avg_sq,
                        )
                        tx_param.wait()
                        tx_grad.wait()
                        tx_exp_avg.wait()
                        tx_exp_avg_sq.wait()

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node_index * blocks_per_node + local_block
                if block < total_blocks:
                    row_start = block * rows_per_block
                    row_stop = row_start + rows_per_block
                    with (
                        param_out_dfb.wait() as param,
                        exp_avg_out_dfb.wait() as exp_avg,
                        exp_avg_sq_out_dfb.wait() as exp_avg_sq,
                        sqrt_exp_avg_sq_out_dfb.wait()
                        as sqrt_exp_avg_sq,
                    ):
                        tx_param = ttl.copy(
                            param,
                            state_out[row_start:row_stop, 0:1],
                        )
                        tx_exp_avg = ttl.copy(
                            exp_avg,
                            state_out[
                                row_start
                                + state_region_rows : row_stop
                                + state_region_rows,
                                0:1,
                            ],
                        )
                        tx_exp_avg_sq = ttl.copy(
                            exp_avg_sq,
                            state_out[
                                row_start
                                + 2 * state_region_rows : row_stop
                                + 2 * state_region_rows,
                                0:1,
                            ],
                        )
                        tx_sqrt_exp_avg_sq = ttl.copy(
                            sqrt_exp_avg_sq,
                            state_out[
                                row_start
                                + 3 * state_region_rows : row_stop
                                + 3 * state_region_rows,
                                0:1,
                            ],
                        )
                        tx_param.wait()
                        tx_exp_avg.wait()
                        tx_exp_avg_sq.wait()
                        tx_sqrt_exp_avg_sq.wait()

    @tt_lang_operation(
        operation_id=_operation_id(
            "moments", beta1, beta2, granularity,
            gradient_page_offset, gradient_pages,
        ),
        arg_roles=("in", "in", "out"),
        version_tag="adamw-exact-moments-v1",
    )
    def moments(state_in, grad_in, state_out):
        _clear_ttl_compile_wrapper_cache(moments_program)
        result = moments_program(state_in, grad_in, state_out)
        state_region_rows = (
            grad_in.shape[0] // TILE_SIZE
            if gradient_pages is None else gradient_pages
        )
        total_blocks = state_region_rows // granularity
        _install_native_compute(
            moments_program,
            source_name="adamw_exact_moments_compute.cpp",
            expected_cbs=17,
            replacements={
                "@TOTAL_BLOCKS@": str(total_blocks),
                "@BLOCKS_PER_NODE@": str(-(-total_blocks // 64)),
                "@ROWS_PER_BLOCK@": str(granularity),
                "@ONE_MINUS_BETA1_BITS@": str(
                    _float_bits(one_minus_beta1)
                ),
                "@BETA2_BITS@": str(_float_bits(beta2_bf16)),
                "@ONE_MINUS_BETA2_BITS@": str(
                    _float_bits(one_minus_beta2)
                ),
            },
        )
        return result

    @ttl.operation(
        grid=(8, 8),
        fp32_dest_acc_en=True,
        options="--no-ttl-maximize-dst",
    )
    def denominator_program(
        state_in,
        denominator_scale_in,
        denominator_out,
    ):
        rows_per_block = granularity
        state_region_rows = denominator_out.shape[0] // TILE_SIZE
        total_blocks = state_region_rows // rows_per_block
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        blocks_per_node = -(-total_blocks // total_nodes)
        block_shape = (rows_per_block, 1)

        exp_avg_sq_dfb = ttl.make_dataflow_buffer_like(
            state_in, shape=block_shape, block_count=2
        )
        scale_dfb = ttl.make_dataflow_buffer_like(
            denominator_scale_in, shape=(1, 1), block_count=1
        )
        denominator_out_dfb = ttl.make_dataflow_buffer_like(
            denominator_out, shape=block_shape, block_count=2
        )
        sqrt_f32_dfb = ttl.make_dataflow_buffer_like(
            denominator_out, shape=block_shape, block_count=2
        )

        @ttl.compute()
        def compute():
            scale_tile = scale_dfb.wait()
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node_index * blocks_per_node + local_block
                if block < total_blocks:
                    with (
                        exp_avg_sq_dfb.wait() as exp_avg_sq,
                        denominator_out_dfb.reserve() as denominator_out_block,
                    ):
                        # Compile-safe same-dtype shell. Native TT-Metal
                        # consumes exp_avg_sq and replaces this placeholder
                        # value with BF16->FP32/divide. The shell cannot
                        # express the mixed-type edge in TT-Lang 1.1.5.
                        denominator_out_block.store(scale_tile)
            if node_index < 0:
                with sqrt_f32_dfb.reserve() as sqrt_f32:
                    sqrt_f32.store(scale_tile)
            scale_tile.pop()

        @ttl.datamovement()
        def read():
            scale_block = scale_dfb.reserve()
            tx_scale = ttl.copy(
                denominator_scale_in[0:1, 0:1], scale_block
            )
            tx_scale.wait()
            scale_block.push()

            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node_index * blocks_per_node + local_block
                if block < total_blocks:
                    row_start = block * rows_per_block
                    row_stop = row_start + rows_per_block
                    with exp_avg_sq_dfb.reserve() as exp_avg_sq:
                        tx = ttl.copy(
                            state_in[
                                row_start
                                + 3 * state_region_rows : row_stop
                                + 3 * state_region_rows,
                                0:1,
                            ],
                            exp_avg_sq,
                        )
                        tx.wait()

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node_index * blocks_per_node + local_block
                if block < total_blocks:
                    row_start = block * rows_per_block
                    row_stop = row_start + rows_per_block
                    with denominator_out_dfb.wait() as denominator:
                        tx = ttl.copy(
                            denominator,
                            denominator_out[row_start:row_stop, 0:1],
                        )
                        tx.wait()

    @tt_lang_operation(
        operation_id=_operation_id(
            "denominator", beta1, beta2, granularity,
            gradient_page_offset, gradient_pages,
        ),
        arg_roles=("in", "in", "out"),
        version_tag="adamw-exact-denominator-v1",
    )
    def denominator(
        state_in,
        denominator_scale_in,
        denominator_out,
    ):
        _clear_ttl_compile_wrapper_cache(denominator_program)
        result = denominator_program(
            state_in,
            denominator_scale_in,
            denominator_out,
        )
        state_region_rows = denominator_out.shape[0] // TILE_SIZE
        total_blocks = state_region_rows // granularity
        _install_native_compute(
            denominator_program,
            source_name="adamw_exact_denominator_compute.cpp",
            expected_cbs=4,
            replacements={
                "@TOTAL_BLOCKS@": str(total_blocks),
                "@BLOCKS_PER_NODE@": str(-(-total_blocks // 64)),
                "@ROWS_PER_BLOCK@": str(granularity),
            },
        )
        _set_native_unpack_fp32(denominator_program, {1, 2, 3})
        return result

    @ttl.operation(
        grid=(8, 8),
        fp32_dest_acc_en=True,
        options="--no-ttl-maximize-dst",
    )
    def epsilon_program(
        denominator_in,
        scaled_epsilon_in,
        denominator_out,
    ):
        rows_per_block = granularity
        total_blocks = (
            denominator_in.shape[0] // TILE_SIZE // rows_per_block
        )
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        blocks_per_node = -(-total_blocks // total_nodes)
        block_shape = (rows_per_block, 1)

        denominator_dfb = ttl.make_dataflow_buffer_like(
            denominator_in, shape=block_shape, block_count=2
        )
        epsilon_dfb = ttl.make_dataflow_buffer_like(
            scaled_epsilon_in, shape=(1, 1), block_count=1
        )
        denominator_out_dfb = ttl.make_dataflow_buffer_like(
            denominator_out, shape=block_shape, block_count=2
        )

        @ttl.compute()
        def compute():
            epsilon_tile = epsilon_dfb.wait()
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node_index * blocks_per_node + local_block
                if block < total_blocks:
                    with (
                        denominator_dfb.wait() as denominator,
                        denominator_out_dfb.reserve()
                        as denominator_out_block,
                    ):
                        denominator_out_block.store(
                            ttl.math.typecast(
                                denominator, torch.float32
                            )
                        )
            # Preserve the FP32 epsilon scalar as a live compute operand.  A
            # native source replacement performs the exact addition.
            if node_index < 0:
                with denominator_out_dfb.reserve() as marker:
                    marker.store(epsilon_tile)
            epsilon_tile.pop()

        @ttl.datamovement()
        def read():
            epsilon_block = epsilon_dfb.reserve()
            tx_epsilon = ttl.copy(
                scaled_epsilon_in[0:1, 0:1], epsilon_block
            )
            tx_epsilon.wait()
            epsilon_block.push()

            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node_index * blocks_per_node + local_block
                if block < total_blocks:
                    row_start = block * rows_per_block
                    row_stop = row_start + rows_per_block
                    with denominator_dfb.reserve() as denominator:
                        tx = ttl.copy(
                            denominator_in[
                                row_start:row_stop, 0:1
                            ],
                            denominator,
                        )
                        tx.wait()

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node_index * blocks_per_node + local_block
                if block < total_blocks:
                    row_start = block * rows_per_block
                    row_stop = row_start + rows_per_block
                    with denominator_out_dfb.wait() as denominator:
                        tx = ttl.copy(
                            denominator,
                            denominator_out[row_start:row_stop, 0:1],
                        )
                        tx.wait()

    @tt_lang_operation(
        operation_id=_operation_id(
            "epsilon", beta1, beta2, granularity,
            gradient_page_offset, gradient_pages,
        ),
        arg_roles=("in", "in", "out"),
        version_tag="adamw-exact-epsilon-v1",
    )
    def epsilon(
        denominator_in,
        scaled_epsilon_in,
        denominator_out,
    ):
        _clear_ttl_compile_wrapper_cache(epsilon_program)
        result = epsilon_program(
            denominator_in,
            scaled_epsilon_in,
            denominator_out,
        )
        total_blocks = (
            denominator_in.shape[0] // TILE_SIZE // granularity
        )
        _install_native_compute(
            epsilon_program,
            source_name="adamw_exact_epsilon_compute.cpp",
            expected_cbs=3,
            replacements={
                "@TOTAL_BLOCKS@": str(total_blocks),
                "@BLOCKS_PER_NODE@": str(-(-total_blocks // 64)),
                "@ROWS_PER_BLOCK@": str(granularity),
            },
        )
        _set_native_unpack_fp32(epsilon_program, {0, 1, 2})
        return result

    @ttl.operation(
        grid=(8, 8),
        fp32_dest_acc_en=True,
        options="--no-ttl-maximize-dst",
    )
    def parameter_program(
        state_in,
        denominator_in,
        decay_factor_in,
        state_out,
    ):
        rows_per_block = granularity
        state_region_rows = denominator_in.shape[0] // TILE_SIZE
        total_blocks = state_region_rows // rows_per_block
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        blocks_per_node = -(-total_blocks // total_nodes)
        block_shape = (rows_per_block, 1)

        param_dfb = ttl.make_dataflow_buffer_like(
            state_in, shape=block_shape, block_count=2
        )
        exp_avg_dfb = ttl.make_dataflow_buffer_like(
            state_in, shape=block_shape, block_count=2
        )
        exp_avg_sq_dfb = ttl.make_dataflow_buffer_like(
            state_in, shape=block_shape, block_count=2
        )
        denominator_dfb = ttl.make_dataflow_buffer_like(
            denominator_in, shape=block_shape, block_count=2
        )
        decay_factor_dfb = ttl.make_dataflow_buffer_like(
            decay_factor_in, shape=(1, 1), block_count=1
        )
        param_out_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=block_shape, block_count=2
        )
        exp_avg_out_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=block_shape, block_count=2
        )
        exp_avg_sq_out_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=block_shape, block_count=2
        )
        decay_bf16_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=block_shape, block_count=2
        )
        update_f32_dfb = ttl.make_dataflow_buffer_like(
            denominator_in, shape=block_shape, block_count=2
        )
        param_f32_dfb = ttl.make_dataflow_buffer_like(
            denominator_in, shape=block_shape, block_count=2
        )
        m_f32_dfb = ttl.make_dataflow_buffer_like(
            denominator_in, shape=block_shape, block_count=2
        )
        sum_f32_dfb = ttl.make_dataflow_buffer_like(
            denominator_in, shape=block_shape, block_count=2
        )

        @ttl.compute()
        def compute():
            decay_factor_tile = decay_factor_dfb.wait()
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node_index * blocks_per_node + local_block
                if block < total_blocks:
                    with (
                        param_dfb.wait() as param,
                        exp_avg_dfb.wait() as exp_avg,
                        exp_avg_sq_dfb.wait() as exp_avg_sq,
                        denominator_dfb.wait() as denominator,
                        param_out_dfb.reserve() as param_out,
                        exp_avg_out_dfb.reserve() as exp_avg_out,
                        exp_avg_sq_out_dfb.reserve() as exp_avg_sq_out,
                    ):
                        # Compile-safe same-dtype shell.  Native TT-Metal
                        # replaces it with the exact mixed-precision parameter
                        # phase while retaining these reader/writer ABIs.
                        param_out.store(param)
                        exp_avg_out.store(exp_avg)
                        exp_avg_sq_out.store(exp_avg_sq)
                        if node_index < 0:
                            with decay_bf16_dfb.reserve() as decay_bf16:
                                decay_bf16.store(decay_factor_tile)
                            with update_f32_dfb.reserve() as update_f32:
                                update_f32.store(denominator)
                            with param_f32_dfb.reserve() as param_f32:
                                param_f32.store(denominator)
                            with m_f32_dfb.reserve() as m_f32:
                                m_f32.store(denominator)
                            with sum_f32_dfb.reserve() as sum_f32:
                                sum_f32.store(denominator)
            decay_factor_tile.pop()

        @ttl.datamovement()
        def read():
            decay_factor_block = decay_factor_dfb.reserve()
            tx_decay_factor = ttl.copy(
                decay_factor_in[0:1, 0:1], decay_factor_block
            )
            tx_decay_factor.wait()
            decay_factor_block.push()

            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node_index * blocks_per_node + local_block
                if block < total_blocks:
                    row_start = block * rows_per_block
                    row_stop = row_start + rows_per_block
                    with (
                        param_dfb.reserve() as param,
                        exp_avg_dfb.reserve() as exp_avg,
                        exp_avg_sq_dfb.reserve() as exp_avg_sq,
                        denominator_dfb.reserve() as denominator,
                    ):
                        tx_param = ttl.copy(
                            state_in[row_start:row_stop, 0:1], param
                        )
                        tx_exp_avg = ttl.copy(
                            state_in[
                                row_start
                                + state_region_rows : row_stop
                                + state_region_rows,
                                0:1,
                            ],
                            exp_avg,
                        )
                        tx_exp_avg_sq = ttl.copy(
                            state_in[
                                row_start
                                + 2 * state_region_rows : row_stop
                                + 2 * state_region_rows,
                                0:1,
                            ],
                            exp_avg_sq,
                        )
                        tx_denominator = ttl.copy(
                            denominator_in[
                                row_start:row_stop, 0:1
                            ],
                            denominator,
                        )
                        tx_param.wait()
                        tx_exp_avg.wait()
                        tx_exp_avg_sq.wait()
                        tx_denominator.wait()

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node_index * blocks_per_node + local_block
                if block < total_blocks:
                    row_start = block * rows_per_block
                    row_stop = row_start + rows_per_block
                    with (
                        param_out_dfb.wait() as param,
                        exp_avg_out_dfb.wait() as exp_avg,
                        exp_avg_sq_out_dfb.wait() as exp_avg_sq,
                    ):
                        tx_param = ttl.copy(
                            param,
                            state_out[row_start:row_stop, 0:1],
                        )
                        tx_exp_avg = ttl.copy(
                            exp_avg,
                            state_out[
                                row_start
                                + state_region_rows : row_stop
                                + state_region_rows,
                                0:1,
                            ],
                        )
                        tx_exp_avg_sq = ttl.copy(
                            exp_avg_sq,
                            state_out[
                                row_start
                                + 2 * state_region_rows : row_stop
                                + 2 * state_region_rows,
                                0:1,
                            ],
                        )
                        tx_param.wait()
                        tx_exp_avg.wait()
                        tx_exp_avg_sq.wait()

    @tt_lang_operation(
        operation_id=_operation_id(
            "parameter", beta1, beta2, granularity,
            gradient_page_offset, gradient_pages,
        ),
        arg_roles=("in", "in", "in", "out"),
        version_tag="adamw-exact-parameter-v1",
    )
    def parameter(
        state_in,
        denominator_in,
        decay_factor_in,
        state_out,
    ):
        _clear_ttl_compile_wrapper_cache(parameter_program)
        result = parameter_program(
            state_in,
            denominator_in,
            decay_factor_in,
            state_out,
        )
        total_blocks = (
            denominator_in.shape[0] // TILE_SIZE // granularity
        )
        _install_native_compute(
            parameter_program,
            source_name="adamw_exact_parameter_compute.cpp",
            expected_cbs=13,
            replacements={
                "@TOTAL_BLOCKS@": str(total_blocks),
                "@BLOCKS_PER_NODE@": str(-(-total_blocks // 64)),
                "@ROWS_PER_BLOCK@": str(granularity),
            },
        )
        _set_native_unpack_fp32(
            parameter_program, {3, 9, 10, 11, 12}
        )
        return result

    @ttl.operation(
        grid=(8, 8),
        fp32_dest_acc_en=True,
        options="--no-ttl-maximize-dst",
    )
    def parameter_prepare_program(
        state_in,
        denominator_in,
        decay_factor_in,
        decayed_out,
        update_out,
    ):
        rows_per_block = granularity
        state_region_rows = denominator_in.shape[0] // TILE_SIZE
        total_blocks = state_region_rows // rows_per_block
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        blocks_per_node = -(-total_blocks // total_nodes)
        block_shape = (rows_per_block, 1)

        param_dfb = ttl.make_dataflow_buffer_like(
            state_in, shape=block_shape, block_count=2
        )
        exp_avg_dfb = ttl.make_dataflow_buffer_like(
            state_in, shape=block_shape, block_count=2
        )
        denominator_dfb = ttl.make_dataflow_buffer_like(
            denominator_in, shape=block_shape, block_count=2
        )
        decay_factor_dfb = ttl.make_dataflow_buffer_like(
            decay_factor_in, shape=(1, 1), block_count=1
        )
        decay_bf16_dfb = ttl.make_dataflow_buffer_like(
            state_in, shape=block_shape, block_count=2
        )
        decayed_out_dfb = ttl.make_dataflow_buffer_like(
            decayed_out, shape=block_shape, block_count=2
        )
        update_out_dfb = ttl.make_dataflow_buffer_like(
            update_out, shape=block_shape, block_count=2
        )

        @ttl.compute()
        def compute():
            decay_factor = decay_factor_dfb.wait()
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node_index * blocks_per_node + local_block
                if block < total_blocks:
                    with (
                        param_dfb.wait() as param,
                        exp_avg_dfb.wait() as exp_avg,
                        denominator_dfb.wait() as denominator,
                        decayed_out_dfb.reserve() as decayed,
                        update_out_dfb.reserve() as update,
                    ):
                        # Compile-safe FP32 shell; native compute supplies the
                        # exact mixed BF16/F32 schedule.
                        decayed.store(denominator)
                        update.store(denominator)
                        if node_index < 0:
                            with decay_bf16_dfb.reserve() as checkpoint:
                                checkpoint.store(param)
            decay_factor.pop()

        @ttl.datamovement()
        def read():
            decay_block = decay_factor_dfb.reserve()
            tx_decay = ttl.copy(
                decay_factor_in[0:1, 0:1], decay_block
            )
            tx_decay.wait()
            decay_block.push()

            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node_index * blocks_per_node + local_block
                if block < total_blocks:
                    row_start = block * rows_per_block
                    row_stop = row_start + rows_per_block
                    with (
                        param_dfb.reserve() as param,
                        exp_avg_dfb.reserve() as exp_avg,
                        denominator_dfb.reserve() as denominator,
                    ):
                        tx_param = ttl.copy(
                            state_in[row_start:row_stop, 0:1], param
                        )
                        tx_exp_avg = ttl.copy(
                            state_in[
                                row_start + state_region_rows :
                                row_stop + state_region_rows,
                                0:1,
                            ],
                            exp_avg,
                        )
                        tx_denominator = ttl.copy(
                            denominator_in[row_start:row_stop, 0:1],
                            denominator,
                        )
                        tx_param.wait()
                        tx_exp_avg.wait()
                        tx_denominator.wait()

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node_index * blocks_per_node + local_block
                if block < total_blocks:
                    row_start = block * rows_per_block
                    row_stop = row_start + rows_per_block
                    with (
                        decayed_out_dfb.wait() as decayed,
                        update_out_dfb.wait() as update,
                    ):
                        tx_decayed = ttl.copy(
                            decayed,
                            decayed_out[row_start:row_stop, 0:1],
                        )
                        tx_update = ttl.copy(
                            update,
                            update_out[row_start:row_stop, 0:1],
                        )
                        tx_decayed.wait()
                        tx_update.wait()

    @tt_lang_operation(
        operation_id=_operation_id(
            "parameter_prepare", beta1, beta2, granularity,
            gradient_page_offset, gradient_pages,
        ),
        arg_roles=("in", "in", "in", "out", "out"),
        version_tag="adamw-exact-parameter-prepare-v1",
    )
    def parameter_prepare(
        state_in,
        denominator_in,
        decay_factor_in,
        decayed_out,
        update_out,
    ):
        _clear_ttl_compile_wrapper_cache(parameter_prepare_program)
        result = parameter_prepare_program(
            state_in,
            denominator_in,
            decay_factor_in,
            decayed_out,
            update_out,
        )
        total_blocks = (
            denominator_in.shape[0] // TILE_SIZE // granularity
        )
        _install_native_compute(
            parameter_prepare_program,
            source_name="adamw_exact_parameter_prepare_compute.cpp",
            expected_cbs=7,
            replacements={
                "@TOTAL_BLOCKS@": str(total_blocks),
                "@BLOCKS_PER_NODE@": str(-(-total_blocks // 64)),
                "@ROWS_PER_BLOCK@": str(granularity),
            },
        )
        return result

    @ttl.operation(
        grid=(8, 8),
        fp32_dest_acc_en=True,
        options="--no-ttl-maximize-dst",
    )
    def parameter_finalize_program(
        decayed_in,
        update_in,
        state_in,
        state_out,
    ):
        rows_per_block = granularity
        state_region_rows = decayed_in.shape[0] // TILE_SIZE
        total_blocks = state_region_rows // rows_per_block
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        blocks_per_node = -(-total_blocks // total_nodes)
        block_shape = (rows_per_block, 1)

        decayed_dfb = ttl.make_dataflow_buffer_like(
            decayed_in, shape=block_shape, block_count=2
        )
        update_dfb = ttl.make_dataflow_buffer_like(
            update_in, shape=block_shape, block_count=2
        )
        exp_avg_dfb = ttl.make_dataflow_buffer_like(
            state_in, shape=block_shape, block_count=2
        )
        exp_avg_sq_dfb = ttl.make_dataflow_buffer_like(
            state_in, shape=block_shape, block_count=2
        )
        param_out_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=block_shape, block_count=2
        )
        exp_avg_out_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=block_shape, block_count=2
        )
        exp_avg_sq_out_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=block_shape, block_count=2
        )
        sum_f32_dfb = ttl.make_dataflow_buffer_like(
            decayed_in, shape=block_shape, block_count=2
        )

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node_index * blocks_per_node + local_block
                if block < total_blocks:
                    with (
                        decayed_dfb.wait() as decayed,
                        update_dfb.wait() as update,
                        exp_avg_dfb.wait() as exp_avg,
                        exp_avg_sq_dfb.wait() as exp_avg_sq,
                        param_out_dfb.reserve() as param_out,
                        exp_avg_out_dfb.reserve() as exp_avg_out,
                        exp_avg_sq_out_dfb.reserve() as exp_avg_sq_out,
                    ):
                        param_out.store(
                            ttl.math.typecast(decayed, torch.bfloat16)
                        )
                        exp_avg_out.store(exp_avg)
                        exp_avg_sq_out.store(exp_avg_sq)
                        if node_index < 0:
                            with sum_f32_dfb.reserve() as checkpoint:
                                checkpoint.store(update)

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node_index * blocks_per_node + local_block
                if block < total_blocks:
                    row_start = block * rows_per_block
                    row_stop = row_start + rows_per_block
                    with (
                        decayed_dfb.reserve() as decayed,
                        update_dfb.reserve() as update,
                        exp_avg_dfb.reserve() as exp_avg,
                        exp_avg_sq_dfb.reserve() as exp_avg_sq,
                    ):
                        tx_decayed = ttl.copy(
                            decayed_in[row_start:row_stop, 0:1], decayed
                        )
                        tx_update = ttl.copy(
                            update_in[row_start:row_stop, 0:1], update
                        )
                        tx_exp_avg = ttl.copy(
                            state_in[
                                row_start + state_region_rows :
                                row_stop + state_region_rows,
                                0:1,
                            ],
                            exp_avg,
                        )
                        tx_exp_avg_sq = ttl.copy(
                            state_in[
                                row_start + 2 * state_region_rows :
                                row_stop + 2 * state_region_rows,
                                0:1,
                            ],
                            exp_avg_sq,
                        )
                        tx_decayed.wait()
                        tx_update.wait()
                        tx_exp_avg.wait()
                        tx_exp_avg_sq.wait()

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node_index * blocks_per_node + local_block
                if block < total_blocks:
                    row_start = block * rows_per_block
                    row_stop = row_start + rows_per_block
                    with (
                        param_out_dfb.wait() as param,
                        exp_avg_out_dfb.wait() as exp_avg,
                        exp_avg_sq_out_dfb.wait() as exp_avg_sq,
                    ):
                        tx_param = ttl.copy(
                            param, state_out[row_start:row_stop, 0:1]
                        )
                        tx_exp_avg = ttl.copy(
                            exp_avg,
                            state_out[
                                row_start + state_region_rows :
                                row_stop + state_region_rows,
                                0:1,
                            ],
                        )
                        tx_exp_avg_sq = ttl.copy(
                            exp_avg_sq,
                            state_out[
                                row_start + 2 * state_region_rows :
                                row_stop + 2 * state_region_rows,
                                0:1,
                            ],
                        )
                        tx_param.wait()
                        tx_exp_avg.wait()
                        tx_exp_avg_sq.wait()

    @tt_lang_operation(
        operation_id=_operation_id(
            "parameter_finalize", beta1, beta2, granularity,
            gradient_page_offset, gradient_pages,
        ),
        arg_roles=("in", "in", "in", "out"),
        version_tag="adamw-exact-parameter-finalize-v1",
    )
    def parameter_finalize(
        decayed_in,
        update_in,
        state_in,
        state_out,
    ):
        _clear_ttl_compile_wrapper_cache(parameter_finalize_program)
        result = parameter_finalize_program(
            decayed_in,
            update_in,
            state_in,
            state_out,
        )
        total_blocks = (
            decayed_in.shape[0] // TILE_SIZE // granularity
        )
        _install_native_compute(
            parameter_finalize_program,
            source_name="adamw_exact_parameter_finalize_compute.cpp",
            expected_cbs=8,
            replacements={
                "@TOTAL_BLOCKS@": str(total_blocks),
                "@BLOCKS_PER_NODE@": str(-(-total_blocks // 64)),
                "@ROWS_PER_BLOCK@": str(granularity),
            },
        )
        return result

    @ttl.operation(
        grid=(8, 8),
        fp32_dest_acc_en=True,
        options="--no-ttl-maximize-dst",
    )
    def parameter_add_program(lhs_in, rhs_in, sum_out):
        rows_per_block = granularity
        total_blocks = lhs_in.shape[0] // TILE_SIZE // rows_per_block
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        blocks_per_node = -(-total_blocks // (grid_cols * grid_rows))
        block_shape = (rows_per_block, 1)
        lhs_dfb = ttl.make_dataflow_buffer_like(
            lhs_in, shape=block_shape, block_count=2
        )
        rhs_dfb = ttl.make_dataflow_buffer_like(
            rhs_in, shape=block_shape, block_count=2
        )
        sum_dfb = ttl.make_dataflow_buffer_like(
            sum_out, shape=block_shape, block_count=2
        )

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            node = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node * blocks_per_node + local_block
                if block < total_blocks:
                    with (
                        lhs_dfb.wait() as lhs,
                        rhs_dfb.wait() as rhs,
                        sum_dfb.reserve() as result,
                    ):
                        result.store(lhs + rhs)

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            node = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node * blocks_per_node + local_block
                if block < total_blocks:
                    start = block * rows_per_block
                    stop = start + rows_per_block
                    with lhs_dfb.reserve() as lhs, rhs_dfb.reserve() as rhs:
                        tx_lhs = ttl.copy(lhs_in[start:stop, 0:1], lhs)
                        tx_rhs = ttl.copy(rhs_in[start:stop, 0:1], rhs)
                        tx_lhs.wait()
                        tx_rhs.wait()

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node * blocks_per_node + local_block
                if block < total_blocks:
                    start = block * rows_per_block
                    stop = start + rows_per_block
                    with sum_dfb.wait() as result:
                        tx = ttl.copy(result, sum_out[start:stop, 0:1])
                        tx.wait()

    @tt_lang_operation(
        operation_id=_operation_id(
            "parameter_add", beta1, beta2, granularity,
            gradient_page_offset, gradient_pages,
        ),
        arg_roles=("in", "in", "out"),
        version_tag="adamw-exact-parameter-add-v1",
    )
    def parameter_add(lhs_in, rhs_in, sum_out):
        _clear_ttl_compile_wrapper_cache(parameter_add_program)
        result = parameter_add_program(lhs_in, rhs_in, sum_out)
        total_blocks = lhs_in.shape[0] // TILE_SIZE // granularity
        _install_native_compute(
            parameter_add_program,
            source_name="adamw_exact_parameter_add_compute.cpp",
            expected_cbs=3,
            replacements={
                "@TOTAL_BLOCKS@": str(total_blocks),
                "@BLOCKS_PER_NODE@": str(-(-total_blocks // 64)),
                "@ROWS_PER_BLOCK@": str(granularity),
            },
        )
        return result

    @ttl.operation(grid=(8, 8), fp32_dest_acc_en=True)
    def parameter_commit_program(sum_in, state_in, state_out):
        rows_per_block = granularity
        state_region_rows = sum_in.shape[0] // TILE_SIZE
        total_blocks = state_region_rows // rows_per_block
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        blocks_per_node = -(-total_blocks // (grid_cols * grid_rows))
        block_shape = (rows_per_block, 1)
        sum_dfb = ttl.make_dataflow_buffer_like(
            sum_in, shape=block_shape, block_count=2
        )
        exp_avg_dfb = ttl.make_dataflow_buffer_like(
            state_in, shape=block_shape, block_count=2
        )
        exp_avg_sq_dfb = ttl.make_dataflow_buffer_like(
            state_in, shape=block_shape, block_count=2
        )
        param_out_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=block_shape, block_count=2
        )
        exp_avg_out_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=block_shape, block_count=2
        )
        exp_avg_sq_out_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=block_shape, block_count=2
        )

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            node = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node * blocks_per_node + local_block
                if block < total_blocks:
                    with (
                        sum_dfb.wait() as summed,
                        exp_avg_dfb.wait() as exp_avg,
                        exp_avg_sq_dfb.wait() as exp_avg_sq,
                        param_out_dfb.reserve() as param_out,
                        exp_avg_out_dfb.reserve() as exp_avg_out,
                        exp_avg_sq_out_dfb.reserve() as exp_avg_sq_out,
                    ):
                        param_out.store(
                            ttl.math.typecast(summed, torch.bfloat16)
                        )
                        exp_avg_out.store(exp_avg)
                        exp_avg_sq_out.store(exp_avg_sq)

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            node = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node * blocks_per_node + local_block
                if block < total_blocks:
                    start = block * rows_per_block
                    stop = start + rows_per_block
                    with (
                        sum_dfb.reserve() as summed,
                        exp_avg_dfb.reserve() as exp_avg,
                        exp_avg_sq_dfb.reserve() as exp_avg_sq,
                    ):
                        tx_sum = ttl.copy(sum_in[start:stop, 0:1], summed)
                        tx_m = ttl.copy(
                            state_in[
                                start + state_region_rows :
                                stop + state_region_rows,
                                0:1,
                            ], exp_avg,
                        )
                        tx_v = ttl.copy(
                            state_in[
                                start + 2 * state_region_rows :
                                stop + 2 * state_region_rows,
                                0:1,
                            ], exp_avg_sq,
                        )
                        tx_sum.wait()
                        tx_m.wait()
                        tx_v.wait()

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            node = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node * blocks_per_node + local_block
                if block < total_blocks:
                    start = block * rows_per_block
                    stop = start + rows_per_block
                    with (
                        param_out_dfb.wait() as param,
                        exp_avg_out_dfb.wait() as exp_avg,
                        exp_avg_sq_out_dfb.wait() as exp_avg_sq,
                    ):
                        tx_param = ttl.copy(
                            param, state_out[start:stop, 0:1]
                        )
                        tx_m = ttl.copy(
                            exp_avg,
                            state_out[
                                start + state_region_rows :
                                stop + state_region_rows,
                                0:1,
                            ],
                        )
                        tx_v = ttl.copy(
                            exp_avg_sq,
                            state_out[
                                start + 2 * state_region_rows :
                                stop + 2 * state_region_rows,
                                0:1,
                            ],
                        )
                        tx_param.wait()
                        tx_m.wait()
                        tx_v.wait()

    @tt_lang_operation(
        operation_id=_operation_id(
            "parameter_commit", beta1, beta2, granularity,
            gradient_page_offset, gradient_pages,
        ),
        arg_roles=("in", "in", "out"),
        version_tag="adamw-exact-parameter-commit-v1",
    )
    def parameter_commit(sum_in, state_in, state_out):
        _clear_ttl_compile_wrapper_cache(parameter_commit_program)
        result = parameter_commit_program(sum_in, state_in, state_out)
        total_blocks = sum_in.shape[0] // TILE_SIZE // granularity
        _install_native_compute(
            parameter_commit_program,
            source_name="adamw_exact_parameter_commit_compute.cpp",
            expected_cbs=6,
            replacements={
                "@TOTAL_BLOCKS@": str(total_blocks),
                "@BLOCKS_PER_NODE@": str(-(-total_blocks // 64)),
                "@ROWS_PER_BLOCK@": str(granularity),
            },
        )
        return result

    def adamw_exact(
        state_in,
        grad_in,
        decay_factor_in,
        denominator_scale_in,
        scaled_epsilon_in,
        compatibility_in,
        state_out,
    ):
        del compatibility_in
        selected_gradient_rows = (
            grad_in.shape[0]
            if gradient_pages is None else gradient_pages * TILE_SIZE
        )
        moments_out = torch.zeros(
            (
                state_in.shape[0] + selected_gradient_rows,
                state_in.shape[1],
            ),
            dtype=state_in.dtype,
            device=state_in.device,
        )
        denominator_rounded_out = torch.zeros(
            (selected_gradient_rows, grad_in.shape[1]),
            dtype=torch.float32,
            device=grad_in.device,
        )
        denominator_out = torch.zeros(
            (selected_gradient_rows, grad_in.shape[1]),
            dtype=torch.float32,
            device=grad_in.device,
        )
        moments_out = moments(state_in, grad_in, moments_out)
        denominator_rounded_out = denominator(
            moments_out,
            denominator_scale_in,
            denominator_rounded_out,
        )
        denominator_out = epsilon(
            denominator_rounded_out,
            scaled_epsilon_in,
            denominator_out,
        )
        if parameter_backend == "native":
            return parameter(
                moments_out,
                denominator_out,
                decay_factor_in,
                state_out,
            )

        if parameter_backend == "native_split":
            decayed_out = torch.zeros(
                (selected_gradient_rows, grad_in.shape[1]),
                dtype=torch.float32,
                device=grad_in.device,
            )
            update_out = torch.zeros_like(decayed_out)
            decayed_out, update_out = parameter_prepare(
                moments_out,
                denominator_out,
                decay_factor_in,
                decayed_out,
                update_out,
            )
            sum_out = torch.zeros_like(decayed_out)
            sum_out = parameter_add(decayed_out, update_out, sum_out)
            return parameter_commit(
                sum_out,
                moments_out,
                state_out,
            )

        # Diagnostic oracle for the final mixed-precision boundary.  This is
        # intentionally expressed with framework primitives so its lowering
        # is identical to capturable AdamW.  It proves whether divergence is
        # confined to the native final parameter program; it is not a
        # production fast path and remains explicitly selectable.
        region_rows = selected_gradient_rows
        parameter_in = moments_out[:region_rows]
        exp_avg = moments_out[region_rows : 2 * region_rows]
        exp_avg_sq = moments_out[2 * region_rows : 3 * region_rows]
        next_parameter = (
            (parameter_in * decay_factor_in[0, 0]).float()
            + exp_avg.float() / denominator_out
        ).bfloat16()
        return torch.cat((next_parameter, exp_avg, exp_avg_sq), dim=0)

    adamw_exact.gradient_page_offset = gradient_page_offset
    adamw_exact.gradient_pages = gradient_pages
    adamw_exact.parameter_backend = parameter_backend
    # Artifact-v4 composition consumes qualified physical children directly.
    # Expose each private TT-Lang program so it can be serialized once without
    # launching a device graph or reopening its source in a parent schedule.
    moments._ttl_program = moments_program
    denominator._ttl_program = denominator_program
    epsilon._ttl_program = epsilon_program
    parameter._ttl_program = parameter_program
    parameter_prepare._ttl_program = parameter_prepare_program
    parameter_finalize._ttl_program = parameter_finalize_program
    parameter_add._ttl_program = parameter_add_program
    parameter_commit._ttl_program = parameter_commit_program
    adamw_exact.debug_phases = (
        moments,
        denominator,
        epsilon,
        parameter,
        parameter_prepare,
        parameter_finalize,
        parameter_add,
        parameter_commit,
    )
    return adamw_exact
