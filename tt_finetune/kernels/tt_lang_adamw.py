"""Experimental fused AdamW kernel for TT-XLA through TT-Lang.

The public trainer does not select this kernel yet.  It is deliberately kept
behind a small factory while its numerical behaviour and Blackhole performance
are validated against the existing tile-packed PyTorch/XLA optimizer.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Callable

import torch


TILE_SIZE = 32
ttl = None
tt_lang_operation = None
SUPPORTED_SEMANTICS = ("textbook", "capturable_xla")


def _clear_ttl_compile_wrapper_cache(operation: Callable) -> None:
    """Force the TT-XLA resolver to observe a compile on repeated resolution.

    This development TT-XLA wheel can resolve the same custom op twice.  Its
    resolver captures calls to ``ttl._compile_kernel``, while the TT-Lang
    operation wrapper has its own Python cache.  A second resolution otherwise
    returns the cached kernel without calling the capture hook, which the
    resolver misdiagnoses as a failed compile.

    This clears only TT-Lang's Python compile-wrapper dictionary.  The emitted
    TT-Metal program and runtime program caches are unaffected.
    """

    closure = operation.__closure__ or ()
    for name, cell in zip(operation.__code__.co_freevars, closure):
        if name == "cache" and isinstance(cell.cell_contents, dict):
            cell.cell_contents.clear()
            return


def _variant_id(
    beta1: float,
    beta2: float,
    granularity: int,
    semantics: str,
) -> str:
    payload = (
        f"{beta1:.17g}:{beta2:.17g}:{granularity}:{semantics}:v2"
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]


@lru_cache(maxsize=32)
def build_tt_lang_adamw(
    beta1: float = 0.9,
    beta2: float = 0.999,
    granularity: int = 2,
    semantics: str = "textbook",
) -> Callable:
    """Build a double-buffered fused AdamW operation.

    The ``textbook`` variant accepts ``lr_decay``,
    ``inv_sqrt_bias_correction2``, ``eps`` and ``step_size`` as its four
    one-tile runtime tensors.

    The ``capturable_xla`` variant follows the operation order and explicit
    BF16/F32 casts in PyTorch's capturable AdamW graph.  Its four runtime
    tensors are ``decay_factor`` (BF16), ``denominator_scale`` (F32),
    ``scaled_epsilon`` (F32), and an unused F32 compatibility tile.  Keeping
    changing values in tensors lets one compiled program serve every step.

    The returned callable accepts inputs first and its one output buffer last:

    ``(state, grad, lr_decay, inv_sqrt_bc2, eps, step_size, state_out)``.

    ``state`` is a contiguous tensor with three equal row regions:
    ``[parameter | first moment | second moment]``.  One result is required
    because the current TT-XLA Shardy frontend rejects tuple-valued custom
    calls.  Keeping all optimizer state contiguous is also the desired
    long-term representation for residency and prefetching.
    """

    if not 0.0 <= beta1 < 1.0:
        raise ValueError("beta1 must be in [0, 1)")
    if not 0.0 <= beta2 < 1.0:
        raise ValueError("beta2 must be in [0, 1)")
    if granularity <= 0:
        raise ValueError("granularity must be positive")
    if semantics not in SUPPORTED_SEMANTICS:
        raise ValueError(
            f"semantics must be one of {SUPPORTED_SEMANTICS}, got "
            f"{semantics!r}"
        )

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import (
                tt_lang_operation as tt_lang_operation_decorator,
            )
        except ImportError as exc:  # pragma: no cover - host-only environment
            raise RuntimeError(
                "The TT-Lang AdamW kernel requires the TT-XLA worker "
                "environment"
            ) from exc
        ttl = ttl_module
        tt_lang_operation = tt_lang_operation_decorator

    one_minus_beta1 = float(
        torch.tensor(1.0 - beta1, dtype=torch.bfloat16).float().item()
    )
    one_minus_beta2 = float(
        torch.tensor(1.0 - beta2, dtype=torch.bfloat16).float().item()
    )
    beta2_bf16 = float(
        torch.tensor(beta2, dtype=torch.bfloat16).float().item()
    )
    variant = _variant_id(beta1, beta2, granularity, semantics)

    @ttl.operation(
        grid="full",
        fp32_dest_acc_en=semantics == "capturable_xla",
        dst_full_sync_en=semantics == "capturable_xla",
        options=(
            "--no-ttl-maximize-dst"
            if semantics == "capturable_xla"
            else None
        ),
    )
    def _adamw_program(
        state_in,
        grad_in,
        lr_decay_in,
        inv_sqrt_bc2_in,
        eps_in,
        step_size_in,
        state_out,
    ):
        row_tiles_per_block = granularity
        total_blocks = (
            grad_in.shape[0] // TILE_SIZE // row_tiles_per_block
        )
        state_region_row_tiles = grad_in.shape[0] // TILE_SIZE
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_nodes = grid_cols * grid_rows
        blocks_per_node = -(-total_blocks // total_nodes)

        # Two entries are intentional: while compute consumes one block, the
        # reader can fill the next and the writer can drain the previous one.
        param_dfb = ttl.make_dataflow_buffer_like(
            state_in, shape=(row_tiles_per_block, 1), block_count=2
        )
        grad_dfb = ttl.make_dataflow_buffer_like(
            grad_in, shape=(row_tiles_per_block, 1), block_count=2
        )
        exp_avg_dfb = ttl.make_dataflow_buffer_like(
            state_in, shape=(row_tiles_per_block, 1), block_count=2
        )
        exp_avg_sq_dfb = ttl.make_dataflow_buffer_like(
            state_in, shape=(row_tiles_per_block, 1), block_count=2
        )

        # Runtime scalars are loaded once per core and kept resident for the
        # full local loop.  They are one physical tile each.
        lr_decay_dfb = ttl.make_dataflow_buffer_like(
            lr_decay_in, shape=(1, 1), block_count=1
        )
        inv_sqrt_bc2_dfb = ttl.make_dataflow_buffer_like(
            inv_sqrt_bc2_in, shape=(1, 1), block_count=1
        )
        eps_dfb = ttl.make_dataflow_buffer_like(
            eps_in, shape=(1, 1), block_count=1
        )
        step_size_dfb = ttl.make_dataflow_buffer_like(
            step_size_in, shape=(1, 1), block_count=1
        )

        param_out_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=(row_tiles_per_block, 1), block_count=2
        )
        exp_avg_out_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=(row_tiles_per_block, 1), block_count=2
        )
        exp_avg_sq_out_dfb = ttl.make_dataflow_buffer_like(
            state_out, shape=(row_tiles_per_block, 1), block_count=2
        )

        if semantics == "capturable_xla":

            @ttl.compute()
            def compute():
                decay_factor_tile = lr_decay_dfb.wait()
                denominator_scale_tile = inv_sqrt_bc2_dfb.wait()
                scaled_epsilon_tile = eps_dfb.wait()
                compatibility_tile = step_size_dfb.wait()

                block_shape = (row_tiles_per_block, 1)
                decay_factor = ttl.block.broadcast(
                    decay_factor_tile, dims=[0], shape=block_shape
                )
                denominator_scale = ttl.block.broadcast(
                    denominator_scale_tile, dims=[0], shape=block_shape
                )
                scaled_epsilon = ttl.block.broadcast(
                    scaled_epsilon_tile, dims=[0], shape=block_shape
                )

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
                            param_out_dfb.reserve() as updated_param,
                            exp_avg_out_dfb.reserve() as updated_exp_avg,
                            exp_avg_sq_out_dfb.reserve()
                            as updated_exp_avg_sq,
                        ):
                            # Match torch.optim.AdamW(capturable=True):
                            # lerp is m + (g - m) * (1 - beta1), with
                            # every operation performed at the BF16 tensor
                            # dtype before the explicit F32 conversion.
                            next_exp_avg = (
                                exp_avg
                                + (grad - exp_avg) * one_minus_beta1
                            )
                            next_exp_avg_sq = (
                                exp_avg_sq * beta2_bf16
                                + (grad * grad) * one_minus_beta2
                            )

                            # StableHLO explicitly rounds the division result
                            # to BF16, then converts it back to F32 before
                            # adding the scaled epsilon.
                            sqrt_exp_avg_sq = ttl.sqrt(next_exp_avg_sq)
                            denominator_fp32 = (
                                ttl.math.typecast(
                                    sqrt_exp_avg_sq, torch.float32
                                )
                                / denominator_scale
                            )
                            denominator_rounded = ttl.math.typecast(
                                denominator_fp32, torch.bfloat16
                            )
                            denominator_fp32 = (
                                ttl.math.typecast(
                                    denominator_rounded, torch.float32
                                )
                                + scaled_epsilon
                            )
                            next_param_fp32 = (
                                ttl.math.typecast(
                                    param * decay_factor, torch.float32
                                )
                                + ttl.math.typecast(
                                    next_exp_avg, torch.float32
                                )
                                / denominator_fp32
                            )
                            next_param = ttl.math.typecast(
                                next_param_fp32, torch.bfloat16
                            )

                            updated_param.store(next_param)
                            updated_exp_avg.store(next_exp_avg)
                            updated_exp_avg_sq.store(next_exp_avg_sq)

                decay_factor_tile.pop()
                denominator_scale_tile.pop()
                scaled_epsilon_tile.pop()
                compatibility_tile.pop()

        else:

            @ttl.compute()
            def compute():
                lr_decay_tile = lr_decay_dfb.wait()
                inv_sqrt_bc2_tile = inv_sqrt_bc2_dfb.wait()
                eps_tile = eps_dfb.wait()
                step_size_tile = step_size_dfb.wait()

                block_shape = (row_tiles_per_block, 1)
                lr_decay = ttl.block.broadcast(
                    lr_decay_tile, dims=[0], shape=block_shape
                )
                inv_sqrt_bc2 = ttl.block.broadcast(
                    inv_sqrt_bc2_tile, dims=[0], shape=block_shape
                )
                eps = ttl.block.broadcast(
                    eps_tile, dims=[0], shape=block_shape
                )
                step_size = ttl.block.broadcast(
                    step_size_tile, dims=[0], shape=block_shape
                )

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
                            param_out_dfb.reserve() as updated_param,
                            exp_avg_out_dfb.reserve()
                            as updated_exp_avg,
                            exp_avg_sq_out_dfb.reserve()
                            as updated_exp_avg_sq,
                        ):
                            next_exp_avg = (
                                exp_avg * beta1
                                + grad * one_minus_beta1
                            )
                            next_exp_avg_sq = (
                                exp_avg_sq * beta2
                                + (grad * grad) * one_minus_beta2
                            )
                            denominator = (
                                ttl.sqrt(next_exp_avg_sq)
                                * inv_sqrt_bc2
                                + eps
                            )
                            decayed_param = param - param * lr_decay
                            next_param = (
                                decayed_param
                                - (next_exp_avg / denominator) * step_size
                            )

                            updated_param.store(next_param)
                            updated_exp_avg.store(next_exp_avg)
                            updated_exp_avg_sq.store(next_exp_avg_sq)

                lr_decay_tile.pop()
                inv_sqrt_bc2_tile.pop()
                eps_tile.pop()
                step_size_tile.pop()

        @ttl.datamovement()
        def read():
            # Issue all scalar reads before waiting so their NoC transactions
            # can overlap.  Each scalar remains in L1 until compute finishes.
            lr_decay_block = lr_decay_dfb.reserve()
            inv_sqrt_bc2_block = inv_sqrt_bc2_dfb.reserve()
            eps_block = eps_dfb.reserve()
            step_size_block = step_size_dfb.reserve()
            tx_lr_decay = ttl.copy(
                lr_decay_in[0:1, 0:1], lr_decay_block
            )
            tx_inv_sqrt_bc2 = ttl.copy(
                inv_sqrt_bc2_in[0:1, 0:1], inv_sqrt_bc2_block
            )
            tx_eps = ttl.copy(eps_in[0:1, 0:1], eps_block)
            tx_step_size = ttl.copy(
                step_size_in[0:1, 0:1], step_size_block
            )
            tx_lr_decay.wait()
            tx_inv_sqrt_bc2.wait()
            tx_eps.wait()
            tx_step_size.wait()
            lr_decay_block.push()
            inv_sqrt_bc2_block.push()
            eps_block.push()
            step_size_block.push()

            node_col, node_row = ttl.node(dims=2)
            node_index = node_row * grid_cols + node_col
            for local_block in range(blocks_per_node):
                block = node_index * blocks_per_node + local_block
                if block < total_blocks:
                    row_start = block * row_tiles_per_block
                    row_stop = row_start + row_tiles_per_block
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
                            grad_in[row_start:row_stop, 0:1], grad
                        )
                        tx_exp_avg = ttl.copy(
                            state_in[
                                row_start
                                + state_region_row_tiles : row_stop
                                + state_region_row_tiles,
                                0:1,
                            ],
                            exp_avg,
                        )
                        tx_exp_avg_sq = ttl.copy(
                            state_in[
                                row_start
                                + 2 * state_region_row_tiles : row_stop
                                + 2 * state_region_row_tiles,
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
                    row_start = block * row_tiles_per_block
                    row_stop = row_start + row_tiles_per_block
                    with (
                        param_out_dfb.wait() as updated_param,
                        exp_avg_out_dfb.wait() as updated_exp_avg,
                        exp_avg_sq_out_dfb.wait() as updated_exp_avg_sq,
                    ):
                        tx_param = ttl.copy(
                            updated_param,
                            state_out[row_start:row_stop, 0:1],
                        )
                        tx_exp_avg = ttl.copy(
                            updated_exp_avg,
                            state_out[
                                row_start
                                + state_region_row_tiles : row_stop
                                + state_region_row_tiles,
                                0:1,
                            ],
                        )
                        tx_exp_avg_sq = ttl.copy(
                            updated_exp_avg_sq,
                            state_out[
                                row_start
                                + 2 * state_region_row_tiles : row_stop
                                + 2 * state_region_row_tiles,
                                0:1,
                            ],
                        )
                        tx_param.wait()
                        tx_exp_avg.wait()
                        tx_exp_avg_sq.wait()

    @tt_lang_operation(
        operation_id=f"tt_finetune.adamw.{variant}",
        arg_roles=(
            "in",
            "in",
            "in",
            "in",
            "in",
            "in",
            "out",
        ),
        version_tag=f"adamw-{variant}",
    )
    def adamw(
        state_in,
        grad_in,
        lr_decay_in,
        inv_sqrt_bc2_in,
        eps_in,
        step_size_in,
        state_out,
    ):
        _clear_ttl_compile_wrapper_cache(_adamw_program)
        return _adamw_program(
            state_in,
            grad_in,
            lr_decay_in,
            inv_sqrt_bc2_in,
            eps_in,
            step_size_in,
            state_out,
        )

    return adamw
