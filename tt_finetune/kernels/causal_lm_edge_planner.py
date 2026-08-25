"""Blackhole plan for a resident final-norm/LM-head/loss owner.

The owner streams frozen vocabulary weights and never publishes logits or
dLogits.  It is deliberately a planner first: exact reduction order, L1
budget and NoC ownership must be fixed before a large TT-Metal program is
generated.
"""

from __future__ import annotations

from dataclasses import dataclass


TILE_SIZE = 32
BF16_TILE_BYTES = TILE_SIZE * TILE_SIZE * 2
FP32_TILE_BYTES = TILE_SIZE * TILE_SIZE * 4
P150A_WORKERS = 110
# Below the documented 1.5 MiB physical L1. This retains room for firmware,
# kernel text, semaphores and runtime-owned buffers.
P150A_SAFE_L1_BYTES = 1_148_190


@dataclass(frozen=True)
class CausalLMEdgePlan:
    tokens: int
    row_tiles: int
    hidden_features: int
    hidden_tiles: int
    vocabulary_size: int
    padded_vocabulary_size: int
    vocabulary_tiles: int
    active_workers: int
    workers_per_row_tile: tuple[int, ...]
    vocabulary_tiles_per_worker: tuple[int, ...]
    resident_logit_tiles_per_worker_max: int
    retains_distributed_logits_in_l1: bool
    vocabulary_tiles_per_block: int
    hidden_tiles_per_phase: int
    phases: tuple[str, ...]
    reduction_contract: str
    peak_l1_bytes_per_worker: int
    l1_safety_limit_bytes: int
    reference_materialized_logits_bytes: int
    reference_materialized_dlogits_bytes: int
    resident_intermediate_dram_bytes: int
    eliminated_intermediate_dram_bytes: int
    streamed_weight_bytes_per_step: int
    final_dhidden_bytes: int
    final_loss_bytes: int


def _balanced_workers(total: int, groups: int) -> tuple[int, ...]:
    quotient, remainder = divmod(total, groups)
    return tuple(
        quotient + (1 if index < remainder else 0)
        for index in range(groups)
    )


def plan_causal_lm_edge(
    *,
    tokens: int,
    hidden_features: int,
    vocabulary_size: int,
    workers: int = P150A_WORKERS,
    vocabulary_tiles_per_block: int = 4,
    hidden_tiles_per_phase: int = 2,
    l1_safety_limit_bytes: int = P150A_SAFE_L1_BYTES,
) -> CausalLMEdgePlan:
    """Plan an exact streamed vocabulary forward/loss/dHidden schedule.

    Each token-row tile owns a disjoint core group. Workers stream vocabulary
    blocks in ascending tile order, keep online max/sum/target state in FP32,
    and replay the same weight stream to form dHidden. A deterministic NoC
    reduction joins per-worker dHidden partials before the accepted RMSNorm
    backward consumes it.
    """

    if tokens <= 0 or tokens % TILE_SIZE:
        raise ValueError("tokens must be positive and tile aligned")
    if hidden_features <= 0 or hidden_features % TILE_SIZE:
        raise ValueError("hidden size must be positive and tile aligned")
    if vocabulary_size <= 0:
        raise ValueError("vocabulary size must be positive")
    if not 1 <= workers <= P150A_WORKERS:
        raise ValueError("workers must be within the P150a worker count")
    if vocabulary_tiles_per_block <= 0:
        raise ValueError("vocabulary block must contain at least one tile")
    if hidden_tiles_per_phase <= 0:
        raise ValueError("hidden phase must contain at least one tile")

    row_tiles = tokens // TILE_SIZE
    if row_tiles > workers:
        raise ValueError("each token-row tile requires at least one worker")
    hidden_tiles = hidden_features // TILE_SIZE
    padded_vocabulary = (
        (vocabulary_size + TILE_SIZE - 1) // TILE_SIZE * TILE_SIZE
    )
    vocabulary_tiles = padded_vocabulary // TILE_SIZE
    active_workers = min(workers, row_tiles * vocabulary_tiles)
    workers_by_row = _balanced_workers(active_workers, row_tiles)
    vocabulary_by_worker: list[int] = []
    for group_workers in workers_by_row:
        quotient, remainder = divmod(vocabulary_tiles, group_workers)
        vocabulary_by_worker.extend(
            quotient + (1 if index < remainder else 0)
            for index in range(group_workers)
        )

    n_block = vocabulary_tiles_per_block
    k_phase = min(hidden_tiles, hidden_tiles_per_phase)
    resident_logit_tiles = max(vocabulary_by_worker)
    # Conservative concurrent footprint on one worker: all locally-owned BF16
    # logits remain resident across the exact max/sum/target reductions; the
    # current block also needs BF16/FP32 compute buffers, hidden and weights are
    # double-buffered, dHidden has FP32/BF16 staging, and 64 KiB remains for
    # control/scratch CBs. Retaining the complete local logit shard is what
    # permits one logits matmul followed by in-place dLogits construction.
    peak_l1 = (
        resident_logit_tiles * BF16_TILE_BYTES
        + 2 * k_phase * BF16_TILE_BYTES
        + 2 * n_block * k_phase * BF16_TILE_BYTES
        + n_block * BF16_TILE_BYTES
        + n_block * FP32_TILE_BYTES
        + 4 * FP32_TILE_BYTES
        + hidden_tiles * FP32_TILE_BYTES
        + hidden_tiles * BF16_TILE_BYTES
        + 64 * 1024
    )
    if peak_l1 > l1_safety_limit_bytes:
        raise ValueError(
            f"causal-LM edge needs {peak_l1} L1 bytes per worker, safety "
            f"limit is {l1_safety_limit_bytes}"
        )

    logits_bytes = tokens * padded_vocabulary * 2
    dlogits_bytes = logits_bytes
    weight_bytes = padded_vocabulary * hidden_features * 2
    return CausalLMEdgePlan(
        tokens=tokens,
        row_tiles=row_tiles,
        hidden_features=hidden_features,
        hidden_tiles=hidden_tiles,
        vocabulary_size=vocabulary_size,
        padded_vocabulary_size=padded_vocabulary,
        vocabulary_tiles=vocabulary_tiles,
        active_workers=active_workers,
        workers_per_row_tile=workers_by_row,
        vocabulary_tiles_per_worker=tuple(vocabulary_by_worker),
        resident_logit_tiles_per_worker_max=resident_logit_tiles,
        retains_distributed_logits_in_l1=True,
        vocabulary_tiles_per_block=n_block,
        hidden_tiles_per_phase=k_phase,
        phases=(
            "final_rmsnorm_forward",
            "streaming_vocab_online_cross_entropy",
            "streaming_vocab_dhidden",
            "final_rmsnorm_backward",
        ),
        reduction_contract=(
            "ascending_vocab_tiles_fp32_online_state_then_fixed_noc_tree_"
            "and_reference_bf16_pack_boundaries"
        ),
        peak_l1_bytes_per_worker=peak_l1,
        l1_safety_limit_bytes=l1_safety_limit_bytes,
        reference_materialized_logits_bytes=logits_bytes,
        reference_materialized_dlogits_bytes=dlogits_bytes,
        resident_intermediate_dram_bytes=0,
        eliminated_intermediate_dram_bytes=logits_bytes + dlogits_bytes,
        streamed_weight_bytes_per_step=2 * weight_bytes,
        final_dhidden_bytes=tokens * hidden_features * 2,
        final_loss_bytes=4,
    )


__all__ = [
    "CausalLMEdgePlan",
    "P150A_SAFE_L1_BYTES",
    "P150A_WORKERS",
    "plan_causal_lm_edge",
]
