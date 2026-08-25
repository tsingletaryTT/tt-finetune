"""Blackhole schedules and L1 budgets for resident training attention.

There are deliberately two schedules.  ``reference_*`` reproduces the
physical TT-XLA/TT-NN lowering used by the exact framework path: query-head
tile rows are distributed over many Tensix workers and GQA dK/dV partials are
reduced afterwards.  ``resident_kv_workers`` describes the earlier, very fast
one-KV-head owner prototype.  Keeping both names explicit prevents a fast but
numerically different schedule from being mistaken for the exact path.

The planner is strict: unsupported geometry must use the framework fallback
instead of silently selecting a poor or numerically different schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

from .tt_lang_lora import TILE_SIZE


BLACKHOLE_WORKER_CORES = 110
BLACKHOLE_L1_BYTES = int(1.46 * 1024 * 1024)
DEFAULT_L1_FRACTION = 0.75
FP32_TILE_BYTES = TILE_SIZE * TILE_SIZE * 4
BF16_TILE_BYTES = TILE_SIZE * TILE_SIZE * 2


@dataclass(frozen=True)
class AttentionBlockPlan:
    batch_size: int
    sequence_length: int
    query_heads: int
    kv_heads: int
    head_dim: int
    query_chunks: int
    kv_group_size: int
    reference_dpdqdv_workers: int
    reference_dk_workers: int
    reference_qt_staging_workers: int
    reference_peak_workers: int
    resident_kv_workers: int
    active_workers: int
    waves: int
    score_tiles_per_head: int
    vector_tiles_per_head: int
    resident_tiles_per_worker: int
    estimated_peak_l1_bytes: int
    l1_budget_bytes: int
    gqa_noc_partial_tiles: int
    backward_custom_calls: int = 1

    @property
    def utilization(self) -> float:
        return min(1.0, self.active_workers / BLACKHOLE_WORKER_CORES)

    @property
    def reference_utilization(self) -> float:
        return min(1.0, self.reference_peak_workers / BLACKHOLE_WORKER_CORES)


@dataclass(frozen=True)
class ResidentAttentionAccounting:
    """Exact v50 BF16 buffer/traffic accounting for one owner invocation."""

    circular_buffers: int
    device_semaphores: int
    configured_cb_l1_bytes_per_active_core: int
    conservative_peak_l1_bytes_per_active_core: int
    input_dram_read_bytes: int
    final_dram_write_bytes: int
    intermediate_dram_write_bytes: int
    ds_transpose_noc_bytes: int
    gqa_peer_noc_bytes: int
    gqa_local_gather_bytes: int
    total_noc_l1_payload_bytes: int


def resident_attention_bf16_accounting(
    plan: AttentionBlockPlan,
) -> ResidentAttentionAccounting:
    """Account for the qualified S=128 BF16 resident implementation.

    The configured-CB value is the exact sum of all 32 circular-buffer
    allocations. It deliberately excludes kernel text, runtime bookkeeping,
    and semaphore storage; the planner's conservative peak remains the
    admission-control number.
    """

    seq_tiles = plan.sequence_length // TILE_SIZE
    dim_tiles = plan.head_dim // TILE_SIZE
    vector_tiles = seq_tiles * dim_tiles
    group = plan.kv_group_size
    # Block counts in ABI order. CBs 0/1/12/13/14/15/18 are physical FP32;
    # every other resident buffer is BF16 in the qualified training path.
    blocks = (
        seq_tiles, seq_tiles, dim_tiles, vector_tiles, vector_tiles,
        vector_tiles, vector_tiles, 1, seq_tiles, dim_tiles,
        vector_tiles, seq_tiles, seq_tiles, 1, seq_tiles, seq_tiles,
        seq_tiles, dim_tiles, seq_tiles, seq_tiles, seq_tiles,
        vector_tiles, dim_tiles, (group - 1) * dim_tiles,
        (group - 1) * dim_tiles, 2 * dim_tiles * group, 1, 1,
        dim_tiles, dim_tiles, dim_tiles, seq_tiles,
    )
    fp32_cbs = {0, 1, 12, 13, 14, 15, 18}
    cb_bytes = sum(
        block_count * (
            FP32_TILE_BYTES if index in fp32_cbs else BF16_TILE_BYTES
        )
        for index, block_count in enumerate(blocks)
    )
    jobs = plan.reference_dpdqdv_workers
    # Each query-row worker reads two probability orientations, its full dO
    # and Q head, its local dO row, and the compact K/V head it consumes.
    dram_reads = (
        jobs * 2 * seq_tiles * FP32_TILE_BYTES
        + jobs * (2 * vector_tiles + dim_tiles) * BF16_TILE_BYTES
        + jobs * 2 * vector_tiles * BF16_TILE_BYTES
    )
    final_writes = (
        (plan.query_heads + 2 * plan.kv_heads)
        * vector_tiles
        * BF16_TILE_BYTES
    )
    ds_noc = jobs * seq_tiles * BF16_TILE_BYTES
    peer_noc = plan.gqa_noc_partial_tiles * BF16_TILE_BYTES
    gather_noc = (
        plan.kv_heads * seq_tiles * 2 * dim_tiles * group
        * BF16_TILE_BYTES
    )
    return ResidentAttentionAccounting(
        circular_buffers=len(blocks),
        device_semaphores=12,
        configured_cb_l1_bytes_per_active_core=cb_bytes,
        conservative_peak_l1_bytes_per_active_core=(
            plan.estimated_peak_l1_bytes
        ),
        input_dram_read_bytes=dram_reads,
        final_dram_write_bytes=final_writes,
        intermediate_dram_write_bytes=0,
        ds_transpose_noc_bytes=ds_noc,
        gqa_peer_noc_bytes=peer_noc,
        gqa_local_gather_bytes=gather_noc,
        total_noc_l1_payload_bytes=ds_noc + peer_noc + gather_noc,
    )


def _positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def plan_attention_block(
    *,
    batch_size: int,
    sequence_length: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    is_causal: bool = True,
    dropout: float = 0.0,
    has_dense_mask: bool = False,
    l1_fraction: float = DEFAULT_L1_FRACTION,
) -> AttentionBlockPlan:
    """Plan one resident SDPA backward program on Blackhole P150a.

    Exact mode currently admits causal, dropout-free training with no external
    dense mask.  Head dimensions and sequence dimensions are tile aligned.
    Longer sequences are split into 128-token query chunks while K/V stream
    through the worker; sequence 128 remains the initial qualified schedule.
    """

    for name, value in (
        ("batch_size", batch_size),
        ("sequence_length", sequence_length),
        ("query_heads", query_heads),
        ("kv_heads", kv_heads),
        ("head_dim", head_dim),
    ):
        _positive(name, value)
    if sequence_length % TILE_SIZE:
        raise ValueError("sequence_length must be divisible by 32")
    if head_dim % TILE_SIZE:
        raise ValueError("head_dim must be divisible by 32")
    if query_heads % kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")
    if not is_causal:
        raise ValueError("resident exact attention currently requires causal attention")
    if dropout != 0.0:
        raise ValueError("resident exact attention does not support dropout")
    if has_dense_mask:
        raise ValueError("resident exact attention does not yet support a dense mask")
    if not 0.0 < l1_fraction <= 0.85:
        raise ValueError("l1_fraction must be in (0, 0.85]")

    seq_tiles = sequence_length // TILE_SIZE
    dim_tiles = head_dim // TILE_SIZE
    query_chunk_tiles = min(seq_tiles, 4)
    query_chunks = (seq_tiles + query_chunk_tiles - 1) // query_chunk_tiles
    score_tiles = query_chunk_tiles * seq_tiles
    vector_tiles = query_chunk_tiles * dim_tiles

    # Per worker peak during backward:
    # Q, K, V and dO streams; resident probability; dP and dS; dQ; local
    # dK/dV partials; two reduction/scalar tiles; plus two-deep reader and
    # writer CBs.  This is intentionally conservative and excludes no runtime
    # overhead by pretending that the full nominal L1 is available.
    k_or_v_tiles = seq_tiles * dim_tiles
    resident_tiles = (
        4 * vector_tiles
        + 2 * k_or_v_tiles
        + 3 * score_tiles
        + vector_tiles
        + 2 * k_or_v_tiles
        + 8
        + 12
    )
    # AOTAutograd keeps P, dP, dS, Q/K/V and their gradients in FP32 across
    # this boundary.  Accounting them as BF16 would understate L1 by 2x and
    # could generate a schedule that only fails after kernel compilation.
    estimated = resident_tiles * FP32_TILE_BYTES
    budget = int(BLACKHOLE_L1_BYTES * l1_fraction)
    if estimated > budget:
        raise ValueError(
            "attention worker L1 budget exceeded: "
            f"{estimated} > {budget} bytes; reduce the query chunk"
        )

    # The optimization-level-2 TT-MLIR lowering maps one query-head sequence
    # tile row to each dP/dQ/dV worker, and one query-head head-dimension tile
    # to each dK worker.  For the qualified sequence-128/head-dim-64 shapes
    # this is 36/18 workers for SmolLM2 and 56/28 for Qwen2.5.
    reference_dpdqdv_workers = batch_size * query_heads * seq_tiles
    # TTNN stages Q.T across query_heads*dim_tiles cores, but the exact final
    # dK output schedule is query-head sequence-row sharded, just like dV.
    # Treating the operand-staging grid as the dK compute grid was measured
    # nonexact and slower.
    reference_dk_workers = batch_size * query_heads * seq_tiles
    reference_qt_staging_workers = batch_size * query_heads * dim_tiles
    reference_peak_workers = max(
        reference_dpdqdv_workers, reference_dk_workers
    )
    waves = (
        reference_peak_workers + BLACKHOLE_WORKER_CORES - 1
    ) // BLACKHOLE_WORKER_CORES
    active_workers = min(reference_peak_workers, BLACKHOLE_WORKER_CORES)
    resident_kv_workers = min(
        batch_size * kv_heads, BLACKHOLE_WORKER_CORES
    )
    group = query_heads // kv_heads
    # Every non-owner Q head sends one dK and one dV partial to the KV owner.
    gqa_noc_tiles = batch_size * kv_heads * (group - 1) * 2 * k_or_v_tiles
    return AttentionBlockPlan(
        batch_size=batch_size,
        sequence_length=sequence_length,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        query_chunks=query_chunks,
        kv_group_size=group,
        reference_dpdqdv_workers=reference_dpdqdv_workers,
        reference_dk_workers=reference_dk_workers,
        reference_qt_staging_workers=reference_qt_staging_workers,
        reference_peak_workers=reference_peak_workers,
        resident_kv_workers=resident_kv_workers,
        active_workers=active_workers,
        waves=waves,
        score_tiles_per_head=score_tiles,
        vector_tiles_per_head=vector_tiles,
        resident_tiles_per_worker=resident_tiles,
        estimated_peak_l1_bytes=estimated,
        l1_budget_bytes=budget,
        gqa_noc_partial_tiles=gqa_noc_tiles,
    )


def attention_block_support_error(**kwargs: object) -> str | None:
    try:
        plan_attention_block(**kwargs)  # type: ignore[arg-type]
    except ValueError as error:
        return str(error)
    return None
