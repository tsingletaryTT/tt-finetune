"""Blackhole plan for one exact resident Transformer training layer.

This planner composes the already-qualified attention and gated-MLP owners.
It deliberately models *phase-aliased per-core L1*: attention, normalization,
and MLP do not occupy one core's SRAM at the same time.  Treating their L1
requirements as additive would reject Qwen even though the sequential owner
fits; treating all worker L1 as one shared cache would be equally incorrect.

The Stage-3 physical boundary is one forward and one backward custom call per
decoder layer.  Backward-lifetime checkpoints may live in DRAM between the
model-wide forward and backward sweeps, but no forward- or backward-internal
edge is a PJRT-visible tensor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .attention_block_planner import BLACKHOLE_L1_BYTES
from .full_attention_block_planner import (
    FullAttentionBlockPlan,
    plan_full_attention_block,
)
from .full_attention_forward_planner import (
    BF16_TILE_BYTES,
    FP32_TILE_BYTES,
    FullAttentionForwardPlan,
    plan_full_attention_forward,
)
from .model_workloads import EXACT_BASE_K_PHASES
from .resident_mlp_planner import (
    ResidentMLPPlan,
    plan_resident_gated_mlp,
)
from .tt_lang_resident_swiglu_down import (
    estimate_resident_swiglu_down_l1_bytes,
)
from .tt_lang_lora import TILE_SIZE


@dataclass(frozen=True)
class ResidentRMSNormPlan:
    """Streaming exact RMSNorm budget for one token-row owner.

    The values are admission-control bounds, not an assertion that the
    generic StableHLO reduction has already been reproduced.  Exactness still
    requires qualification of the hidden-width reduction tree and rsqrt mode.
    """

    hidden_tiles: int
    row_owners: int
    rows_per_owner: int
    stream_depth: int
    forward_l1_bytes_per_owner: int
    backward_l1_bytes_per_owner: int
    exact_forward_contract: tuple[str, ...]
    exact_backward_contract: tuple[str, ...]


@dataclass(frozen=True)
class ResidentLayerArenaSlice:
    """One page-addressed region in the unified BF16 layer arena."""

    name: str
    page_offset: int
    page_count: int
    logical_tiles: int
    pages_per_tile: int
    dtype: str


@dataclass(frozen=True)
class ResidentLayerArenaPlan:
    """Unified forward-output and backward-checkpoint storage.

    The physical tensor is BF16 because PJRT needs one ordinary output ABI.
    FP32 row statistics and attention probabilities occupy two consecutive
    2-KiB pages per logical tile and are never interpreted by PJRT.
    """

    physical_shape: tuple[int, int]
    hidden_tiles: int
    used_pages: int
    allocated_pages: int
    output_pages: int
    checkpoint_pages: int
    checkpoint_dram_bytes: int
    padding_bytes: int
    slices: tuple[ResidentLayerArenaSlice, ...]

    def slice(self, name: str) -> ResidentLayerArenaSlice:
        for region in self.slices:
            if region.name == name:
                return region
        raise KeyError(name)


@dataclass(frozen=True)
class ResidentLayerPhase:
    """Device epoch and ownership contract for a layer phase."""

    name: str
    worker_cores: int
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    l1_handoffs: tuple[str, ...]
    barrier_epoch: int


@dataclass(frozen=True)
class ResidentTransformerLayerTraffic:
    """Logical traffic at boundaries removed by the Stage-3 owner.

    This is intentionally named *logical* traffic.  StableHLO exposes these
    producer/consumer edges, but only a device profiler can prove how many of
    the bytes reached DRAM after compiler scheduling.  The separate private
    checkpoint count is physical data that must survive the model-wide
    forward/backward gap unless a later recompute policy removes it.
    """

    hidden_tensor_bytes: int
    gate_up_checkpoint_bytes: int
    attention_checkpoint_bytes: int
    attention_residual_checkpoint_bytes: int
    layer_input_checkpoint_bytes: int
    rmsnorm_checkpoint_bytes: int
    private_checkpoint_dram_bytes: int
    current_visible_intermediate_edge_bytes: int
    target_visible_intermediate_edge_bytes: int
    logical_boundary_bytes_eliminated: int


@dataclass(frozen=True)
class ResidentLayerForwardPhysicalPlan:
    """First physical one-call forward schedule on the existing two islands.

    Attention already uses a 50-core projection island and a 60-core
    attention island.  Once the post-attention normalization barrier closes,
    both islands rejoin as one 110-core MLP grid.  The four normalized-row
    owners remain the source of truth in L1; all workers fetch only the rows
    they need over NoC.  This is the silicon-qualified schedule used by the
    physical layer owner, rather than leaving 60 cores idle during MLP.
    """

    projection_workers: int
    attention_workers: int
    mlp_workers: int
    gate_up_total_jobs: int
    gate_up_jobs_per_mlp_core: int
    normalized_row_core_assignments: int
    gate_up_rank_reductions: int
    product_total_jobs: int
    product_jobs_per_mlp_core: int
    down_total_jobs: int
    down_jobs_per_mlp_core: int
    direct_normalized_handoff_noc_bytes: int
    final_residual_handoff_noc_bytes: int
    grouped_projection_l1_bytes_per_core: int
    swiglu_down_l1_bytes_per_core: int
    phase_aliased_peak_l1_bytes_per_core: int
    cb_epoch_order: tuple[str, ...]
    normalized_source_cb: int
    gate_resident_cb: int
    up_resident_cb: int


@dataclass(frozen=True)
class ResidentTransformerLayerPlan:
    tokens: int
    hidden_features: int
    intermediate_features: int
    query_heads: int
    kv_heads: int
    head_dim: int
    logical_rank: int
    physical_rank: int
    qkv_bias: bool
    rms_norm_epsilon: float
    current_custom_calls_per_layer: int
    target_custom_calls_per_layer: int
    current_custom_calls_for_depth: int
    target_custom_calls_for_depth: int
    model_depth: int
    l1_budget_bytes_per_core: int
    forward_peak_l1_bytes_per_core: int
    backward_peak_l1_bytes_per_core: int
    phase_handoff_l1_bytes_per_core: int
    forward_phases: tuple[str, ...]
    backward_phases: tuple[str, ...]
    final_forward_outputs: tuple[str, ...]
    final_backward_outputs: tuple[str, ...]
    attention_forward: FullAttentionForwardPlan
    attention_backward: FullAttentionBlockPlan
    mlp: ResidentMLPPlan
    input_norm: ResidentRMSNormPlan
    post_attention_norm: ResidentRMSNormPlan
    arena: ResidentLayerArenaPlan
    forward_schedule: tuple[ResidentLayerPhase, ...]
    backward_schedule: tuple[ResidentLayerPhase, ...]
    traffic: ResidentTransformerLayerTraffic
    physical_forward: ResidentLayerForwardPhysicalPlan

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _plan_rmsnorm(*, tokens: int, hidden_features: int) -> ResidentRMSNormPlan:
    hidden_tiles = hidden_features // TILE_SIZE
    # A tile-row owner handles 32 logical tokens across the complete hidden
    # width. Sequence-128 therefore uses four owners, not 110 scalar-row
    # owners. This is the ownership used by the accepted native kernels.
    row_owners = tokens // TILE_SIZE
    rows_per_owner = TILE_SIZE
    stream_depth = 2

    # Forward streams x and gamma through depth-two BF16 CBs. Squared x and
    # the normalized product use two FP32 tiles each; mean/rstd use two scalar
    # tiles. Output staging is depth two BF16. Backward additionally streams
    # dY and retains two FP32 row-reduction accumulators. These deliberately
    # conservative bounds leave code/semaphore headroom under the shared layer
    # admission budget.
    forward_bf16_tiles = 2 * stream_depth + stream_depth
    forward_fp32_tiles = 2 * stream_depth + 2
    backward_bf16_tiles = 3 * stream_depth + stream_depth
    backward_fp32_tiles = 3 * stream_depth + 4
    return ResidentRMSNormPlan(
        hidden_tiles=hidden_tiles,
        row_owners=row_owners,
        rows_per_owner=rows_per_owner,
        stream_depth=stream_depth,
        forward_l1_bytes_per_owner=(
            forward_bf16_tiles * BF16_TILE_BYTES
            + forward_fp32_tiles * FP32_TILE_BYTES
        ),
        backward_l1_bytes_per_owner=(
            backward_bf16_tiles * BF16_TILE_BYTES
            + backward_fp32_tiles * FP32_TILE_BYTES
        ),
        exact_forward_contract=(
            "BF16 x converts to FP32 before square",
            "hidden-width sum uses the qualified TTNN tile traversal tree",
            "mean and epsilon remain FP32 through the qualified rsqrt",
            "x*rstd converts to BF16 before BF16 gamma multiply",
        ),
        exact_backward_contract=(
            "autograd reductions retain the qualified hidden-tile order",
            "FP32 and BF16 spill/pack boundaries match the StableHLO path",
            "dX is packed once at the decoder-layer gradient edge",
        ),
    )


def _plan_layer_arena(
    *, tokens: int, hidden_features: int, intermediate_features: int,
    physical_rank: int, attention_forward: FullAttentionForwardPlan,
) -> ResidentLayerArenaPlan:
    """Lay out every cross-forward/backward lifetime in one arena."""

    sequence_tiles = tokens // TILE_SIZE
    hidden_tiles = hidden_features // TILE_SIZE
    intermediate_tiles = intermediate_features // TILE_SIZE
    rank_tiles = physical_rank // TILE_SIZE
    regions: list[ResidentLayerArenaSlice] = []
    cursor = 0

    def append(
        name: str, logical_tiles: int, *, pages_per_tile: int = 1,
        dtype: str = "bfloat16",
    ) -> None:
        nonlocal cursor
        page_count = logical_tiles * pages_per_tile
        regions.append(
            ResidentLayerArenaSlice(
                name=name,
                page_offset=cursor,
                page_count=page_count,
                logical_tiles=logical_tiles,
                pages_per_tile=pages_per_tile,
                dtype=dtype,
            )
        )
        cursor += page_count

    hidden_tensor_tiles = sequence_tiles * hidden_tiles
    append("layer_output", hidden_tensor_tiles)
    # The layer input and post-attention residual are the two residual sources
    # and the exact RMSNorm recompute inputs during backward.
    append("layer_input", hidden_tensor_tiles)
    append("post_attention_residual", hidden_tensor_tiles)
    # Row statistics survive the model-wide forward/backward gap. They are raw
    # FP32 inside the BF16 arena and therefore occupy two pages per tile.
    append(
        "input_rmsnorm_rstd_fp32", sequence_tiles,
        pages_per_tile=2, dtype="float32_raw",
    )
    append(
        "post_attention_rmsnorm_rstd_fp32", sequence_tiles,
        pages_per_tile=2, dtype="float32_raw",
    )

    for source in attention_forward.arena.slices:
        if source.name == "output":
            continue
        append(
            f"attention_{source.name}", source.logical_tiles,
            pages_per_tile=source.pages_per_tile, dtype=source.dtype,
        )

    # Exact full-MLP backward consumes gate/up outputs, their rank-H fields,
    # and the normalized MLP input. The previous traffic model counted only
    # gate/up outputs and therefore understated the real private checkpoint.
    member_tiles = sequence_tiles * intermediate_tiles
    append("mlp_gate", member_tiles)
    append("mlp_up", member_tiles)
    append("mlp_gate_rank_hidden", sequence_tiles * rank_tiles)
    append("mlp_up_rank_hidden", sequence_tiles * rank_tiles)
    append("mlp_normalized_input", hidden_tensor_tiles)

    used_pages = cursor
    allocated_pages = _ceil_div(used_pages, hidden_tiles) * hidden_tiles
    physical_tile_rows = allocated_pages // hidden_tiles
    output_pages = hidden_tensor_tiles
    checkpoint_pages = used_pages - output_pages
    return ResidentLayerArenaPlan(
        physical_shape=(physical_tile_rows * TILE_SIZE, hidden_features),
        hidden_tiles=hidden_tiles,
        used_pages=used_pages,
        allocated_pages=allocated_pages,
        output_pages=output_pages,
        checkpoint_pages=checkpoint_pages,
        checkpoint_dram_bytes=checkpoint_pages * BF16_TILE_BYTES,
        padding_bytes=(allocated_pages - used_pages) * BF16_TILE_BYTES,
        slices=tuple(regions),
    )


def _plan_physical_forward(
    *, tokens: int, hidden_features: int, intermediate_features: int,
    physical_rank: int, base_k_tiles_per_phase: int,
    attention_projection_l1_bytes: int,
) -> ResidentLayerForwardPhysicalPlan:
    """Plan the first source-compatible attention→MLP device schedule."""

    sequence_tiles = tokens // TILE_SIZE
    hidden_tiles = hidden_features // TILE_SIZE
    intermediate_tiles = intermediate_features // TILE_SIZE
    rank_tiles = physical_rank // TILE_SIZE
    projection_workers = 50
    attention_workers = 60
    mlp_workers = projection_workers + attention_workers
    gate_up_jobs = sequence_tiles * 2 * intermediate_tiles
    product_jobs = sequence_tiles * intermediate_tiles
    down_jobs = sequence_tiles * hidden_tiles
    gate_up_per_core = _ceil_div(gate_up_jobs, mlp_workers)
    product_per_core = _ceil_div(product_jobs, mlp_workers)
    down_per_core = _ceil_div(down_jobs, mlp_workers)
    if gate_up_per_core > 32:
        raise ValueError(
            "all-core direct gate/up schedule exceeds the 32-block CB limit"
        )
    jobs_per_row = 2 * intermediate_tiles
    row_assignments: set[tuple[int, int]] = set()
    for core in range(mlp_workers):
        first = core * gate_up_per_core
        stop = min(first + gate_up_per_core, gate_up_jobs)
        for job in range(first, stop):
            row_assignments.add((core, job // jobs_per_row))

    # Exact grouped frozen+LoRA forward CBs. FP32 accumulator CBs are counted
    # at 4 KiB/tile; everything else is BF16. The input checkpoint CBs are
    # omitted because the post-RMSNorm arena region is already the exact
    # backward checkpoint.
    grouped_bf16_tiles = (
        4 + 4 + 2 + 1
        + 4 * base_k_tiles_per_phase
        + max(2, gate_up_per_core)
        + 2 + 2 + max(2, gate_up_per_core) + 2
        + 2 * product_per_core  # routed gate and up resident windows
        + 2 * hidden_tiles  # depth-two persistent normalized row cache
    )
    grouped_fp32_tiles = 4  # depth-two H and frozen-base accumulators
    grouped_l1 = (
        grouped_bf16_tiles * BF16_TILE_BYTES
        + grouped_fp32_tiles * FP32_TILE_BYTES
    )
    swiglu_down_l1 = estimate_resident_swiglu_down_l1_bytes(
        intermediate_features=intermediate_features,
        product_jobs_per_core=product_per_core,
        output_jobs_per_core=down_per_core,
        rank_tiles=rank_tiles,
    ) + down_per_core * BF16_TILE_BYTES  # final residual input window
    phase_peak = max(
        attention_projection_l1_bytes, grouped_l1, swiglu_down_l1,
    )
    return ResidentLayerForwardPhysicalPlan(
        projection_workers=projection_workers,
        attention_workers=attention_workers,
        mlp_workers=mlp_workers,
        gate_up_total_jobs=gate_up_jobs,
        gate_up_jobs_per_mlp_core=gate_up_per_core,
        normalized_row_core_assignments=len(row_assignments),
        # Gate and up each reduce one X@A tile per token tile-row. The older
        # grouped program redundantly repeated this long-K reduction for every
        # output tile.
        gate_up_rank_reductions=2 * sequence_tiles,
        product_total_jobs=product_jobs,
        product_jobs_per_mlp_core=product_per_core,
        down_total_jobs=down_jobs,
        down_jobs_per_mlp_core=down_per_core,
        # A core caches each token row it owns exactly once. Both the frozen
        # projection and the separately multicast rank-H reuse that local row;
        # there is no per-output re-read and no duplicate H reduction.
        direct_normalized_handoff_noc_bytes=(
            len(row_assignments) * hidden_tiles * BF16_TILE_BYTES
        ),
        final_residual_handoff_noc_bytes=(
            down_jobs * BF16_TILE_BYTES
        ),
        grouped_projection_l1_bytes_per_core=grouped_l1,
        swiglu_down_l1_bytes_per_core=swiglu_down_l1,
        phase_aliased_peak_l1_bytes_per_core=phase_peak,
        cb_epoch_order=(
            "input_norm_qkv", "attention_o", "post_attention_norm",
            "grouped_gate_up", "swiglu_down", "final_residual",
        ),
        normalized_source_cb=22,
        gate_resident_cb=13,
        up_resident_cb=14,
    )


def plan_resident_transformer_layer(
    *,
    tokens: int,
    hidden_features: int,
    intermediate_features: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    logical_rank: int,
    model_depth: int,
    rms_norm_epsilon: float,
    qkv_bias: bool,
    physical_rank: int = TILE_SIZE,
    l1_fraction: float = 0.75,
    activation_checkpoint: str = "save_gate_up",
) -> ResidentTransformerLayerPlan:
    """Compose the accepted owners into a two-call decoder-layer plan."""

    dimensions = (
        tokens,
        hidden_features,
        intermediate_features,
        head_dim,
        physical_rank,
        model_depth,
    )
    if any(value <= 0 for value in dimensions):
        raise ValueError("layer dimensions and model depth must be positive")
    if any(
        value % TILE_SIZE
        for value in (
            tokens,
            hidden_features,
            intermediate_features,
            head_dim,
            physical_rank,
        )
    ):
        raise ValueError("resident layer dimensions must be tile aligned")
    if not 0 < logical_rank <= physical_rank:
        raise ValueError("logical rank must fit the physical rank tile")
    if rms_norm_epsilon <= 0.0:
        raise ValueError("RMSNorm epsilon must be positive")
    if not 0.0 < l1_fraction <= 0.80:
        raise ValueError("l1_fraction must be in (0, 0.80]")

    kv_features = kv_heads * head_dim
    attention_forward = plan_full_attention_forward(
        tokens=tokens,
        hidden_features=hidden_features,
        kv_features=kv_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        physical_rank=physical_rank,
        qkv_bias=qkv_bias,
    )
    attention_backward = plan_full_attention_block(
        batch_size=1,
        sequence_length=tokens,
        hidden_features=hidden_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        logical_rank=logical_rank,
        l1_fraction=l1_fraction,
    )
    mlp = plan_resident_gated_mlp(
        tokens=tokens,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        physical_rank=physical_rank,
        activation_checkpoint=activation_checkpoint,
        l1_fraction=l1_fraction,
    )
    input_norm = _plan_rmsnorm(
        tokens=tokens, hidden_features=hidden_features
    )
    post_attention_norm = _plan_rmsnorm(
        tokens=tokens, hidden_features=hidden_features
    )
    arena = _plan_layer_arena(
        tokens=tokens,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        physical_rank=physical_rank,
        attention_forward=attention_forward,
    )
    physical_forward = _plan_physical_forward(
        tokens=tokens,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        physical_rank=physical_rank,
        base_k_tiles_per_phase=EXACT_BASE_K_PHASES[
            (hidden_features, intermediate_features)
        ],
        # The accepted attention builder exposes the exact phase-aliased
        # projection-island footprint used by the same physical program.
        attention_projection_l1_bytes=max(
            attention_forward.attention_l1_bytes_per_query_owner,
            post_attention_norm.forward_l1_bytes_per_owner,
        ),
    )

    # One depth-two hidden stream plus FP32 scalar/reduction scratch crosses a
    # phase boundary at a time. The same addresses are reset and reused after
    # an on-device barrier; they are never added for all phases at once.
    phase_handoff_l1 = 4 * BF16_TILE_BYTES + 2 * FP32_TILE_BYTES
    forward_peak = max(
        attention_forward.attention_l1_bytes_per_query_owner,
        input_norm.forward_l1_bytes_per_owner,
        post_attention_norm.forward_l1_bytes_per_owner,
        mlp.estimated_backward_peak_l1_bytes,
    ) + phase_handoff_l1
    backward_peak = max(
        attention_backward.phase_peak_l1_bytes,
        input_norm.backward_l1_bytes_per_owner,
        post_attention_norm.backward_l1_bytes_per_owner,
        mlp.estimated_backward_peak_l1_bytes,
    ) + phase_handoff_l1
    budget = int(BLACKHOLE_L1_BYTES * l1_fraction)
    if max(forward_peak, backward_peak) > budget:
        raise ValueError(
            "resident Transformer layer exceeds per-core L1 budget: "
            f"{max(forward_peak, backward_peak)} > {budget} bytes"
        )
    if physical_forward.phase_aliased_peak_l1_bytes_per_core > budget:
        raise ValueError(
            "physical 50-core layer-forward schedule exceeds per-core L1 "
            f"budget: {physical_forward.phase_aliased_peak_l1_bytes_per_core} "
            f"> {budget} bytes"
        )

    hidden_bytes = tokens * hidden_features * 2
    gate_up_checkpoint_bytes = sum(
        region.page_count * BF16_TILE_BYTES
        for region in arena.slices
        if region.name.startswith("mlp_")
    )
    attention_checkpoint_bytes = sum(
        region.page_count * BF16_TILE_BYTES
        for region in arena.slices
        if region.name.startswith("attention_")
    )
    # The post-attention residual is both the second residual source and the
    # exact post-attention RMSNorm recompute input during backward.
    attention_residual_checkpoint_bytes = hidden_bytes
    layer_input_checkpoint_bytes = hidden_bytes
    rmsnorm_checkpoint_bytes = sum(
        region.page_count * BF16_TILE_BYTES
        for region in arena.slices
        if "rmsnorm_rstd" in region.name
    )
    private_checkpoint_bytes = arena.checkpoint_dram_bytes

    # Exposed edges removed from the current five-call graph:
    # attention output, both normalized streams, MLP output, two backward
    # dHidden handoffs (six hidden edges, each write+read), plus the read of
    # saved gate/up tiles by the separate SwiGLU/down call. These are logical
    # upper-bound bytes until device profiling reports physical DRAM traffic.
    visible_edges = 12 * hidden_bytes + gate_up_checkpoint_bytes
    traffic = ResidentTransformerLayerTraffic(
        hidden_tensor_bytes=hidden_bytes,
        gate_up_checkpoint_bytes=gate_up_checkpoint_bytes,
        attention_checkpoint_bytes=attention_checkpoint_bytes,
        attention_residual_checkpoint_bytes=(
            attention_residual_checkpoint_bytes
        ),
        layer_input_checkpoint_bytes=layer_input_checkpoint_bytes,
        rmsnorm_checkpoint_bytes=rmsnorm_checkpoint_bytes,
        private_checkpoint_dram_bytes=private_checkpoint_bytes,
        current_visible_intermediate_edge_bytes=visible_edges,
        target_visible_intermediate_edge_bytes=0,
        logical_boundary_bytes_eliminated=visible_edges,
    )

    forward_schedule = (
        ResidentLayerPhase(
            name="input_rmsnorm", worker_cores=input_norm.row_owners,
            reads=("external_layer_input", "input_norm_gamma"),
            writes=("layer_input", "input_rmsnorm_rstd_fp32"),
            l1_handoffs=("normalized_attention_input",), barrier_epoch=1,
        ),
        ResidentLayerPhase(
            name="attention_forward", worker_cores=110,
            reads=("normalized_attention_input", "attention_parameters"),
            writes=tuple(
                region.name for region in arena.slices
                if region.name.startswith("attention_")
            ),
            l1_handoffs=("attention_output",), barrier_epoch=2,
        ),
        ResidentLayerPhase(
            name="attention_residual", worker_cores=50,
            reads=("attention_output", "layer_input"),
            writes=("post_attention_residual",),
            l1_handoffs=("post_attention_hidden",), barrier_epoch=3,
        ),
        ResidentLayerPhase(
            name="post_attention_rmsnorm",
            worker_cores=post_attention_norm.row_owners,
            reads=("post_attention_hidden", "post_attention_norm_gamma"),
            writes=("post_attention_rmsnorm_rstd_fp32",),
            l1_handoffs=("normalized_mlp_input",), barrier_epoch=4,
        ),
        ResidentLayerPhase(
            name="grouped_gate_up", worker_cores=110,
            reads=("normalized_mlp_input", "gate_up_parameters"),
            writes=(
                "mlp_gate", "mlp_up", "mlp_gate_rank_hidden",
                "mlp_up_rank_hidden", "mlp_normalized_input",
            ),
            l1_handoffs=("gate_up_tiles",), barrier_epoch=5,
        ),
        ResidentLayerPhase(
            name="swiglu_down", worker_cores=110,
            reads=("gate_up_tiles", "down_parameters"),
            writes=(), l1_handoffs=("mlp_output",), barrier_epoch=6,
        ),
        ResidentLayerPhase(
            name="mlp_residual", worker_cores=100,
            reads=("mlp_output", "post_attention_residual"),
            writes=("layer_output",), l1_handoffs=(), barrier_epoch=7,
        ),
    )
    backward_schedule = (
        ResidentLayerPhase(
            name="mlp_residual_split", worker_cores=100,
            reads=("external_output_gradient",), writes=(),
            l1_handoffs=("mlp_dy", "residual_dy"), barrier_epoch=1,
        ),
        ResidentLayerPhase(
            name="full_mlp_backward", worker_cores=110,
            reads=(
                "mlp_dy", "mlp_gate", "mlp_up",
                "mlp_normalized_input", "mlp_parameters",
            ),
            writes=("mlp_adapter_gradients",),
            l1_handoffs=("normalized_mlp_dx",), barrier_epoch=2,
        ),
        ResidentLayerPhase(
            name="post_attention_rmsnorm_backward",
            worker_cores=post_attention_norm.row_owners,
            reads=(
                "normalized_mlp_dx", "post_attention_residual",
                "post_attention_rmsnorm_rstd_fp32",
            ),
            writes=(), l1_handoffs=("post_attention_norm_dx",),
            barrier_epoch=3,
        ),
        ResidentLayerPhase(
            name="attention_residual_accumulation", worker_cores=50,
            reads=("post_attention_norm_dx", "residual_dy"), writes=(),
            l1_handoffs=("attention_dy", "outer_residual_dy"),
            barrier_epoch=4,
        ),
        ResidentLayerPhase(
            name="full_attention_backward", worker_cores=110,
            reads=("attention_dy", "attention_checkpoints"),
            writes=("attention_adapter_gradients",),
            l1_handoffs=("normalized_attention_dx",), barrier_epoch=5,
        ),
        ResidentLayerPhase(
            name="input_rmsnorm_backward", worker_cores=input_norm.row_owners,
            reads=(
                "normalized_attention_dx", "layer_input",
                "input_rmsnorm_rstd_fp32",
            ),
            writes=(), l1_handoffs=("input_norm_dx",), barrier_epoch=6,
        ),
        ResidentLayerPhase(
            name="final_hidden_gradient", worker_cores=50,
            reads=("input_norm_dx", "outer_residual_dy"),
            writes=("input_gradient",), l1_handoffs=(), barrier_epoch=7,
        ),
    )

    return ResidentTransformerLayerPlan(
        tokens=tokens,
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        logical_rank=logical_rank,
        physical_rank=physical_rank,
        qkv_bias=bool(qkv_bias),
        rms_norm_epsilon=float(rms_norm_epsilon),
        current_custom_calls_per_layer=5,
        target_custom_calls_per_layer=2,
        current_custom_calls_for_depth=5 * model_depth,
        target_custom_calls_for_depth=2 * model_depth,
        model_depth=model_depth,
        l1_budget_bytes_per_core=budget,
        forward_peak_l1_bytes_per_core=forward_peak,
        backward_peak_l1_bytes_per_core=backward_peak,
        phase_handoff_l1_bytes_per_core=phase_handoff_l1,
        forward_phases=(
            "input_rmsnorm",
            "qkv_base_lora",
            "rope_causal_attention_output_projection",
            "attention_residual",
            "post_attention_rmsnorm",
            "gate_up_base_lora",
            "swiglu_down_base_lora",
            "mlp_residual",
        ),
        backward_phases=(
            "mlp_residual_split",
            "full_mlp_backward",
            "post_attention_rmsnorm_backward",
            "attention_residual_accumulation",
            "full_attention_backward",
            "input_rmsnorm_backward",
            "final_hidden_gradient_pack",
        ),
        final_forward_outputs=(
            "layer_hidden_state",
            "private_backward_checkpoint_arena",
        ),
        final_backward_outputs=(
            "input_gradient",
            "q_lora_a_gradient", "q_lora_b_gradient",
            "k_lora_a_gradient", "k_lora_b_gradient",
            "v_lora_a_gradient", "v_lora_b_gradient",
            "o_lora_a_gradient", "o_lora_b_gradient",
            "gate_lora_a_gradient", "gate_lora_b_gradient",
            "up_lora_a_gradient", "up_lora_b_gradient",
            "down_lora_a_gradient", "down_lora_b_gradient",
        ),
        attention_forward=attention_forward,
        attention_backward=attention_backward,
        mlp=mlp,
        input_norm=input_norm,
        post_attention_norm=post_attention_norm,
        arena=arena,
        forward_schedule=forward_schedule,
        backward_schedule=backward_schedule,
        traffic=traffic,
        physical_forward=physical_forward,
    )


def resident_transformer_layer_support_error(
    **kwargs: object,
) -> str | None:
    try:
        plan_resident_transformer_layer(**kwargs)  # type: ignore[arg-type]
    except ValueError as error:
        return str(error)
    return None
