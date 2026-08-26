"""Parametric native geometry and backend planning.

Qualification points are evidence, not implementation branches. This module
derives rank, batch and sequence tiling from geometry and resource limits. The
only fixed execution fingerprints are the frozen resident runtime golden
specializations; every other admitted shape takes the scalable exact route (or
an explicit resource rejection) until a resident implementation is selected.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


TILE_SIZE = 32
DEFAULT_SEQUENCE_BLOCK = 128


@dataclass(frozen=True)
class ModelGeometry:
    model_family: str
    layers: int
    hidden_features: int
    intermediate_features: int
    query_heads: int
    kv_heads: int
    head_dim: int
    vocabulary_size: int

    @property
    def kv_features(self) -> int:
        return self.kv_heads * self.head_dim

    def validate(self) -> None:
        values = {
            "layers": self.layers,
            "hidden_features": self.hidden_features,
            "intermediate_features": self.intermediate_features,
            "query_heads": self.query_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "vocabulary_size": self.vocabulary_size,
        }
        bad = {name: value for name, value in values.items() if value <= 0}
        if bad:
            raise ValueError(f"model geometry must be positive: {bad}")
        if self.query_heads % self.kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if self.hidden_features != self.query_heads * self.head_dim:
            raise ValueError("hidden size must equal query_heads * head_dim")
        for name in (
            "hidden_features", "intermediate_features", "kv_features",
            "head_dim",
        ):
            if getattr(self, name) % TILE_SIZE:
                raise ValueError(f"{name} must be divisible by {TILE_SIZE}")


@dataclass(frozen=True)
class GeometryHardwareLimits:
    architecture: str = "blackhole"
    worker_cores: int = 110
    l1_bytes_per_core: int = 1_500_000
    l1_usable_fraction: float = 0.75
    dram_bytes: int = 24 * 1024**3
    sequence_block_tokens: int = DEFAULT_SEQUENCE_BLOCK
    native_sdpa_backward_available: bool = True

    @property
    def usable_l1_bytes_per_core(self) -> int:
        return int(self.l1_bytes_per_core * self.l1_usable_fraction)

    def validate(self) -> None:
        if self.architecture != "blackhole":
            raise ValueError("native currently targets Blackhole")
        if self.worker_cores <= 0 or self.l1_bytes_per_core <= 0:
            raise ValueError("hardware core/L1 limits must be positive")
        if not 0.0 < self.l1_usable_fraction < 1.0:
            raise ValueError("L1 usable fraction must be within (0, 1)")
        if self.dram_bytes <= 0:
            raise ValueError("DRAM capacity must be positive")
        if (
            self.sequence_block_tokens <= 0
            or self.sequence_block_tokens % TILE_SIZE
        ):
            raise ValueError("sequence block must be a positive tile multiple")


@dataclass(frozen=True)
class GeometryPlan:
    model: ModelGeometry
    logical_rank: int
    rank_tiles: int
    physical_rank: int
    rank_tail: int
    batch_size: int
    sequence_length: int
    logical_tokens: int
    physical_tokens: int
    token_tiles: int
    sequence_block_tokens: int
    sequence_blocks: int
    sequence_tail: int
    batch_layout: str
    attention_strategy: str
    attention_backward_backend: str
    attention_numerical_contract: str
    attention_backend_selection: str
    attention_full_sxs_dram: bool
    checkpoint_strategy: str
    loss_strategy: str
    loss_token_block: int
    estimated_dense_logits_f32_bytes: int
    optimizer_gradient_prefix_pages: int
    optimizer_gradient_pages: int
    optimizer_state_rows: int
    estimated_mutable_bytes: int
    l1_rank_working_set_bytes_per_core: int
    l1_state_policy: str
    backend: str
    qualification: str
    admitted: bool
    safe_fallback: bool
    admission_reason: str


@dataclass(frozen=True)
class RankPlan:
    """Compatibility view retained for existing rank-only callers."""

    model_family: str
    logical_rank: int
    physical_rank: int
    backend: str
    qualification: str
    safe_fallback: bool


_MODEL_GEOMETRIES = {
    "smollm2": ModelGeometry(
        model_family="smollm2",
        layers=30,
        hidden_features=576,
        intermediate_features=1536,
        query_heads=9,
        kv_heads=3,
        head_dim=64,
        vocabulary_size=49152,
    ),
    "qwen": ModelGeometry(
        model_family="qwen",
        layers=24,
        hidden_features=896,
        intermediate_features=4864,
        query_heads=14,
        kv_heads=2,
        head_dim=64,
        vocabulary_size=151936,
    ),
}


def model_geometry(model_family: str) -> ModelGeometry:
    normalized = model_family.lower()
    try:
        geometry = _MODEL_GEOMETRIES[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported model family: {model_family}") from exc
    geometry.validate()
    return geometry


def _ceil_tile(value: int) -> int:
    return math.ceil(value / TILE_SIZE) * TILE_SIZE


def _layer_gradient_pages(
    geometry: ModelGeometry, rank_tiles: int,
) -> int:
    hidden_tiles = geometry.hidden_features // TILE_SIZE
    intermediate_tiles = geometry.intermediate_features // TILE_SIZE
    kv_tiles = geometry.kv_features // TILE_SIZE
    # The canonical 14-matrix order in optimizer_gradient_arena_planner.
    return rank_tiles * (
        9 * hidden_tiles
        + 3 * intermediate_tiles
        + 2 * kv_tiles
    )


def plan_geometry(
    model: str | ModelGeometry,
    *,
    rank: int,
    batch_size: int,
    sequence_length: int,
    dtype: str = "bfloat16",
    hardware: GeometryHardwareLimits | None = None,
) -> GeometryPlan:
    """Derive one R×B×S plan without anchor-specific implementation cases."""

    geometry = model_geometry(model) if isinstance(model, str) else model
    geometry.validate()
    hardware = hardware or GeometryHardwareLimits()
    hardware.validate()
    if rank <= 0:
        raise ValueError("LoRA rank must be positive")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if sequence_length <= 0:
        raise ValueError("sequence length must be positive")
    if dtype != "bfloat16":
        raise ValueError("the exact native contract currently requires bfloat16")

    rank_tiles = math.ceil(rank / TILE_SIZE)
    physical_rank = rank_tiles * TILE_SIZE
    rank_tail = rank - (rank_tiles - 1) * TILE_SIZE
    logical_tokens = batch_size * sequence_length
    physical_tokens = _ceil_tile(logical_tokens)
    token_tiles = physical_tokens // TILE_SIZE
    block = hardware.sequence_block_tokens
    sequence_blocks = math.ceil(sequence_length / block)
    sequence_tail = sequence_length - (sequence_blocks - 1) * block

    hidden_tiles = geometry.hidden_features // TILE_SIZE
    scratch_pages = token_tiles * hidden_tiles
    layer_gradient_pages = _layer_gradient_pages(geometry, rank_tiles)
    gradient_pages = geometry.layers * layer_gradient_pages
    state_rows = gradient_pages * TILE_SIZE
    # Parameter, gradient and the two exact moment regions are BF16-backed;
    # this conservative estimate also reserves two full temporary regions.
    estimated_mutable_bytes = (
        (6 * state_rows + (scratch_pages * TILE_SIZE))
        * TILE_SIZE
        * 2
    )

    # Conservative per-core hot set for one rank block and two CB epochs.
    rank_block_tiles = min(rank_tiles, 4)
    l1_rank_working_set = (
        rank_block_tiles * (hidden_tiles + geometry.kv_features // TILE_SIZE)
        * TILE_SIZE * TILE_SIZE * 2 * 4
    )
    l1_policy = (
        "hot_rank_block_l1"
        if l1_rank_working_set <= hardware.usable_l1_bytes_per_core
        else "warm_layer_streamed_dram"
    )

    admitted = estimated_mutable_bytes <= hardware.dram_bytes
    attention_strategy = (
        "dense_causal"
        if sequence_length <= block
        else "blocked_streamed_causal"
    )
    if attention_strategy == "dense_causal":
        attention_backward_backend = "framework_exact"
        attention_contract = "dense_exact"
        attention_selection = "dense sequence fits one planner block"
    elif (
        hardware.native_sdpa_backward_available
        and sequence_tail == block
    ):
        # The native TT-Train primitive admits complete 128-token epochs.  A
        # partial terminal epoch remains on the deterministic safe recompute
        # path until a masked native tail is available.  This is a derived
        # backend capability, not a qualification-length allowlist.
        attention_backward_backend = "native_streamed_exact"
        attention_contract = "native_streamed_v1"
        attention_selection = (
            "native causal SDPA backward available and sequence has no "
            "partial terminal block"
        )
    else:
        attention_backward_backend = "safe_streamed_exact"
        attention_contract = "safe_streamed_v1"
        attention_selection = (
            "safe deterministic recompute selected for a partial sequence "
            "block or unavailable native runtime capability"
        )

    if not admitted:
        backend = "rejected_resource_limit"
        fallback = False
        reason = (
            f"estimated mutable state {estimated_mutable_bytes} exceeds "
            f"DRAM budget {hardware.dram_bytes}"
        )
    elif (
        geometry.model_family == "smollm2"
        and rank == 8 and batch_size == 1 and sequence_length == 128
    ) or (
        geometry.model_family == "qwen"
        and rank == 16 and batch_size == 1 and sequence_length == 128
    ):
        backend = "resident_exact"
        fallback = False
        reason = "frozen resident runtime golden fingerprint"
    else:
        backend = (
            "native_streamed_attention_device_packed_update"
            if attention_backward_backend == "native_streamed_exact"
            else "generic_tt_autograd_device_packed_update_fallback"
        )
        fallback = True
        reason = (
            "admitted native streamed attention with safe exact fallback"
            if attention_backward_backend == "native_streamed_exact"
            else "admitted exact scalable fallback"
        )
    checkpoint_strategy = (
        "resident_layer_checkpoints"
        if sequence_blocks == 1 and batch_size == 1
        else "layerwise_recompute_streamed"
    )
    dense_logits_f32_bytes = (
        logical_tokens * geometry.vocabulary_size * 4
    )
    # Keep the dense compatibility edge only while its single FP32 logits
    # allocation is comfortably below one GiB.  Larger vocab/token products
    # are processed in ascending 128-token blocks so no runtime allocation is
    # proportional to T*V.  The threshold is a resource policy, not a model or
    # qualification-shape fingerprint.
    loss_strategy = (
        "streamed_vocab_causal_loss"
        if dense_logits_f32_bytes > 1024**3
        else "dense_causal_loss"
    )
    qualification = (
        "golden_20_step_exact"
        if backend == "resident_exact"
        else "requires_evolving_qualification"
        if admitted
        else "not_admitted"
    )
    return GeometryPlan(
        model=geometry,
        logical_rank=rank,
        rank_tiles=rank_tiles,
        physical_rank=physical_rank,
        rank_tail=rank_tail,
        batch_size=batch_size,
        sequence_length=sequence_length,
        logical_tokens=logical_tokens,
        physical_tokens=physical_tokens,
        token_tiles=token_tiles,
        sequence_block_tokens=block,
        sequence_blocks=sequence_blocks,
        sequence_tail=sequence_tail,
        batch_layout="device_batch_sequence_hidden",
        attention_strategy=attention_strategy,
        attention_backward_backend=attention_backward_backend,
        attention_numerical_contract=attention_contract,
        attention_backend_selection=attention_selection,
        attention_full_sxs_dram=False,
        checkpoint_strategy=checkpoint_strategy,
        loss_strategy=loss_strategy,
        loss_token_block=block,
        estimated_dense_logits_f32_bytes=dense_logits_f32_bytes,
        optimizer_gradient_prefix_pages=scratch_pages,
        optimizer_gradient_pages=gradient_pages,
        optimizer_state_rows=state_rows,
        estimated_mutable_bytes=estimated_mutable_bytes,
        l1_rank_working_set_bytes_per_core=l1_rank_working_set,
        l1_state_policy=l1_policy,
        backend=backend,
        qualification=qualification,
        admitted=admitted,
        safe_fallback=fallback,
        admission_reason=reason,
    )


def plan_rank_backend(model_family: str, rank: int) -> RankPlan:
    """Compatibility wrapper over the parametric R×B×S planner."""

    plan = plan_geometry(
        model_family,
        rank=rank,
        batch_size=1,
        sequence_length=128,
    )
    return RankPlan(
        model_family=plan.model.model_family,
        logical_rank=plan.logical_rank,
        physical_rank=plan.physical_rank,
        backend=plan.backend,
        qualification=plan.qualification,
        safe_fallback=plan.safe_fallback,
    )
