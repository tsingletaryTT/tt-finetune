"""Canonical packed state for the experimental resident runtime Transformer runtime.

The resident multi-layer TT-Metal schedule does not consume a Python list of
PEFT Parameters.  It consumes two dense tile-page tensors:

* immutable/frozen layer data, addressed through per-layer page offsets; and
* mutable LoRA state, ordered exactly like the optimizer gradient arena.

This module is the model-facing ABI between those representations.  Keeping
the map here (rather than in a benchmark) is important: the packed tensor must
remain the canonical state updated by the resident AdamW owner, while exports,
fallbacks and numerical audits still need stable logical parameter names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from tt_finetune.kernels.optimizer_gradient_arena_planner import (
    MultiLayerGradientArenaPlan,
    plan_multi_layer_optimizer_gradient_arena,
)
from tt_finetune.kernels.tt_lang_lora import pad_lora_weights, padded_rank


_PROJECTIONS = ("q", "k", "v", "o", "gate", "up", "down")
_PACKED_STATE_ORDER = (
    "o_a", "o_b", "q_a", "k_a", "v_a", "q_b", "k_b", "v_b",
    "gate_a", "gate_b", "up_a", "up_b", "down_a", "down_b",
)


@dataclass(frozen=True)
class ResidentModelShape:
    layers: int
    batch_size: int
    sequence_length: int
    hidden_features: int
    intermediate_features: int
    kv_features: int
    query_heads: int
    kv_heads: int
    head_dim: int
    logical_rank: int
    physical_rank: int
    lora_scale: float
    attention_scale: float
    rmsnorm_epsilon: float
    has_bias: bool


@dataclass(frozen=True)
class PackedStateRegion:
    """One logical LoRA matrix inside the dense packed state."""

    layer: int
    short_name: str
    parameter_name: str
    logical_shape: tuple[int, int]
    physical_shape: tuple[int, int]
    page_offset: int
    page_count: int


@dataclass(frozen=True)
class PackedTransformerStack:
    """Host-side materialization of the resident stack ABI."""

    shape: ResidentModelShape
    frozen_pages: torch.Tensor
    trainable_pages: torch.Tensor
    block_page_offsets: tuple[tuple[int, ...], ...]
    block_template: tuple[torch.Tensor, ...]
    gradient_plan: MultiLayerGradientArenaPlan
    regions: tuple[PackedStateRegion, ...]

    def region_by_parameter(self) -> Mapping[str, PackedStateRegion]:
        return {region.parameter_name: region for region in self.regions}

    def logical_state(self, packed: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Return logical, unpadded matrix views keyed by model parameter name."""

        source = self.trainable_pages if packed is None else packed
        pages = source.reshape(-1, 32, 32)
        result: dict[str, torch.Tensor] = {}
        for region in self.regions:
            rows, columns = region.physical_shape
            row_tiles, column_tiles = rows // 32, columns // 32
            matrix = pages[
                region.page_offset : region.page_offset + region.page_count
            ].reshape(row_tiles, column_tiles, 32, 32).permute(
                0, 2, 1, 3
            ).reshape(rows, columns)
            logical_rows, logical_columns = region.logical_shape
            result[region.parameter_name] = matrix[
                :logical_rows, :logical_columns
            ]
        return result

    def logical_gradients(self, arena: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode the packed backward arena using the canonical state names."""

        if tuple(arena.shape) != self.gradient_plan.physical_shape:
            raise ValueError(
                "gradient arena has the wrong physical shape: "
                f"{tuple(arena.shape)} != {self.gradient_plan.physical_shape}"
            )
        pages = arena.reshape(-1, 32, 32)
        result: dict[str, torch.Tensor] = {}
        for region in self.regions:
            gradient_slice = self.gradient_plan.slice(
                region.layer, region.short_name
            )
            rows, columns = region.physical_shape
            row_tiles, column_tiles = rows // 32, columns // 32
            matrix = pages[
                gradient_slice.page_offset : gradient_slice.page_stop
            ].reshape(row_tiles, column_tiles, 32, 32).permute(
                0, 2, 1, 3
            ).reshape(rows, columns)
            logical_rows, logical_columns = region.logical_shape
            result[region.parameter_name] = matrix[
                :logical_rows, :logical_columns
            ]
        return result

    def pack_logical_gradients(
        self, values: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Pack logical gradients on their current device into the arena ABI."""

        expected = {region.parameter_name for region in self.regions}
        if values.keys() != expected:
            missing = sorted(expected - values.keys())
            extra = sorted(values.keys() - expected)
            raise ValueError(
                f"gradient names differ; missing={missing}, extra={extra}"
            )
        first = next(iter(values.values()))
        chunks = [
            first.new_zeros((self.gradient_plan.scratch_pages * 32, 32))
        ]
        page_cursor = self.gradient_plan.scratch_pages
        for region in self.regions:
            logical = values[region.parameter_name]
            if tuple(logical.shape) != region.logical_shape:
                raise ValueError(
                    f"{region.parameter_name} gradient shape changed: "
                    f"{tuple(logical.shape)} != {region.logical_shape}"
                )
            physical_rows, physical_columns = region.physical_shape
            padded = F.pad(
                logical,
                (
                    0, physical_columns - logical.shape[1],
                    0, physical_rows - logical.shape[0],
                ),
            )
            row_tiles = physical_rows // 32
            column_tiles = physical_columns // 32
            tiled = (
                padded.reshape(row_tiles, 32, column_tiles, 32)
                .permute(0, 2, 1, 3)
                .reshape(-1, 32)
            )
            target = self.gradient_plan.slice(
                region.layer, region.short_name
            )
            if target.page_offset != page_cursor:
                raise RuntimeError(
                    "gradient arena is not a dense canonical stream"
                )
            if tiled.shape[0] != target.page_count * 32:
                raise RuntimeError("packed gradient page count changed")
            chunks.append(tiled)
            page_cursor = target.page_stop
        arena = torch.cat(chunks, dim=0)
        if tuple(arena.shape) != self.gradient_plan.physical_shape:
            raise RuntimeError("packed gradient arena shape changed")
        return arena


@dataclass(frozen=True)
class _Projection:
    weight: torch.Tensor
    weight_t: torch.Tensor
    bias: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    logical_a_shape: tuple[int, int]
    logical_b_shape: tuple[int, int]
    a_name: str
    b_name: str
    scale: float


def _active_adapter(projection: nn.Module) -> str:
    active = getattr(projection, "active_adapter", "default")
    if isinstance(active, str):
        return active
    values = tuple(active)
    if len(values) != 1:
        raise ValueError("resident runtime requires exactly one active LoRA adapter")
    return values[0]


def _projection(
    projection: nn.Module, *, path: str, dtype: torch.dtype,
) -> _Projection:
    adapter = _active_adapter(projection)
    try:
        weight = projection.base_layer.weight.detach()
        a_parameter = projection.lora_A[adapter].weight
        b_parameter = projection.lora_B[adapter].weight
        scale = float(projection.scaling[adapter])
        dropout = projection.lora_dropout[adapter]
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(f"{path} is not an ordinary PEFT LoRA linear") from exc
    if float(getattr(dropout, "p", 0.0)) != 0.0:
        raise ValueError(f"{path} has nonzero LoRA dropout")
    if bool(getattr(projection, "use_dora", {}).get(adapter, False)):
        raise ValueError(f"{path} uses unsupported DoRA")
    if weight.dtype != dtype or a_parameter.dtype != dtype or b_parameter.dtype != dtype:
        raise ValueError(f"{path} must use {dtype} weights and adapters")
    a, b = pad_lora_weights(a_parameter.detach(), b_parameter.detach())
    bias = getattr(projection.base_layer, "bias", None)
    if bias is None:
        bias = torch.zeros(
            projection.out_features, dtype=dtype, device=weight.device
        )
    elif bias.dtype != dtype:
        raise ValueError(f"{path} bias must use {dtype}")
    return _Projection(
        weight=weight,
        weight_t=weight.T.contiguous(),
        bias=bias.detach().reshape(1, -1).expand(32, -1).contiguous(),
        a=a,
        b=b,
        logical_a_shape=tuple(a_parameter.shape),
        logical_b_shape=tuple(b_parameter.shape),
        a_name=f"{path}.lora_A.{adapter}.weight",
        b_name=f"{path}.lora_B.{adapter}.weight",
        scale=scale,
    )


def _packed_tile_pages(*tensors: torch.Tensor) -> torch.Tensor:
    pages = []
    for tensor in tensors:
        if tensor.ndim != 2 or tensor.shape[0] % 32 or tensor.shape[1] % 32:
            raise ValueError(
                f"resident runtime tensor is not tile aligned: {tuple(tensor.shape)}"
            )
        row_tiles, column_tiles = tensor.shape[0] // 32, tensor.shape[1] // 32
        pages.append(
            tensor.reshape(row_tiles, 32, column_tiles, 32)
            .permute(0, 2, 1, 3)
            .reshape(row_tiles * column_tiles, 32, 32)
        )
    return torch.cat(pages).reshape(-1, 32).contiguous()


def _decoder_layers(model: nn.Module) -> tuple[tuple[str, nn.Module], ...]:
    candidates = (
        "base_model.model.model.layers",
        "model.model.layers",
        "model.layers",
        "layers",
    )
    for prefix in candidates:
        value: object = model
        try:
            for component in prefix.split("."):
                value = getattr(value, component)
        except AttributeError:
            continue
        if isinstance(value, (nn.ModuleList, list, tuple)) and value:
            return tuple(
                (f"{prefix}.{index}", layer)
                for index, layer in enumerate(value)
            )
    raise ValueError("could not locate decoder layers on the model")


def _matrix(value: torch.Tensor, *, rows: int, columns: int, name: str) -> torch.Tensor:
    value = value.detach()
    if value.numel() != rows * columns:
        raise ValueError(
            f"{name} has {value.numel()} values, expected {rows * columns}"
        )
    return value.reshape(rows, columns).contiguous()


def pack_transformer_stack(
    model: nn.Module,
    *,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    attention_mask: torch.Tensor | None,
    batch_size: int = 1,
    sequence_length: int = 128,
    physical_rank: int | None = None,
    include_frozen_state: bool = True,
) -> PackedTransformerStack:
    """Pack a homogeneous PEFT decoder stack for the resident runtime.

    The function performs no device transfer and registers no duplicate model
    state.  Callers decide whether the returned host tensors become persistent
    device buffers or are used only for an exactness oracle.
    """

    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if sequence_length <= 0 or sequence_length % 32:
        raise ValueError("sequence length must be a positive tile multiple")
    layers = _decoder_layers(model)
    dtype = torch.bfloat16
    first_path, first = layers[0]
    if not all(
        hasattr(first, name)
        for name in ("self_attn", "mlp", "input_layernorm", "post_attention_layernorm")
    ):
        raise ValueError(f"{first_path} is not a supported decoder layer")
    hidden = int(first.self_attn.q_proj.in_features)
    intermediate = int(first.mlp.gate_proj.out_features)
    head_dim = int(first.self_attn.head_dim)
    query_heads = int(first.self_attn.q_proj.out_features) // head_dim
    kv_features = int(first.self_attn.k_proj.out_features)
    kv_heads = kv_features // head_dim
    if any(value <= 0 or value % 32 for value in (hidden, intermediate, kv_features, head_dim)):
        raise ValueError("resident runtime model dimensions must be positive tile multiples")

    cosine_matrix = _matrix(
        cosine, rows=sequence_length, columns=head_dim, name="cosine"
    ).to(dtype=dtype)
    sine_matrix = _matrix(
        sine, rows=sequence_length, columns=head_dim, name="sine"
    ).to(dtype=dtype)
    if include_frozen_state:
        if attention_mask is None:
            raise ValueError("resident frozen state requires an attention mask")
        mask_matrix = _matrix(
            attention_mask,
            rows=sequence_length,
            columns=sequence_length,
            name="attention_mask",
        ).to(dtype=dtype)
    else:
        # The scalable update-only path owns no model forward state.  Avoid
        # constructing or replicating an SxS causal mask merely to discover
        # and pack LoRA parameters for clipping/AdamW.
        mask_matrix = None

    blocks: list[tuple[torch.Tensor, ...]] = []
    projection_maps: list[dict[str, _Projection]] = []
    epsilons: set[float] = set()
    scales: set[float] = set()
    bias_contracts: set[bool] = set()
    for layer_index, (layer_path, layer) in enumerate(layers):
        attention, mlp = layer.self_attn, layer.mlp
        projection_modules = {
            "q": attention.q_proj,
            "k": attention.k_proj,
            "v": attention.v_proj,
            "o": attention.o_proj,
            "gate": mlp.gate_proj,
            "up": mlp.up_proj,
            "down": mlp.down_proj,
        }
        projections = {
            name: _projection(
                module, path=f"{layer_path}.{'self_attn' if name in ('q', 'k', 'v', 'o') else 'mlp'}.{name}_proj",
                dtype=dtype,
            )
            for name, module in projection_modules.items()
        }
        if (
            int(attention.q_proj.in_features) != hidden
            or int(mlp.gate_proj.out_features) != intermediate
            or int(attention.k_proj.out_features) != kv_features
            or int(attention.head_dim) != head_dim
        ):
            raise ValueError(f"{layer_path} changes the homogeneous resident runtime shape")
        logical_ranks = {
            projection.logical_a_shape[0] for projection in projections.values()
        }
        if len(logical_ranks) != 1:
            raise ValueError(f"{layer_path} requires one uniform logical rank")
        layer_rank = next(iter(logical_ranks))
        if physical_rank is None:
            physical_rank = padded_rank(layer_rank)
        if layer_rank <= 0 or layer_rank > physical_rank:
            raise ValueError(
                f"{layer_path} logical rank {layer_rank} does not fit "
                f"physical rank {physical_rank}"
            )
        layer_scales = {projection.scale for projection in projections.values()}
        if len(layer_scales) != 1:
            raise ValueError(f"{layer_path} has nonuniform LoRA scales")
        scales.update(layer_scales)
        qkv_bias = tuple(
            getattr(projection_modules[name].base_layer, "bias", None) is not None
            for name in ("q", "k", "v")
        )
        if len(set(qkv_bias)) != 1:
            raise ValueError(f"{layer_path} has mixed Q/K/V bias presence")
        if getattr(projection_modules["o"].base_layer, "bias", None) is not None:
            raise ValueError(f"{layer_path} has unsupported O bias")
        bias_contracts.add(qkv_bias[0])
        input_gamma = layer.input_layernorm.weight.detach().reshape(
            1, hidden
        ).expand(32, hidden).contiguous()
        post_gamma = layer.post_attention_layernorm.weight.detach().reshape(
            1, hidden
        ).expand(32, hidden).contiguous()
        if input_gamma.dtype != dtype or post_gamma.dtype != dtype:
            raise ValueError(f"{layer_path} RMSNorm weights must use {dtype}")
        grouped_weight_t = torch.cat(
            (projections["gate"].weight, projections["up"].weight), dim=0
        ).T.contiguous()
        packed_post = None
        if include_frozen_state:
            assert mask_matrix is not None
            packed_post = _packed_tile_pages(
                post_gamma,
                mask_matrix,
                grouped_weight_t,
                projections["gate"].a,
                projections["gate"].b,
                projections["up"].a,
                projections["up"].b,
                projections["down"].weight_t,
                projections["down"].a,
                projections["down"].b,
            )
        q, k, v, o = (projections[name] for name in ("q", "k", "v", "o"))
        gate, up, down = (projections[name] for name in ("gate", "up", "down"))
        if include_frozen_state:
            assert mask_matrix is not None and packed_post is not None
            block = (
                q.weight_t, q.bias, q.a, q.b,
                k.weight_t, k.bias, k.a, k.b,
                v.weight_t, v.bias, v.a, v.b,
                o.weight_t, o.a, o.b,
                cosine_matrix, sine_matrix, mask_matrix, input_gamma, packed_post,
                gate.a, gate.b, up.a, up.b, down.a, down.b,
                post_gamma,
                o.weight, q.weight, k.weight, v.weight,
                down.weight, gate.weight, up.weight,
            )
            if len(block) != 34:
                raise AssertionError("resident runtime static block ABI changed")
            blocks.append(block)
        projection_maps.append(projections)
        epsilons.add(float(layer.input_layernorm.variance_epsilon))
        epsilons.add(float(layer.post_attention_layernorm.variance_epsilon))

    if len(scales) != 1 or len(epsilons) != 1 or len(bias_contracts) != 1:
        raise ValueError("resident runtime requires uniform scale, RMS epsilon and bias ABI")
    logical_rank = projection_maps[0]["q"].logical_a_shape[0]
    assert physical_rank is not None
    shape = ResidentModelShape(
        layers=len(layers),
        batch_size=batch_size,
        sequence_length=sequence_length,
        hidden_features=hidden,
        intermediate_features=intermediate,
        kv_features=kv_features,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        logical_rank=logical_rank,
        physical_rank=physical_rank,
        lora_scale=next(iter(scales)),
        attention_scale=float(first.self_attn.scaling),
        rmsnorm_epsilon=next(iter(epsilons)),
        has_bias=next(iter(bias_contracts)),
    )
    plan = plan_multi_layer_optimizer_gradient_arena(
        layer_count=len(layers),
        tokens=batch_size * sequence_length,
        hidden_features=hidden,
        intermediate_features=intermediate,
        kv_features=kv_features,
        physical_rank=physical_rank,
    )

    frozen_chunks: list[torch.Tensor] = []
    block_offsets: list[tuple[int, ...]] = []
    next_frozen_page = 0
    for block in blocks:
        offsets = []
        for tensor in block:
            packed = _packed_tile_pages(tensor)
            offsets.append(next_frozen_page)
            frozen_chunks.append(packed)
            next_frozen_page += packed.shape[0] // 32
        block_offsets.append(tuple(offsets))

    state_chunks: list[torch.Tensor] = []
    regions: list[PackedStateRegion] = []
    next_state_page = 0
    for layer_index, projections in enumerate(projection_maps):
        values = {
            f"{name}_a": projection.a
            for name, projection in projections.items()
        }
        values.update({
            f"{name}_b": projection.b
            for name, projection in projections.items()
        })
        for short_name in _PACKED_STATE_ORDER:
            projection_name, side = short_name.rsplit("_", 1)
            projection = projections[projection_name]
            value = values[short_name]
            packed = _packed_tile_pages(value)
            page_count = packed.shape[0] // 32
            regions.append(PackedStateRegion(
                layer=layer_index,
                short_name=short_name,
                parameter_name=(
                    projection.a_name if side == "a" else projection.b_name
                ),
                logical_shape=(
                    projection.logical_a_shape
                    if side == "a" else projection.logical_b_shape
                ),
                physical_shape=tuple(value.shape),
                page_offset=next_state_page,
                page_count=page_count,
            ))
            state_chunks.append(packed)
            next_state_page += page_count

    trainable_pages = torch.cat(state_chunks)
    expected_rows = len(layers) * plan.layer_gradient_pages * 32
    if tuple(trainable_pages.shape) != (expected_rows, 32):
        raise RuntimeError(
            "packed trainable state does not match the optimizer arena: "
            f"{tuple(trainable_pages.shape)} != {(expected_rows, 32)}"
        )
    return PackedTransformerStack(
        shape=shape,
        frozen_pages=(
            torch.cat(frozen_chunks)
            if frozen_chunks else torch.empty((0, 32), dtype=dtype)
        ),
        trainable_pages=trainable_pages,
        block_page_offsets=tuple(block_offsets),
        block_template=blocks[0] if blocks else (),
        gradient_plan=plan,
        regions=tuple(regions),
    )


__all__ = [
    "ResidentModelShape",
    "PackedTransformerStack",
    "PackedStateRegion",
    "pack_transformer_stack",
]
