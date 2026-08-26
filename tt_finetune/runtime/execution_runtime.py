"""Automatic execution bridge from a architecture plan to the frozen resident ABI.

The resident owner is deliberately not disguised as ordinary Hugging Face
autograd: it owns packed parameters, checkpoints, gradients, clipping and
AdamW state.  This bridge makes that ABI selectable from the canonical architecture
plan without introducing a model-name switch or changing frozen resident runtime code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .causal_lm import CausalLMDeviceEdges, CausalLMHostEdges
from .resident_engine import AdamWScalars, ResidentStackEngine, ResidentStepResult
from .packed_state import PackedTransformerStack, pack_transformer_stack
from .execution_planner import (
    ExecutionPlan,
    ResidentQualification,
    resident_qualification_for_plan,
)


@dataclass(frozen=True)
class ResidentBuildReport:
    semantic_fingerprint: str
    contract: str
    source_bundle: str
    descriptor_bundle: str
    public_custom_calls_per_step: int
    physical_programs_per_step: int
    batch_size: int
    sequence_length: int
    rank: int
    host_batch_loop: bool = False
    host_gradient_materialization: bool = False

    def to_dict(self) -> dict[str, Any]:
        return vars(self)


class ResidentCausalLMExecutor:
    """Narrow tokens-to-update executor for an admitted resident contract."""

    def __init__(
        self,
        *,
        plan: ExecutionPlan,
        qualification: ResidentQualification,
        packed: PackedTransformerStack,
        edges: CausalLMDeviceEdges,
        engine: ResidentStackEngine,
    ) -> None:
        self.plan = plan
        self.qualification = qualification
        self.packed = packed
        self.edges = edges
        self.engine = engine
        self.report = ResidentBuildReport(
            semantic_fingerprint=plan.architecture.semantic_fingerprint,
            contract=qualification.contract,
            source_bundle=qualification.source_bundle,
            descriptor_bundle=qualification.descriptor_bundle,
            public_custom_calls_per_step=engine.public_custom_calls_per_step,
            physical_programs_per_step=engine.physical_programs_per_step,
            batch_size=plan.batch_size,
            sequence_length=plan.sequence_length,
            rank=plan.rank,
        )

    def step_tokens(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        scalars: AdamWScalars,
    ) -> ResidentStepResult:
        return self.engine.step_tokens(input_ids, labels, self.edges, scalars)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _rotary_owner(model: nn.Module, decoder_path: str) -> nn.Module:
    decoder_parent = decoder_path.rsplit(".", 1)[0]
    decoder = model.get_submodule(decoder_parent)
    direct = getattr(decoder, "rotary_emb", None)
    if isinstance(direct, nn.Module):
        return direct
    candidates = [
        module
        for name, module in decoder.named_modules()
        if name and name.rsplit(".", 1)[-1] == "rotary_emb"
    ]
    identities = {id(module): module for module in candidates}
    if len(identities) != 1:
        raise ValueError(
            "resident runtime requires one discoverable shared rotary embedding"
        )
    return next(iter(identities.values()))


def build_resident_executor(
    model: nn.Module,
    *,
    plan: ExecutionPlan,
    decoder_path: str,
    device: torch.device,
    project_root: str | Path | None = None,
) -> ResidentCausalLMExecutor:
    """Build the fastest qualified full-stack executor selected by ``plan``.

    The input model must still be host-resident because packing consumes its
    frozen BF16 weights.  Only exact structural/R/B/S matches are admitted.
    Unsupported plans fail before submitting a device program, leaving the
    caller free to use the safe autograd implementation.
    """

    plan.require_admitted()
    if plan.selected_stack_backend != "resident_exact":
        raise ValueError(
            "the architecture plan did not select a qualified resident stack"
        )
    qualification = resident_qualification_for_plan(plan)
    if qualification is None:
        raise RuntimeError("resident plan has no frozen qualification bundle")
    root = Path(project_root) if project_root is not None else _project_root()
    source = root / qualification.source_bundle
    descriptor = root / qualification.descriptor_bundle
    for kind, path in (("source", source), ("descriptor", descriptor)):
        if not (path / "manifest.json").is_file():
            raise FileNotFoundError(
                f"qualified resident {kind} bundle is missing: {path}"
            )

    if next(model.parameters()).device.type != "cpu":
        raise ValueError("resident executor must pack the model before device transfer")
    model.to(dtype=torch.bfloat16)
    rotary = _rotary_owner(model, decoder_path)
    positions_h = torch.arange(plan.sequence_length).reshape(1, -1)
    dummy_h = torch.zeros(
        (plan.batch_size, plan.sequence_length, plan.architecture.hidden_size),
        dtype=torch.bfloat16,
    )
    cosine_xla, sine_xla = rotary(dummy_h.to(device), positions_h.to(device))
    import torch_xla

    torch_xla.sync(wait=True)
    cosine = cosine_xla.detach().cpu()
    sine = sine_xla.detach().cpu()
    attention_mask = torch.triu(
        torch.full(
            (plan.sequence_length, plan.sequence_length),
            torch.finfo(torch.bfloat16).min,
            dtype=torch.bfloat16,
        ),
        diagonal=1,
    )
    packed = pack_transformer_stack(
        model,
        cosine=cosine,
        sine=sine,
        attention_mask=attention_mask,
        batch_size=plan.batch_size,
        sequence_length=plan.sequence_length,
        physical_rank=plan.physical_rank,
    )
    edges = CausalLMHostEdges.from_model(model).to(device)
    engine = ResidentStackEngine(
        packed,
        source_bundle=source,
        descriptor_bundle=descriptor,
        device=device,
        compact_forward=True,
        canonical_exact_attention=True,
    )
    return ResidentCausalLMExecutor(
        plan=plan,
        qualification=qualification,
        packed=packed,
        edges=edges,
        engine=engine,
    )


__all__ = [
    "ResidentBuildReport",
    "ResidentCausalLMExecutor",
    "build_resident_executor",
]
