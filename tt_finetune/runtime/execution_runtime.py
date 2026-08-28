"""Automatic execution bridge from a architecture plan to the frozen resident ABI.

The resident owner is deliberately not disguised as ordinary Hugging Face
autograd: it owns packed parameters, checkpoints, gradients, clipping and
AdamW state.  This bridge makes that ABI selectable from the canonical architecture
plan without introducing a model-name switch or changing frozen resident runtime code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
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
    source_bundle_sha256: str
    descriptor_bundle_sha256: str
    runtime_source_hash: str
    physical_plan_fingerprint: str
    public_custom_calls_per_step: int
    physical_programs_per_step: int
    batch_size: int
    sequence_length: int
    rank: int
    packing_audit: dict[str, Any]
    host_batch_loop: bool = False
    host_gradient_materialization: bool = False

    def to_dict(self) -> dict[str, Any]:
        return vars(self)


@dataclass(frozen=True)
class PublicResidentAdmission:
    """Product-level admission for the existing frozen resident ABI.

    The low-level planner only knows architecture and tensor geometry.  The
    public trainer also has to preserve optimizer, adapter, dropout and
    batching semantics.  Keeping that second gate explicit prevents the
    planner from advertising a resident plan that the worker later replaces
    with ordinary autograd.
    """

    admitted: bool
    backend: str
    reason: str

    def require_admitted(self) -> None:
        if not self.admitted:
            raise ValueError(self.reason)

    def to_dict(self) -> dict[str, Any]:
        return vars(self)


def public_resident_admission(
    config: Any,
    plan: ExecutionPlan,
) -> PublicResidentAdmission:
    """Return whether the normal trainer can execute ``plan`` exactly.

    These checks describe the currently qualified ABI rather than a model
    identity.  Unsupported training semantics remain on the safe backend and
    receive one concrete, user-visible reason.
    """

    if plan.selected_stack_backend != "resident_exact":
        reason = plan.admission_reason
        selection = getattr(plan, "selection", None)
        if callable(selection):
            try:
                reason = selection("transformer_stack").reason
            except (KeyError, AttributeError):
                pass
        return PublicResidentAdmission(
            admitted=False,
            backend=plan.selected_stack_backend,
            reason=reason,
        )
    checks = (
        (bool(getattr(config, "use_tt", False)), "resident execution requires a TT device"),
        (
            str(getattr(config, "training_model_type", "")) in {"lora", "peft"},
            "resident execution currently owns standard PEFT LoRA training only",
        ),
        (
            str(getattr(config, "adapter_type", "lora")) == "lora",
            "resident execution currently supports LoRA adapters only",
        ),
        (
            str(getattr(config, "dtype", "")) == "torch.bfloat16",
            "resident execution requires the qualified BF16 numerical contract",
        ),
        (
            int(getattr(config, "gradient_accumulation_steps", 1)) == 1,
            "resident execution does not yet own multi-microbatch gradient accumulation",
        ),
        (
            not bool(getattr(config, "gradient_checkpointing", False)),
            "resident execution owns its activation checkpoints and cannot combine with framework checkpointing",
        ),
        (
            float(getattr(config, "lora_dropout", 0.0)) == 0.0,
            "resident execution requires zero LoRA dropout",
        ),
        (
            not bool(getattr(config, "lora_use_rslora", False)),
            "rsLoRA is not yet qualified by the frozen resident contract",
        ),
        (
            not bool(getattr(config, "lora_use_dora", False)),
            "DoRA is not represented by the frozen resident state ABI",
        ),
        (
            getattr(config, "lora_plus_lr_ratio", None) is None,
            "LoRA+ per-matrix learning-rate groups are not represented by the frozen resident optimizer ABI",
        ),
        (
            str(getattr(config, "lora_bias", "none")) == "none",
            "trainable LoRA bias is not represented by the frozen resident state ABI",
        ),
        (
            getattr(config, "lora_modules_to_save", None) in (None, ()),
            "modules_to_save parameters are not represented by the frozen resident state ABI",
        ),
        (
            getattr(config, "lora_layers_to_transform", None) in (None, ()),
            "partial-layer LoRA targeting is not represented by the frozen resident state ABI",
        ),
        (
            getattr(config, "max_grad_norm", None) == 1.0,
            "the qualified resident clipping contract requires max_grad_norm=1.0",
        ),
        (
            str(getattr(config, "memory_mode", "auto")) in {"auto", "bf16"},
            "the current resident weight ABI stores frozen weights in BF16",
        ),
        (
            getattr(config, "weight_dtype_overrides", None) is None,
            "per-weight precision overrides are not represented by the frozen resident ABI",
        ),
        (
            getattr(config, "experimental_weight_dtype", None) is None,
            "compiler-wide reduced weight precision is not represented by the frozen resident ABI",
        ),
    )
    for accepted, reason in checks:
        if not accepted:
            return PublicResidentAdmission(False, "safe_torch_xla_autograd", reason)
    return PublicResidentAdmission(
        True,
        "resident_exact",
        "architecture, geometry and public training semantics match the qualified resident ABI",
    )


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
        model: nn.Module,
        packing_audit: dict[str, Any],
        source_bundle_sha256: str,
        descriptor_bundle_sha256: str,
        runtime_source_hash: str,
    ) -> None:
        self.plan = plan
        self.qualification = qualification
        self.packed = packed
        self.edges = edges
        self.engine = engine
        self.model = model
        physical_document = {
            "schema": "tt-finetune-physical-plan-v1",
            "semantic_fingerprint": plan.architecture.semantic_fingerprint,
            "rank": plan.rank,
            "batch_size": plan.batch_size,
            "sequence_length": plan.sequence_length,
            "targets": [vars(item) for item in plan.lora_targets],
            "contract": qualification.contract,
            "source_bundle_sha256": source_bundle_sha256,
            "descriptor_bundle_sha256": descriptor_bundle_sha256,
            "runtime_source_hash": runtime_source_hash,
            "public_custom_calls_per_step": engine.public_custom_calls_per_step,
            "physical_programs_per_step": engine.physical_programs_per_step,
        }
        self.physical_plan_fingerprint = hashlib.sha256(
            json.dumps(
                physical_document, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()[:24]
        self.report = ResidentBuildReport(
            semantic_fingerprint=plan.architecture.semantic_fingerprint,
            contract=qualification.contract,
            source_bundle=qualification.source_bundle,
            descriptor_bundle=qualification.descriptor_bundle,
            source_bundle_sha256=source_bundle_sha256,
            descriptor_bundle_sha256=descriptor_bundle_sha256,
            runtime_source_hash=runtime_source_hash,
            physical_plan_fingerprint=self.physical_plan_fingerprint,
            public_custom_calls_per_step=engine.public_custom_calls_per_step,
            physical_programs_per_step=engine.physical_programs_per_step,
            batch_size=plan.batch_size,
            sequence_length=plan.sequence_length,
            rank=plan.rank,
            packing_audit=packing_audit,
        )

    def step_tokens(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        scalars: AdamWScalars,
    ) -> ResidentStepResult:
        return self.engine.step_tokens(input_ids, labels, self.edges, scalars)

    def packed_state_dict(self) -> dict[str, Any]:
        """Materialize the canonical resident optimizer state for a checkpoint."""

        return {
            "format": "tt_finetune_resident_adamw_v1",
            "semantic_fingerprint": self.plan.architecture.semantic_fingerprint,
            "rank": self.plan.rank,
            "batch_size": self.plan.batch_size,
            "sequence_length": self.plan.sequence_length,
            "physical_plan_fingerprint": self.physical_plan_fingerprint,
            "optimizer_state": self.engine.optimizer_state.detach().cpu(),
        }

    def load_packed_state_dict(self, state: dict[str, Any]) -> None:
        """Restore parameters and AdamW moments without rebuilding the owner."""

        if state.get("format") != "tt_finetune_resident_adamw_v1":
            raise ValueError("checkpoint does not contain resident optimizer state")
        expected = {
            "semantic_fingerprint": self.plan.architecture.semantic_fingerprint,
            "rank": self.plan.rank,
            "batch_size": self.plan.batch_size,
            "sequence_length": self.plan.sequence_length,
        }
        differing = {
            name: (state.get(name), value)
            for name, value in expected.items()
            if state.get(name) != value
        }
        checkpoint_fingerprint = state.get("physical_plan_fingerprint")
        if (
            checkpoint_fingerprint is not None
            and checkpoint_fingerprint != self.physical_plan_fingerprint
        ):
            differing["physical_plan_fingerprint"] = (
                checkpoint_fingerprint,
                self.physical_plan_fingerprint,
            )
        if differing:
            raise ValueError(
                "resident checkpoint geometry differs from the executable: "
                f"{differing}"
            )
        value = state.get("optimizer_state")
        if not isinstance(value, torch.Tensor):
            raise TypeError("resident checkpoint optimizer_state is not a tensor")
        if tuple(value.shape) != tuple(self.engine.optimizer_state.shape):
            raise ValueError(
                "resident checkpoint optimizer state shape differs: "
                f"{tuple(value.shape)} != {tuple(self.engine.optimizer_state.shape)}"
            )
        self.engine.optimizer_state = value.to(
            device=self.engine.device,
            dtype=self.engine.optimizer_state.dtype,
        )

    def sync_logical_model(self) -> dict[str, torch.Tensor]:
        """Copy current packed parameters into ordinary PEFT Parameters.

        This is a checkpoint/export boundary only; it is never called from the
        timed training step.  The packed device tensor remains authoritative
        between steps.
        """

        rows = self.packed.trainable_pages.shape[0]
        logical = self.packed.logical_state(
            self.engine.optimizer_state[:rows].detach().cpu()
        )
        named = dict(self.model.named_parameters())
        missing = sorted(set(logical) - set(named))
        if missing:
            raise RuntimeError(
                f"resident parameter map is absent from the PEFT model: {missing[:8]}"
            )
        with torch.no_grad():
            for name, value in logical.items():
                parameter = named[name]
                parameter.copy_(value.to(parameter.device, parameter.dtype))
        return logical


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _qualified_bundle(root: Path, relative: str) -> Path:
    """Resolve development or wheel-vendored resident source bundles."""

    development = root / relative
    if (development / "manifest.json").is_file():
        return development
    vendored = Path(__file__).resolve().parents[1] / "_resident_bundles" / Path(relative).name
    if (vendored / "manifest.json").is_file():
        return vendored
    return development


def _bundle_sha256(path: Path) -> str:
    """Hash one frozen execution bundle independent of its install path."""

    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise FileNotFoundError(f"resident execution bundle is empty: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
    return digest.hexdigest()


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
    runtime_source_hash: str | None = None,
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
    source = _qualified_bundle(root, qualification.source_bundle)
    descriptor = _qualified_bundle(root, qualification.descriptor_bundle)
    for kind, path in (("source", source), ("descriptor", descriptor)):
        if not (path / "manifest.json").is_file():
            raise FileNotFoundError(
                f"qualified resident {kind} bundle is missing: {path}"
            )
    source_bundle_sha256 = _bundle_sha256(source)
    descriptor_bundle_sha256 = _bundle_sha256(descriptor)
    runtime_source_hash = str(runtime_source_hash or "unreported")

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
        logical_rank=plan.rank,
        physical_rank=plan.physical_rank,
    )
    packed_logical = packed.logical_state(packed.trainable_pages)
    named_parameters = dict(model.named_parameters())
    packing_exact = {
        name: (
            name in named_parameters
            and torch.equal(
                value,
                named_parameters[name].detach().cpu().to(value.dtype),
            )
        )
        for name, value in packed_logical.items()
    }
    packing_audit = {
        "contract": "logical_peft_to_resident_pages_bit_exact_v1",
        "exact_tensors": sum(packing_exact.values()),
        "total_tensors": len(packing_exact),
        "all_exact": all(packing_exact.values()),
        "first_mismatches": [
            name for name, exact in packing_exact.items() if not exact
        ][:16],
    }
    if not packing_audit["all_exact"]:
        raise RuntimeError(
            "resident packing changed logical PEFT parameters: "
            f"{packing_audit['first_mismatches']}"
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
        model=model,
        packing_audit=packing_audit,
        source_bundle_sha256=source_bundle_sha256,
        descriptor_bundle_sha256=descriptor_bundle_sha256,
        runtime_source_hash=runtime_source_hash,
    )


__all__ = [
    "ResidentBuildReport",
    "ResidentCausalLMExecutor",
    "PublicResidentAdmission",
    "build_resident_executor",
    "public_resident_admission",
]
