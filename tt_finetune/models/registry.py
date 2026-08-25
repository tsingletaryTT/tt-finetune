"""Declarative architecture knowledge; runtime truth still comes from preflight."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelFamily:
    name: str
    model_types: tuple[str, ...]
    attention_layout: str
    default_lora_policy: str = "auto"
    public_hardware_status: str = "not_tested"


MODEL_FAMILIES = (
    ModelFamily(
        "llama",
        ("llama", "mistral", "qwen2", "qwen3", "gemma", "gemma2", "phi3"),
        "separate_qkv",
        public_hardware_status="passed_representatives",
    ),
    ModelFamily(
        "gpt2",
        ("gpt2",),
        "fused_qkv_conv1d",
        public_hardware_status="passed_representative",
    ),
    ModelFamily("gpt_neox", ("gpt_neox",), "fused_qkv_linear"),
    ModelFamily("falcon", ("falcon",), "fused_qkv_linear"),
    ModelFamily("bloom", ("bloom",), "fused_qkv_linear"),
)


def family_for_model_type(model_type: str) -> ModelFamily | None:
    for family in MODEL_FAMILIES:
        if model_type in family.model_types:
            return family
    return None

