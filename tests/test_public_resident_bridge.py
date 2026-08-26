from types import SimpleNamespace

import pytest
import torch
import transformers
from transformers import AutoModelForCausalLM

from tt_finetune.architecture import describe_model
from tt_finetune.runtime import LoRATargetGeometry, plan_execution
from tt_finetune.runtime.execution_runtime import public_resident_admission


def _config(**changes):
    values = {
        "use_tt": True,
        "training_model_type": "lora",
        "adapter_type": "lora",
        "dtype": "torch.bfloat16",
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "lora_dropout": 0.0,
        "lora_use_rslora": False,
        "lora_use_dora": False,
        "lora_plus_lr_ratio": None,
        "lora_bias": "none",
        "lora_modules_to_save": None,
        "lora_layers_to_transform": None,
        "max_grad_norm": 1.0,
        "memory_mode": "auto",
        "weight_dtype_overrides": None,
        "experimental_weight_dtype": None,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _plan(backend="resident_exact"):
    return SimpleNamespace(
        selected_stack_backend=backend,
        admission_reason="safe architecture plan",
    )


def test_exact_public_contract_admits_resident_executor():
    result = public_resident_admission(_config(), _plan())
    assert result.admitted
    assert result.backend == "resident_exact"
    assert "match" in result.reason


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"gradient_accumulation_steps": 2}, "accumulation"),
        ({"lora_dropout": 0.1}, "dropout"),
        ({"lora_use_dora": True}, "DoRA"),
        ({"lora_plus_lr_ratio": 16.0}, "LoRA+"),
        ({"max_grad_norm": 0.3}, "max_grad_norm=1.0"),
        ({"memory_mode": "bf8_frozen"}, "BF16"),
    ],
)
def test_unrepresented_training_semantics_select_explicit_safe_fallback(
    change, reason,
):
    result = public_resident_admission(_config(**change), _plan())
    assert not result.admitted
    assert result.backend == "safe_torch_xla_autograd"
    assert reason in result.reason


def test_nonresident_geometry_preserves_planner_backend():
    result = public_resident_admission(
        _config(), _plan("safe_torch_xla_autograd")
    )
    assert not result.admitted
    assert result.backend == "safe_torch_xla_autograd"
    assert result.reason == "safe architecture plan"


def test_unqualified_llama_geometry_has_explicit_semantic_safe_reason():
    config = transformers.LlamaConfig(
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=22,
        num_attention_heads=32,
        num_key_value_heads=4,
        vocab_size=32000,
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    descriptor, _ = describe_model(model)
    plan = plan_execution(
        descriptor, rank=8, batch_size=1, sequence_length=128
    )
    result = public_resident_admission(_config(), plan)

    assert plan.selected_stack_backend == "safe_torch_xla_autograd"
    assert not result.admitted
    assert result.backend == "safe_torch_xla_autograd"
    assert result.reason == (
        "no faster full-stack implementation is qualified for this geometry"
    )


def test_layernorm_ordinary_mlp_is_semantically_supported_but_not_misadmitted():
    config = transformers.GPTNeoXConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        vocab_size=50304,
        rotary_pct=0.25,
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    descriptor, _ = describe_model(model)
    plan = plan_execution(
        descriptor, rank=8, batch_size=1, sequence_length=32
    )

    assert descriptor.norm_kind == "layer_norm"
    assert descriptor.mlp_kind == "ordinary"
    assert plan.capabilities.supported
    assert plan.selected_stack_backend == "safe_torch_xla_autograd"
    assert plan.selection("normalization").implementation == (
        "safe_explicit_layer_norm"
    )
    assert not public_resident_admission(_config(), plan).admitted


def test_resident_qualification_zero_fills_uniform_adapter_subset():
    config = transformers.Qwen2Config(
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        vocab_size=151936,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        attention_bias=True,
        tie_word_embeddings=True,
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    descriptor, _ = describe_model(model)
    full = plan_execution(
        descriptor, rank=16, batch_size=1, sequence_length=128
    )
    assert full.selected_stack_backend == "resident_exact"

    qv_only = plan_execution(
        descriptor,
        rank=16,
        batch_size=1,
        sequence_length=128,
        lora_targets=(
            LoRATargetGeometry("q", 896, 896, 24),
            LoRATargetGeometry("v", 896, 128, 24),
        ),
    )
    assert qv_only.selected_stack_backend == "resident_exact"
    assert qv_only.selection("lora").implementation == "device_packed_resident"

    partial_qv = plan_execution(
        descriptor,
        rank=16,
        batch_size=1,
        sequence_length=128,
        lora_targets=(
            LoRATargetGeometry("q", 896, 896, 12),
            LoRATargetGeometry("v", 896, 128, 12),
        ),
    )
    assert partial_qv.selected_stack_backend == "safe_torch_xla_autograd"
    assert partial_qv.selection("lora").implementation == "peft_device_exact"
