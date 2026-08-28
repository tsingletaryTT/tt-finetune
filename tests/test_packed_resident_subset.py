from __future__ import annotations

import torch
from torch import nn

from tt_finetune.optim import TilePackedAdamW
from tt_finetune.runtime.packed_state import pack_transformer_stack


class _LoRALinear(nn.Module):
    def __init__(
        self, input_features: int, output_features: int, rank: int, *, bias: bool
    ) -> None:
        super().__init__()
        self.in_features = input_features
        self.out_features = output_features
        self.base_layer = nn.Linear(
            input_features, output_features, bias=bias, dtype=torch.bfloat16
        )
        self.lora_A = nn.ModuleDict({
            "default": nn.Linear(
                input_features, rank, bias=False, dtype=torch.bfloat16
            )
        })
        self.lora_B = nn.ModuleDict({
            "default": nn.Linear(
                rank, output_features, bias=False, dtype=torch.bfloat16
            )
        })
        self.lora_dropout = nn.ModuleDict({"default": nn.Identity()})
        self.scaling = {"default": 2.0}
        self.use_dora = {"default": False}
        self.active_adapter = "default"


class _Norm(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden, dtype=torch.bfloat16))
        self.variance_epsilon = 1.0e-5


class _Layer(nn.Module):
    def __init__(self, hidden: int, intermediate: int, rank: int) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = _LoRALinear(hidden, hidden, rank, bias=False)
        self.self_attn.k_proj = _LoRALinear(hidden, 32, rank, bias=False)
        self.self_attn.v_proj = _LoRALinear(hidden, 32, rank, bias=False)
        self.self_attn.o_proj = _LoRALinear(hidden, hidden, rank, bias=False)
        self.self_attn.head_dim = 32
        self.self_attn.scaling = 32**-0.5
        self.mlp = nn.Module()
        self.mlp.gate_proj = _LoRALinear(hidden, intermediate, rank, bias=False)
        self.mlp.up_proj = _LoRALinear(hidden, intermediate, rank, bias=False)
        self.mlp.down_proj = _LoRALinear(intermediate, hidden, rank, bias=False)
        self.input_layernorm = _Norm(hidden)
        self.post_attention_layernorm = _Norm(hidden)


class _Model(nn.Module):
    def __init__(self, rank: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer(64, 128, rank) for _ in range(2)])
        for layer in self.layers:
            # Match ordinary PEFT Q/V targeting: non-target projections remain
            # plain frozen Linear modules rather than LoRA wrappers.
            layer.self_attn.k_proj = layer.self_attn.k_proj.base_layer
            layer.self_attn.o_proj = layer.self_attn.o_proj.base_layer
            layer.mlp.gate_proj = layer.mlp.gate_proj.base_layer
            layer.mlp.up_proj = layer.mlp.up_proj.base_layer
            layer.mlp.down_proj = layer.mlp.down_proj.base_layer


def test_qv_subset_round_trips_only_logical_parameters_and_zero_fills_abi():
    model = _Model()
    cosine = torch.zeros((128, 32), dtype=torch.bfloat16)
    mask = torch.zeros((128, 128), dtype=torch.bfloat16)
    packed = pack_transformer_stack(
        model,
        cosine=cosine,
        sine=cosine,
        attention_mask=mask,
        logical_rank=8,
        physical_rank=32,
    )

    logical = packed.logical_state()
    expected = {
        name: parameter.detach()
        for name, parameter in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }
    assert len(packed.regions) == 28
    assert len(logical) == 8
    assert logical.keys() == expected.keys()
    assert all(torch.equal(logical[name], value) for name, value in expected.items())

    pages = packed.trainable_pages.reshape(-1, 32, 32)
    inactive = [region for region in packed.regions if region.parameter_name is None]
    assert len(inactive) == 20
    assert all(
        pages[region.page_offset : region.page_offset + region.page_count]
        .count_nonzero()
        .item()
        == 0
        for region in inactive
    )

    gradients = {
        name: torch.ones_like(value) for name, value in logical.items()
    }
    arena = packed.pack_logical_gradients(gradients).reshape(-1, 32, 32)
    for region in inactive:
        target = packed.gradient_plan.slice(region.layer, region.short_name)
        assert arena[target.page_offset : target.page_stop].count_nonzero() == 0


def test_tile_packed_gradient_audit_reconstructs_logical_order():
    first = nn.Parameter(torch.arange(17, dtype=torch.float32))
    second = nn.Parameter(torch.arange(35, dtype=torch.float32).reshape(5, 7))
    optimizer = TilePackedAdamW(
        [first, second],
        max_pack_bytes=4096,
        capturable=False,
        lr=1.0e-3,
    )
    first.grad = torch.arange(17, dtype=torch.float32) + 0.25
    second.grad = torch.arange(35, dtype=torch.float32).reshape(5, 7) - 0.5
    expected = (first.grad.clone(), second.grad.clone())
    optimizer.pack_gradients()

    actual = optimizer.logical_gradient_tensors()
    assert len(actual) == 2
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
