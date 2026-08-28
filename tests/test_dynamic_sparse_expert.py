import torch

from tt_finetune.kernels import dynamic_sparse_expert
from tt_finetune.kernels.dynamic_sparse_expert import (
    dynamic_pack_expert_assignments,
    dynamic_restore_expert_assignments,
    dynamic_expert_input_linear,
    dynamic_shared_input_expert_linear,
    expert_block_sparsity,
)
from tt_finetune.integration.sparse_moe import GenericDeviceSparseMoELoRA


def _routing(tokens: int, experts: int, top_k: int):
    selected = torch.stack(
        [torch.randperm(experts)[:top_k] for _ in range(tokens)]
    )
    exact, blocks = expert_block_sparsity(
        selected, expert_count=experts, dtype=torch.bfloat16
    )
    return selected, exact, blocks


def test_dynamic_shared_projection_registered_autograd_matches_selected_bmm():
    torch.manual_seed(41)
    tokens, experts, top_k, width = 64, 4, 2, 32
    selected, _exact, blocks = _routing(tokens, experts, top_k)
    token_ids = torch.arange(tokens).unsqueeze(1)
    weight = torch.randn(
        experts, width, width, dtype=torch.bfloat16
    )

    values = torch.randn(
        tokens, width, dtype=torch.bfloat16, requires_grad=True
    )
    output = dynamic_shared_input_expert_linear(
        values.unsqueeze(1).expand(-1, top_k, -1),
        weight,
        blocks,
        selected,
    )[token_ids, selected]
    output.float().square().sum().backward()
    dynamic_gradient = values.grad.clone()

    reference_values = values.detach().clone().requires_grad_()
    reference = torch.bmm(
        reference_values.unsqueeze(1)
        .expand(-1, top_k, -1)
        .reshape(-1, 1, width),
        weight[selected].reshape(-1, width, width).transpose(1, 2),
    ).reshape(tokens, top_k, width)
    reference.float().square().sum().backward()

    assert torch.equal(output, reference)
    assert torch.equal(dynamic_gradient, reference_values.grad)


def test_dynamic_expert_projection_registered_autograd_matches_selected_bmm():
    torch.manual_seed(42)
    tokens, experts, top_k, width = 64, 4, 2, 32
    selected, exact, blocks = _routing(tokens, experts, top_k)
    token_ids = torch.arange(tokens).unsqueeze(1)
    weight = torch.randn(
        experts, width, width, dtype=torch.bfloat16
    )

    values = (
        torch.randn(tokens, experts, width, dtype=torch.bfloat16)
        * exact.unsqueeze(-1)
    ).detach().requires_grad_()
    output = dynamic_expert_input_linear(
        values, weight, blocks, selected
    )[token_ids, selected]
    output.float().square().sum().backward()
    dynamic_gradient = values.grad.clone()

    reference_values = values.detach().clone().requires_grad_()
    reference = torch.bmm(
        reference_values[token_ids, selected].reshape(-1, 1, width),
        weight[selected].reshape(-1, width, width).transpose(1, 2),
    ).reshape(tokens, top_k, width)
    reference.float().square().sum().backward()

    assert torch.equal(output, reference)
    assert torch.equal(dynamic_gradient, reference_values.grad)


def test_dynamic_registered_owner_survives_fullgraph_dynamo():
    torch.manual_seed(43)
    tokens, experts, top_k, width = 32, 4, 2, 32
    selected, _exact, blocks = _routing(tokens, experts, top_k)
    token_ids = torch.arange(tokens).unsqueeze(1)
    weight = torch.randn(
        experts, width, width, dtype=torch.bfloat16
    )

    class Projection(torch.nn.Module):
        def forward(self, values):
            output = dynamic_shared_input_expert_linear(
                values.unsqueeze(1).expand(-1, top_k, -1),
                weight,
                blocks,
                selected,
            )
            return output[token_ids, selected].float().square().sum()

    values = torch.randn(
        tokens, width, dtype=torch.bfloat16, requires_grad=True
    )
    compiled = torch.compile(Projection(), backend="eager", fullgraph=True)
    compiled(values).backward()
    compiled_gradient = values.grad.clone()

    reference_values = values.detach().clone().requires_grad_()
    Projection()(reference_values).backward()
    assert torch.equal(compiled_gradient, reference_values.grad)


def _dynamic_moe_projection_fixture():
    experts, width, rank = 4, 32, 8
    generator = torch.Generator().manual_seed(44)
    weights = {
        role: torch.randn(
            experts, width, width, generator=generator, dtype=torch.bfloat16
        )
        for role in ("up", "down")
    }
    module = GenericDeviceSparseMoELoRA(
        torch.nn.Identity(),
        router_layer=torch.nn.Linear(width, experts, bias=False),
        routed_weights=weights,
        shared_weights=None,
        shared_gate=None,
        rank=rank,
        alpha=rank,
        top_k=2,
        backend="dynamic",
        router_normalization="softmax_selected",
        output_contract="hidden",
    )
    with torch.no_grad():
        module.routed_lora_B["down"].normal_(generator=generator)
    return module


def test_shared_swiglu_lora_exposes_every_parameter_to_compiled_autograd():
    experts, width, intermediate, rank = 4, 32, 64, 8
    generator = torch.Generator().manual_seed(441)
    routed = {
        "up": torch.randn(
            experts, width, width, generator=generator, dtype=torch.float32
        ),
        "down": torch.randn(
            experts, width, width, generator=generator, dtype=torch.float32
        ),
    }
    shared = {
        "gate": torch.randn(
            1, intermediate, width, generator=generator, dtype=torch.float32
        ),
        "up": torch.randn(
            1, intermediate, width, generator=generator, dtype=torch.float32
        ),
        "down": torch.randn(
            1, width, intermediate, generator=generator, dtype=torch.float32
        ),
    }
    shared_gate = torch.nn.Linear(width, 1, bias=False)
    module = GenericDeviceSparseMoELoRA(
        torch.nn.Identity(),
        router_layer=torch.nn.Linear(width, experts, bias=False),
        routed_weights=routed,
        shared_weights=shared,
        shared_gate=shared_gate,
        rank=rank,
        alpha=rank,
        top_k=2,
        backend="dynamic",
        router_normalization="softmax_selected",
        output_contract="hidden",
    )
    with torch.no_grad():
        for value in module.shared_lora_B.values():
            value.normal_(generator=generator)

    class Shared(torch.nn.Module):
        def __init__(self, owner):
            super().__init__()
            self.owner = owner

        def forward(self, values):
            return self.owner._shared_experts(values).float().square().sum()

    values = torch.randn(32, width, generator=generator, dtype=torch.float32)
    eager = Shared(module)(values).detach()
    compiled = torch.compile(Shared(module), backend="eager", fullgraph=True)(
        values
    )
    assert torch.equal(compiled.detach(), eager)
    compiled.backward()

    for role in ("gate", "up", "down"):
        assert module.shared_lora_A[role].grad is not None
        assert module.shared_lora_B[role].grad is not None


def test_dynamic_no_capacity_packing_restores_every_assignment_exactly():
    module = _dynamic_moe_projection_fixture()
    tokens, top_k, width = 64, 2, 32
    selected, _exact, _blocks = _routing(tokens, module.expert_count, top_k)
    positions = module._routing_positions(selected)
    values = torch.randn(tokens, top_k, width, dtype=torch.bfloat16)

    packed, blocks = module._pack_dynamic_assignments(
        values, selected, positions
    )
    restored = packed[positions, selected]

    assert torch.equal(restored, values)
    assert int(positions.max()) < tokens
    assert blocks.shape == (1, 1, tokens // 32, module.expert_count)
    expected_blocks = sum(
        (int((selected == expert).sum()) + 31) // 32
        for expert in range(module.expert_count)
    )
    assert int(blocks.sum()) == expected_blocks


def test_dynamic_pack_restore_custom_ops_survive_fullgraph_backward():
    module = _dynamic_moe_projection_fixture()
    tokens, top_k, width = 64, 2, 32
    selected, _exact, _blocks = _routing(tokens, module.expert_count, top_k)
    positions = module._routing_positions(selected)

    class RoundTrip(torch.nn.Module):
        def forward(self, values):
            packed = dynamic_pack_expert_assignments(
                values, selected, positions, module.expert_count
            )
            return dynamic_restore_expert_assignments(
                packed, selected, positions
            ).float().square().sum()

    values = torch.randn(
        tokens, top_k, width, dtype=torch.bfloat16, requires_grad=True
    )
    torch.compile(RoundTrip(), backend="eager", fullgraph=True)(values).backward()
    expected_gradient = (values.detach().float() * 2).to(torch.bfloat16)
    assert torch.equal(values.grad, expected_gradient)


def test_gather_pack_backward_matches_safe_dispatch_exactly(monkeypatch):
    module = _dynamic_moe_projection_fixture()
    tokens, top_k, width = 64, 2, 32
    selected, _exact, _blocks = _routing(tokens, module.expert_count, top_k)
    positions = module._routing_positions(selected)
    values = torch.randn(tokens, top_k, width, dtype=torch.bfloat16)

    monkeypatch.setattr(
        dynamic_sparse_expert, "USE_GATHER_PACK_BACKWARD", False
    )
    safe = dynamic_sparse_expert._backward_pack_assignments(
        values,
        selected,
        positions,
        expert_count=module.expert_count,
    )
    monkeypatch.setattr(
        dynamic_sparse_expert, "USE_GATHER_PACK_BACKWARD", True
    )
    candidate = dynamic_sparse_expert._backward_pack_assignments(
        values,
        selected,
        positions,
        expert_count=module.expert_count,
    )

    assert torch.equal(candidate, safe)


def test_dense_packed_forward_diagnostic_is_explicit(monkeypatch):
    monkeypatch.setattr(
        dynamic_sparse_expert,
        "USE_DENSE_PACKED_FORWARD_DIAGNOSTIC",
        True,
    )
    values = torch.randn(32, 2, 32, dtype=torch.bfloat16)
    weight = torch.zeros(2, 32, 32, dtype=torch.bfloat16)
    diagonal = torch.arange(32)
    weight[:, diagonal, diagonal] = 1
    bitmap = torch.ones(1, 1, 1, 2, dtype=torch.bfloat16)
    selected = torch.zeros(32, 1, dtype=torch.int64)
    actual = dynamic_sparse_expert.dynamic_expert_input_linear(
        values, weight, bitmap, selected
    )
    assert torch.equal(actual, values)


def test_block_batched_projection_is_selected_only_and_differentiable():
    module = _dynamic_moe_projection_fixture()
    tokens, top_k, width = 64, 2, 32
    selected, _exact, _blocks = _routing(tokens, module.expert_count, top_k)
    positions = module._routing_positions(selected)
    values = torch.randn(
        tokens, top_k, width, dtype=torch.bfloat16, requires_grad=True
    )
    weight = module.routed_weights["down"]
    actual = dynamic_sparse_expert.dynamic_block_batched_expert_linear(
        values, weight, selected, positions
    )
    token_ids = torch.arange(tokens).unsqueeze(1)
    expected = torch.bmm(
        values.reshape(-1, 1, width),
        weight[selected].reshape(-1, width, width).transpose(1, 2),
    ).reshape(tokens, top_k, width)
    assert torch.equal(actual, expected)

    actual.float().square().sum().backward()
    actual_gradient = values.grad.clone()
    values.grad = None
    expected.float().square().sum().backward()
    assert torch.equal(actual_gradient, values.grad)


def test_no_drop_assignment_block_bound_covers_awkward_routing():
    assert dynamic_sparse_expert.maximum_no_drop_assignment_blocks(
        tokens=128, assignments=1024, expert_count=32
    ) == 63
    assert dynamic_sparse_expert.maximum_no_drop_assignment_blocks(
        tokens=128, assignments=128, expert_count=32
    ) == 35


def test_block_batched_admission_is_geometry_and_memory_bounded(monkeypatch):
    monkeypatch.setattr(
        dynamic_sparse_expert,
        "MAX_BLOCK_BATCHED_WEIGHT_BYTES",
        256 << 20,
    )
    assert dynamic_sparse_expert.dynamic_block_batched_support_error(
        tokens=128,
        top_k=8,
        expert_count=32,
        input_features=1024,
        output_features=1024,
        element_size=2,
    ) is None
    error = dynamic_sparse_expert.dynamic_block_batched_support_error(
        tokens=128,
        top_k=8,
        expert_count=32,
        input_features=4096,
        output_features=4096,
        element_size=2,
    )
    assert error is not None
    assert "weight view exceeds" in error


def test_dynamic_no_capacity_projection_matches_assignment_reference_backward():
    module = _dynamic_moe_projection_fixture()
    tokens, top_k, width = 64, 2, 32
    selected, _exact, _blocks = _routing(tokens, module.expert_count, top_k)
    positions = module._routing_positions(selected)
    upstream = torch.randn(tokens, top_k, width, dtype=torch.bfloat16)

    dynamic_inputs = torch.randn(
        tokens, top_k, width, dtype=torch.bfloat16, requires_grad=True
    )
    dynamic_output = module._project_dynamic_selected(
        "down", dynamic_inputs, selected, positions
    )
    (dynamic_output.float() * upstream.float()).sum().backward()
    dynamic_input_gradient = dynamic_inputs.grad.clone()
    dynamic_a_gradient = module.routed_lora_A["down"].grad.clone()
    dynamic_b_gradient = module.routed_lora_B["down"].grad.clone()

    module.zero_grad(set_to_none=True)
    reference_inputs = dynamic_inputs.detach().clone().requires_grad_()
    reference_output = module._project_selected(
        "down", reference_inputs, selected
    )
    (reference_output.float() * upstream.float()).sum().backward()

    assert torch.equal(dynamic_output, reference_output)
    assert torch.equal(dynamic_input_gradient, reference_inputs.grad)
    assert torch.equal(dynamic_a_gradient, module.routed_lora_A["down"].grad)
    assert torch.equal(dynamic_b_gradient, module.routed_lora_B["down"].grad)
