import torch

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
