import torch

from moe_learning import MoELayer, TopKRouter


def test_topk_router_returns_expected_shapes() -> None:
    router = TopKRouter(hidden_size=8, num_experts=4, top_k=2)
    hidden_states = torch.randn(6, 8)

    output = router(hidden_states)

    assert output.logits.shape == (6, 4)
    assert output.probs.shape == (6, 4)
    assert output.expert_indices.shape == (6, 2)
    assert output.expert_weights.shape == (6, 2)
    assert torch.allclose(output.expert_weights.sum(dim=-1), torch.ones(6))


def test_moe_layer_preserves_hidden_state_shape() -> None:
    layer = MoELayer(
        hidden_size=8,
        expert_hidden_size=16,
        num_experts=4,
        top_k=2,
    )
    hidden_states = torch.randn(2, 3, 8)

    output = layer(hidden_states)

    assert output.hidden_states.shape == hidden_states.shape


def test_moe_layer_matches_simple_dispatch_path() -> None:
    torch.manual_seed(0)
    layer = MoELayer(
        hidden_size=8,
        expert_hidden_size=16,
        num_experts=4,
        top_k=2,
    )
    hidden_states = torch.randn(2, 3, 8)

    output = layer(hidden_states)
    flat_states = hidden_states.reshape(-1, 8)
    router = output.router_output
    expected = torch.zeros_like(flat_states)

    for expert_id, expert in enumerate(layer.experts):
        for route_rank in range(layer.top_k):
            token_mask = router.expert_indices[:, route_rank] == expert_id
            if not torch.any(token_mask):
                continue

            expert_output = expert(flat_states[token_mask])
            expert_weight = router.expert_weights[token_mask, route_rank]
            expected[token_mask] += expert_output * expert_weight.unsqueeze(-1)

    expected = expected.reshape_as(hidden_states)
    assert torch.allclose(output.hidden_states, expected, atol=1e-6)


def test_moe_layer_backpropagates_through_router_and_experts() -> None:
    layer = MoELayer(
        hidden_size=8,
        expert_hidden_size=16,
        num_experts=4,
        top_k=2,
    )
    hidden_states = torch.randn(2, 3, 8, requires_grad=True)

    output = layer(hidden_states)
    loss = output.hidden_states.square().mean()
    loss.backward()

    assert hidden_states.grad is not None
    assert layer.router.gate.weight.grad is not None
    assert any(
        expert.net[0].weight.grad is not None
        for expert in layer.experts
    )
