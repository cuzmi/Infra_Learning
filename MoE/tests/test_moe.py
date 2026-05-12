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
