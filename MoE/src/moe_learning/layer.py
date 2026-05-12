from __future__ import annotations

import torch
from torch import nn

from .experts import FeedForwardExpert
from .outputs import MoEOutput
from .router import TopKRouter


class MoELayer(nn.Module):
    """
    A simple token-level MoE feed-forward layer.

    Core flow:
        hidden states -> router -> selected experts -> weighted combine
    """

    def __init__(
        self,
        hidden_size: int,
        expert_hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
        )
        self.experts = nn.ModuleList(
            FeedForwardExpert(
                hidden_size=hidden_size,
                expert_hidden_size=expert_hidden_size,
                dropout=dropout,
            )
            for _ in range(num_experts)
        )

    def forward(self, hidden_states: torch.Tensor) -> MoEOutput:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        flat_states = hidden_states.reshape(-1, hidden_size)

        router_output = self.router(flat_states)
        combined = torch.zeros_like(flat_states)

        for expert_id, expert in enumerate(self.experts):
            for route_rank in range(self.top_k):
                token_mask = router_output.expert_indices[:, route_rank] == expert_id
                if not torch.any(token_mask):
                    continue

                expert_input = flat_states[token_mask]
                expert_output = expert(expert_input)
                expert_weight = router_output.expert_weights[token_mask, route_rank]
                combined[token_mask] += expert_output * expert_weight.unsqueeze(-1)

        output_states = combined.reshape(batch_size, sequence_length, hidden_size)
        return MoEOutput(
            hidden_states=output_states,
            router_output=router_output,
        )

