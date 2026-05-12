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

        token_count = flat_states.shape[0]
        token_ids = torch.arange(
            token_count,
            device=flat_states.device,
        ).repeat_interleave(self.top_k)
        expert_ids = router_output.expert_indices.reshape(-1)
        expert_weights = router_output.expert_weights.reshape(-1)

        order = torch.argsort(expert_ids)
        sorted_expert_ids = expert_ids[order]
        sorted_token_ids = token_ids[order]
        sorted_weights = expert_weights[order]
        sorted_states = flat_states[sorted_token_ids]

        expert_counts = torch.bincount(
            sorted_expert_ids,
            minlength=self.num_experts,
        )
        expert_offsets = torch.cumsum(expert_counts, dim=0)

        start = 0
        for expert_id, end in enumerate(expert_offsets.tolist()):
            if end == start:
                continue

            expert_input = sorted_states[start:end]
            expert_output = self.experts[expert_id](expert_input)
            expert_weight = sorted_weights[start:end].unsqueeze(-1)
            original_token_ids = sorted_token_ids[start:end]

            combined.index_add_(
                0,
                original_token_ids,
                expert_output * expert_weight,
            )
            start = end

        output_states = combined.reshape(batch_size, sequence_length, hidden_size)
        return MoEOutput(
            hidden_states=output_states,
            router_output=router_output,
        )
