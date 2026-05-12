from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .outputs import RouterOutput


class TopKRouter(nn.Module):
    """
    Score every expert for every token, then keep the top-k choices.

    The router does not run experts by itself. It only answers:
        1. Which experts should process each token?
        2. What weight should each selected expert get?
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> RouterOutput:
        logits = self.gate(hidden_states)
        probs = F.softmax(logits, dim=-1)

        topk_probs, expert_indices = torch.topk(probs, k=self.top_k, dim=-1)
        expert_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        return RouterOutput(
            logits=logits,
            probs=probs,
            expert_indices=expert_indices,
            expert_weights=expert_weights,
        )

