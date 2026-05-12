from __future__ import annotations

import torch
from torch import nn


class FeedForwardExpert(nn.Module):
    """A small MLP expert similar to a Transformer feed-forward block."""

    def __init__(
        self,
        hidden_size: int,
        expert_hidden_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(hidden_size, expert_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_size, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)

