from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RouterOutput:
    """
    Core routing results for flattened tokens.

    Shapes:
        logits: [tokens, num_experts]
        probs: [tokens, num_experts]
        expert_indices: [tokens, top_k]
        expert_weights: [tokens, top_k]
    """

    logits: torch.Tensor
    probs: torch.Tensor
    expert_indices: torch.Tensor
    expert_weights: torch.Tensor


@dataclass
class MoEOutput:
    """Output bundle returned by MoELayer."""

    hidden_states: torch.Tensor
    router_output: RouterOutput

