"""
Run a small Mixture of Experts routing demo.

Before running:
    pip install -e .

Example:
    python MoE_Learning/examples/run_moe_demo.py --batch-size 2 --sequence-length 8
"""

from __future__ import annotations

import argparse

import torch

from moe_learning import MoELayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple MoE layer demo")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--expert-hidden-size", type=int, default=64)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    moe = MoELayer(
        hidden_size=args.hidden_size,
        expert_hidden_size=args.expert_hidden_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
    )
    hidden_states = torch.randn(
        args.batch_size,
        args.sequence_length,
        args.hidden_size,
    )

    output = moe(hidden_states)
    router = output.router_output

    print("MoE demo")
    print(f"input shape:  {tuple(hidden_states.shape)}")
    print(f"output shape: {tuple(output.hidden_states.shape)}")
    print(f"top-k: {args.top_k}")
    print()
    print("first token routes")
    print(f"expert ids: {router.expert_indices[0].tolist()}")
    print(f"weights:    {[round(x, 4) for x in router.expert_weights[0].tolist()]}")
    print()
    print("expert selection counts")
    print("expert | selected")
    selected_counts = torch.bincount(
        router.expert_indices.reshape(-1),
        minlength=args.num_experts,
    )
    for expert_id in range(args.num_experts):
        print(f"{expert_id:>6} | {selected_counts[expert_id].item():>8}")


if __name__ == "__main__":
    main()
