"""
Run a small simple EAGLE-style decoding demo.

Before running:
    pip install -e .

Example:
    python examples/run_eagle_demo.py --max-new-tokens 20 --max-draft-tokens 4
"""

from __future__ import annotations

import argparse

from speculative_decoding.config import default_config
from speculative_decoding.eagle import eagle_decode_with_loaded_draft
from speculative_decoding.metrics import acceptance_rate, measure_decode
from speculative_decoding.models import assert_tokenizers_match, load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    config = default_config()
    parser = argparse.ArgumentParser(description="Simple EAGLE decoding demo")
    parser.add_argument(
        "--draft-model",
        default=config.models.draft_model,
        help="Draft model name or local path.",
    )
    parser.add_argument(
        "--target-model",
        default=config.models.target_model,
        help="Target model name or local path.",
    )
    parser.add_argument(
        "--prompt",
        default=config.generation.prompt,
        help="Prompt text.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=config.generation.max_new_tokens,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--max-draft-tokens",
        type=int,
        default=config.generation.max_draft_tokens,
        help="Maximum draft tokens per EAGLE step.",
    )
    parser.add_argument(
        "--tree-top-k",
        type=int,
        default=2,
        help="Number of candidate tokens to branch at each EAGLE tree depth.",
    )
    parser.add_argument(
        "--device",
        default=config.models.device,
        help="Device to use, such as cpu or cuda. Defaults to auto.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading models...")
    draft = load_model_and_tokenizer(args.draft_model, device=args.device)
    target = load_model_and_tokenizer(args.target_model, device=args.device)
    assert_tokenizers_match(draft, target)

    print("\nRunning simple EAGLE decoding...")
    (text, stats), metrics = measure_decode(
        eagle_decode_with_loaded_draft,
        args.max_new_tokens,
        draft,
        target,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_draft_tokens=args.max_draft_tokens,
        tree_top_k=args.tree_top_k,
        collect_stats=True,
    )

    accept_rate = acceptance_rate(stats.accepted_tokens, stats.drafted_tokens)

    print("\nEAGLE output:")
    print(text)
    print(f"EAGLE latency: {metrics.latency_seconds:.3f}s")
    print(f"EAGLE tokens/s: {metrics.tokens_per_second:.2f}")
    print(f"EAGLE accepted/drafted: {stats.accepted_tokens}/{stats.drafted_tokens}")
    print(f"EAGLE breakpoint tokens: {stats.breakpoint_tokens}")
    print(f"EAGLE target forwards: {stats.target_forwards}")
    print(f"EAGLE draft forwards: {stats.draft_forwards}")
    print(f"EAGLE tree attention verifications: {stats.tree_attention_verifications}")
    print(f"EAGLE path fallback verifications: {stats.path_fallback_verifications}")
    print(f"EAGLE accept rate: {accept_rate:.2%}")


if __name__ == "__main__":
    main()
