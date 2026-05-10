"""
Run a small baseline vs speculative decoding demo.

Before running:
    pip install -e .

Example:
    python examples/run_demo.py --max-new-tokens 20 --max-draft-tokens 4
"""

from __future__ import annotations

import argparse

from speculative_decoding.baseline import sample_decode
from speculative_decoding.metrics import measure_decode, speedup
from speculative_decoding.models import load_model_and_tokenizer
from speculative_decoding.speculative import speculative_decode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speculative decoding demo")
    parser.add_argument(
        "--draft-model",
        default="distilgpt2",
        help="Draft model name or local path.",
    )
    parser.add_argument(
        "--target-model",
        default="gpt2",
        help="Target model name or local path.",
    )
    parser.add_argument(
        "--prompt",
        default="The future of artificial intelligence is",
        help="Prompt text.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=30,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--max-draft-tokens",
        type=int,
        default=5,
        help="Maximum draft tokens per speculative decoding step.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for baseline decoding.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use, such as cpu or cuda. Defaults to auto.",
    )
    return parser.parse_args()


def print_metrics(name: str, latency_seconds: float, tokens_per_second: float) -> None:
    print(f"{name} latency: {latency_seconds:.3f}s")
    print(f"{name} tokens/s: {tokens_per_second:.2f}")


def main() -> None:
    args = parse_args()

    print("Loading models...")
    draft = load_model_and_tokenizer(args.draft_model, device=args.device)
    target = load_model_and_tokenizer(args.target_model, device=args.device)

    print("\nRunning baseline decoding...")
    baseline_text, baseline_metrics = measure_decode(
        sample_decode,
        args.max_new_tokens,
        target,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print("\nRunning speculative decoding...")
    speculative_text, speculative_metrics = measure_decode(
        speculative_decode,
        args.max_new_tokens,
        draft,
        target,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_draft_tokens=args.max_draft_tokens,
    )

    print("\nBaseline output:")
    print(baseline_text)
    print_metrics(
        "Baseline",
        baseline_metrics.latency_seconds,
        baseline_metrics.tokens_per_second,
    )

    print("\nSpeculative output:")
    print(speculative_text)
    print_metrics(
        "Speculative",
        speculative_metrics.latency_seconds,
        speculative_metrics.tokens_per_second,
    )

    ratio = speedup(
        baseline_metrics.latency_seconds,
        speculative_metrics.latency_seconds,
    )
    print(f"\nSpeedup: {ratio:.2f}x")


if __name__ == "__main__":
    main()
