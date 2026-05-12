"""
Run a small baseline vs speculative decoding demo.

Before running:
    pip install -e .

Example:
    python Speculative_Decoding/examples/run_demo.py --max-new-tokens 20 --max-draft-tokens 4
"""

from __future__ import annotations

import argparse

from speculative_decoding.baseline import sample_decode
from speculative_decoding.config import default_config
from speculative_decoding.metrics import acceptance_rate, measure_decode, speedup
from speculative_decoding.models import assert_tokenizers_match, load_model_and_tokenizer
from speculative_decoding.speculative import speculative_decode


def parse_args() -> argparse.Namespace:
    config = default_config()
    parser = argparse.ArgumentParser(description="Speculative decoding demo")
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
        help="Maximum draft tokens per speculative decoding step.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.generation.temperature,
        help="Sampling temperature for baseline decoding.",
    )
    parser.add_argument(
        "--device",
        default=config.models.device,
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
    assert_tokenizers_match(draft, target)

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
    (speculative_text, speculative_stats), speculative_metrics = measure_decode(
        speculative_decode,
        args.max_new_tokens,
        draft,
        target,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_draft_tokens=args.max_draft_tokens,
        collect_stats=True,
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
    accept_rate = acceptance_rate(
        speculative_stats.accepted_tokens,
        speculative_stats.drafted_tokens,
    )
    print(f"Speculative accepted/drafted: {speculative_stats.accepted_tokens}/{speculative_stats.drafted_tokens}")
    print(f"Speculative rejected: {speculative_stats.rejected_tokens}")
    print(f"Speculative accept rate: {accept_rate:.2%}")

    ratio = speedup(
        baseline_metrics.latency_seconds,
        speculative_metrics.latency_seconds,
    )
    print(f"\nSpeedup: {ratio:.2f}x")


if __name__ == "__main__":
    main()
