"""
Metrics helpers for decoding demos.
core evaluation methods: tokens_per_second ; acceptance_rate
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, TypeVar


T = TypeVar("T")


@dataclass
class DecodeMetrics:
    """
    Basic performance metrics for one decoding run.
    """

    latency_seconds: float
    num_new_tokens: int
    tokens_per_second: float


def tokens_per_second(
    num_tokens: int,
    latency_seconds: float,
) -> float:
    """
    Compute token throughput.
    """
    if latency_seconds <= 0:
        return 0.0

    return num_tokens / latency_seconds


def measure_latency(
    fn: Callable[..., T],
    *args,
    **kwargs,
) -> tuple[T, float]:
    """
    Run a callable and return its result plus elapsed wall-clock time.
    """
    start_time = time.perf_counter()
    result = fn(*args, **kwargs)
    end_time = time.perf_counter()

    return result, end_time - start_time


def measure_decode(
    fn: Callable[..., T],
    num_new_tokens: int,
    *args,
    **kwargs,
) -> tuple[T, DecodeMetrics]:
    """
    Run a decode function and return its result plus basic metrics.
    """
    result, latency_seconds = measure_latency(fn, *args, **kwargs)
    metrics = DecodeMetrics(
        latency_seconds=latency_seconds,
        num_new_tokens=num_new_tokens,
        tokens_per_second=tokens_per_second(num_new_tokens, latency_seconds),
    )

    return result, metrics


def speedup(
    baseline_latency_seconds: float,
    speculative_latency_seconds: float,
) -> float:
    """
    Compute speedup as baseline latency divided by speculative latency.
    """
    if speculative_latency_seconds <= 0:
        return 0.0

    return baseline_latency_seconds / speculative_latency_seconds


def acceptance_rate(
    accepted_tokens: int,
    drafted_tokens: int,
) -> float:
    """
    Compute the fraction of draft tokens accepted by the target model.
    """
    if drafted_tokens <= 0:
        return 0.0

    return accepted_tokens / drafted_tokens
