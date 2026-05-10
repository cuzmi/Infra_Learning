"""
Configuration objects for the speculative decoding demo.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Model loading configuration.
    """

    draft_model: str = "distilgpt2"
    target_model: str = "gpt2"
    device: str | None = None


@dataclass
class GenerationConfig:
    """
    Text generation configuration.
    """

    prompt: str = "The future of artificial intelligence is"
    max_new_tokens: int = 30
    max_draft_tokens: int = 5
    temperature: float = 1.0


@dataclass
class DemoConfig:
    """
    Full demo configuration.
    """

    models: ModelConfig
    generation: GenerationConfig


def default_config() -> DemoConfig:
    """
    Return the default demo configuration.
    """
    return DemoConfig(
        models=ModelConfig(),
        generation=GenerationConfig(),
    )
