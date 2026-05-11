"""
Speculative decoding demo package.
"""

from .baseline import greedy_decode, sample_decode
from .models import LoadedModel, assert_tokenizers_match, encode_prompt, load_model_and_tokenizer
from .speculative import speculative_decode

__all__ = [
    "LoadedModel",
    "assert_tokenizers_match",
    "encode_prompt",
    "greedy_decode",
    "load_model_and_tokenizer",
    "sample_decode",
    "speculative_decode",
]
