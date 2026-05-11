"""
Speculative decoding demo package.
"""

from .baseline import greedy_decode, sample_decode
from .eagle import (
    EagleCandidateTree,
    EagleDecodeStats,
    SimpleEagleDraftModel,
    build_candidate_tree,
    build_tree_attention_mask,
    build_tree_position_ids,
    eagle_decode,
    eagle_decode_with_loaded_draft,
)
from .models import LoadedModel, assert_tokenizers_match, encode_prompt, load_model_and_tokenizer
from .speculative import SpeculativeDecodeStats, speculative_decode

__all__ = [
    "EagleDecodeStats",
    "EagleCandidateTree",
    "LoadedModel",
    "SpeculativeDecodeStats",
    "SimpleEagleDraftModel",
    "assert_tokenizers_match",
    "build_candidate_tree",
    "build_tree_attention_mask",
    "build_tree_position_ids",
    "encode_prompt",
    "eagle_decode",
    "eagle_decode_with_loaded_draft",
    "greedy_decode",
    "load_model_and_tokenizer",
    "sample_decode",
    "speculative_decode",
]
