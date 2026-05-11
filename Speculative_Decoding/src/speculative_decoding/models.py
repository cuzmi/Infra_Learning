from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class LoadedModel:
    """A causal language model bundled with its tokenizer."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device


def get_device(device: str | None = None) -> torch.device:
    """Return the requested device, or choose a sensible default."""
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_tokenizer(model_name_or_path: str) -> PreTrainedTokenizerBase:
    """Load a tokenizer from a Hugging Face model id or local path."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model(
    model_name_or_path: str,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> PreTrainedModel:
    """Load a causal LM from a Hugging Face model id or local path."""
    target_device = get_device(str(device) if device is not None else None)
    model_dtype = dtype

    if model_dtype is None and target_device.type == "cuda":
        model_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=model_dtype,
    )
    model.to(target_device)
    model.eval()

    return model


def load_model_and_tokenizer(
    model_name_or_path: str,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> LoadedModel:
    """Load a causal LM and tokenizer from the same model id or local path."""
    target_device = get_device(str(device) if device is not None else None)
    tokenizer = load_tokenizer(model_name_or_path)
    model = load_model(model_name_or_path, device=target_device, dtype=dtype)

    return LoadedModel(model=model, tokenizer=tokenizer, device=target_device)


def encode_prompt(loaded_model: LoadedModel, prompt: str) -> torch.Tensor:
    """Encode a prompt and move the input ids to the model device."""
    input_ids = loaded_model.tokenizer(prompt, return_tensors="pt").input_ids
    return input_ids.to(loaded_model.device)


def assert_tokenizers_match(
    draft: LoadedModel,
    target: LoadedModel,
) -> None:
    """
    Raise if draft and target tokenizers are not compatible for speculative decoding.
    """
    draft_signature = tokenizer_signature(draft.tokenizer)
    target_signature = tokenizer_signature(target.tokenizer)

    if draft_signature != target_signature:
        raise ValueError(
            "Draft and target tokenizers must match for speculative decoding. "
            f"Got draft tokenizer {draft.tokenizer.name_or_path!r} and "
            f"target tokenizer {target.tokenizer.name_or_path!r}."
        )


def tokenizer_signature(tokenizer: PreTrainedTokenizerBase) -> dict[str, Any]:
    """
    Return the tokenizer fields that must agree for token ids to be interchangeable.
    """
    signature: dict[str, Any] = {
        "class": tokenizer.__class__.__name__,
        "length": len(tokenizer),
        "vocab": tokenizer.get_vocab(),
        "special_tokens_map": tokenizer.special_tokens_map,
    }

    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    if backend_tokenizer is not None:
        signature["backend_tokenizer"] = backend_tokenizer.to_str()

    return signature
