"""
Baseline autoregressive decoding.
Token & content
"""

import torch

from .models import LoadedModel, encode_prompt


def next_token_logits(
    loaded_model: LoadedModel,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Return logits for the next token.
    """
    with torch.no_grad():
        outputs = loaded_model.model(input_ids.to(loaded_model.device))
        return outputs.logits[:, -1, :]


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Sample one token from next-token logits.
    """
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def greedy_next_token(logits: torch.Tensor) -> torch.Tensor:
    """
    Select the highest-probability next token.
    """
    return torch.argmax(logits, dim=-1, keepdim=True)


def generate_tokens(
    loaded_model: LoadedModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    do_sample: bool = True,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Generate token ids autoregressively.
    """
    current_ids = input_ids.to(loaded_model.device)

    for _ in range(max_new_tokens):
        logits = next_token_logits(loaded_model, current_ids)

        if do_sample:
            next_token_id = sample_next_token(logits, temperature=temperature)
        else:
            next_token_id = greedy_next_token(logits)

        current_ids = torch.cat([current_ids, next_token_id], dim=-1)

    return current_ids


def sample_decode(
    loaded_model: LoadedModel,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> str:
    """
    Decode text using autoregressive sampling.
    """
    input_ids = encode_prompt(loaded_model, prompt)
    output_ids = generate_tokens(
        loaded_model,
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
    )
    return loaded_model.tokenizer.decode(output_ids[0], skip_special_tokens=True)


def greedy_decode(
    loaded_model: LoadedModel,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """
    Decode text using greedy autoregressive decoding.
    """
    input_ids = encode_prompt(loaded_model, prompt)
    output_ids = generate_tokens(
        loaded_model,
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return loaded_model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
