"""
Speculative Decoding
"""

import torch

from .models import LoadedModel, encode_prompt


def speculative_decode(
    draft: LoadedModel,
    target: LoadedModel,
    prompt: str,
    max_new_tokens: int,
    max_draft_tokens: int,
) -> str:
    """
    Generate text with speculative decoding.
    """
    input_ids = encode_prompt(draft, prompt)

    while max_new_tokens > 0:
        num_draft_tokens = min(max_new_tokens, max_draft_tokens)
        generated_tokens, generated_probs = draft_tokens_and_probs(
            draft,
            input_ids,
            num_draft_tokens,
        )

        index = verify_draft_tokens(generated_probs, target, input_ids, generated_tokens)
        accepted_tokens = generated_tokens[:, :index]
        input_ids = torch.cat([input_ids, accepted_tokens], dim=-1)
        max_new_tokens -= accepted_tokens.shape[-1]

        if index == generated_tokens.shape[-1] or max_new_tokens <= 0:
            continue
        
        # TODO: replaced by distribution - [By: Weijie] - 2026/05/10
        continue_tokens = min(max_new_tokens, max_draft_tokens - index + 1)
        tokens = target_tokens(target, input_ids, continue_tokens)

        input_ids = torch.cat([input_ids, tokens], dim=-1)
        max_new_tokens -= tokens.shape[-1]

    return target.tokenizer.decode(input_ids[0], skip_special_tokens=True)


def draft_tokens_and_probs(
    draft: LoadedModel,
    input_ids: torch.Tensor,
    num_draft_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate next n draft tokens with their probs.
    """
    generated_ids = []
    generated_probs = []
    current_ids = input_ids

    with torch.no_grad():
        for _ in range(num_draft_tokens):
            output = draft.model(current_ids)
            logits = output.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            # TODO: switch to a configurable sampling strategy.
            next_token_id = torch.multinomial(probs, num_samples=1)
            next_token_prob = probs.gather(dim=-1, index=next_token_id)

            generated_ids.append(next_token_id)
            generated_probs.append(next_token_prob)
            current_ids = torch.cat([current_ids, next_token_id], dim=-1)

    return torch.cat(generated_ids, dim=-1), torch.cat(generated_probs, dim=-1)


def verify_draft_tokens(
    draft_probs: torch.Tensor,
    target: LoadedModel,
    prefix_ids: torch.Tensor,
    draft_ids: torch.Tensor,
) -> int:
    """
    Return the first rejected draft token index.
    """
    target_input = torch.cat([prefix_ids, draft_ids], dim=-1).to(target.device)

    with torch.no_grad():
        prefix_len = prefix_ids.shape[-1]
        draft_len = draft_ids.shape[-1]

        logits = target.model(target_input).logits
        target_logits = logits[:, prefix_len - 1 : prefix_len - 1 + draft_len, :]
        logits_probs = torch.softmax(target_logits, dim=-1)
        target_probs = logits_probs.gather(
            dim=-1,
            index=draft_ids.to(target.device).unsqueeze(-1),
        ).squeeze(-1)
        decline_idx = accept_or_reject(draft_probs, target_probs)

    return decline_idx


def accept_or_reject(
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
) -> int:
    """
    Return the first rejected token index.

    accept_prob = min(1, p(x) / q(x))
    """
    draft_probs = draft_probs.to(target_probs.device)

    accept_probs = torch.minimum(
        torch.ones_like(target_probs),
        target_probs / torch.clamp(draft_probs, min=1e-12),
    )

    random_values = torch.rand_like(accept_probs)
    accepts = random_values <= accept_probs

    for idx, accepted in enumerate(accepts[0]):
        if not bool(accepted):
            return idx

    return accepts.shape[-1]


def target_tokens(
    target: LoadedModel,
    input_ids: torch.Tensor,
    continue_tokens: int,
) -> torch.Tensor:
    generated_tokens = []
    current_ids = input_ids.to(target.device)

    if continue_tokens <= 0:
        return torch.empty((input_ids.shape[0], 0), dtype=input_ids.dtype, device=input_ids.device)

    with torch.no_grad():
        for _ in range(continue_tokens):
            logits = target.model(current_ids).logits
            output = logits[:, -1, :]
            probs = torch.softmax(output, dim=-1)

            next_token_id = torch.multinomial(probs, num_samples=1)

            generated_tokens.append(next_token_id)
            current_ids = torch.cat([current_ids, next_token_id], dim=-1)

    return torch.cat(generated_tokens, dim=-1).to(input_ids.device)
