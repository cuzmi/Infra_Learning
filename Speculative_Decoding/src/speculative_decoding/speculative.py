"""
Speculative Decoding
"""

from dataclasses import dataclass

import torch

from .models import LoadedModel, assert_tokenizers_match, encode_prompt


@dataclass
class SpeculativeDecodeStats:
    """
    Runtime stats for one speculative decoding run.
    """

    drafted_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0


def speculative_decode(
    draft: LoadedModel,
    target: LoadedModel,
    prompt: str,
    max_new_tokens: int,
    max_draft_tokens: int,
    collect_stats: bool = False,
) -> str | tuple[str, SpeculativeDecodeStats]:
    """
    Generate text with speculative decoding.
    """
    assert_tokenizers_match(draft, target)
    input_ids = encode_prompt(draft, prompt)
    stats = SpeculativeDecodeStats()

    while max_new_tokens > 0:
        num_draft_tokens = min(max_new_tokens, max_draft_tokens)
        generated_tokens, generated_probs, generated_distributions = draft_tokens_and_probs(
            draft,
            input_ids,
            num_draft_tokens,
        )

        index, target_distributions = verify_draft_tokens(
            generated_probs,
            target,
            input_ids,
            generated_tokens,
        )
        stats.drafted_tokens += generated_tokens.shape[-1]
        stats.accepted_tokens += index

        accepted_tokens = generated_tokens[:, :index]
        input_ids = torch.cat([input_ids, accepted_tokens], dim=-1)
        max_new_tokens -= accepted_tokens.shape[-1]

        if index == generated_tokens.shape[-1] or max_new_tokens <= 0:
            continue

        stats.rejected_tokens += 1
        next_token_id = sample_from_adjusted_distribution(
            target_distributions[:, index, :],
            generated_distributions[:, index, :],
        ).to(input_ids.device)

        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        max_new_tokens -= 1

    text = target.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    if collect_stats:
        return text, stats

    return text


def draft_tokens_and_probs(
    draft: LoadedModel,
    input_ids: torch.Tensor,
    num_draft_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate next n draft tokens with their token probs and full distributions.
    """
    generated_ids = []
    generated_probs = []
    generated_distributions = []
    current_ids = input_ids

    with torch.no_grad():
        for _ in range(num_draft_tokens):
            output = draft.model(current_ids)
            logits = output.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            next_token_prob = probs.gather(dim=-1, index=next_token_id)

            generated_ids.append(next_token_id)
            generated_probs.append(next_token_prob)
            generated_distributions.append(probs)
            current_ids = torch.cat([current_ids, next_token_id], dim=-1)

    return (
        torch.cat(generated_ids, dim=-1),
        torch.cat(generated_probs, dim=-1),
        torch.stack(generated_distributions, dim=1),
    )


def verify_draft_tokens(
    draft_probs: torch.Tensor,
    target: LoadedModel,
    prefix_ids: torch.Tensor,
    draft_ids: torch.Tensor,
) -> tuple[int, torch.Tensor]:
    """
    Return the first rejected draft token index and target distributions.
    """
    target_input = torch.cat([prefix_ids, draft_ids], dim=-1).to(target.device)

    with torch.no_grad():
        prefix_len = prefix_ids.shape[-1]
        draft_len = draft_ids.shape[-1]

        logits = target.model(target_input).logits
        target_logits = logits[:, prefix_len - 1 : prefix_len - 1 + draft_len, :]
        target_distributions = torch.softmax(target_logits, dim=-1)
        target_probs = target_distributions.gather(
            dim=-1,
            index=draft_ids.to(target.device).unsqueeze(-1),
        ).squeeze(-1)
        decline_idx = accept_or_reject(draft_probs, target_probs)

    return decline_idx, target_distributions


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


def sample_from_adjusted_distribution(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Sample from the corrected rejection distribution: norm(max(p - q, 0)).
    """
    draft_probs = draft_probs.to(target_probs.device)
    adjusted_probs = torch.clamp(target_probs - draft_probs, min=0.0)
    normalizer = adjusted_probs.sum(dim=-1, keepdim=True)

    adjusted_probs = torch.where(
        normalizer > 0,
        adjusted_probs / torch.clamp(normalizer, min=1e-12),
        target_probs,
    )

    return torch.multinomial(adjusted_probs, num_samples=1)
