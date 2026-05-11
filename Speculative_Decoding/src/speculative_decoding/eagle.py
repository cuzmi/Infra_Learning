"""
Simple EAGLE-style decoding.

This module keeps the EAGLE data flow explicit:

    target hidden state + current token
    -> draft model predicts future hidden states
    -> target lm_head turns hidden states into draft tokens
    -> flatten a candidate tree
    -> target verifies candidates with tree attention when supported
    -> greedy continuous-match commit
    -> target logits at the breakpoint produce the next token

The implementation intentionally keeps the draft model simple and does not try
to implement KV-cache selection yet. The candidate tree, tree attention mask,
depth-based position ids, and longest-path commit are implemented explicitly so
the main EAGLE mechanics are visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .models import LoadedModel, assert_tokenizers_match, encode_prompt


@dataclass
class EagleDecodeStats:
    """
    Runtime stats for one simple EAGLE decoding run.
    """

    drafted_tokens: int = 0
    accepted_tokens: int = 0
    breakpoint_tokens: int = 0
    target_forwards: int = 0
    draft_forwards: int = 0
    tree_attention_verifications: int = 0
    path_fallback_verifications: int = 0


@dataclass
class EagleCandidateTree:
    """
    Flattened candidate tree.

    Nodes are stored in parent-before-child order. parent_indices[i] is -1 for a
    root candidate and otherwise points to another flattened node.
    """

    token_ids: torch.Tensor
    parent_indices: list[int]
    depths: list[int]


@dataclass
class EagleVerificationResult:
    """
    Result of greedy target verification over flattened candidates.
    """

    match_length: int
    accepted_token_ids: torch.Tensor
    accepted_node_indices: list[int]
    target_argmax_ids: torch.Tensor
    breakpoint_logits: torch.Tensor
    target_forwards: int = 1
    used_tree_attention: bool = True


class EagleDraftModel(Protocol):
    """
    Interface for an EAGLE draft head/model.

    A real EAGLE model should predict future hidden states from the latest
    target hidden state and current token. The prefix_ids argument is included
    for this simple implementation and can be ignored by a trained head.
    """

    def predict_future_hidden_states(
        self,
        prefix_ids: torch.Tensor,
        target_hidden_state: torch.Tensor,
        current_token_id: torch.Tensor,
        num_draft_tokens: int,
    ) -> torch.Tensor:
        """
        Return future hidden states shaped [batch, num_draft_tokens, hidden].
        """


class SimpleEagleDraftModel:
    """
    Small stand-in for a trained EAGLE draft model.

    Instead of learning the hidden-state transition, it runs a normal causal LM
    and collects the last hidden state at each draft step. The target lm_head is
    still used to turn those hidden states into draft tokens, which keeps the
    EAGLE verification path visible.
    """

    def __init__(
        self,
        loaded_model: LoadedModel,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0")

        self.loaded_model = loaded_model
        self.do_sample = do_sample
        self.temperature = temperature
        self.forward_count = 0

    def predict_future_hidden_states(
        self,
        prefix_ids: torch.Tensor,
        target_hidden_state: torch.Tensor,
        current_token_id: torch.Tensor,
        num_draft_tokens: int,
    ) -> torch.Tensor:
        del target_hidden_state, current_token_id

        current_ids = prefix_ids.to(self.loaded_model.device)
        hidden_states = []

        with torch.no_grad():
            for _ in range(num_draft_tokens):
                outputs = self.loaded_model.model(
                    current_ids,
                    output_hidden_states=True,
                )
                self.forward_count += 1

                step_hidden = outputs.hidden_states[-1][:, -1, :]
                hidden_states.append(step_hidden)

                logits = outputs.logits[:, -1, :]
                if self.do_sample:
                    probs = torch.softmax(logits / self.temperature, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

                current_ids = torch.cat([current_ids, next_token_id], dim=-1)

        return torch.stack(hidden_states, dim=1)


def eagle_decode(
    eagle_draft: EagleDraftModel,
    target: LoadedModel,
    prompt: str,
    max_new_tokens: int,
    max_draft_tokens: int,
    tree_top_k: int = 2,
    collect_stats: bool = False,
) -> str | tuple[str, EagleDecodeStats]:
    """
    Generate text with a simple EAGLE-style greedy decoder.

    This function is deterministic because verification uses target argmax
    matching and the breakpoint token is also selected with argmax.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be greater than or equal to 0")
    if max_draft_tokens <= 0:
        raise ValueError("max_draft_tokens must be greater than 0")
    if tree_top_k <= 0:
        raise ValueError("tree_top_k must be greater than 0")

    input_ids = encode_prompt(target, prompt)
    stats = EagleDecodeStats()
    generated_new_tokens = 0

    while generated_new_tokens < max_new_tokens:
        remaining_tokens = max_new_tokens - generated_new_tokens
        num_draft_tokens = min(remaining_tokens, max_draft_tokens)

        target_hidden_state, current_token_id = latest_target_state(target, input_ids)
        stats.target_forwards += 1

        draft_start_forwards = getattr(eagle_draft, "forward_count", 0)
        future_hidden_states = eagle_draft.predict_future_hidden_states(
            input_ids,
            target_hidden_state,
            current_token_id,
            num_draft_tokens,
        )
        stats.draft_forwards += (
            getattr(eagle_draft, "forward_count", 0) - draft_start_forwards
        )

        tree = build_candidate_tree(
            target,
            future_hidden_states,
            top_k=tree_top_k,
        )
        verification = verify_eagle_tree_greedy(target, input_ids, tree)
        stats.target_forwards += verification.target_forwards
        if verification.used_tree_attention:
            stats.tree_attention_verifications += 1
        else:
            stats.path_fallback_verifications += 1

        stats.drafted_tokens += tree.token_ids.shape[-1]
        stats.accepted_tokens += verification.match_length

        accepted_ids = verification.accepted_token_ids.to(input_ids.device)
        tokens_to_commit = min(accepted_ids.shape[-1], remaining_tokens)
        if tokens_to_commit > 0:
            input_ids = torch.cat([input_ids, accepted_ids[:, :tokens_to_commit]], dim=-1)
            generated_new_tokens += tokens_to_commit

        if generated_new_tokens >= max_new_tokens:
            break

        # If the draft path breaks, or if a whole draft chunk matched, the
        # target logits at that breakpoint provide the next committed token.
        breakpoint_token_id = torch.argmax(
            verification.breakpoint_logits,
            dim=-1,
            keepdim=True,
        ).to(input_ids.device)
        input_ids = torch.cat([input_ids, breakpoint_token_id], dim=-1)
        stats.breakpoint_tokens += 1
        generated_new_tokens += 1

    text = target.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    if collect_stats:
        return text, stats

    return text


def eagle_decode_with_loaded_draft(
    draft: LoadedModel,
    target: LoadedModel,
    prompt: str,
    max_new_tokens: int,
    max_draft_tokens: int,
    tree_top_k: int = 2,
    collect_stats: bool = False,
) -> str | tuple[str, EagleDecodeStats]:
    """
    Convenience wrapper using a normal causal LM as the simple EAGLE draft.
    """
    assert_tokenizers_match(draft, target)
    eagle_draft = SimpleEagleDraftModel(draft)
    return eagle_decode(
        eagle_draft=eagle_draft,
        target=target,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        max_draft_tokens=max_draft_tokens,
        tree_top_k=tree_top_k,
        collect_stats=collect_stats,
    )


def latest_target_state(
    target: LoadedModel,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return the latest target hidden state and current token id.
    """
    with torch.no_grad():
        outputs = target.model(
            input_ids.to(target.device),
            output_hidden_states=True,
        )

    hidden_state = outputs.hidden_states[-1][:, -1, :]
    current_token_id = input_ids[:, -1:].to(target.device)
    return hidden_state, current_token_id


def build_candidate_tree(
    target: LoadedModel,
    future_hidden_states: torch.Tensor,
    top_k: int = 2,
) -> EagleCandidateTree:
    """
    Convert predicted future hidden states into a flattened top-k token tree.

    The simple draft predicts one hidden state per depth, not one hidden state
    per branch. We therefore use the depth-wise top-k tokens from target lm_head
    and expand every node at the previous depth with those candidates. A trained
    EAGLE head can later replace this with branch-conditioned hidden states.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    lm_head = target.model.get_output_embeddings()
    if lm_head is None:
        raise ValueError("target model does not expose output embeddings/lm_head")

    future_hidden_states = future_hidden_states.to(target.device)
    expected_hidden_size = lm_head.weight.shape[-1]
    actual_hidden_size = future_hidden_states.shape[-1]
    if actual_hidden_size != expected_hidden_size:
        raise ValueError(
            "EAGLE draft hidden size must match target lm_head input size. "
            f"Got draft hidden size {actual_hidden_size}, target size {expected_hidden_size}."
        )

    with torch.no_grad():
        logits = lm_head(future_hidden_states)
        vocab_size = logits.shape[-1]
        k = min(top_k, vocab_size)
        top_token_ids = torch.topk(logits, k=k, dim=-1).indices

    if top_token_ids.shape[0] != 1:
        raise ValueError("simple EAGLE currently supports batch size 1")

    token_values: list[int] = []
    parent_indices: list[int] = []
    depths: list[int] = []
    previous_depth_nodes: list[int] = []

    for depth in range(top_token_ids.shape[1]):
        current_depth_nodes = []
        depth_tokens = top_token_ids[0, depth].tolist()

        if depth == 0:
            parents = [-1]
        else:
            parents = previous_depth_nodes

        for parent_idx in parents:
            for token_id in depth_tokens:
                node_idx = len(token_values)
                token_values.append(int(token_id))
                parent_indices.append(parent_idx)
                depths.append(depth)
                current_depth_nodes.append(node_idx)

        previous_depth_nodes = current_depth_nodes

    token_ids = torch.tensor(
        [token_values],
        dtype=torch.long,
        device=target.device,
    )

    return EagleCandidateTree(
        token_ids=token_ids,
        parent_indices=parent_indices,
        depths=depths,
    )


def verify_eagle_tree_greedy(
    target: LoadedModel,
    prefix_ids: torch.Tensor,
    tree: EagleCandidateTree,
) -> EagleVerificationResult:
    """
    Verify flattened candidates with greedy argmax matching.

    The fast path uses a 4D tree attention mask so every flattened tree node can
    attend only to the prefix and its ancestors. Some Hugging Face model classes
    do not accept this mask shape, so the fallback verifies each path separately
    with ordinary causal attention. The fallback is slower but semantically
    matches tree attention.
    """
    try:
        return verify_tree_with_attention(target, prefix_ids, tree)
    except (RuntimeError, TypeError, ValueError):
        return verify_tree_by_paths_greedy(target, prefix_ids, tree)


def verify_tree_with_attention(
    target: LoadedModel,
    prefix_ids: torch.Tensor,
    tree: EagleCandidateTree,
) -> EagleVerificationResult:
    """
    Verify the flattened tree in one target forward with a tree attention mask.
    """
    candidate_ids = tree.token_ids.to(target.device)
    target_input = torch.cat([prefix_ids.to(target.device), candidate_ids], dim=-1)

    with torch.no_grad():
        prefix_len = prefix_ids.shape[-1]
        attention_mask = build_tree_attention_mask(
            prefix_len=prefix_len,
            tree=tree,
            device=target.device,
            dtype=target.model.dtype,
        )
        position_ids = build_tree_position_ids(
            prefix_len=prefix_len,
            tree=tree,
            device=target.device,
        )
        outputs = target.model(
            target_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = outputs.logits

        prediction_logits = gather_tree_prediction_logits(logits, prefix_len, tree)
        node_output_logits = logits[
            :,
            prefix_len : prefix_len + candidate_ids.shape[-1],
            :,
        ]
        target_argmax_ids = torch.argmax(prediction_logits, dim=-1)
        best = select_longest_matching_path(
            tree,
            prediction_logits,
            node_output_logits,
            target_argmax_ids,
        )

    return EagleVerificationResult(
        match_length=best.match_length,
        accepted_token_ids=best.accepted_token_ids,
        accepted_node_indices=best.accepted_node_indices,
        target_argmax_ids=target_argmax_ids,
        breakpoint_logits=best.breakpoint_logits,
        target_forwards=1,
        used_tree_attention=True,
    )


def verify_tree_by_paths_greedy(
    target: LoadedModel,
    prefix_ids: torch.Tensor,
    tree: EagleCandidateTree,
) -> EagleVerificationResult:
    """
    Verify every root-to-leaf path separately when tree attention is unavailable.
    """
    prefix_ids = prefix_ids.to(target.device)
    paths = root_to_leaf_paths(tree)
    node_argmax: dict[int, torch.Tensor] = {}
    best_result: PathMatchResult | None = None
    target_forwards = 0

    with torch.no_grad():
        for path in paths:
            path_token_ids = tokens_for_path(tree, path).to(target.device)
            target_input = torch.cat([prefix_ids, path_token_ids], dim=-1)
            outputs = target.model(target_input)
            target_forwards += 1

            prefix_len = prefix_ids.shape[-1]
            path_len = len(path)
            path_logits = outputs.logits[:, prefix_len - 1 : prefix_len - 1 + path_len, :]
            path_argmax_ids = torch.argmax(path_logits, dim=-1)

            for offset, node_idx in enumerate(path):
                node_argmax.setdefault(node_idx, path_argmax_ids[:, offset])

            match_length = continuous_match_length(path_token_ids == path_argmax_ids)
            if match_length < path_len:
                breakpoint_logits = path_logits[:, match_length, :]
            else:
                breakpoint_logits = outputs.logits[:, -1, :]

            accepted_node_indices = path[:match_length]
            accepted_token_ids = path_token_ids[:, :match_length]
            current_result = PathMatchResult(
                match_length=match_length,
                accepted_token_ids=accepted_token_ids,
                accepted_node_indices=accepted_node_indices,
                breakpoint_logits=breakpoint_logits,
            )

            if best_result is None or current_result.match_length > best_result.match_length:
                best_result = current_result

    if best_result is None:
        raise ValueError("candidate tree must contain at least one path")

    target_argmax_ids = torch.empty_like(tree.token_ids, device=target.device)
    for node_idx in range(tree.token_ids.shape[-1]):
        target_argmax_ids[:, node_idx] = node_argmax[node_idx]

    return EagleVerificationResult(
        match_length=best_result.match_length,
        accepted_token_ids=best_result.accepted_token_ids,
        accepted_node_indices=best_result.accepted_node_indices,
        target_argmax_ids=target_argmax_ids,
        breakpoint_logits=best_result.breakpoint_logits,
        target_forwards=target_forwards,
        used_tree_attention=False,
    )


@dataclass
class PathMatchResult:
    """
    Best matching path selected from the candidate tree.
    """

    match_length: int
    accepted_token_ids: torch.Tensor
    accepted_node_indices: list[int]
    breakpoint_logits: torch.Tensor


def build_tree_attention_mask(
    prefix_len: int,
    tree: EagleCandidateTree,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Return an additive 4D attention mask for prefix + flattened tree nodes.

    Allowed entries are 0.0. Blocked entries use the minimum finite value for
    the model dtype, matching the convention used inside transformer models.
    """
    num_nodes = tree.token_ids.shape[-1]
    total_len = prefix_len + num_nodes
    allowed = torch.zeros((total_len, total_len), dtype=torch.bool, device=device)

    for row in range(prefix_len):
        allowed[row, : row + 1] = True

    ancestor_sets = ancestor_node_sets(tree)
    for node_idx in range(num_nodes):
        row = prefix_len + node_idx
        allowed[row, :prefix_len] = True
        for ancestor_idx in ancestor_sets[node_idx]:
            allowed[row, prefix_len + ancestor_idx] = True
        allowed[row, prefix_len + node_idx] = True

    blocked_value = torch.finfo(dtype).min
    mask = torch.full((total_len, total_len), blocked_value, dtype=dtype, device=device)
    mask = mask.masked_fill(allowed, 0.0)
    return mask.unsqueeze(0).unsqueeze(0)


def build_tree_position_ids(
    prefix_len: int,
    tree: EagleCandidateTree,
    device: torch.device,
) -> torch.Tensor:
    """
    Return position ids where tree node positions are based on tree depth.
    """
    prefix_positions = list(range(prefix_len))
    tree_positions = [prefix_len + depth for depth in tree.depths]
    return torch.tensor(
        [prefix_positions + tree_positions],
        dtype=torch.long,
        device=device,
    )


def gather_tree_prediction_logits(
    logits: torch.Tensor,
    prefix_len: int,
    tree: EagleCandidateTree,
) -> torch.Tensor:
    """
    Gather the logits that predict each flattened tree node.

    A root node is predicted by the final prefix position. A non-root node is
    predicted by its parent node's output position.
    """
    prediction_positions = []
    for parent_idx in tree.parent_indices:
        if parent_idx == -1:
            prediction_positions.append(prefix_len - 1)
        else:
            prediction_positions.append(prefix_len + parent_idx)

    position_tensor = torch.tensor(
        prediction_positions,
        dtype=torch.long,
        device=logits.device,
    )
    return logits.index_select(dim=1, index=position_tensor)


def select_longest_matching_path(
    tree: EagleCandidateTree,
    prediction_logits: torch.Tensor,
    node_output_logits: torch.Tensor,
    target_argmax_ids: torch.Tensor,
) -> PathMatchResult:
    """
    Select the root-to-leaf path with the longest greedy prefix match.
    """
    best_result: PathMatchResult | None = None
    paths = root_to_leaf_paths(tree)

    for path in paths:
        path_token_ids = tokens_for_path(tree, path).to(prediction_logits.device)
        path_argmax_ids = target_argmax_ids[:, path]
        match_length = continuous_match_length(path_token_ids == path_argmax_ids)

        if match_length < len(path):
            next_node_idx = path[match_length]
            breakpoint_logits = prediction_logits[:, next_node_idx, :]
        else:
            last_node_idx = path[-1]
            breakpoint_logits = node_output_logits[:, last_node_idx, :]

        current_result = PathMatchResult(
            match_length=match_length,
            accepted_token_ids=path_token_ids[:, :match_length],
            accepted_node_indices=path[:match_length],
            breakpoint_logits=breakpoint_logits,
        )

        if best_result is None or current_result.match_length > best_result.match_length:
            best_result = current_result

    if best_result is None:
        raise ValueError("candidate tree must contain at least one path")

    return best_result


def root_to_leaf_paths(tree: EagleCandidateTree) -> list[list[int]]:
    """
    Return every root-to-leaf path in flattened node indices.
    """
    children: dict[int, list[int]] = {idx: [] for idx in range(tree.token_ids.shape[-1])}
    roots = []
    for node_idx, parent_idx in enumerate(tree.parent_indices):
        if parent_idx == -1:
            roots.append(node_idx)
        else:
            children[parent_idx].append(node_idx)

    paths: list[list[int]] = []

    def visit(node_idx: int, prefix: list[int]) -> None:
        path = prefix + [node_idx]
        if not children[node_idx]:
            paths.append(path)
            return
        for child_idx in children[node_idx]:
            visit(child_idx, path)

    for root_idx in roots:
        visit(root_idx, [])

    return paths


def ancestor_node_sets(tree: EagleCandidateTree) -> list[set[int]]:
    """
    Return ancestor node indices for every flattened tree node.
    """
    ancestors: list[set[int]] = []
    for node_idx in range(tree.token_ids.shape[-1]):
        node_ancestors = set()
        parent_idx = tree.parent_indices[node_idx]
        while parent_idx != -1:
            node_ancestors.add(parent_idx)
            parent_idx = tree.parent_indices[parent_idx]
        ancestors.append(node_ancestors)

    return ancestors


def tokens_for_path(tree: EagleCandidateTree, path: list[int]) -> torch.Tensor:
    """
    Return token ids for a root-to-leaf path.
    """
    path_indices = torch.tensor(
        path,
        dtype=torch.long,
        device=tree.token_ids.device,
    )
    return tree.token_ids.index_select(dim=1, index=path_indices)


def continuous_match_length(matches: torch.Tensor) -> int:
    """
    Return the number of consecutive True values from the first position.
    """
    for idx, matched in enumerate(matches[0]):
        if not bool(matched):
            return idx

    return matches.shape[-1]
