from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


def oracle_action_indices(instance, reward_source: str = "oracle") -> np.ndarray:
    rewards_by_t = _get_reward_lists(instance, reward_source)
    return np.asarray([int(np.argmax(r)) for r in rewards_by_t], dtype=int)


def random_action_indices(instance, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(
        [int(rng.integers(0, X_t.shape[0])) for X_t in instance.X_by_t],
        dtype=int,
    )


def greedy_action_indices_from_reward_source(instance, reward_source: str = "linearized") -> np.ndarray:
    rewards_by_t = _get_reward_lists(instance, reward_source)
    return np.asarray([int(np.argmax(r)) for r in rewards_by_t], dtype=int)


def evaluate_action_indices(
    instance,
    chosen_action_indices: Sequence[int],
    *,
    evaluation_source: str = "oracle",
    realized_source: Optional[str] = None,
    ks: Iterable[int] = (1, 3),
) -> Dict[str, object]:
    """
    Evaluate a sequence of chosen arm indices.

    evaluation_source:
        source used for regret / rank / top-k calculations
    realized_source:
        source used for reported chosen reward; if None, same as evaluation_source
    """
    chosen_action_indices = np.asarray(chosen_action_indices, dtype=int)
    if len(chosen_action_indices) != instance.num_times:
        raise ValueError("chosen_action_indices must have length instance.num_times.")

    eval_rewards_by_t = _get_reward_lists(instance, evaluation_source)
    real_rewards_by_t = _get_reward_lists(instance, realized_source or evaluation_source)

    ks = sorted(set(int(k) for k in ks))
    topk_hits = {k: [] for k in ks}

    chosen_rewards = []
    oracle_rewards = []
    regrets = []
    normalized_regrets = []
    chosen_ranks = []
    oracle_action_indices = []

    for t in range(instance.num_times):
        eval_rewards = np.asarray(eval_rewards_by_t[t], dtype=float)
        real_rewards = np.asarray(real_rewards_by_t[t], dtype=float)
        a = int(chosen_action_indices[t])

        if a < 0 or a >= len(eval_rewards):
            raise IndexError(f"Chosen action index out of range at t={t}: {a}")

        order = np.argsort(-eval_rewards)
        oracle_a = int(order[0])
        oracle_action_indices.append(oracle_a)

        chosen_reward = float(real_rewards[a])
        oracle_reward_eval = float(eval_rewards[oracle_a])
        chosen_reward_eval = float(eval_rewards[a])

        regret = oracle_reward_eval - chosen_reward_eval
        reward_range = float(eval_rewards[order[0]] - eval_rewards[order[-1]])

        chosen_rank = int(np.where(order == a)[0][0]) + 1

        chosen_rewards.append(chosen_reward)
        oracle_rewards.append(float(real_rewards[oracle_a]))
        regrets.append(regret)
        normalized_regrets.append(0.0 if reward_range <= 1e-12 else regret / reward_range)
        chosen_ranks.append(chosen_rank)

        for k in ks:
            topk_hits[k].append(float(chosen_rank <= k))

    chosen_rewards = np.asarray(chosen_rewards, dtype=float)
    oracle_rewards = np.asarray(oracle_rewards, dtype=float)
    regrets = np.asarray(regrets, dtype=float)
    normalized_regrets = np.asarray(normalized_regrets, dtype=float)
    chosen_ranks = np.asarray(chosen_ranks, dtype=float)
    oracle_action_indices = np.asarray(oracle_action_indices, dtype=int)

    summary = {
        "avg_chosen_reward": float(chosen_rewards.mean()),
        "avg_oracle_reward": float(oracle_rewards.mean()),
        "avg_regret": float(regrets.mean()),
        "avg_norm_regret": float(normalized_regrets.mean()),
        "mean_chosen_rank": float(chosen_ranks.mean()),
        "median_chosen_rank": float(np.median(chosen_ranks)),
        "cumulative_regret": float(regrets.sum()),
    }
    for k in ks:
        summary[f"top{k}_rate"] = float(np.mean(topk_hits[k]))

    return {
        "summary": summary,
        "chosen_action_indices": chosen_action_indices,
        "oracle_action_indices": oracle_action_indices,
        "chosen_rewards": chosen_rewards,
        "oracle_rewards": oracle_rewards,
        "regrets": regrets,
        "normalized_regrets": normalized_regrets,
        "chosen_ranks": chosen_ranks,
    }


def evaluate_score_lists(
    instance,
    scores_by_t: Sequence[np.ndarray],
    *,
    evaluation_source: str = "oracle",
    ks: Iterable[int] = (1, 3),
) -> Dict[str, object]:
    """
    Evaluate per-time score vectors. This is useful for offline ranking diagnostics.

    It returns:
    - action-choice metrics based on argmax(score)
    - retrieval metrics of whether the true best action is in predicted top-k
    """
    if len(scores_by_t) != instance.num_times:
        raise ValueError("scores_by_t must have length instance.num_times.")

    eval_rewards_by_t = _get_reward_lists(instance, evaluation_source)
    ks = sorted(set(int(k) for k in ks))
    true_best_in_pred_topk = {k: [] for k in ks}

    chosen_action_indices = []

    for t in range(instance.num_times):
        scores = np.asarray(scores_by_t[t], dtype=float)
        eval_rewards = np.asarray(eval_rewards_by_t[t], dtype=float)

        if len(scores) != len(eval_rewards):
            raise ValueError(f"Length mismatch at t={t}: scores vs rewards.")

        pred_order = np.argsort(-scores)
        true_order = np.argsort(-eval_rewards)
        chosen_action_indices.append(int(pred_order[0]))

        true_best = int(true_order[0])
        for k in ks:
            true_best_in_pred_topk[k].append(float(true_best in pred_order[:k]))

    chosen_eval = evaluate_action_indices(
        instance,
        chosen_action_indices,
        evaluation_source=evaluation_source,
        realized_source=evaluation_source,
        ks=ks,
    )

    retrieval = {f"true_best_in_pred_top{k}": float(np.mean(true_best_in_pred_topk[k])) for k in ks}

    return {
        "choice_eval": chosen_eval,
        "retrieval_summary": retrieval,
        "chosen_action_indices": np.asarray(chosen_action_indices, dtype=int),
    }


def cumulative_sum(x: Sequence[float]) -> np.ndarray:
    return np.cumsum(np.asarray(x, dtype=float))


def _get_reward_lists(instance, source: str):
    if source == "oracle":
        return instance.oracle_rewards_by_t
    if source == "linearized":
        return instance.linearized_rewards_by_t
    if source == "active":
        return instance.rewards_by_t
    raise ValueError("source must be one of {'oracle', 'linearized', 'active'}.")