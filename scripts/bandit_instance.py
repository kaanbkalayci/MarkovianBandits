from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd


RewardMode = Literal["oracle", "linearized"]


@dataclass
class EmbeddingRun:
    run_dir: Path
    examples_df: pd.DataFrame
    ridge_model: Any
    meta: Dict[str, Any]


@dataclass
class BanditInstance:
    times: np.ndarray
    X_by_t: List[np.ndarray]                 # arm features at each time (possibly with intercept column)
    X_base_by_t: List[np.ndarray]            # learned embeddings only, no intercept
    oracle_rewards_by_t: List[np.ndarray]    # utility from examples_df["utility"]
    linearized_rewards_by_t: List[np.ndarray]  # ridge prediction on learned embedding
    rewards_by_t: List[np.ndarray]           # chosen reward source for this instance
    misspec_by_t: List[np.ndarray]           # oracle - linearized
    subset_strs_by_t: List[List[str]]
    subsets_by_t: List[List[Any]]
    theta_star: np.ndarray                   # includes intercept if include_intercept=True
    reward_mode: str
    feature_dim: int
    num_times: int
    num_actions_per_time: np.ndarray
    meta: Dict[str, Any]
    run_dir: Optional[Path] = None


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def load_embedding_run(run_dir: str | Path) -> EmbeddingRun:
    run_dir = Path(run_dir)

    examples_path = _require_file(run_dir / "examples_with_embed.pkl")
    ridge_path = _require_file(run_dir / "ridge_model.pkl")
    meta_path = _require_file(run_dir / "meta.json")

    examples_df = pd.read_pickle(examples_path)

    with open(ridge_path, "rb") as f:
        ridge_model = pickle.load(f)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    return EmbeddingRun(
        run_dir=run_dir,
        examples_df=examples_df,
        ridge_model=ridge_model,
        meta=meta,
    )


def extract_theta_star(ridge_model: Any, include_intercept: bool = True) -> np.ndarray:
    coef = np.asarray(ridge_model.coef_, dtype=float).reshape(-1)
    intercept = float(np.asarray(ridge_model.intercept_).reshape(()))

    if include_intercept:
        return np.concatenate([[intercept], coef])
    return coef


def add_intercept_column(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    ones = np.ones((X.shape[0], 1), dtype=float)
    return np.concatenate([ones, X], axis=1)


def predict_linearized_rewards(X_base: np.ndarray, ridge_model: Any) -> np.ndarray:
    intercept = float(np.asarray(ridge_model.intercept_).reshape(()))
    coef = np.asarray(ridge_model.coef_, dtype=float).reshape(-1)
    return intercept + X_base @ coef


def build_bandit_instance(
    examples_df: pd.DataFrame,
    ridge_model: Any,
    *,
    reward_mode: RewardMode = "oracle",
    include_intercept: bool = True,
    sort_rows: bool = True,
    meta: Optional[Dict[str, Any]] = None,
    run_dir: Optional[str | Path] = None,
) -> BanditInstance:
    if reward_mode not in {"oracle", "linearized"}:
        raise ValueError("reward_mode must be 'oracle' or 'linearized'.")

    if "datetime" not in examples_df.columns:
        raise ValueError("examples_df must contain a 'datetime' column.")
    if "utility" not in examples_df.columns:
        raise ValueError("examples_df must contain a 'utility' column.")
    if "learned_embed" not in examples_df.columns:
        raise ValueError("examples_df must contain a 'learned_embed' column.")

    df = examples_df.copy()

    if sort_rows:
        sort_cols = ["datetime"]
        if "subset_size" in df.columns:
            sort_cols.append("subset_size")
        if "subset_str" in df.columns:
            sort_cols.append("subset_str")
        df = df.sort_values(sort_cols).reset_index(drop=True)

    theta_star = extract_theta_star(ridge_model, include_intercept=include_intercept)

    times: List[Any] = []
    X_by_t: List[np.ndarray] = []
    X_base_by_t: List[np.ndarray] = []
    oracle_rewards_by_t: List[np.ndarray] = []
    linearized_rewards_by_t: List[np.ndarray] = []
    rewards_by_t: List[np.ndarray] = []
    misspec_by_t: List[np.ndarray] = []
    subset_strs_by_t: List[List[str]] = []
    subsets_by_t: List[List[Any]] = []
    num_actions_per_time: List[int] = []

    for dt, g in df.groupby("datetime", sort=False):
        X_base = np.stack(g["learned_embed"].to_numpy()).astype(float)
        X = add_intercept_column(X_base) if include_intercept else X_base

        oracle_rewards = g["utility"].to_numpy(dtype=float)
        linearized_rewards = predict_linearized_rewards(X_base, ridge_model)
        rewards = oracle_rewards if reward_mode == "oracle" else linearized_rewards
        misspec = oracle_rewards - linearized_rewards

        times.append(dt)
        X_by_t.append(X)
        X_base_by_t.append(X_base)
        oracle_rewards_by_t.append(oracle_rewards)
        linearized_rewards_by_t.append(linearized_rewards)
        rewards_by_t.append(rewards)
        misspec_by_t.append(misspec)

        if "subset_str" in g.columns:
            subset_strs_by_t.append(list(g["subset_str"].astype(str)))
        else:
            subset_strs_by_t.append([str(i) for i in range(len(g))])

        if "subset" in g.columns:
            subsets_by_t.append(list(g["subset"]))
        else:
            subsets_by_t.append(list(range(len(g))))

        num_actions_per_time.append(len(g))

    feature_dim = int(X_by_t[0].shape[1]) if X_by_t else 0

    return BanditInstance(
        times=np.array(times),
        X_by_t=X_by_t,
        X_base_by_t=X_base_by_t,
        oracle_rewards_by_t=oracle_rewards_by_t,
        linearized_rewards_by_t=linearized_rewards_by_t,
        rewards_by_t=rewards_by_t,
        misspec_by_t=misspec_by_t,
        subset_strs_by_t=subset_strs_by_t,
        subsets_by_t=subsets_by_t,
        theta_star=theta_star,
        reward_mode=reward_mode,
        feature_dim=feature_dim,
        num_times=len(times),
        num_actions_per_time=np.asarray(num_actions_per_time, dtype=int),
        meta={} if meta is None else meta,
        run_dir=None if run_dir is None else Path(run_dir),
    )


def build_bandit_instance_from_run(
    run_dir: str | Path,
    *,
    reward_mode: RewardMode = "oracle",
    include_intercept: bool = True,
    sort_rows: bool = True,
) -> BanditInstance:
    run = load_embedding_run(run_dir)
    return build_bandit_instance(
        run.examples_df,
        run.ridge_model,
        reward_mode=reward_mode,
        include_intercept=include_intercept,
        sort_rows=sort_rows,
        meta=run.meta,
        run_dir=run.run_dir,
    )


def summarize_instance(instance: BanditInstance) -> Dict[str, Any]:
    oracle_flat = np.concatenate(instance.oracle_rewards_by_t)
    linearized_flat = np.concatenate(instance.linearized_rewards_by_t)
    misspec_flat = np.concatenate(instance.misspec_by_t)

    return {
        "num_times": instance.num_times,
        "feature_dim": instance.feature_dim,
        "num_actions_min": int(instance.num_actions_per_time.min()),
        "num_actions_max": int(instance.num_actions_per_time.max()),
        "num_actions_mean": float(instance.num_actions_per_time.mean()),
        "reward_mode": instance.reward_mode,
        "oracle_reward_mean": float(oracle_flat.mean()),
        "oracle_reward_std": float(oracle_flat.std()),
        "linearized_reward_mean": float(linearized_flat.mean()),
        "linearized_reward_std": float(linearized_flat.std()),
        "misspec_mae": float(np.mean(np.abs(misspec_flat))),
        "misspec_rmse": float(np.sqrt(np.mean(misspec_flat ** 2))),
        "misspec_abs_p95": float(np.quantile(np.abs(misspec_flat), 0.95)),
        "misspec_abs_p99": float(np.quantile(np.abs(misspec_flat), 0.99)),
        "misspec_abs_max": float(np.max(np.abs(misspec_flat))),
    }