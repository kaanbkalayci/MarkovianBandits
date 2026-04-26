import math
from itertools import combinations
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ============================================================
# Utility helpers
# ============================================================

def _safe_std(x: np.ndarray) -> float:
    if len(x) <= 1:
        return 0.0
    return float(np.std(x, ddof=0))


def _linear_slope(x: np.ndarray) -> float:
    """
    Simple slope of y against equally spaced time indices 0,1,...,n-1.
    Returns 0.0 for length < 2.
    """
    n = len(x)
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=float)
    y = np.asarray(x, dtype=float)
    t_mean = t.mean()
    y_mean = y.mean()
    denom = np.sum((t - t_mean) ** 2)
    if denom <= 0:
        return 0.0
    return float(np.sum((t - t_mean) * (y - y_mean)) / denom)


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x / max(temperature, 1e-8)
    x = x - np.nanmax(x)
    ex = np.exp(x)
    s = np.nansum(ex)
    if s <= 0:
        return np.ones_like(x) / len(x)
    return ex / s


def _pad_list(vals: List[float], target_len: int, pad_value: float = 0.0) -> List[float]:
    vals = list(vals)
    if len(vals) >= target_len:
        return vals[:target_len]
    return vals + [pad_value] * (target_len - len(vals))


def _triangle_area(points: List[Tuple[float, float]]) -> float:
    """
    Area of triangle formed by 3 points. Returns 0 if not exactly 3 points.
    """
    if len(points) != 3:
        return 0.0
    (x1, y1), (x2, y2), (x3, y3) = points
    return float(abs(0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))))


def _normalize_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    denom = s.max() - s.min()
    if denom <= 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / denom


def _node_col(node: int) -> str:
    return f"rpi{node}"


def _dist_col(node: int) -> str:
    return f"distance_to_{node}"


def _subset_to_str(subset: Tuple[int, ...]) -> str:
    return "-".join(map(str, subset))


# ============================================================
# Main builder
# ============================================================

def build_bandit_dataset(
    gdf_cleaned: pd.DataFrame,
    gdf_nodes: pd.DataFrame,
    valid_indices: List[int],
    node_list: List[int],
    *,
    history_steps: int = 5,
    max_subset_size: int = 3,
    trigger_prob_threshold: float = 0.12,
    trigger_min_nodes: int = 2,
    trigger_max_nodes: int = 5,
    trigger_temperature: float = 4.0,
    utility_second_weight: float = 0.3,
    rho: Optional[float] = None,
) -> Dict[str, object]:
    """
    Builds a bandit-ready dataset from the cleaned synchronized dataframe.

    Inputs
    ------
    gdf_cleaned:
        Output from new_dataloader.load_data(...). Must contain:
            - DatetimeIndex
            - geometry column
            - rpi{node} columns in dB
            - distance_to_{node} columns in meters
    gdf_nodes:
        Sensor metadata in projected coordinates. Must contain:
            - "Node #" column
            - geometry column
    valid_indices:
        Valid chain indices from new_dataloader
    node_list:
        Kept nodes, e.g. [1,2,3,4,5,7,8,9,10]

    Returns
    -------
    dict with:
        - "sequence_df": one row per valid time, with context vectors and triggered sets
        - "examples_df": one row per (time, admissible subset)
        - "meta": dimensions and configuration
    """
    if len(valid_indices) == 0:
        raise ValueError("valid_indices is empty.")

    # --------------------------------------------------------
    # 1) Restrict to valid sequential chain
    # --------------------------------------------------------
    seq = gdf_cleaned.iloc[valid_indices].copy()
    seq = seq.sort_index()

    # Keep only nodes that actually exist in the frame
    node_list = [
        n for n in node_list
        if _node_col(n) in seq.columns and _dist_col(n) in seq.columns
    ]
    if len(node_list) == 0:
        raise ValueError("No usable nodes found in gdf_cleaned.")

    # --------------------------------------------------------
    # 2) Prepare node coordinate table
    # --------------------------------------------------------
    if "Node #" not in gdf_nodes.columns:
        raise ValueError('gdf_nodes must contain a "Node #" column.')

    nodes_sub = gdf_nodes[gdf_nodes["Node #"].isin(node_list)].copy()
    nodes_sub = nodes_sub.sort_values("Node #")

    node_x = {int(row["Node #"]): float(row.geometry.x) for _, row in nodes_sub.iterrows()}
    node_y = {int(row["Node #"]): float(row.geometry.y) for _, row in nodes_sub.iterrows()}

    x_vals = pd.Series(node_x)
    y_vals = pd.Series(node_y)
    x_norm = _normalize_series(x_vals).to_dict()
    y_norm = _normalize_series(y_vals).to_dict()

    # --------------------------------------------------------
    # 3) Choose rho for utility
    # --------------------------------------------------------
    if rho is None:
        nearest_dists = []
        for _, row in seq.iterrows():
            dists = [row[_dist_col(n)] for n in node_list if pd.notna(row[_dist_col(n)])]
            if len(dists) > 0:
                nearest_dists.append(min(dists))
        if len(nearest_dists) == 0:
            rho = 1.0
        else:
            rho = float(np.median(nearest_dists))
            rho = max(rho, 1.0)

    # --------------------------------------------------------
    # 4) Build per-node temporal features
    # --------------------------------------------------------
    # We work only with observed quantities here. No GPS leakage.
    rss_cols = [_node_col(n) for n in node_list]

    node_feature_rows = []

    for t_idx in range(len(seq)):
        row = seq.iloc[t_idx]
        dt = seq.index[t_idx]

        # current RSSI values across nodes
        current_vals = np.array([float(row[_node_col(n)]) for n in node_list], dtype=float)

        # softmax-style acoustic shares
        shares = _softmax(current_vals, temperature=trigger_temperature)

        # ranks by descending current acoustic strength
        order_desc = np.argsort(-current_vals)
        rank_position = np.empty(len(node_list), dtype=int)
        rank_position[order_desc] = np.arange(len(node_list))
        rank_percentile = 1.0 - (rank_position / max(len(node_list) - 1, 1))

        for local_idx, n in enumerate(node_list):
            hist_start = max(0, t_idx - history_steps + 1)
            hist_vals = seq.iloc[hist_start : t_idx + 1][_node_col(n)].to_numpy(dtype=float)

            node_feature_rows.append(
                {
                    "datetime": dt,
                    "node": n,
                    "rssi_db": float(current_vals[local_idx]),
                    "rssi_mean": float(np.mean(hist_vals)),
                    "rssi_std": _safe_std(hist_vals),
                    "rssi_slope": _linear_slope(hist_vals),
                    "rssi_delta1": float(hist_vals[-1] - hist_vals[-2]) if len(hist_vals) >= 2 else 0.0,
                    "energy_share": float(shares[local_idx]),
                    "rank_percentile": float(rank_percentile[local_idx]),
                    "sensor_x_norm": float(x_norm[n]),
                    "sensor_y_norm": float(y_norm[n]),
                    # offline only, kept for utility and diagnostics
                    "distance_to_vehicle": float(row[_dist_col(n)]),
                }
            )

    node_feature_df = pd.DataFrame(node_feature_rows)

    # --------------------------------------------------------
    # 5) Build one context vector per timestamp
    # --------------------------------------------------------
    sequence_rows = []
    examples_rows = []

    node_feature_cols = [
        "rssi_db",
        "rssi_mean",
        "rssi_std",
        "rssi_slope",
        "rssi_delta1",
        "energy_share",
        "rank_percentile",
        "sensor_x_norm",
        "sensor_y_norm",
    ]

    # final order of nodes inside context vector
    ordered_nodes = list(node_list)

    for dt, frame_t in node_feature_df.groupby("datetime", sort=True):
        frame_t = frame_t.sort_values("node").set_index("node")

        # --- global scene features
        shares_t = frame_t["energy_share"].to_numpy(dtype=float)
        rssi_t = frame_t["rssi_db"].to_numpy(dtype=float)

        entropy = float(-np.sum(shares_t * np.log(shares_t + 1e-12)))
        order = np.argsort(-shares_t)
        top1_node = int(frame_t.index[order[0]])
        top1_share = float(shares_t[order[0]])
        top2_share = float(shares_t[order[1]]) if len(order) >= 2 else 0.0
        top1_top2_gap = float(top1_share - top2_share)

        acoustic_com_x = float(np.sum(shares_t * frame_t["sensor_x_norm"].to_numpy(dtype=float)))
        acoustic_com_y = float(np.sum(shares_t * frame_t["sensor_y_norm"].to_numpy(dtype=float)))

        global_features = [
            entropy,
            top1_top2_gap,
            acoustic_com_x,
            acoustic_com_y,
        ]

        # --- context vector: flatten node features in fixed node order + global
        context_vec = []
        for n in ordered_nodes:
            vals = frame_t.loc[n, node_feature_cols].to_numpy(dtype=float).tolist()
            context_vec.extend(vals)
        context_vec.extend(global_features)
        context_vec = np.asarray(context_vec, dtype=float)

        # --- triggered set B_t
        triggered = [
            n for n in ordered_nodes
            if float(frame_t.loc[n, "energy_share"]) >= trigger_prob_threshold
        ]

        if len(triggered) < trigger_min_nodes:
            ranked = frame_t.sort_values("energy_share", ascending=False).index.tolist()
            triggered = ranked[:trigger_min_nodes]

        if len(triggered) > trigger_max_nodes:
            ranked = frame_t.loc[triggered].sort_values("energy_share", ascending=False).index.tolist()
            triggered = ranked[:trigger_max_nodes]

        triggered = sorted(triggered)

        # --- available super-arms A_t
        subsets = []
        for k in range(1, min(max_subset_size, len(triggered)) + 1):
            subsets.extend(list(combinations(triggered, k)))

        sequence_rows.append(
            {
                "datetime": dt,
                "context_vec": context_vec,
                "triggered_nodes": tuple(triggered),
                "num_triggered": len(triggered),
                "top1_node": top1_node,
                "num_actions": len(subsets),
            }
        )

        # --- one row per admissible subset
        for subset in subsets:
            subset = tuple(sorted(subset))
            subset_str = _subset_to_str(subset)

            subset_frame = frame_t.loc[list(subset)].copy()
            subset_frame = subset_frame.sort_values("energy_share", ascending=False)

            # binary membership mask over kept nodes
            mask = [1.0 if n in subset else 0.0 for n in ordered_nodes]

            # selected-node descriptors, padded to 3 nodes
            selected_desc = []
            for _, r in subset_frame.iterrows():
                selected_desc.extend(
                    [
                        float(r["sensor_x_norm"]),
                        float(r["sensor_y_norm"]),
                        float(r["rssi_db"]),
                        float(r["rssi_mean"]),
                        float(r["rssi_slope"]),
                        float(r["energy_share"]),
                    ]
                )
            selected_desc = _pad_list(selected_desc, target_len=3 * 6, pad_value=0.0)

            # geometry
            pts = [(node_x[n], node_y[n]) for n in subset]
            pairwise_d = []
            for a, b in combinations(subset, 2):
                dx = node_x[a] - node_x[b]
                dy = node_y[a] - node_y[b]
                pairwise_d.append(float(np.sqrt(dx * dx + dy * dy)))
            pairwise_d = _pad_list(pairwise_d, target_len=3, pad_value=0.0)

            centroid_x = float(np.mean([x_norm[n] for n in subset]))
            centroid_y = float(np.mean([y_norm[n] for n in subset]))
            max_spread = float(max(pairwise_d)) if len(pairwise_d) > 0 else 0.0
            tri_area = _triangle_area([(x_norm[n], y_norm[n]) for n in subset])

            geometry_features = pairwise_d + [centroid_x, centroid_y, max_spread, tri_area]

            # subset acoustic aggregates
            subset_shares = subset_frame["energy_share"].to_numpy(dtype=float)
            subset_rssi = subset_frame["rssi_db"].to_numpy(dtype=float)
            subset_slopes = subset_frame["rssi_slope"].to_numpy(dtype=float)

            acoustic_agg = [
                float(len(subset)),
                float(np.sum(subset_shares)),
                float(np.max(subset_shares)),
                float(np.min(subset_shares)),
                float(np.mean(subset_rssi)),
                float(np.var(subset_rssi)) if len(subset_rssi) > 1 else 0.0,
                float(np.mean(subset_slopes)),
                float(1.0 if top1_node in subset else 0.0),
            ]

            action_raw_vec = np.asarray(mask + selected_desc + geometry_features + acoustic_agg, dtype=float)

            # --- oracle utility from GPS-derived distances
            dists = sorted(float(frame_t.loc[n, "distance_to_vehicle"]) for n in subset)
            d1 = dists[0]
            d2 = dists[1] if len(dists) >= 2 else np.inf

            utility = float(
                np.exp(-(d1 ** 2) / (2 * rho ** 2))
                + utility_second_weight * np.exp(-(d2 ** 2) / (2 * rho ** 2))
            )

            examples_rows.append(
                {
                    "datetime": dt,
                    "subset": subset,
                    "subset_str": subset_str,
                    "subset_size": len(subset),
                    "context_vec": context_vec,
                    "action_raw_vec": action_raw_vec,
                    "utility": utility,
                    "d1": float(d1),
                    "d2": float(d2) if np.isfinite(d2) else np.nan,
                    "triggered_nodes": tuple(triggered),
                    "top1_node": top1_node,
                }
            )

    sequence_df = pd.DataFrame(sequence_rows).sort_values("datetime").reset_index(drop=True)
    examples_df = pd.DataFrame(examples_rows).sort_values(["datetime", "subset_size", "subset_str"]).reset_index(drop=True)

    context_dim = len(sequence_df.iloc[0]["context_vec"])
    action_raw_dim = len(examples_df.iloc[0]["action_raw_vec"])

    meta = {
        "ordered_nodes": ordered_nodes,
        "history_steps": history_steps,
        "max_subset_size": max_subset_size,
        "trigger_prob_threshold": trigger_prob_threshold,
        "trigger_min_nodes": trigger_min_nodes,
        "trigger_max_nodes": trigger_max_nodes,
        "trigger_temperature": trigger_temperature,
        "utility_second_weight": utility_second_weight,
        "rho": rho,
        "context_dim": context_dim,
        "action_raw_dim": action_raw_dim,
        "num_times": len(sequence_df),
        "num_examples": len(examples_df),
    }

    return {
        "node_feature_df": node_feature_df,
        "sequence_df": sequence_df,
        "examples_df": examples_df,
        "meta": meta,
    }