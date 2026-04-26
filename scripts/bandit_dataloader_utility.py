from itertools import combinations
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


def _linear_slope(x: np.ndarray) -> float:
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
    temperature = max(float(temperature), 1e-8)
    z = x / temperature
    z = z - np.nanmax(z)
    ex = np.exp(z)
    s = np.nansum(ex)
    if s <= 0:
        return np.ones_like(x) / len(x)
    return ex / s


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


def _triangle_area(points: List[Tuple[float, float]]) -> float:
    if len(points) != 3:
        return 0.0
    (x1, y1), (x2, y2), (x3, y3) = points
    return float(abs(0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))))


def build_bandit_dataset_utility(
    gdf_cleaned: pd.DataFrame,
    gdf_nodes: pd.DataFrame,
    valid_indices: List[int],
    node_list: List[int],
    *,
    history_steps: int = 5,
    max_subset_size: int = 3,
    utility_second_weight: float = 0.45,
    utility_third_weight: float = 0.20,
    rho: Optional[float] = None,
    softmax_temperature: float = 4.0,
    verbose: bool = True,
    progress_every: int = 500,
) -> Dict[str, object]:
    """
    Full 129-arm builder with a less-compressive rational utility:
        u(S) = 1/(1 + d1/rho)
             + w2/(1 + d2/rho)
             + w3/(1 + d3/rho)
    """

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    if len(valid_indices) == 0:
        raise ValueError("valid_indices is empty.")

    vprint("[1/8] Restricting to valid sequential chain...")
    seq = gdf_cleaned.iloc[valid_indices].copy()
    seq = seq.sort_index()

    ordered_nodes = [
        n for n in node_list
        if _node_col(n) in seq.columns and _dist_col(n) in seq.columns
    ]
    if len(ordered_nodes) == 0:
        raise ValueError("No usable nodes found in gdf_cleaned.")

    T = len(seq)
    N = len(ordered_nodes)
    vprint(f"      valid timesteps: {T}")
    vprint(f"      kept nodes: {ordered_nodes}")

    vprint("[2/8] Loading sensor coordinates and normalizing geometry...")
    if "Node #" not in gdf_nodes.columns:
        raise ValueError('gdf_nodes must contain a "Node #" column.')

    nodes_sub = gdf_nodes[gdf_nodes["Node #"].isin(ordered_nodes)].copy()
    nodes_sub = nodes_sub.sort_values("Node #")

    node_x = {int(row["Node #"]): float(row.geometry.x) for _, row in nodes_sub.iterrows()}
    node_y = {int(row["Node #"]): float(row.geometry.y) for _, row in nodes_sub.iterrows()}

    x_vals = pd.Series(node_x)
    y_vals = pd.Series(node_y)
    x_norm_map = _normalize_series(x_vals).to_dict()
    y_norm_map = _normalize_series(y_vals).to_dict()

    x_norm = np.array([x_norm_map[n] for n in ordered_nodes], dtype=float)
    y_norm = np.array([y_norm_map[n] for n in ordered_nodes], dtype=float)
    x_metric = np.array([node_x[n] for n in ordered_nodes], dtype=float)
    y_metric = np.array([node_y[n] for n in ordered_nodes], dtype=float)

    node_to_idx = {n: i for i, n in enumerate(ordered_nodes)}

    vprint("[3/8] Extracting per-time RSSI and distance arrays...")
    datetimes = seq.index.to_numpy()
    rssi_mat = np.stack([seq[_node_col(n)].to_numpy(dtype=float) for n in ordered_nodes], axis=1)
    dist_mat = np.stack([seq[_dist_col(n)].to_numpy(dtype=float) for n in ordered_nodes], axis=1)

    vprint("[4/8] Choosing utility scale rho...")
    if rho is None:
        nearest_dists = np.min(dist_mat, axis=1)
        nearest_dists = nearest_dists[np.isfinite(nearest_dists)]
        if len(nearest_dists) == 0:
            rho = 1.0
        else:
            rho = float(np.median(nearest_dists))
            rho = max(rho, 1.0)
    vprint(f"      rho = {rho:.4f}")

    vprint("[5/8] Precomputing full subset universe and static subset geometry...")
    full_subset_universe: List[Tuple[int, ...]] = []
    for k in range(1, min(max_subset_size, N) + 1):
        full_subset_universe.extend(list(combinations(ordered_nodes, k)))

    M = len(full_subset_universe)
    vprint(f"      num actions per time = {M}")

    subset_node_ids: List[Tuple[int, ...]] = []
    subset_node_idxs: List[np.ndarray] = []
    subset_strs: List[str] = []
    subset_sizes: List[int] = []
    subset_masks: List[np.ndarray] = []
    subset_geometry_static: List[List[float]] = []

    for subset in full_subset_universe:
        subset = tuple(sorted(subset))
        idxs = np.array([node_to_idx[n] for n in subset], dtype=int)

        subset_node_ids.append(subset)
        subset_node_idxs.append(idxs)
        subset_strs.append(_subset_to_str(subset))
        subset_sizes.append(len(subset))

        mask = np.zeros(N, dtype=float)
        mask[idxs] = 1.0
        subset_masks.append(mask)

        pairwise_d = []
        for a, b in combinations(idxs, 2):
            dx = x_metric[a] - x_metric[b]
            dy = y_metric[a] - y_metric[b]
            pairwise_d.append(float(np.sqrt(dx * dx + dy * dy)))
        while len(pairwise_d) < 3:
            pairwise_d.append(0.0)

        centroid_x = float(np.mean(x_norm[idxs]))
        centroid_y = float(np.mean(y_norm[idxs]))
        max_spread = float(max(pairwise_d)) if len(pairwise_d) > 0 else 0.0
        tri_area = _triangle_area([(x_norm[i], y_norm[i]) for i in idxs])

        subset_geometry_static.append(pairwise_d[:3] + [centroid_x, centroid_y, max_spread, tri_area])

    subset_sizes = np.array(subset_sizes, dtype=int)

    vprint("[6/8] Building per-node temporal features...")
    rssi_mean = np.zeros((T, N), dtype=float)
    rssi_std = np.zeros((T, N), dtype=float)
    rssi_slope = np.zeros((T, N), dtype=float)
    rssi_delta1 = np.zeros((T, N), dtype=float)
    energy_share = np.zeros((T, N), dtype=float)
    rank_percentile = np.zeros((T, N), dtype=float)

    for t in range(T):
        if verbose and (t == 0 or (t + 1) % progress_every == 0 or t == T - 1):
            print(f"      temporal features: {t+1}/{T}")

        current_vals = rssi_mat[t]
        shares = _softmax(current_vals, temperature=softmax_temperature)
        energy_share[t] = shares

        order_desc = np.argsort(-current_vals)
        rank_pos = np.empty(N, dtype=int)
        rank_pos[order_desc] = np.arange(N)
        rank_percentile[t] = 1.0 - (rank_pos / max(N - 1, 1))

        h0 = max(0, t - history_steps + 1)
        hist = rssi_mat[h0:t + 1]

        rssi_mean[t] = hist.mean(axis=0)
        rssi_std[t] = hist.std(axis=0, ddof=0)

        for j in range(N):
            hcol = hist[:, j]
            rssi_slope[t, j] = _linear_slope(hcol)
            rssi_delta1[t, j] = float(hcol[-1] - hcol[-2]) if len(hcol) >= 2 else 0.0

    vprint("[7/8] Materializing node_feature_df...")
    node_feature_rows = []
    for t in range(T):
        if verbose and (t == 0 or (t + 1) % progress_every == 0 or t == T - 1):
            print(f"      node_feature_df rows: {t+1}/{T}")

        dt = datetimes[t]
        for j, node in enumerate(ordered_nodes):
            node_feature_rows.append(
                {
                    "datetime": dt,
                    "node": node,
                    "rssi_db": float(rssi_mat[t, j]),
                    "rssi_mean": float(rssi_mean[t, j]),
                    "rssi_std": float(rssi_std[t, j]),
                    "rssi_slope": float(rssi_slope[t, j]),
                    "rssi_delta1": float(rssi_delta1[t, j]),
                    "energy_share": float(energy_share[t, j]),
                    "rank_percentile": float(rank_percentile[t, j]),
                    "sensor_x_norm": float(x_norm[j]),
                    "sensor_y_norm": float(y_norm[j]),
                    "distance_to_vehicle": float(dist_mat[t, j]),
                }
            )

    node_feature_df = pd.DataFrame(node_feature_rows)

    vprint("[8/8] Building sequence_df and examples_df...")
    sequence_rows = []
    examples_rows = []

    for t in range(T):
        if verbose and (t == 0 or (t + 1) % progress_every == 0 or t == T - 1):
            print(f"      examples/sequence: {t+1}/{T}")

        dt = datetimes[t]
        shares_t = energy_share[t]

        entropy = float(-np.sum(shares_t * np.log(shares_t + 1e-12)))
        order = np.argsort(-shares_t)
        top1_idx = int(order[0])
        top1_node = int(ordered_nodes[top1_idx])
        top1_share = float(shares_t[order[0]])
        top2_share = float(shares_t[order[1]]) if len(order) >= 2 else 0.0
        top1_top2_gap = float(top1_share - top2_share)

        acoustic_com_x = float(np.sum(shares_t * x_norm))
        acoustic_com_y = float(np.sum(shares_t * y_norm))

        context_vec = []
        for j in range(N):
            context_vec.extend(
                [
                    float(rssi_mat[t, j]),
                    float(rssi_mean[t, j]),
                    float(rssi_std[t, j]),
                    float(rssi_slope[t, j]),
                    float(rssi_delta1[t, j]),
                    float(energy_share[t, j]),
                    float(rank_percentile[t, j]),
                    float(x_norm[j]),
                    float(y_norm[j]),
                ]
            )
        context_vec.extend([entropy, top1_top2_gap, acoustic_com_x, acoustic_com_y])
        context_vec = np.asarray(context_vec, dtype=float)

        sequence_rows.append(
            {
                "datetime": dt,
                "context_vec": context_vec,
                "num_actions": M,
                "top1_node": top1_node,
            }
        )

        for m in range(M):
            idxs = subset_node_idxs[m]
            subset = subset_node_ids[m]

            sort_local = idxs[np.argsort(-shares_t[idxs])]

            selected_desc = []
            for j in sort_local:
                selected_desc.extend(
                    [
                        float(x_norm[j]),
                        float(y_norm[j]),
                        float(rssi_mat[t, j]),
                        float(rssi_mean[t, j]),
                        float(rssi_slope[t, j]),
                        float(energy_share[t, j]),
                    ]
                )
            while len(selected_desc) < 18:
                selected_desc.append(0.0)

            subset_shares = shares_t[idxs]
            subset_rssi_vals = rssi_mat[t, idxs]
            subset_slopes = rssi_slope[t, idxs]

            acoustic_agg = [
                float(len(idxs)),
                float(np.sum(subset_shares)),
                float(np.max(subset_shares)),
                float(np.min(subset_shares)),
                float(np.mean(subset_rssi_vals)),
                float(np.var(subset_rssi_vals)) if len(subset_rssi_vals) > 1 else 0.0,
                float(np.mean(subset_slopes)),
                float(1.0 if top1_idx in idxs else 0.0),
            ]

            action_raw_vec = np.asarray(
                subset_masks[m].tolist()
                + selected_desc
                + subset_geometry_static[m]
                + acoustic_agg,
                dtype=float,
            )

            dists = np.sort(dist_mat[t, idxs])
            d1 = float(dists[0])
            d2 = float(dists[1]) if len(dists) >= 2 else np.inf
            d3 = float(dists[2]) if len(dists) >= 3 else np.inf

            term1 = 1.0 / (1.0 + d1 / rho)
            term2 = utility_second_weight / (1.0 + d2 / rho) if np.isfinite(d2) else 0.0
            term3 = utility_third_weight / (1.0 + d3 / rho) if np.isfinite(d3) else 0.0
            utility = float(term1 + term2 + term3)

            examples_rows.append(
                {
                    "datetime": dt,
                    "subset": subset,
                    "subset_str": subset_strs[m],
                    "subset_size": int(subset_sizes[m]),
                    "context_vec": context_vec,
                    "action_raw_vec": action_raw_vec,
                    "utility": utility,
                    "d1": d1,
                    "d2": float(d2) if np.isfinite(d2) else np.nan,
                    "d3": float(d3) if np.isfinite(d3) else np.nan,
                    "top1_node": top1_node,
                }
            )

    sequence_df = pd.DataFrame(sequence_rows).sort_values("datetime").reset_index(drop=True)
    examples_df = (
        pd.DataFrame(examples_rows)
        .sort_values(["datetime", "subset_size", "subset_str"])
        .reset_index(drop=True)
    )

    context_dim = len(sequence_df.iloc[0]["context_vec"])
    action_raw_dim = len(examples_df.iloc[0]["action_raw_vec"])

    meta = {
        "ordered_nodes": ordered_nodes,
        "history_steps": history_steps,
        "max_subset_size": max_subset_size,
        "utility_second_weight": utility_second_weight,
        "utility_third_weight": utility_third_weight,
        "rho": rho,
        "softmax_temperature": softmax_temperature,
        "context_dim": context_dim,
        "action_raw_dim": action_raw_dim,
        "num_times": len(sequence_df),
        "num_examples": len(examples_df),
        "num_actions_per_time": M,
        "full_subset_universe": full_subset_universe,
    }

    vprint("Done.")
    vprint(f"      sequence_df shape: {sequence_df.shape}")
    vprint(f"      examples_df shape: {examples_df.shape}")
    vprint(f"      context_dim: {context_dim}")
    vprint(f"      action_raw_dim: {action_raw_dim}")

    return {
        "node_feature_df": node_feature_df,
        "sequence_df": sequence_df,
        "examples_df": examples_df,
        "meta": meta,
    }
