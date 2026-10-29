from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from condchemsymm.metrics import classification_metrics
from ensemble_components import (
    GRAPH_MAIN,
    POSITIVE_TRANSFORMS,
    SEEDS,
    ci95,
    load_component_view_predictions,
    paired_stats,
    paired_t_p,
)
from fusion_components import (
    PAIRED_METRICS,
    SPECS,
    component_columns,
    fast_auc_scores,
    fast_false_invariance_scores,
    graph_baseline_seed_rows,
    load_sphys_predictions,
    merge_for_spec,
    pair_indices,
    sphys_baseline_seed_rows,
    simplex_weights,
)
from analyze_sphysnet_positive_views import sphys_sample_predictions


OBJECTIVE_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.5]
SCALE_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
VAL_OBJECTIVE_SCALE_GRID = [0.5, 0.75, 0.9, 1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0]
PROJECTION_POLICIES = [
    "prob_pair_norm",
    "pair_logitdiff_scale1",
    "pair_logitdiff_scale1p2",
    "pair_logitdiff_scale1p25",
    "pair_logitdiff_valobjscale",
    "pair_logitdiff_valnllscale",
]


def safe_logit(prob: np.ndarray) -> np.ndarray:
    clipped = np.clip(prob.astype(float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(score: np.ndarray) -> np.ndarray:
    clipped = np.clip(score, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def objective_label(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def projection_label(policy: str) -> str:
    return {
        "prob_pair_norm": "probpairnorm",
        "pair_logitdiff_scale1": "pairlogit_s1",
        "pair_logitdiff_scale1p2": "pairlogit_s1p2",
        "pair_logitdiff_scale1p25": "pairlogit_s1p25",
        "pair_logitdiff_valobjscale": "pairlogit_valobjscale",
        "pair_logitdiff_valnllscale": "pairlogit_valnllscale",
    }[policy]


def candidate_name(row: pd.Series | dict) -> str:
    return (
        f"{row['fusion']}_n{int(row['sphys_num_confs'])}_"
        f"valobj{objective_label(float(row['val_selection_false_inv_weight']))}_"
        f"{projection_label(str(row['projection_policy']))}"
    )


def project_probability_norm(df: pd.DataFrame, raw_predictions: np.ndarray) -> np.ndarray:
    projected = raw_predictions.copy()
    left, right = pair_indices(df)
    if len(left) == 0:
        return projected
    left_prob = np.clip(raw_predictions[:, left], 1e-7, 1.0)
    right_prob = np.clip(raw_predictions[:, right], 1e-7, 1.0)
    pair_prob = left_prob / (left_prob + right_prob)
    projected[:, left] = pair_prob
    projected[:, right] = 1.0 - pair_prob
    return np.clip(projected, 1e-7, 1.0 - 1e-7)


def project_logit_difference(df: pd.DataFrame, scores: np.ndarray, scale: float) -> np.ndarray:
    projected = sigmoid(float(scale) * scores)
    left, right = pair_indices(df)
    if len(left) == 0:
        return projected
    pair_prob = sigmoid(float(scale) * (scores[:, left] - scores[:, right]))
    projected[:, left] = pair_prob
    projected[:, right] = 1.0 - pair_prob
    return np.clip(projected, 1e-7, 1.0 - 1e-7)


def vector_nll(y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    clipped = np.clip(predictions, 1e-7, 1.0 - 1e-7)
    return np.mean(-(y[None, :] * np.log(clipped) + (1.0 - y[None, :]) * np.log(1.0 - clipped)), axis=1)


def vector_brier(y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    return np.mean((predictions - y[None, :]) ** 2, axis=1)


def policy_predictions(
    df: pd.DataFrame,
    weights: np.ndarray,
    policy: str,
    objective_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    columns = [c for c in df.columns if c.endswith("_mechanistic") or c == "sphysnet_prediction"]
    components = df[columns].to_numpy(dtype=float)
    if policy == "prob_pair_norm":
        raw = weights @ components.T
        return project_probability_norm(df, raw), np.ones(len(weights), dtype=float)

    logits = safe_logit(components)
    scores = weights @ logits.T
    if policy == "pair_logitdiff_scale1":
        return project_logit_difference(df, scores, 1.0), np.ones(len(weights), dtype=float)
    if policy == "pair_logitdiff_scale1p2":
        return project_logit_difference(df, scores, 1.2), np.full(len(weights), 1.2, dtype=float)
    if policy == "pair_logitdiff_scale1p25":
        return project_logit_difference(df, scores, 1.25), np.full(len(weights), 1.25, dtype=float)
    if policy == "pair_logitdiff_valobjscale":
        y = df["y"].to_numpy(dtype=float)
        left, right = pair_indices(df)
        best_predictions: np.ndarray | None = None
        best_score = np.full(len(weights), -np.inf, dtype=float)
        best_nll = np.full(len(weights), np.inf, dtype=float)
        best_brier = np.full(len(weights), np.inf, dtype=float)
        best_scale = np.ones(len(weights), dtype=float)
        for scale in VAL_OBJECTIVE_SCALE_GRID:
            predictions = project_logit_difference(df, scores, scale)
            aucs = fast_auc_scores(y, predictions)
            false_inv = fast_false_invariance_scores(predictions, left, right)
            nll = vector_nll(y, predictions)
            brier = vector_brier(y, predictions)
            objective = aucs - float(objective_weight) * false_inv
            improve = objective > best_score + 1e-12
            tied = np.isclose(objective, best_score, rtol=0.0, atol=1e-12)
            tie_improve = tied & (
                (nll < best_nll - 1e-12)
                | (np.isclose(nll, best_nll, rtol=0.0, atol=1e-12) & (brier < best_brier - 1e-12))
            )
            update = improve | tie_improve
            if best_predictions is None:
                best_predictions = predictions.copy()
                update = np.ones(len(weights), dtype=bool)
            else:
                best_predictions[update] = predictions[update]
            best_score[update] = objective[update]
            best_nll[update] = nll[update]
            best_brier[update] = brier[update]
            best_scale[update] = float(scale)
        if best_predictions is None:
            raise RuntimeError("empty scale grid")
        return best_predictions, best_scale
    if policy == "pair_logitdiff_valnllscale":
        y = df["y"].to_numpy(dtype=float)
        best_predictions: np.ndarray | None = None
        best_nll = np.full(len(weights), np.inf, dtype=float)
        best_scale = np.ones(len(weights), dtype=float)
        for scale in SCALE_GRID:
            predictions = project_logit_difference(df, scores, scale)
            nll = vector_nll(y, predictions)
            improve = nll < best_nll
            if best_predictions is None:
                best_predictions = predictions.copy()
            else:
                best_predictions[improve] = predictions[improve]
            best_nll[improve] = nll[improve]
            best_scale[improve] = float(scale)
        if best_predictions is None:
            raise RuntimeError("empty scale grid")
        return best_predictions, best_scale
    raise ValueError(f"unknown projection policy: {policy}")


def evaluate_predictions(df: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    eval_df = df[["sample_id", "y", "paired_id"]].copy()
    eval_df["prediction"] = prediction
    metrics = classification_metrics(eval_df["y"].to_numpy(dtype=float), prediction)
    pair = paired_stats(eval_df, "prediction")
    return {
        "test_accuracy": metrics["accuracy"],
        "test_brier": metrics["brier"],
        "test_nll": metrics["nll"],
        "test_ece10": metrics["ece10"],
        "test_mce10": metrics["mce10"],
        "test_roc_auc": metrics["roc_auc"],
        "test_pr_auc": metrics["pr_auc"],
        **pair,
    }


def choose_candidate(
    val: pd.DataFrame,
    columns: list[str],
    objective_weight: float,
    policy: str,
) -> tuple[np.ndarray, float, float, dict[str, float]]:
    weights = simplex_weights(len(columns))
    work = val[["sample_id", "y", "paired_id", *columns]].copy()
    predictions, scales = policy_predictions(work, weights, policy, objective_weight)
    y = work["y"].to_numpy(dtype=float)
    aucs = fast_auc_scores(y, predictions)
    left, right = pair_indices(work)
    false_inv = fast_false_invariance_scores(predictions, left, right)
    nll = vector_nll(y, predictions)
    brier = vector_brier(y, predictions)
    scores = aucs - float(objective_weight) * false_inv
    best_score = float(np.nanmax(scores))
    candidate_idx = np.flatnonzero(np.isclose(scores, best_score, rtol=0.0, atol=1e-12))

    def tie_key(idx: int) -> tuple[float, float, float, float]:
        # After the declared objective ties, prefer better calibration and less external dependence.
        sphys_weight = float(weights[idx, -1])
        graph_weight = float(weights[idx, 0])
        return (float(nll[idx]), float(brier[idx]), sphys_weight, -graph_weight)

    best_idx = min(candidate_idx, key=tie_key)
    selected_weights = weights[best_idx]
    selected_scale = float(scales[best_idx])
    selected_metrics = evaluate_predictions(work, predictions[best_idx])
    return selected_weights, selected_scale, best_score, selected_metrics


def apply_candidate(df: pd.DataFrame, columns: list[str], weights: np.ndarray, policy: str, scale: float) -> np.ndarray:
    work = df[["sample_id", "y", "paired_id", *columns]].copy()
    components = work[columns].to_numpy(dtype=float)
    weights = np.asarray(weights, dtype=float)
    if policy == "prob_pair_norm":
        raw = weights[None, :] @ components.T
        return project_probability_norm(work, raw)[0]
    logits = safe_logit(components)
    scores = weights[None, :] @ logits.T
    return project_logit_difference(work, scores, scale)[0]


def summarize_seed_rows(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "candidate",
        "fusion",
        "sphys_num_confs",
        "sphys_mode",
        "components",
        "projection_policy",
        "val_selection_false_inv_weight",
        "selection_protocol",
        "note",
    ]
    metric_cols = [
        "test_accuracy",
        "test_brier",
        "test_nll",
        "test_ece10",
        "test_mce10",
        "test_roc_auc",
        "test_pr_auc",
        "false_invariance_rate_test_pairs",
        "pair_order_accuracy",
        "pair_gap_mean",
        "pair_gap_p10",
        "selected_pair_logit_scale",
    ]
    grouped = frame.groupby(keys, dropna=False, sort=True)
    out = grouped[metric_cols].agg(["mean", "std", "count"])
    out.columns = [f"{name}_{stat}" for name, stat in out.columns]
    out = out.reset_index()
    weight_cols = [col for col in frame.columns if col.startswith("selected_") and col.endswith("_weight")]
    chosen = grouped[weight_cols].agg(["mean", "std"])
    chosen.columns = [f"{name}_{stat}" for name, stat in chosen.columns]
    chosen = chosen.reset_index()
    weight_rows = []
    for key_values, group in frame.groupby(keys, dropna=False, sort=True):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = dict(zip(keys, key_values))
        row["selected_weights_by_seed"] = ",".join(
            "/".join(f"{float(record[col]):.2f}" for col in weight_cols)
            for _, record in group.sort_values("seed").iterrows()
        )
        row["selected_pair_logit_scale_by_seed"] = ",".join(
            f"{float(record['selected_pair_logit_scale']):.2f}" for _, record in group.sort_values("seed").iterrows()
        )
        weight_rows.append(row)
    return out.merge(chosen, on=keys, how="left").merge(pd.DataFrame(weight_rows), on=keys, how="left")


def paired_tests(root: Path, selected_raw: pd.DataFrame, num_confs: int) -> pd.DataFrame:
    raw_path = root / "reports" / "tables" / f"main_sphysnet_numconf{num_confs}_fusion_val_selected_seed_raw.csv"
    raw_fusion = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    baseline_map = {
        "graph_main": graph_baseline_seed_rows(root),
        "sphys_boltzmann": sphys_baseline_seed_rows(root, num_confs, "boltzmann_rt_298k"),
        "sphys_valtemp": sphys_baseline_seed_rows(root, num_confs, "val_nll_temperature"),
    }
    rows = []
    group_keys = ["candidate", "fusion", "projection_policy", "val_selection_false_inv_weight"]
    for key_values, group in selected_raw.groupby(group_keys, sort=True):
        candidate, fusion, projection_policy, objective_weight = key_values
        baselines = dict(baseline_map)
        if not raw_fusion.empty:
            raw_group = raw_fusion[
                raw_fusion["fusion"].eq(fusion)
                & np.isclose(raw_fusion["val_selection_false_inv_weight"].astype(float), float(objective_weight))
            ].copy()
            if not raw_group.empty:
                baselines["raw_probability_fusion_same_obj"] = raw_group
        for baseline_name, baseline in baselines.items():
            merged = baseline.merge(group, on="seed", suffixes=("_baseline", "_candidate"))
            for metric, direction in PAIRED_METRICS:
                base_col = f"{metric}_baseline"
                cand_col = f"{metric}_candidate"
                if base_col not in merged.columns or cand_col not in merged.columns:
                    continue
                metric_df = merged[["seed", base_col, cand_col]].replace([np.inf, -np.inf], np.nan).dropna()
                if metric_df.empty:
                    continue
                baseline_values = metric_df[base_col].to_numpy(dtype=float)
                candidate_values = metric_df[cand_col].to_numpy(dtype=float)
                improvement = (candidate_values - baseline_values) * direction
                low, high = ci95(improvement)
                wins = int(np.sum(improvement > 0))
                losses = int(np.sum(improvement < 0))
                rows.append(
                    {
                        "comparison": f"{candidate}_vs_{baseline_name}",
                        "candidate": candidate,
                        "fusion": fusion,
                        "projection_policy": projection_policy,
                        "baseline": baseline_name,
                        "val_selection_false_inv_weight": float(objective_weight),
                        "metric": metric,
                        "n_paired_seeds": len(metric_df),
                        "baseline_mean": float(np.mean(baseline_values)),
                        "candidate_mean": float(np.mean(candidate_values)),
                        "improvement_mean": float(np.mean(improvement)),
                        "improvement_std": float(np.std(improvement, ddof=1)) if len(improvement) > 1 else np.nan,
                        "improvement_ci95_low": low,
                        "improvement_ci95_high": high,
                        "paired_t_p_two_sided": paired_t_p(baseline_values, candidate_values),
                        "sign_wins": wins,
                        "sign_losses": losses,
                        "sign_ties": int(np.sum(improvement == 0)),
                        "all_seeds_better": bool(wins == len(metric_df)),
                    }
                )
    return pd.DataFrame(rows)


def merge_view_for_spec(root: Path, spec, num_confs: int, seed: int, transform: str) -> pd.DataFrame:
    graph = load_component_view_predictions(root, spec.graph, seed, transform)
    sphys = sphys_sample_predictions(root, num_confs, transform, spec.sphys_mode)
    merged = graph.merge(sphys, on=["sample_id", "y", "paired_id"], how="inner")
    if len(merged) != len(graph) or len(merged) != len(sphys):
        raise ValueError(f"sPhysNet view row mismatch for {spec.label} seed={seed} transform={transform}")
    if spec.mlp is not None:
        mlp = load_component_view_predictions(root, spec.mlp, seed, transform)
        merged = merged.merge(mlp, on=["sample_id", "y", "paired_id"], how="inner")
        if len(merged) != len(graph) or len(merged) != len(mlp):
            raise ValueError(f"MLP view row mismatch for {spec.label} seed={seed} transform={transform}")
    return merged


def positive_view_rows(root: Path, raw: pd.DataFrame, num_confs: int) -> pd.DataFrame:
    rows = []
    spec_by_label = {spec.label: spec for spec in SPECS}
    for _, selected in raw.iterrows():
        spec = spec_by_label[str(selected["fusion"])]
        columns = component_columns(spec)
        weights = []
        for col in columns:
            if col == spec.graph.label:
                weights.append(float(selected["selected_graph_weight"]))
            elif spec.mlp is not None and col == spec.mlp.label:
                weights.append(float(selected["selected_mlp_weight"]))
            elif col == "sphysnet_prediction":
                weights.append(float(selected["selected_sphysnet_weight"]))
            else:
                raise ValueError(f"unexpected component column: {col}")
        for transform in POSITIVE_TRANSFORMS:
            try:
                view = merge_view_for_spec(root, spec, num_confs, int(selected["seed"]), transform)
            except FileNotFoundError:
                continue
            prediction = apply_candidate(
                view,
                columns,
                np.asarray(weights, dtype=float),
                str(selected["projection_policy"]),
                float(selected["selected_pair_logit_scale"]),
            )
            metrics = evaluate_predictions(view, prediction)
            rows.append(
                {
                    "candidate": candidate_name(selected),
                    "fusion": selected["fusion"],
                    "seed": int(selected["seed"]),
                    "sphys_num_confs": num_confs,
                    "sphys_mode": selected["sphys_mode"],
                    "projection_policy": selected["projection_policy"],
                    "val_selection_false_inv_weight": selected["val_selection_false_inv_weight"],
                    "transform_type": transform,
                    "selected_pair_logit_scale": selected["selected_pair_logit_scale"],
                    "selected_graph_weight": selected["selected_graph_weight"],
                    "selected_mlp_weight": selected["selected_mlp_weight"],
                    "selected_sphysnet_weight": selected["selected_sphysnet_weight"],
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def summarize_positive_views(raw_views: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw_views.empty:
        return pd.DataFrame(), pd.DataFrame()
    keys = [
        "candidate",
        "fusion",
        "sphys_num_confs",
        "sphys_mode",
        "projection_policy",
        "val_selection_false_inv_weight",
        "transform_type",
    ]
    metric_cols = [
        "test_roc_auc",
        "test_brier",
        "test_nll",
        "test_ece10",
        "false_invariance_rate_test_pairs",
        "pair_order_accuracy",
    ]
    by_transform = raw_views.groupby(keys, dropna=False, sort=True)[metric_cols].agg(["mean", "std", "count"])
    by_transform.columns = [f"{name}_{stat}" for name, stat in by_transform.columns]
    by_transform = by_transform.reset_index()
    worst_rows = []
    group_keys = [key for key in keys if key != "transform_type"]
    for key_values, group in by_transform.groupby(group_keys, dropna=False, sort=True):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = dict(zip(group_keys, key_values))
        raw_group = raw_views
        for key, value in row.items():
            raw_group = raw_group[raw_group[key].eq(value)]
        row.update(
            {
                "selected_graph_weight_mean": raw_group["selected_graph_weight"].mean(),
                "selected_graph_weight_std": raw_group.groupby("seed")["selected_graph_weight"].first().std(ddof=1),
                "selected_mlp_weight_mean": raw_group["selected_mlp_weight"].mean(),
                "selected_mlp_weight_std": raw_group.groupby("seed")["selected_mlp_weight"].first().std(ddof=1),
                "selected_sphysnet_weight_mean": raw_group["selected_sphysnet_weight"].mean(),
                "selected_sphysnet_weight_std": raw_group.groupby("seed")["selected_sphysnet_weight"].first().std(ddof=1),
                "selected_pair_logit_scale_mean": raw_group["selected_pair_logit_scale"].mean(),
                "selected_pair_logit_scale_std": raw_group.groupby("seed")["selected_pair_logit_scale"].first().std(ddof=1),
                "worst_positive_transform_roc_auc_mean": group["test_roc_auc_mean"].min(),
                "worst_positive_transform_roc_auc_std": group.loc[group["test_roc_auc_mean"].idxmin(), "test_roc_auc_std"],
                "worst_positive_transform_brier_mean": group["test_brier_mean"].max(),
                "worst_positive_transform_nll_mean": group["test_nll_mean"].max(),
                "worst_positive_transform_ece10_mean": group["test_ece10_mean"].max(),
                "worst_positive_transform_false_invariance_mean": group[
                    "false_invariance_rate_test_pairs_mean"
                ].max(),
                "worst_positive_transform_pair_order_accuracy_mean": group["pair_order_accuracy_mean"].min(),
                "positive_view_count": int(group["test_roc_auc_count"].min()),
            }
        )
        worst_rows.append(row)
    return by_transform, pd.DataFrame(worst_rows)


def fmt(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def make_markdown(summary: pd.DataFrame) -> str:
    cols = [
        "candidate",
        "projection_policy",
        "sphys_mode",
        "val_selection_false_inv_weight",
        "selected_graph_weight_mean",
        "selected_mlp_weight_mean",
        "selected_sphysnet_weight_mean",
        "selected_pair_logit_scale_mean",
        "selected_weights_by_seed",
        "selected_pair_logit_scale_by_seed",
        "test_roc_auc_mean",
        "test_roc_auc_std",
        "test_brier_mean",
        "test_nll_mean",
        "false_invariance_rate_test_pairs_mean",
        "pair_order_accuracy_mean",
        "pair_gap_mean_mean",
    ]
    view = summary.sort_values(["test_roc_auc_mean", "test_brier_mean"], ascending=[False, True]).copy()
    view = view[[col for col in cols if col in view.columns]]
    for col in view.columns:
        view[col] = view[col].map(fmt)
    return (
        "# Phase 1 sPhysNet Pair-Projected Fusion Results\n\n"
        "Seedwise validation-selected fusion followed by a tautomer-pair projection. "
        "`pair_logitdiff_*` converts component probabilities to logits, averages them by validation-selected weights, "
        "then projects each paired tautomer decision with a Bradley-Terry style logit difference. "
        "Selection uses validation data only; test metrics are reported once after selection.\n\n"
        + view.to_markdown(index=False)
        + "\n"
    )


def make_paired_markdown(tests: pd.DataFrame) -> str:
    focus = tests[
        tests["metric"].isin(
            [
                "test_roc_auc",
                "test_brier",
                "test_nll",
                "false_invariance_rate_test_pairs",
                "pair_order_accuracy",
                "pair_gap_mean",
            ]
        )
    ].copy()
    focus = focus.sort_values(["baseline", "candidate", "metric"])
    cols = [
        "candidate",
        "baseline",
        "metric",
        "baseline_mean",
        "candidate_mean",
        "improvement_mean",
        "paired_t_p_two_sided",
        "sign_wins",
        "sign_losses",
        "all_seeds_better",
    ]
    view = focus[cols].copy()
    for col in view.columns:
        view[col] = view[col].map(fmt)
    return (
        "# Phase 1 sPhysNet Pair-Projected Fusion Paired Tests\n\n"
        "`improvement_mean` is signed so positive is better. "
        "The raw-probability-fusion baseline is the same fusion/objective before pair projection.\n\n"
        + view.to_markdown(index=False)
        + "\n"
    )


def make_positive_worst_markdown(worst: pd.DataFrame) -> str:
    if worst.empty:
        return "# Phase 1 sPhysNet Pair-Projected Positive-View Worst Results\n\nNo complete positive-view rows were available.\n"
    cols = [
        "candidate",
        "sphys_mode",
        "projection_policy",
        "val_selection_false_inv_weight",
        "selected_pair_logit_scale_mean",
        "worst_positive_transform_roc_auc_mean",
        "worst_positive_transform_brier_mean",
        "worst_positive_transform_nll_mean",
        "worst_positive_transform_false_invariance_mean",
        "worst_positive_transform_pair_order_accuracy_mean",
    ]
    view = worst.sort_values(
        ["worst_positive_transform_roc_auc_mean", "worst_positive_transform_brier_mean"],
        ascending=[False, True],
    ).copy()
    view = view[[col for col in cols if col in view.columns]]
    for col in view.columns:
        view[col] = view[col].map(fmt)
    return (
        "# Phase 1 sPhysNet Pair-Projected Positive-View Worst Results\n\n"
        "Worst positive-transform values across identity, atom-renumber, and random-SMILES views when all inputs are available.\n\n"
        + view.to_markdown(index=False)
        + "\n"
    )


def run(num_confs: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(__file__).resolve().parents[1]
    rows = []
    for spec in SPECS:
        sphys_by_mode = load_sphys_predictions(root, num_confs, spec.sphys_mode)
        columns = component_columns(spec)
        for objective_weight in OBJECTIVE_WEIGHTS:
            for policy in PROJECTION_POLICIES:
                for seed in SEEDS:
                    val = merge_for_spec(root, sphys_by_mode, spec, seed, "val")
                    weights, scale, best_score, val_metrics = choose_candidate(val, columns, objective_weight, policy)
                    test = merge_for_spec(root, sphys_by_mode, spec, seed, "test")
                    prediction = apply_candidate(test, columns, weights, policy, scale)
                    test_metrics = evaluate_predictions(test, prediction)
                    weight_map = dict(zip(columns, weights))
                    record = {
                        "fusion": spec.label,
                        "seed": seed,
                        "sphys_num_confs": num_confs,
                        "sphys_mode": spec.sphys_mode,
                        "components": "+".join(columns),
                        "projection_policy": policy,
                        "selected_graph_weight": float(weight_map.get(spec.graph.label, np.nan)),
                        "selected_mlp_weight": float(weight_map.get(spec.mlp.label, np.nan)) if spec.mlp is not None else np.nan,
                        "selected_sphysnet_weight": float(weight_map["sphysnet_prediction"]),
                        "selected_pair_logit_scale": scale,
                        "val_selection_false_inv_weight": objective_weight,
                        "val_selection_score": best_score,
                        "val_selected_roc_auc": val_metrics["test_roc_auc"],
                        "val_selected_brier": val_metrics["test_brier"],
                        "val_selected_nll": val_metrics["test_nll"],
                        "val_selected_false_invariance": val_metrics["false_invariance_rate_test_pairs"],
                        "selection_protocol": "seedwise_val_selected_pair_projected_sphysnet_fusion",
                        "note": spec.note,
                        **test_metrics,
                    }
                    record["candidate"] = candidate_name(record)
                    rows.append(record)
    raw = pd.DataFrame(rows)
    summary = summarize_seed_rows(raw)
    tests = paired_tests(root, raw, num_confs)
    return raw, summary, tests


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-confs", type=int, default=10)
    parser.add_argument("--positive-views", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    table_dir = root / "reports" / "tables"
    raw, summary, tests = run(args.num_confs)
    raw.to_csv(table_dir / f"main_sphysnet_numconf{args.num_confs}_pair_projected_fusion_seed_raw.csv", index=False)
    summary.to_csv(table_dir / f"main_sphysnet_numconf{args.num_confs}_pair_projected_fusion_val_selected.csv", index=False)
    tests.to_csv(table_dir / f"main_sphysnet_numconf{args.num_confs}_pair_projected_fusion_paired_tests.csv", index=False)
    (table_dir / f"main_sphysnet_numconf{args.num_confs}_pair_projected_fusion_val_selected.md").write_text(
        make_markdown(summary),
        encoding="utf-8",
    )
    (table_dir / f"main_sphysnet_numconf{args.num_confs}_pair_projected_fusion_paired_tests.md").write_text(
        make_paired_markdown(tests),
        encoding="utf-8",
    )
    if args.positive_views:
        raw_views = positive_view_rows(root, raw, args.num_confs)
        by_transform, worst = summarize_positive_views(raw_views)
        raw_views.to_csv(
            table_dir / f"main_sphysnet_numconf{args.num_confs}_pair_projected_positive_view_eval_seed_raw.csv",
            index=False,
        )
        by_transform.to_csv(
            table_dir / f"main_sphysnet_numconf{args.num_confs}_pair_projected_positive_view_eval_by_transform.csv",
            index=False,
        )
        worst.to_csv(
            table_dir / f"main_sphysnet_numconf{args.num_confs}_pair_projected_positive_view_eval_worst.csv",
            index=False,
        )
        (table_dir / f"main_sphysnet_numconf{args.num_confs}_pair_projected_positive_view_eval_worst.md").write_text(
            make_positive_worst_markdown(worst),
            encoding="utf-8",
        )
    print(
        f"wrote pair-projected sPhysNet fusion num_confs={args.num_confs} "
        f"summary={len(summary)} raw={len(raw)} paired={len(tests)}"
    )


if __name__ == "__main__":
    main()
