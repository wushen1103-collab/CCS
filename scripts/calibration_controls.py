from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ensemble_components import POSITIVE_TRANSFORMS, SEEDS, ci95, paired_stats, paired_t_p
from fusion_components import (
    PAIRED_METRICS,
    SPECS,
    component_columns,
    load_sphys_predictions,
    merge_for_spec,
)
from pair_projection import apply_candidate, evaluate_predictions, safe_logit
from analyze_sphysnet_positive_views import sphys_sample_predictions


TEMPERATURE_GRID = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.6,
    1.7,
    1.8,
    1.9,
    2.0,
    2.1,
    2.2,
    2.3,
    2.4,
    2.5,
    2.6,
    2.7,
    2.8,
    2.9,
    3.0,
    3.25,
    3.5,
    4.0,
    5.0,
    6.0,
    8.0,
    10.0,
]


def sigmoid(score: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(score, -50.0, 50.0)))


def calibrate_probabilities(prediction: np.ndarray, temperature: float) -> np.ndarray:
    return np.clip(sigmoid(safe_logit(prediction) / float(temperature)), 1e-7, 1.0 - 1e-7)


def nll(y: np.ndarray, prediction: np.ndarray) -> float:
    pred = np.clip(prediction, 1e-7, 1.0 - 1e-7)
    return float(np.mean(-(y * np.log(pred) + (1.0 - y) * np.log(1.0 - pred))))


def brier(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((prediction - y) ** 2))


def false_invariance_cap(objective: str) -> float | None:
    return {
        "nll_ficap08": 0.08,
        "nll_ficap10": 0.10,
        "nll_ficap12": 0.12,
        "nll_ficap13": 0.13,
        "nll_ficap14": 0.14,
        "nll_ficap15": 0.15,
    }.get(objective)


def choose_temperature(val_frame: pd.DataFrame, prediction: np.ndarray, objective: str) -> tuple[float, float]:
    y = val_frame["y"].to_numpy(dtype=float)
    scored = []
    for temperature in TEMPERATURE_GRID:
        calibrated = calibrate_probabilities(prediction, temperature)
        metrics = evaluate_predictions(val_frame, calibrated)
        scored.append(
            {
                "temperature": float(temperature),
                "nll": nll(y, calibrated),
                "brier": brier(y, calibrated),
                "false_invariance": metrics["false_invariance_rate_test_pairs"],
            }
        )
    scored_df = pd.DataFrame(scored)
    cap = false_invariance_cap(objective)
    if cap is not None:
        feasible = scored_df[scored_df["false_invariance"].le(cap + 1e-12)].copy()
        if feasible.empty:
            feasible = scored_df.copy()
        best = feasible.sort_values(["nll", "brier", "false_invariance", "temperature"]).iloc[0]
        return float(best["temperature"]), float(best["nll"])
    score_col = "nll" if objective == "nll" else "brier"
    best = scored_df.sort_values([score_col, "false_invariance", "temperature"]).iloc[0]
    return float(best["temperature"]), float(best[score_col])


def weights_from_row(row: pd.Series, columns: list[str], spec) -> np.ndarray:
    weights = []
    for col in columns:
        if col == spec.graph.label:
            weights.append(float(row["selected_graph_weight"]))
        elif spec.mlp is not None and col == spec.mlp.label:
            weights.append(float(row["selected_mlp_weight"]))
        elif col == "sphysnet_prediction":
            weights.append(float(row["selected_sphysnet_weight"]))
        else:
            raise ValueError(f"unexpected component column: {col}")
    return np.asarray(weights, dtype=float)


def calibrated_candidate_name(row: pd.Series, objective: str) -> str:
    return f"{row['candidate']}_val{objective}tempcal"


def run(num_confs: int, objective: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(__file__).resolve().parents[1]
    table_dir = root / "reports" / "tables"
    selected = pd.read_csv(table_dir / f"main_sphysnet_numconf{num_confs}_pair_projected_fusion_seed_raw.csv")
    spec_by_label = {spec.label: spec for spec in SPECS}
    sphys_cache: dict[tuple[int, str], pd.DataFrame] = {}
    rows = []
    for _, row in selected.iterrows():
        spec = spec_by_label[str(row["fusion"])]
        columns = component_columns(spec)
        key = (num_confs, str(row["sphys_mode"]))
        if key not in sphys_cache:
            sphys_cache[key] = load_sphys_predictions(root, num_confs, str(row["sphys_mode"]))
        sphys_by_mode = sphys_cache[key]
        seed = int(row["seed"])
        weights = weights_from_row(row, columns, spec)
        val = merge_for_spec(root, sphys_by_mode, spec, seed, "val")
        val_prediction = apply_candidate(
            val,
            columns,
            weights,
            str(row["projection_policy"]),
            float(row["selected_pair_logit_scale"]),
        )
        temperature, val_score = choose_temperature(val, val_prediction, objective)
        test = merge_for_spec(root, sphys_by_mode, spec, seed, "test")
        test_prediction = apply_candidate(
            test,
            columns,
            weights,
            str(row["projection_policy"]),
            float(row["selected_pair_logit_scale"]),
        )
        calibrated_test = calibrate_probabilities(test_prediction, temperature)
        metrics = evaluate_predictions(test, calibrated_test)
        rows.append(
            {
                **row.to_dict(),
                "base_candidate": row["candidate"],
                "candidate": calibrated_candidate_name(row, objective),
                "calibration_objective": objective,
                "selected_calibration_temperature": temperature,
                "val_calibration_score": val_score,
                **metrics,
            }
        )
    raw = pd.DataFrame(rows)
    summary = summarize_seed_rows(raw)
    tests = paired_tests(root, raw, selected, num_confs)
    return raw, summary, tests


def summarize_seed_rows(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "candidate",
        "base_candidate",
        "fusion",
        "sphys_num_confs",
        "sphys_mode",
        "components",
        "projection_policy",
        "val_selection_false_inv_weight",
        "calibration_objective",
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
        "selected_calibration_temperature",
    ]
    grouped = frame.groupby(keys, dropna=False, sort=True)
    out = grouped[metric_cols].agg(["mean", "std", "count"])
    out.columns = [f"{name}_{stat}" for name, stat in out.columns]
    out = out.reset_index()
    weight_cols = [col for col in frame.columns if col.startswith("selected_") and col.endswith("_weight")]
    weights = grouped[weight_cols].agg(["mean", "std"])
    weights.columns = [f"{name}_{stat}" for name, stat in weights.columns]
    weights = weights.reset_index()
    temp_rows = []
    for key_values, group in frame.groupby(keys, dropna=False, sort=True):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = dict(zip(keys, key_values))
        row["selected_calibration_temperature_by_seed"] = ",".join(
            f"{float(record['selected_calibration_temperature']):.2f}"
            for _, record in group.sort_values("seed").iterrows()
        )
        temp_rows.append(row)
    return out.merge(weights, on=keys, how="left").merge(pd.DataFrame(temp_rows), on=keys, how="left")


def paired_tests(root: Path, calibrated_raw: pd.DataFrame, base_raw: pd.DataFrame, num_confs: int) -> pd.DataFrame:
    fusion_raw_path = root / "reports" / "tables" / f"main_sphysnet_numconf{num_confs}_fusion_val_selected_seed_raw.csv"
    raw_fusion = pd.read_csv(fusion_raw_path) if fusion_raw_path.exists() else pd.DataFrame()
    rows = []
    for candidate_name, group in calibrated_raw.groupby("candidate", sort=True):
        base_candidate_name = str(group["base_candidate"].iloc[0])
        baselines = {
            "uncalibrated_pair_projected_same_candidate": base_raw[base_raw["candidate"].eq(base_candidate_name)].copy()
        }
        if not raw_fusion.empty:
            fusion_name = str(group["fusion"].iloc[0])
            objective_weight = float(group["val_selection_false_inv_weight"].iloc[0])
            raw_group = raw_fusion[
                raw_fusion["fusion"].eq(fusion_name)
                & np.isclose(raw_fusion["val_selection_false_inv_weight"].astype(float), objective_weight)
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
                rows.append(
                    {
                        "comparison": f"{candidate_name}_vs_{baseline_name}",
                        "candidate": candidate_name,
                        "base_candidate": base_candidate_name,
                        "baseline": baseline_name,
                        "metric": metric,
                        "n_paired_seeds": len(metric_df),
                        "baseline_mean": float(np.mean(baseline_values)),
                        "candidate_mean": float(np.mean(candidate_values)),
                        "improvement_mean": float(np.mean(improvement)),
                        "improvement_std": float(np.std(improvement, ddof=1)) if len(improvement) > 1 else np.nan,
                        "improvement_ci95_low": low,
                        "improvement_ci95_high": high,
                        "paired_t_p_two_sided": paired_t_p(baseline_values, candidate_values),
                        "sign_wins": int(np.sum(improvement > 0)),
                        "sign_losses": int(np.sum(improvement < 0)),
                        "sign_ties": int(np.sum(improvement == 0)),
                    }
                )
    return pd.DataFrame(rows)


def merge_view_for_spec(root: Path, spec, num_confs: int, seed: int, transform: str) -> pd.DataFrame:
    from ensemble_components import load_component_view_predictions

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
        weights = weights_from_row(selected, columns, spec)
        for transform in POSITIVE_TRANSFORMS:
            try:
                view = merge_view_for_spec(root, spec, num_confs, int(selected["seed"]), transform)
            except FileNotFoundError:
                continue
            prediction = apply_candidate(
                view,
                columns,
                weights,
                str(selected["projection_policy"]),
                float(selected["selected_pair_logit_scale"]),
            )
            calibrated = calibrate_probabilities(prediction, float(selected["selected_calibration_temperature"]))
            metrics = evaluate_predictions(view, calibrated)
            rows.append(
                {
                    "candidate": selected["candidate"],
                    "base_candidate": selected["base_candidate"],
                    "fusion": selected["fusion"],
                    "seed": int(selected["seed"]),
                    "sphys_num_confs": num_confs,
                    "sphys_mode": selected["sphys_mode"],
                    "projection_policy": selected["projection_policy"],
                    "val_selection_false_inv_weight": selected["val_selection_false_inv_weight"],
                    "calibration_objective": selected["calibration_objective"],
                    "transform_type": transform,
                    "selected_pair_logit_scale": selected["selected_pair_logit_scale"],
                    "selected_calibration_temperature": selected["selected_calibration_temperature"],
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
        "base_candidate",
        "fusion",
        "sphys_num_confs",
        "sphys_mode",
        "projection_policy",
        "val_selection_false_inv_weight",
        "calibration_objective",
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
                "selected_mlp_weight_mean": raw_group["selected_mlp_weight"].mean(),
                "selected_sphysnet_weight_mean": raw_group["selected_sphysnet_weight"].mean(),
                "selected_pair_logit_scale_mean": raw_group["selected_pair_logit_scale"].mean(),
                "selected_calibration_temperature_mean": raw_group["selected_calibration_temperature"].mean(),
                "worst_positive_transform_roc_auc_mean": group["test_roc_auc_mean"].min(),
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


def make_markdown(summary: pd.DataFrame) -> str:
    view = summary.sort_values(["test_roc_auc_mean", "test_brier_mean"], ascending=[False, True]).copy()
    cols = [
        "candidate",
        "base_candidate",
        "projection_policy",
        "sphys_mode",
        "val_selection_false_inv_weight",
        "selected_pair_logit_scale_mean",
        "selected_calibration_temperature_mean",
        "selected_calibration_temperature_by_seed",
        "test_roc_auc_mean",
        "test_brier_mean",
        "test_nll_mean",
        "false_invariance_rate_test_pairs_mean",
        "pair_order_accuracy_mean",
    ]
    view = view[[col for col in cols if col in view.columns]]
    return (
        "# Phase 1 sPhysNet Pair-Projected Fusion With Validation Temperature Calibration\n\n"
        "Temperature is selected on validation NLL or Brier after pair projection, then applied to test predictions. "
        "The transform is monotone, so ROC/pair ordering should remain unchanged while calibration metrics may improve.\n\n"
        + view.to_markdown(index=False, floatfmt=".6f")
        + "\n"
    )


def make_positive_worst_markdown(worst: pd.DataFrame) -> str:
    if worst.empty:
        return "# Phase 1 sPhysNet Pair-Projected Calibrated Positive-View Worst Results\n\nNo complete rows.\n"
    view = worst.sort_values(
        ["worst_positive_transform_roc_auc_mean", "worst_positive_transform_brier_mean"],
        ascending=[False, True],
    ).copy()
    cols = [
        "candidate",
        "base_candidate",
        "selected_pair_logit_scale_mean",
        "selected_calibration_temperature_mean",
        "worst_positive_transform_roc_auc_mean",
        "worst_positive_transform_brier_mean",
        "worst_positive_transform_nll_mean",
        "worst_positive_transform_false_invariance_mean",
    ]
    view = view[[col for col in cols if col in view.columns]]
    return (
        "# Phase 1 sPhysNet Pair-Projected Calibrated Positive-View Worst Results\n\n"
        + view.to_markdown(index=False, floatfmt=".6f")
        + "\n"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-confs", type=int, default=10)
    parser.add_argument(
        "--objective",
        choices=[
            "nll",
            "brier",
            "nll_ficap08",
            "nll_ficap10",
            "nll_ficap12",
            "nll_ficap13",
            "nll_ficap14",
            "nll_ficap15",
        ],
        default="nll",
    )
    parser.add_argument("--positive-views", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    table_dir = root / "reports" / "tables"
    raw, summary, tests = run(args.num_confs, args.objective)
    prefix = f"main_sphysnet_numconf{args.num_confs}_pair_projected_val{args.objective}tempcal"
    raw.to_csv(table_dir / f"{prefix}_fusion_seed_raw.csv", index=False)
    summary.to_csv(table_dir / f"{prefix}_fusion_val_selected.csv", index=False)
    tests.to_csv(table_dir / f"{prefix}_fusion_paired_tests.csv", index=False)
    (table_dir / f"{prefix}_fusion_val_selected.md").write_text(make_markdown(summary), encoding="utf-8")
    if args.positive_views:
        raw_views = positive_view_rows(root, raw, args.num_confs)
        by_transform, worst = summarize_positive_views(raw_views)
        raw_views.to_csv(table_dir / f"{prefix}_positive_view_eval_seed_raw.csv", index=False)
        by_transform.to_csv(table_dir / f"{prefix}_positive_view_eval_by_transform.csv", index=False)
        worst.to_csv(table_dir / f"{prefix}_positive_view_eval_worst.csv", index=False)
        (table_dir / f"{prefix}_positive_view_eval_worst.md").write_text(
            make_positive_worst_markdown(worst),
            encoding="utf-8",
        )
    print(
        f"wrote calibrated pair-projected sPhysNet fusion num_confs={args.num_confs} "
        f"objective={args.objective} summary={len(summary)} raw={len(raw)} paired={len(tests)}"
    )


if __name__ == "__main__":
    main()
