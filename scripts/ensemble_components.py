from __future__ import annotations

from dataclasses import dataclass
from math import comb, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from condchemsymm.metrics import classification_metrics

try:
    from scipy import stats
except ImportError:  # pragma: no cover - scipy is optional for this reporting helper.
    stats = None


@dataclass(frozen=True)
class Component:
    label: str
    run_name: str
    family: str
    mode: str


@dataclass(frozen=True)
class EnsembleSpec:
    label: str
    first: Component
    second: Component
    note: str


GRAPH_MAIN = Component(
    "graph_pairm005_select02_mechanistic",
    "main_graph_b256_pairm005_select02_5seed",
    "graph_dp",
    "mechanistic_gate",
)

SPECS = [
    EnsembleSpec(
        "graph_main_plus_mlp_select01_viewpred",
        GRAPH_MAIN,
        Component("mlp_select01_viewpred_mechanistic", "main_concat_select01_viewpred_5seed", "concat", "mechanistic_gate"),
        "Combines graph ROC/Brier strength with the MLP lower false-invariance branch.",
    ),
    EnsembleSpec(
        "graph_main_plus_mlp_select02",
        GRAPH_MAIN,
        Component("mlp_select02_viewpred_mechanistic", "main_concat_select02_viewpred_5seed", "concat", "mechanistic_gate"),
        "Tests whether the stronger MLP selection penalty gives a better ensemble reliability tradeoff.",
    ),
    EnsembleSpec(
        "graph_main_plus_mlp_select03",
        GRAPH_MAIN,
        Component("mlp_select03_viewpred_mechanistic", "main_concat_select03_viewpred_5seed", "concat", "mechanistic_gate"),
        "Tests the low false-invariance end of the MLP branch.",
    ),
    EnsembleSpec(
        "graph_main_plus_mlp_pairm002_select02",
        GRAPH_MAIN,
        Component(
            "mlp_pairm002_select02_viewpred_mechanistic",
            "main_concat_pairm002_select02_viewpred_5seed",
            "concat",
            "mechanistic_gate",
        ),
        "Tests whether MLP pair-margin adds useful pair separation to the graph ensemble.",
    ),
    EnsembleSpec(
        "graph_main_plus_mlp_pairm005_select02",
        GRAPH_MAIN,
        Component(
            "mlp_pairm005_select02_viewpred_mechanistic",
            "main_concat_pairm005_select02_viewpred_5seed",
            "concat",
            "mechanistic_gate",
        ),
        "Tests a stronger MLP pair-margin branch inside the graph ensemble.",
    ),
    EnsembleSpec(
        "graph_main_plus_chemprop_canonical_pos",
        GRAPH_MAIN,
        Component(
            "chemprop_canonical_positive_aug",
            "main_chemprop_v231_tautobase_5seed",
            "chemprop_v231",
            "canonical_positive_aug",
        ),
        "External D-MPNN ensemble check.",
    ),
]

SEEDS = [0, 1, 2, 3, 4]
WEIGHTS = np.round(np.linspace(0.0, 1.0, 101), 2)
VAL_FALSE_INV_OBJECTIVE_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.5]
POSITIVE_TRANSFORMS = ["atom_renumber", "identity", "random_smiles"]
PAIRED_TEST_METRICS = [
    ("test_roc_auc", 1.0),
    ("test_pr_auc", 1.0),
    ("test_brier", -1.0),
    ("test_ece10", -1.0),
    ("false_invariance_rate_test_pairs", -1.0),
    ("pair_order_accuracy", 1.0),
    ("pair_gap_mean", 1.0),
    ("pos_worst_positive_transform_roc_auc", 1.0),
    ("pos_worst_positive_transform_brier", -1.0),
]


def prediction_path(root: Path, component: Component, seed: int, split: str) -> Path:
    return (
        root
        / "runs"
        / component.run_name
        / component.family
        / "tautobase_pref"
        / component.mode
        / f"seed_{seed}"
        / f"{split}_predictions.parquet"
    )


def load_predictions(root: Path, component: Component, seed: int, split: str) -> pd.DataFrame:
    path = prediction_path(root, component, seed, split)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    cols = ["sample_id", "y", "paired_id", "prediction"]
    out = df[cols].copy()
    return out.rename(columns={"prediction": component.label})


def view_prediction_path(root: Path, component: Component, seed: int, transform: str) -> Path:
    return (
        root
        / "runs"
        / component.run_name
        / component.family
        / "tautobase_pref"
        / component.mode
        / f"seed_{seed}"
        / "view_predictions"
        / f"{transform}.parquet"
    )


def load_component_view_predictions(root: Path, component: Component, seed: int, transform: str) -> pd.DataFrame:
    path = view_prediction_path(root, component, seed, transform)
    if path.exists():
        df = pd.read_parquet(path)
        if "y" not in df.columns and "target" in df.columns:
            df["y"] = df["target"]
        cols = ["sample_id", "y", "paired_id", "prediction"]
        return df[cols].copy().rename(columns={"prediction": component.label})
    if component.family == "graph_dp":
        return load_predictions(root, component, seed, "test")
    if transform == "identity":
        return load_predictions(root, component, seed, "test")
    raise FileNotFoundError(path)


def has_split_predictions(root: Path, component: Component, split: str) -> bool:
    return all(prediction_path(root, component, seed, split).exists() for seed in SEEDS)


def merge_components(root: Path, spec: EnsembleSpec, seed: int, split: str) -> pd.DataFrame:
    first = load_predictions(root, spec.first, seed, split)
    second = load_predictions(root, spec.second, seed, split)
    merged = first.merge(second, on=["sample_id", "y", "paired_id"], how="inner")
    if len(merged) != len(first) or len(merged) != len(second):
        raise ValueError(f"prediction row mismatch for {spec.label} seed={seed} split={split}")
    return merged


def paired_stats(df: pd.DataFrame, pred_col: str, threshold: float = 0.25) -> dict[str, float]:
    pred_by_id = dict(zip(df["sample_id"], df[pred_col]))
    y_by_id = dict(zip(df["sample_id"], df["y"]))
    false_flags = []
    order_correct = []
    gaps = []
    seen = set()
    for _, row in df.iterrows():
        sid = row["sample_id"]
        pid = row.get("paired_id", "")
        if not pid or pid not in pred_by_id or sid in seen or pid in seen:
            continue
        seen.add(sid)
        seen.add(pid)
        pred_a = float(pred_by_id[sid])
        pred_b = float(pred_by_id[pid])
        y_a = float(y_by_id[sid])
        y_b = float(y_by_id[pid])
        gap = abs(pred_a - pred_b)
        gaps.append(gap)
        false_flags.append(gap < threshold)
        if y_a != y_b:
            order_correct.append(pred_a > pred_b if y_a > y_b else pred_b > pred_a)
    if not false_flags:
        return {
            "false_invariance_rate_test_pairs": float("nan"),
            "false_invariance_n_pairs": 0,
            "pair_order_accuracy": float("nan"),
            "pair_gap_mean": float("nan"),
            "pair_gap_p10": float("nan"),
        }
    return {
        "false_invariance_rate_test_pairs": float(np.mean(false_flags)),
        "false_invariance_n_pairs": int(len(false_flags)),
        "pair_order_accuracy": float(np.mean(order_correct)) if order_correct else float("nan"),
        "pair_gap_mean": float(np.mean(gaps)),
        "pair_gap_p10": float(np.quantile(gaps, 0.10)),
    }


def evaluate_ensemble(df: pd.DataFrame, spec: EnsembleSpec, first_weight: float) -> dict[str, float]:
    pred = first_weight * df[spec.first.label].to_numpy(dtype=float) + (1.0 - first_weight) * df[
        spec.second.label
    ].to_numpy(dtype=float)
    eval_df = df[["sample_id", "y", "paired_id"]].copy()
    eval_df["ensemble_prediction"] = pred
    metrics = classification_metrics(eval_df["y"].to_numpy(dtype=float), pred)
    pair = paired_stats(eval_df, "ensemble_prediction")
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


def evaluate_positive_views(root: Path, spec: EnsembleSpec, seed: int, first_weight: float) -> dict[str, float | str]:
    view_metrics = {}
    missing = []
    for transform in POSITIVE_TRANSFORMS:
        try:
            first = load_component_view_predictions(root, spec.first, seed, transform)
            second = load_component_view_predictions(root, spec.second, seed, transform)
        except FileNotFoundError as exc:
            missing.append(str(exc))
            continue
        merged = first.merge(second, on=["sample_id", "y", "paired_id"], how="inner")
        if len(merged) != len(first) or len(merged) != len(second):
            raise ValueError(f"view prediction row mismatch for {spec.label} seed={seed} transform={transform}")
        pred = first_weight * merged[spec.first.label].to_numpy(dtype=float) + (1.0 - first_weight) * merged[
            spec.second.label
        ].to_numpy(dtype=float)
        metrics = classification_metrics(merged["y"].to_numpy(dtype=float), pred)
        view_metrics[transform] = metrics

    if len(view_metrics) != len(POSITIVE_TRANSFORMS):
        return {
            "view_eval_status": "missing_positive_view_predictions",
            "pos_worst_positive_transform_roc_auc": float("nan"),
            "pos_worst_positive_transform_brier": float("nan"),
            "pos_view_missing_count": len(missing),
        }

    aucs = [metrics["roc_auc"] for metrics in view_metrics.values()]
    briers = [metrics["brier"] for metrics in view_metrics.values()]
    return {
        "view_eval_status": "positive_views_available",
        "pos_worst_positive_transform_roc_auc": float(np.min(aucs)),
        "pos_worst_positive_transform_brier": float(np.max(briers)),
        "pos_atom_renumber_roc_auc": float(view_metrics["atom_renumber"]["roc_auc"]),
        "pos_identity_roc_auc": float(view_metrics["identity"]["roc_auc"]),
        "pos_random_smiles_roc_auc": float(view_metrics["random_smiles"]["roc_auc"]),
        "pos_atom_renumber_brier": float(view_metrics["atom_renumber"]["brier"]),
        "pos_identity_brier": float(view_metrics["identity"]["brier"]),
        "pos_random_smiles_brier": float(view_metrics["random_smiles"]["brier"]),
        "pos_view_missing_count": 0,
    }


def evaluate_single_component(root: Path, component: Component, seed: int) -> dict[str, float]:
    pred_df = load_predictions(root, component, seed, "test")
    eval_df = pred_df[["sample_id", "y", "paired_id"]].copy()
    eval_df["prediction"] = pred_df[component.label].to_numpy(dtype=float)
    metrics = classification_metrics(eval_df["y"].to_numpy(dtype=float), eval_df["prediction"].to_numpy(dtype=float))
    pair = paired_stats(eval_df, "prediction")
    return {
        "seed": seed,
        "test_accuracy": metrics["accuracy"],
        "test_brier": metrics["brier"],
        "test_nll": metrics["nll"],
        "test_ece10": metrics["ece10"],
        "test_mce10": metrics["mce10"],
        "test_roc_auc": metrics["roc_auc"],
        "test_pr_auc": metrics["pr_auc"],
        **pair,
        "pos_worst_positive_transform_roc_auc": metrics["roc_auc"] if component.family == "graph_dp" else float("nan"),
        "pos_worst_positive_transform_brier": metrics["brier"] if component.family == "graph_dp" else float("nan"),
    }


def summarize(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
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
        "pos_worst_positive_transform_roc_auc",
        "pos_worst_positive_transform_brier",
        "pos_atom_renumber_roc_auc",
        "pos_identity_roc_auc",
        "pos_random_smiles_roc_auc",
        "pos_atom_renumber_brier",
        "pos_identity_brier",
        "pos_random_smiles_brier",
        "pos_view_missing_count",
    ]
    grouped = frame.groupby(keys, dropna=False, sort=True)
    out = grouped[metric_cols].agg(["mean", "std", "count"])
    out.columns = [f"{name}_{stat}" for name, stat in out.columns]
    out = out.reset_index()
    return out


def sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return np.nan
    tail = sum(comb(n, k) for k in range(0, min(wins, losses) + 1)) / (2**n)
    return float(min(1.0, 2.0 * tail))


def paired_t_p(baseline: np.ndarray, candidate: np.ndarray) -> float:
    if stats is None or len(baseline) < 2:
        return np.nan
    return float(stats.ttest_rel(candidate, baseline, nan_policy="omit").pvalue)


def ci95(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return (np.nan, np.nan)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    tcrit = float(stats.t.ppf(0.975, len(values) - 1)) if stats is not None else 2.776
    half = tcrit * sd / sqrt(len(values))
    return (mean - half, mean + half)


def fixed_weight_sweep(root: Path) -> pd.DataFrame:
    rows = []
    for spec in SPECS:
        for seed in SEEDS:
            test = merge_components(root, spec, seed, "test")
            for weight in WEIGHTS:
                metrics = evaluate_ensemble(test, spec, float(weight))
                view_metrics = evaluate_positive_views(root, spec, seed, float(weight))
                rows.append(
                    {
                        "ensemble": spec.label,
                        "seed": seed,
                        "first_component": spec.first.label,
                        "second_component": spec.second.label,
                        "first_weight": float(weight),
                        "selection_protocol": "fixed_test_sweep_exploratory",
                        "view_eval_status": view_metrics["view_eval_status"],
                        **metrics,
                        **view_metrics,
                    }
                )
    raw = pd.DataFrame(rows)
    return summarize(raw, ["ensemble", "first_component", "second_component", "first_weight", "selection_protocol", "view_eval_status"])


def val_selected(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows = []
    for spec in SPECS:
        if not has_split_predictions(root, spec.first, "val") or not has_split_predictions(root, spec.second, "val"):
            continue
        for objective_weight in VAL_FALSE_INV_OBJECTIVE_WEIGHTS:
            for seed in SEEDS:
                val = merge_components(root, spec, seed, "val")
                scored = []
                for weight in WEIGHTS:
                    metrics = evaluate_ensemble(val, spec, float(weight))
                    score = metrics["test_roc_auc"] - objective_weight * metrics["false_invariance_rate_test_pairs"]
                    scored.append((score, float(weight), metrics))
                best_score, best_weight, best_val_metrics = max(scored, key=lambda item: (item[0], item[1]))
                test = merge_components(root, spec, seed, "test")
                test_metrics = evaluate_ensemble(test, spec, best_weight)
                view_metrics = evaluate_positive_views(root, spec, seed, best_weight)
                raw_rows.append(
                    {
                        "ensemble": spec.label,
                        "seed": seed,
                        "first_component": spec.first.label,
                        "second_component": spec.second.label,
                        "selected_first_weight": best_weight,
                        "val_selection_false_inv_weight": objective_weight,
                        "val_selection_score": float(best_score),
                        "val_selected_roc_auc": best_val_metrics["test_roc_auc"],
                        "val_selected_false_invariance": best_val_metrics["false_invariance_rate_test_pairs"],
                        "selection_protocol": "seedwise_val_selected",
                        "view_eval_status": view_metrics["view_eval_status"],
                        "note": spec.note,
                        **test_metrics,
                        **view_metrics,
                    }
                )
    raw = pd.DataFrame(raw_rows)
    summary = summarize(
        raw,
        [
            "ensemble",
            "first_component",
            "second_component",
            "val_selection_false_inv_weight",
            "selection_protocol",
            "view_eval_status",
            "note",
        ],
    )
    chosen = raw.groupby(["ensemble", "val_selection_false_inv_weight"], sort=True)["selected_first_weight"].agg(
        ["mean", "std", lambda x: ",".join(f"{v:.2f}" for v in x)]
    )
    chosen.columns = ["selected_first_weight_mean", "selected_first_weight_std", "selected_first_weights_by_seed"]
    summary = summary.merge(chosen.reset_index(), on=["ensemble", "val_selection_false_inv_weight"], how="left")
    return raw, summary


def ensemble_paired_tests(root: Path, selected_raw: pd.DataFrame) -> pd.DataFrame:
    baseline_rows = [evaluate_single_component(root, GRAPH_MAIN, seed) for seed in SEEDS]
    baseline = pd.DataFrame(baseline_rows)
    rows = []
    for (ensemble, objective_weight), group in selected_raw.groupby(["ensemble", "val_selection_false_inv_weight"], sort=True):
        merged = baseline.merge(group, on="seed", suffixes=("_baseline", "_candidate"))
        for metric, direction in PAIRED_TEST_METRICS:
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
            ties = int(np.sum(improvement == 0))
            rows.append(
                {
                    "comparison": f"{ensemble}_valobj{objective_weight:g}_vs_graph_main",
                    "ensemble": ensemble,
                    "val_selection_false_inv_weight": objective_weight,
                    "baseline": GRAPH_MAIN.label,
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
                    "sign_ties": ties,
                    "all_seeds_better": bool(wins == len(metric_df)),
                }
            )
    return pd.DataFrame(rows)


def fmt(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def make_markdown(summary: pd.DataFrame) -> str:
    view = summary.sort_values(
        ["test_roc_auc_mean", "test_brier_mean", "false_invariance_rate_test_pairs_mean"],
        ascending=[False, True, True],
    ).copy()
    view = view[
        [
            "ensemble",
            "val_selection_false_inv_weight",
            "selected_first_weight_mean",
            "selected_first_weights_by_seed",
            "test_roc_auc_mean",
            "test_roc_auc_std",
            "test_brier_mean",
            "test_brier_std",
            "test_ece10_mean",
            "pos_worst_positive_transform_roc_auc_mean",
            "pos_worst_positive_transform_brier_mean",
            "false_invariance_rate_test_pairs_mean",
            "false_invariance_rate_test_pairs_std",
            "pair_order_accuracy_mean",
            "pair_gap_mean_mean",
            "view_eval_status",
        ]
    ]
    for col in view.columns:
        view[col] = view[col].map(fmt)
    return (
        "# Phase 1 Prediction Ensemble Val-Selected Results\n\n"
        "Seedwise validation selects the ensemble probability weight on duplicate-aware Tautobase; test metrics are then averaged over five seeds. "
        "`selected_first_weight` is the graph-main probability weight for all current rows. Rows marked `positive_views_available` include atom-renumber/random-SMILES view metrics; rows marked missing are kept as identity/pair-only exploratory comparisons.\n\n"
        + view.to_markdown(index=False)
        + "\n"
    )


def make_paired_markdown(tests: pd.DataFrame) -> str:
    focus = tests[
        (tests["ensemble"] == "graph_main_plus_mlp_select01_viewpred")
        & (tests["val_selection_false_inv_weight"].isin([0.0, 0.1, 0.2]))
        & (
            tests["metric"].isin(
                [
                    "test_roc_auc",
                    "test_brier",
                    "false_invariance_rate_test_pairs",
                    "pair_order_accuracy",
                    "pair_gap_mean",
                    "pos_worst_positive_transform_roc_auc",
                    "pos_worst_positive_transform_brier",
                ]
            )
        )
    ].copy()
    focus = focus.sort_values(["val_selection_false_inv_weight", "metric"])
    cols = [
        "ensemble",
        "val_selection_false_inv_weight",
        "metric",
        "baseline_mean",
        "candidate_mean",
        "improvement_mean",
        "improvement_ci95_low",
        "improvement_ci95_high",
        "paired_t_p_two_sided",
        "sign_wins",
        "sign_losses",
        "all_seeds_better",
    ]
    view = focus[cols].copy()
    for col in view.columns:
        view[col] = view[col].map(fmt)
    return (
        "# Phase 1 Prediction Ensemble Paired Tests\n\n"
        "Rows compare val-selected prediction ensembles against the graph main recipe by aligned seed. `improvement_mean` is signed so positive is better.\n\n"
        + view.to_markdown(index=False)
        + "\n"
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    table_dir = root / "reports" / "tables"
    sweep = fixed_weight_sweep(root)
    raw_selected, selected = val_selected(root)
    paired_tests = ensemble_paired_tests(root, raw_selected)
    sweep.to_csv(table_dir / "main_prediction_ensemble_sweep.csv", index=False)
    raw_selected.to_csv(table_dir / "main_prediction_ensemble_val_selected_seed_raw.csv", index=False)
    selected.to_csv(table_dir / "main_prediction_ensemble_val_selected.csv", index=False)
    (table_dir / "main_prediction_ensemble_val_selected.md").write_text(make_markdown(selected), encoding="utf-8")
    paired_tests.to_csv(table_dir / "main_prediction_ensemble_paired_tests.csv", index=False)
    (table_dir / "main_prediction_ensemble_paired_tests.md").write_text(make_paired_markdown(paired_tests), encoding="utf-8")
    print(
        "wrote ensemble tables rows "
        f"sweep={len(sweep)} selected={len(selected)} raw={len(raw_selected)} paired={len(paired_tests)}"
    )


if __name__ == "__main__":
    main()
