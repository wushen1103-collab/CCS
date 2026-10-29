from __future__ import annotations

from dataclasses import dataclass
from math import comb, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from condchemsymm.metrics import classification_metrics
from predict_energy_expert import samples_from_pair_predictions
from ensemble_components import (
    GRAPH_MAIN,
    SEEDS,
    Component,
    ci95,
    load_predictions,
    paired_stats,
    paired_t_p,
)


@dataclass(frozen=True)
class FusionSpec:
    label: str
    graph: Component
    mlp: Component | None
    sphys_mode: str
    note: str


MLP_SELECT01 = Component(
    "mlp_select01_viewpred_mechanistic",
    "main_concat_select01_viewpred_5seed",
    "concat",
    "mechanistic_gate",
)
MLP_SELECT02 = Component(
    "mlp_select02_viewpred_mechanistic",
    "main_concat_select02_viewpred_5seed",
    "concat",
    "mechanistic_gate",
)
MLP_SELECT03 = Component(
    "mlp_select03_viewpred_mechanistic",
    "main_concat_select03_viewpred_5seed",
    "concat",
    "mechanistic_gate",
)

SPECS = [
    FusionSpec(
        "graph_sphys_boltzmann",
        GRAPH_MAIN,
        None,
        "boltzmann_rt_298k",
        "Two-way fusion between the internal graph candidate and the released sPhysNet-Taut pair expert.",
    ),
    FusionSpec(
        "graph_sphys_valtemp",
        GRAPH_MAIN,
        None,
        "val_nll_temperature",
        "Two-way fusion using the validation-temperature-calibrated sPhysNet probabilities.",
    ),
    FusionSpec(
        "graph_mlp_select01_sphys_boltzmann",
        GRAPH_MAIN,
        MLP_SELECT01,
        "boltzmann_rt_298k",
        "Jointly selects graph, MLP select01, and sPhysNet Boltzmann weights.",
    ),
    FusionSpec(
        "graph_mlp_select01_sphys_valtemp",
        GRAPH_MAIN,
        MLP_SELECT01,
        "val_nll_temperature",
        "Jointly selects graph, MLP select01, and calibrated sPhysNet weights.",
    ),
    FusionSpec(
        "graph_mlp_select02_sphys_boltzmann",
        GRAPH_MAIN,
        MLP_SELECT02,
        "boltzmann_rt_298k",
        "Tests whether the stronger MLP reliability branch helps sPhysNet fusion.",
    ),
    FusionSpec(
        "graph_mlp_select03_sphys_boltzmann",
        GRAPH_MAIN,
        MLP_SELECT03,
        "boltzmann_rt_298k",
        "Tests the low false-invariance MLP branch inside sPhysNet fusion.",
    ),
]

VAL_FALSE_INV_OBJECTIVE_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.5]
WEIGHT_DENOM = 50
PAIRED_METRICS = [
    ("test_roc_auc", 1.0),
    ("test_pr_auc", 1.0),
    ("test_brier", -1.0),
    ("test_nll", -1.0),
    ("test_ece10", -1.0),
    ("false_invariance_rate_test_pairs", -1.0),
    ("pair_order_accuracy", 1.0),
    ("pair_gap_mean", 1.0),
]


def simplex_weights(n_components: int) -> np.ndarray:
    if n_components == 2:
        return np.asarray([(i / WEIGHT_DENOM, 1.0 - i / WEIGHT_DENOM) for i in range(WEIGHT_DENOM + 1)], dtype=float)
    if n_components == 3:
        weights = []
        for first in range(WEIGHT_DENOM + 1):
            for second in range(WEIGHT_DENOM - first + 1):
                third = WEIGHT_DENOM - first - second
                weights.append((first / WEIGHT_DENOM, second / WEIGHT_DENOM, third / WEIGHT_DENOM))
        return np.asarray(weights, dtype=float)
    raise ValueError(f"unsupported component count: {n_components}")


def fast_auc_scores(y_true: np.ndarray, predictions: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    y_bool = y_true.astype(float) > 0.5
    pos = predictions[:, y_bool]
    neg = predictions[:, ~y_bool]
    if pos.shape[1] == 0 or neg.shape[1] == 0:
        return np.full(predictions.shape[0], np.nan)
    denom = float(pos.shape[1] * neg.shape[1])
    scores = np.empty(predictions.shape[0], dtype=float)
    for start in range(0, predictions.shape[0], chunk_size):
        stop = min(start + chunk_size, predictions.shape[0])
        diff = pos[start:stop, :, None] - neg[start:stop, None, :]
        scores[start:stop] = ((diff > 0).sum(axis=(1, 2)) + 0.5 * (diff == 0).sum(axis=(1, 2))) / denom
    return scores


def pair_indices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    row_by_id = {sample_id: idx for idx, sample_id in enumerate(df["sample_id"])}
    left = []
    right = []
    seen = set()
    for _, row in df.iterrows():
        sid = row["sample_id"]
        pid = row.get("paired_id", "")
        if not pid or pid not in row_by_id or sid in seen or pid in seen:
            continue
        seen.add(sid)
        seen.add(pid)
        left.append(row_by_id[sid])
        right.append(row_by_id[pid])
    return np.asarray(left, dtype=int), np.asarray(right, dtype=int)


def fast_false_invariance_scores(predictions: np.ndarray, left: np.ndarray, right: np.ndarray, threshold: float = 0.25) -> np.ndarray:
    if len(left) == 0:
        return np.full(predictions.shape[0], np.nan)
    gaps = np.abs(predictions[:, left] - predictions[:, right])
    return np.mean(gaps < threshold, axis=1)


def load_sphys_temperature(table_dir: Path, num_confs: int, mode: str) -> float:
    path = table_dir / f"main_sphysnet_tautobase_numconf{num_confs}_baseline.csv"
    df = pd.read_csv(path)
    rows = df[df["mode"].eq(mode)]
    if rows.empty:
        raise ValueError(f"missing sPhysNet mode={mode} in {path}")
    return float(rows.iloc[0]["temperature"])


def load_sphys_predictions(root: Path, num_confs: int, mode: str) -> pd.DataFrame:
    table_dir = root / "reports" / "tables"
    pair_path = table_dir / f"main_sphysnet_tautobase_numconf{num_confs}_pair_predictions.csv"
    pair_df = pd.read_csv(pair_path)
    pair_df = pair_df[pair_df["status"].eq("ok")].copy()
    temperature = load_sphys_temperature(table_dir, num_confs, mode)
    sample_df = samples_from_pair_predictions(pair_df, temperature)
    sample_df = sample_df[["sample_id", "y", "paired_id", "split", "prediction", "energy_score"]].copy()
    return sample_df.rename(columns={"prediction": "sphysnet_prediction"})


def has_split_predictions(root: Path, component: Component, split: str) -> bool:
    return all(
        (
            root
            / "runs"
            / component.run_name
            / component.family
            / "tautobase_pref"
            / component.mode
            / f"seed_{seed}"
            / f"{split}_predictions.parquet"
        ).exists()
        for seed in SEEDS
    )


def merge_for_spec(root: Path, sphys: pd.DataFrame, spec: FusionSpec, seed: int, split: str) -> pd.DataFrame:
    graph = load_predictions(root, spec.graph, seed, split)
    merged = graph.merge(
        sphys[sphys["split"].eq(split)].drop(columns=["split"]),
        on=["sample_id", "y", "paired_id"],
        how="inner",
    )
    if len(merged) != len(graph):
        raise ValueError(f"sPhysNet row mismatch for {spec.label} seed={seed} split={split}: {len(merged)} != {len(graph)}")
    if spec.mlp is not None:
        mlp = load_predictions(root, spec.mlp, seed, split)
        merged = merged.merge(mlp, on=["sample_id", "y", "paired_id"], how="inner")
        if len(merged) != len(graph) or len(merged) != len(mlp):
            raise ValueError(f"MLP row mismatch for {spec.label} seed={seed} split={split}")
    return merged


def component_columns(spec: FusionSpec) -> list[str]:
    cols = [spec.graph.label]
    if spec.mlp is not None:
        cols.append(spec.mlp.label)
    cols.append("sphysnet_prediction")
    return cols


def evaluate_predictions(df: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    eval_df = df[["sample_id", "y", "paired_id"]].copy()
    eval_df["ensemble_prediction"] = prediction
    metrics = classification_metrics(eval_df["y"].to_numpy(dtype=float), prediction)
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


def evaluate_weighted(df: pd.DataFrame, spec: FusionSpec, weights: np.ndarray) -> dict[str, float]:
    components = df[component_columns(spec)].to_numpy(dtype=float)
    prediction = components @ weights
    return evaluate_predictions(df, prediction)


def select_weights_on_validation(df: pd.DataFrame, spec: FusionSpec, objective_weight: float) -> tuple[np.ndarray, float, dict[str, float]]:
    columns = component_columns(spec)
    weights = simplex_weights(len(columns))
    components = df[columns].to_numpy(dtype=float)
    predictions = weights @ components.T
    aucs = fast_auc_scores(df["y"].to_numpy(dtype=float), predictions)
    left, right = pair_indices(df)
    false_invariance = fast_false_invariance_scores(predictions, left, right)
    scores = aucs - objective_weight * false_invariance
    best_score = float(np.nanmax(scores))
    candidate_idx = np.flatnonzero(np.isclose(scores, best_score, rtol=0.0, atol=1e-12))

    # Prefer more internal weight on exact ties; this avoids gratuitous dependence on the external expert.
    def tie_key(idx: int) -> tuple[float, float]:
        sphys_weight = weights[idx, -1]
        graph_weight = weights[idx, 0]
        return (-float(sphys_weight), float(graph_weight))

    best_idx = max(candidate_idx, key=tie_key)
    best_weights = weights[best_idx]
    best_metrics = evaluate_weighted(df, spec, best_weights)
    return best_weights, best_score, best_metrics


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "fusion",
        "sphys_num_confs",
        "sphys_mode",
        "components",
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
        weight_rows.append(row)
    weight_strings = pd.DataFrame(weight_rows)
    out = out.merge(chosen, on=keys, how="left").merge(weight_strings, on=keys, how="left")
    return out


def sphys_baseline_seed_rows(root: Path, num_confs: int, mode: str) -> pd.DataFrame:
    sphys = load_sphys_predictions(root, num_confs, mode)
    rows = []
    for seed in SEEDS:
        test = sphys[sphys["split"].eq("test")].copy()
        metrics = evaluate_predictions(test, test["sphysnet_prediction"].to_numpy(dtype=float))
        rows.append({"seed": seed, **metrics})
    return pd.DataFrame(rows)


def graph_baseline_seed_rows(root: Path) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        pred = load_predictions(root, GRAPH_MAIN, seed, "test")
        metrics = evaluate_predictions(pred, pred[GRAPH_MAIN.label].to_numpy(dtype=float))
        rows.append({"seed": seed, **metrics})
    return pd.DataFrame(rows)


def paired_tests(root: Path, selected_raw: pd.DataFrame, num_confs: int) -> pd.DataFrame:
    baselines = {
        "graph_main": graph_baseline_seed_rows(root),
        "sphys_boltzmann": sphys_baseline_seed_rows(root, num_confs, "boltzmann_rt_298k"),
        "sphys_valtemp": sphys_baseline_seed_rows(root, num_confs, "val_nll_temperature"),
    }
    rows = []
    for (fusion, objective_weight), group in selected_raw.groupby(["fusion", "val_selection_false_inv_weight"], sort=True):
        for baseline_name, baseline in baselines.items():
            merged = baseline.merge(group, on="seed", suffixes=("_baseline", "_candidate"))
            for metric, direction in PAIRED_METRICS:
                base_col = f"{metric}_baseline"
                cand_col = f"{metric}_candidate"
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
                        "comparison": f"{fusion}_valobj{objective_weight:g}_vs_{baseline_name}",
                        "fusion": fusion,
                        "baseline": baseline_name,
                        "val_selection_false_inv_weight": objective_weight,
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
    cols = [
        "fusion",
        "sphys_num_confs",
        "sphys_mode",
        "val_selection_false_inv_weight",
        "selected_graph_weight_mean",
        "selected_mlp_weight_mean",
        "selected_sphysnet_weight_mean",
        "selected_weights_by_seed",
        "test_roc_auc_mean",
        "test_roc_auc_std",
        "test_brier_mean",
        "test_nll_mean",
        "test_ece10_mean",
        "false_invariance_rate_test_pairs_mean",
        "pair_order_accuracy_mean",
        "pair_gap_mean_mean",
    ]
    view = view[[col for col in cols if col in view.columns]]
    for col in view.columns:
        view[col] = view[col].map(fmt)
    return (
        "# Phase 1 sPhysNet Fusion Results\n\n"
        "Seedwise validation-selected probability fusion between CondChemSymm candidates and the released sPhysNet-Taut pair expert. "
        "`selected_weights_by_seed` follows graph/MLP/sPhysNet for three-component rows and graph/sPhysNet for two-component rows. "
        "sPhysNet is an external deterministic baseline, so these rows are fusion/provenance checks rather than five-seed trained-method evidence.\n\n"
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
    focus = focus.sort_values(["baseline", "fusion", "val_selection_false_inv_weight", "metric"])
    cols = [
        "fusion",
        "baseline",
        "val_selection_false_inv_weight",
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
        "# Phase 1 sPhysNet Fusion Paired Tests\n\n"
        "Rows compare each validation-selected fusion candidate with graph-main and replicated deterministic sPhysNet baselines by aligned seed. "
        "`improvement_mean` is signed so positive is better.\n\n"
        + view.to_markdown(index=False)
        + "\n"
    )


def run(num_confs: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(__file__).resolve().parents[1]
    rows = []
    for spec in SPECS:
        if not has_split_predictions(root, spec.graph, "val") or not has_split_predictions(root, spec.graph, "test"):
            continue
        if spec.mlp is not None and (
            not has_split_predictions(root, spec.mlp, "val") or not has_split_predictions(root, spec.mlp, "test")
        ):
            continue
        sphys = load_sphys_predictions(root, num_confs, spec.sphys_mode)
        for objective_weight in VAL_FALSE_INV_OBJECTIVE_WEIGHTS:
            for seed in SEEDS:
                val = merge_for_spec(root, sphys, spec, seed, "val")
                weights, best_score, val_metrics = select_weights_on_validation(val, spec, objective_weight)
                test = merge_for_spec(root, sphys, spec, seed, "test")
                test_metrics = evaluate_weighted(test, spec, weights)
                columns = component_columns(spec)
                weight_map = dict(zip(columns, weights))
                rows.append(
                    {
                        "fusion": spec.label,
                        "seed": seed,
                        "sphys_num_confs": num_confs,
                        "sphys_mode": spec.sphys_mode,
                        "components": "+".join(columns),
                        "selected_graph_weight": float(weight_map.get(spec.graph.label, np.nan)),
                        "selected_mlp_weight": float(weight_map.get(spec.mlp.label, np.nan)) if spec.mlp is not None else np.nan,
                        "selected_sphysnet_weight": float(weight_map["sphysnet_prediction"]),
                        "val_selection_false_inv_weight": objective_weight,
                        "val_selection_score": float(best_score),
                        "val_selected_roc_auc": val_metrics["test_roc_auc"],
                        "val_selected_false_invariance": val_metrics["false_invariance_rate_test_pairs"],
                        "selection_protocol": "seedwise_val_selected_sphysnet_fusion",
                        "note": spec.note,
                        **test_metrics,
                    }
                )
    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    tests = paired_tests(root, raw, num_confs)
    return raw, summary, tests


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-confs", type=int, default=10)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    table_dir = root / "reports" / "tables"
    raw, summary, tests = run(args.num_confs)
    raw.to_csv(table_dir / f"main_sphysnet_numconf{args.num_confs}_fusion_val_selected_seed_raw.csv", index=False)
    summary.to_csv(table_dir / f"main_sphysnet_numconf{args.num_confs}_fusion_val_selected.csv", index=False)
    tests.to_csv(table_dir / f"main_sphysnet_numconf{args.num_confs}_fusion_paired_tests.csv", index=False)
    (table_dir / f"main_sphysnet_numconf{args.num_confs}_fusion_val_selected.md").write_text(
        make_markdown(summary),
        encoding="utf-8",
    )
    (table_dir / f"main_sphysnet_numconf{args.num_confs}_fusion_paired_tests.md").write_text(
        make_paired_markdown(tests),
        encoding="utf-8",
    )
    print(f"wrote sPhysNet fusion num_confs={args.num_confs} summary={len(summary)} raw={len(raw)} paired={len(tests)}")


if __name__ == "__main__":
    main()
