from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble_components import (
    GRAPH_MAIN,
    SEEDS,
    Component,
    ci95,
    load_predictions,
    paired_t_p,
)
from fusion_components import MLP_SELECT01, PAIRED_METRICS, load_sphys_predictions
from pair_projection import (
    evaluate_predictions,
    fast_auc_scores,
    fast_false_invariance_scores,
    pair_indices,
    project_logit_difference,
    project_probability_norm,
    safe_logit,
    sigmoid,
    vector_brier,
    vector_nll,
)


@dataclass(frozen=True)
class FusionSpec:
    label: str
    component_ids: tuple[str, ...]
    note: str


TP_RUN = "main_tautomer_predictor_baseline"
TP_RAW = "tautomer_predictor_attentivefp_raw"
TP_PAIRNORM = "tautomer_predictor_attentivefp_pairnorm"
CHEMPROP_POS = Component(
    "chemprop_canonical_positive_aug",
    "main_chemprop_v231_tautobase_5seed",
    "chemprop_v231",
    "canonical_positive_aug",
)

SPECS = [
    FusionSpec(
        "tp_pairnorm_sphys",
        ("tp_pairnorm", "sphys_boltzmann"),
        "Released Tautomer-Predictor 2D GNN plus released sPhysNet-Taut Boltzmann energy prior.",
    ),
    FusionSpec(
        "tp_pairnorm_graph",
        ("tp_pairnorm", "graph_main"),
        "Released Tautomer-Predictor 2D GNN plus the best CondChemSymm graph branch.",
    ),
    FusionSpec(
        "tp_pairnorm_graph_sphys",
        ("tp_pairnorm", "graph_main", "sphys_boltzmann"),
        "Three-source fusion of Tautomer-Predictor, CondChemSymm graph, and sPhysNet.",
    ),
    FusionSpec(
        "tp_pairnorm_mlp_sphys",
        ("tp_pairnorm", "mlp_select01", "sphys_boltzmann"),
        "Tests whether the low false-invariance MLP branch complements the two external experts.",
    ),
    FusionSpec(
        "tp_pairnorm_graph_mlp_sphys",
        ("tp_pairnorm", "graph_main", "mlp_select01", "sphys_boltzmann"),
        "Full lightweight fusion among released Tautomer-Predictor, CondChemSymm branches, and sPhysNet.",
    ),
    FusionSpec(
        "tp_raw_pairnorm_sphys",
        ("tp_raw", "tp_pairnorm", "sphys_boltzmann"),
        "Keeps Tautomer-Predictor raw score as a calibration signal while pairnorm supplies pair consistency.",
    ),
    FusionSpec(
        "tp_pairnorm_sphys_chemprop",
        ("tp_pairnorm", "sphys_boltzmann", "chemprop_pos"),
        "Checks whether the external D-MPNN baseline adds complementary pair reliability.",
    ),
]

OBJECTIVE_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.5]
POLICIES = [
    "prob_avg",
    "prob_pair_norm",
    "logit_avg",
    "pair_logitdiff_s1",
    "pair_logitdiff_s1p2",
    "pair_logitdiff_s1p5",
]


def label_float(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def prediction_path(root: Path, method: str, seed: int, split: str) -> Path:
    return root / "runs" / TP_RUN / method / "tautobase_pref" / "identity" / f"seed_{seed}" / f"{split}_predictions.parquet"


def load_tp_prediction(root: Path, method: str, seed: int, split: str) -> pd.DataFrame:
    path = prediction_path(root, method, seed, split)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    return df[["sample_id", "y", "paired_id", "prediction"]].copy()


def load_component(root: Path, component_id: str, seed: int, split: str, sphys: pd.DataFrame) -> pd.DataFrame:
    if component_id == "tp_raw":
        return load_tp_prediction(root, TP_RAW, seed, split).rename(columns={"prediction": "tp_raw"})
    if component_id == "tp_pairnorm":
        return load_tp_prediction(root, TP_PAIRNORM, seed, split).rename(columns={"prediction": "tp_pairnorm"})
    if component_id == "sphys_boltzmann":
        df = sphys[sphys["split"].eq(split)].copy()
        return df[["sample_id", "y", "paired_id", "sphysnet_prediction"]].rename(
            columns={"sphysnet_prediction": "sphys_boltzmann"}
        )
    if component_id == "graph_main":
        return load_predictions(root, GRAPH_MAIN, seed, split).rename(columns={GRAPH_MAIN.label: "graph_main"})
    if component_id == "mlp_select01":
        return load_predictions(root, MLP_SELECT01, seed, split).rename(columns={MLP_SELECT01.label: "mlp_select01"})
    if component_id == "chemprop_pos":
        return load_predictions(root, CHEMPROP_POS, seed, split).rename(columns={CHEMPROP_POS.label: "chemprop_pos"})
    raise ValueError(component_id)


def merge_components(root: Path, spec: FusionSpec, seed: int, split: str, sphys: pd.DataFrame) -> pd.DataFrame:
    frames = [load_component(root, component_id, seed, split, sphys) for component_id in spec.component_ids]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["sample_id", "y", "paired_id"], how="inner")
    if any(len(merged) != len(frame) for frame in frames):
        raise ValueError(f"component row mismatch for {spec.label} seed={seed} split={split}")
    return merged


def simplex_weights(n_components: int, denom: int) -> np.ndarray:
    rows: list[tuple[int, ...]] = []

    def rec(prefix: tuple[int, ...], remaining_components: int, remaining: int) -> None:
        if remaining_components == 1:
            rows.append((*prefix, remaining))
            return
        for value in range(remaining + 1):
            rec((*prefix, value), remaining_components - 1, remaining - value)

    rec((), n_components, int(denom))
    return np.asarray(rows, dtype=float) / float(denom)


def policy_scale(policy: str) -> float:
    if policy == "pair_logitdiff_s1":
        return 1.0
    if policy == "pair_logitdiff_s1p2":
        return 1.2
    if policy == "pair_logitdiff_s1p5":
        return 1.5
    return 1.0


def policy_predictions(df: pd.DataFrame, columns: list[str], weights: np.ndarray, policy: str) -> np.ndarray:
    components = np.clip(df[columns].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    if policy == "prob_avg":
        return np.clip(weights @ components.T, 1e-7, 1.0 - 1e-7)
    if policy == "prob_pair_norm":
        return project_probability_norm(df, np.clip(weights @ components.T, 1e-7, 1.0 - 1e-7))
    if policy == "logit_avg":
        return sigmoid(weights @ safe_logit(components).T)
    if policy.startswith("pair_logitdiff_"):
        scores = weights @ safe_logit(components).T
        return project_logit_difference(df, scores, policy_scale(policy))
    raise ValueError(policy)


def choose_candidate(
    val: pd.DataFrame,
    columns: list[str],
    weights: np.ndarray,
    policy: str,
    objective_weight: float,
) -> tuple[np.ndarray, float, dict[str, float]]:
    predictions = policy_predictions(val, columns, weights, policy)
    y = val["y"].to_numpy(dtype=float)
    left, right = pair_indices(val)
    aucs = fast_auc_scores(y, predictions, chunk_size=256)
    false_inv = fast_false_invariance_scores(predictions, left, right)
    nll = vector_nll(y, predictions)
    brier = vector_brier(y, predictions)
    objective = aucs - float(objective_weight) * false_inv
    best_score = float(np.nanmax(objective))
    candidates = np.flatnonzero(np.isclose(objective, best_score, rtol=0.0, atol=1e-12))

    def tie_key(idx: int) -> tuple[float, float, float, float]:
        tp_weight = float(weights[idx, columns.index("tp_pairnorm")]) if "tp_pairnorm" in columns else 0.0
        external_weight = sum(float(weights[idx, columns.index(col)]) for col in columns if col.startswith(("tp_", "sphys")))
        return (float(nll[idx]), float(brier[idx]), float(false_inv[idx]), -tp_weight + 0.01 * external_weight)

    best_idx = int(min(candidates, key=tie_key))
    selected = weights[best_idx]
    metrics = evaluate_predictions(val, predictions[best_idx])
    metrics.update(
        {
            "val_selection_score": best_score,
            "val_selection_nll": float(nll[best_idx]),
            "val_selection_brier": float(brier[best_idx]),
        }
    )
    return selected, best_score, metrics


def apply_candidate(df: pd.DataFrame, columns: list[str], weights: np.ndarray, policy: str) -> np.ndarray:
    return policy_predictions(df, columns, np.asarray(weights, dtype=float)[None, :], policy)[0]


def write_predictions(frame: pd.DataFrame, prediction: np.ndarray, path: Path) -> None:
    out = frame[["sample_id", "y", "paired_id"]].copy()
    out["prediction"] = prediction
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def run_weight_denom(n_components: int) -> int:
    return 50 if n_components <= 3 else 25


def candidate_name(spec: FusionSpec, policy: str, objective_weight: float) -> str:
    return f"{spec.label}_{policy}_valobj{label_float(objective_weight)}"


def baseline_seed_rows(root: Path) -> dict[str, pd.DataFrame]:
    rows = {}
    tp = pd.read_csv(root / "reports/tables/main_tautomer_predictor_baseline_seed_raw.csv")
    for method in [TP_PAIRNORM, TP_RAW]:
        group = tp[tp["method"].eq(method)].copy()
        rename = {
            "test_false_invariance_rate_test_pairs": "false_invariance_rate_test_pairs",
            "test_pair_order_accuracy": "pair_order_accuracy",
            "test_pair_gap_mean": "pair_gap_mean",
            "test_pair_gap_p10": "pair_gap_p10",
        }
        rows[method] = group.rename(columns=rename)
    global_path = root / "reports/tables/main_sphysnet_numconf10_pair_logit_global_weight_selection_boltzmann_denom500_fixeds12_tempcal_seed_raw.csv"
    if global_path.exists():
        df = pd.read_csv(global_path)
        for cap in ["ficap10", "ficap15"]:
            cap_rows = df[df["candidate"].str.contains(f"valnll_{cap}tempcal", regex=False)].copy()
            if not cap_rows.empty:
                rows[f"condchemsymm_globalw500_{cap}"] = cap_rows
    return rows


def paired_tests(root: Path, raw: pd.DataFrame) -> pd.DataFrame:
    baselines = baseline_seed_rows(root)
    rows = []
    for candidate, group in raw.groupby("candidate", sort=True):
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
                improvement = (candidate_values - baseline_values) * float(direction)
                low, high = ci95(improvement)
                rows.append(
                    {
                        "candidate": candidate,
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
                        "all_seeds_better": bool(np.all(improvement > 0)),
                    }
                )
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "candidate",
        "fusion",
        "policy",
        "components",
        "val_selection_false_inv_weight",
        "weight_denom",
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
        "val_selected_roc_auc",
        "val_selected_brier",
        "val_selected_nll",
        "val_selected_false_invariance",
        "val_selection_score",
    ]
    grouped = raw.groupby(keys, sort=True, dropna=False)
    out = grouped[metric_cols].agg(["mean", "std", "count"])
    out.columns = [f"{name}_{stat}" for name, stat in out.columns]
    out = out.reset_index()
    weight_cols = [col for col in raw.columns if col.startswith("selected_") and col.endswith("_weight")]
    weights = grouped[weight_cols].agg(["mean", "std"]).reset_index()
    weights.columns = keys + [f"{name}_{stat}" for name in weight_cols for stat in ["mean", "std"]]
    strings = []
    for key_values, group in grouped:
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = dict(zip(keys, key_values))
        row["selected_weights_by_seed"] = ",".join(
            "/".join(
                f"{float(record[f'selected_{component}_weight']):.2f}"
                for component in str(record["components"]).split("+")
                if f"selected_{component}_weight" in record.index
                and not pd.isna(record[f"selected_{component}_weight"])
            )
            for _, record in group.sort_values("seed").iterrows()
        )
        strings.append(row)
    return out.merge(weights, on=keys, how="left").merge(pd.DataFrame(strings), on=keys, how="left")


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
    view = summary.sort_values(
        ["test_roc_auc_mean", "test_brier_mean", "false_invariance_rate_test_pairs_mean"],
        ascending=[False, True, True],
    ).copy()
    cols = [
        "candidate",
        "components",
        "weight_denom",
        "selected_weights_by_seed",
        "test_roc_auc_mean",
        "test_roc_auc_std",
        "test_brier_mean",
        "test_nll_mean",
        "false_invariance_rate_test_pairs_mean",
        "pair_order_accuracy_mean",
        "val_selected_roc_auc_mean",
        "val_selected_false_invariance_mean",
    ]
    view = view[cols]
    for col in view.columns:
        view[col] = view[col].map(fmt)
    return (
        "# Tautomer-Predictor Fusion Results\n\n"
        "Exploratory validation-selected fusion after adding the released Tautomer-Predictor checkpoint. "
        "Selection uses only the frozen validation split; test metrics are evaluated once after selection. "
        "Weights are listed in the component order shown in `components`.\n\n"
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
        "# Tautomer-Predictor Fusion Paired Tests\n\n"
        "`improvement_mean` is signed so positive is better. "
        "The external Tautomer-Predictor pair-normalized checkpoint is included as the strongest new baseline.\n\n"
        + view.to_markdown(index=False)
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit validation weights for the fixed B--E policies")
    parser.add_argument("--development-sweep", action="store_true", help="Run the broader historical development grid; requires additional model variants")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sphys = load_sphys_predictions(root, 10, "boltzmann_rt_298k")
    rows = []
    for spec in SPECS:
        fixed_specs = {"tp_pairnorm_sphys": "pair_logitdiff_s1", "tp_pairnorm_graph_mlp_sphys": "pair_logitdiff_s1p2"}
        if not args.development_sweep and spec.label not in fixed_specs:
            continue
        columns = list(spec.component_ids)
        weights = simplex_weights(len(columns), run_weight_denom(len(columns)))
        for policy in POLICIES:
            for objective_weight in OBJECTIVE_WEIGHTS:
                if not args.development_sweep and (policy, objective_weight) not in {
                    ("logit_avg", 0.0), (fixed_specs[spec.label], 0.5)
                }:
                    continue
                name = candidate_name(spec, policy, objective_weight)
                for seed in SEEDS:
                    val = merge_components(root, spec, seed, "val", sphys)
                    selected, _, val_metrics = choose_candidate(val, columns, weights, policy, objective_weight)
                    test = merge_components(root, spec, seed, "test", sphys)
                    prediction = apply_candidate(test, columns, selected, policy)
                    test_metrics = evaluate_predictions(test, prediction)
                    row = {
                        "candidate": name,
                        "fusion": spec.label,
                        "seed": seed,
                        "policy": policy,
                        "components": "+".join(columns),
                        "val_selection_false_inv_weight": float(objective_weight),
                        "weight_denom": run_weight_denom(len(columns)),
                        "selection_protocol": "seedwise_validation_selected_tautomer_predictor_fusion",
                        "note": spec.note,
                        "val_selected_roc_auc": val_metrics["test_roc_auc"],
                        "val_selected_brier": val_metrics["test_brier"],
                        "val_selected_nll": val_metrics["test_nll"],
                        "val_selected_false_invariance": val_metrics["false_invariance_rate_test_pairs"],
                        "val_selection_score": val_metrics["val_selection_score"],
                        **test_metrics,
                    }
                    for component_id, weight in zip(columns, selected):
                        row[f"selected_{component_id}_weight"] = float(weight)
                    rows.append(row)
                    out_dir = root / "runs/main_tautomer_predictor_fusion" / name / "tautobase_pref" / "identity" / f"seed_{seed}"
                    write_predictions(test, prediction, out_dir / "test_predictions.parquet")
                    write_predictions(val, apply_candidate(val, columns, selected, policy), out_dir / "val_predictions.parquet")
    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    tests = paired_tests(root, raw) if args.development_sweep else pd.DataFrame()
    table_dir = root / "reports" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(table_dir / "main_tautomer_predictor_fusion_seed_raw.csv", index=False)
    summary.to_csv(table_dir / "main_tautomer_predictor_fusion.csv", index=False)
    tests.to_csv(table_dir / "main_tautomer_predictor_fusion_paired_tests.csv", index=False)
    (table_dir / "main_tautomer_predictor_fusion.md").write_text(make_markdown(summary), encoding="utf-8")
    if args.development_sweep:
        (table_dir / "main_tautomer_predictor_fusion_paired_tests.md").write_text(make_paired_markdown(tests), encoding="utf-8")
    print(f"WROTE tautomer-predictor fusion summary={len(summary)} raw={len(raw)} paired={len(tests)}")


if __name__ == "__main__":
    main()
