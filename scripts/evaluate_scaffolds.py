from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from condchemsymm.metrics import classification_metrics
from ensemble_components import Component, GRAPH_MAIN, SEEDS, load_predictions, paired_stats
from fusion_components import SPECS, component_columns, load_sphys_predictions, merge_for_spec
from calibration_controls import calibrate_probabilities
from pair_projection import project_logit_difference, safe_logit


GLOBALW500_WEIGHTS = np.asarray([0.336, 0.396, 0.268], dtype=float)
GLOBALW500_SCALE = 1.20
GLOBALW500_CANDIDATE_PREFIX = (
    "graph_mlp_select01_sphys_boltzmann_n10_valobj0p5_"
    "pairlogit_s1p2_globalw500_fixeds12"
)
TP_FUSION_ROC_CANDIDATE = "tp_pairnorm_graph_mlp_sphys_pair_logitdiff_s1p2_valobj0p5"
TP_FUSION_BRIER_CANDIDATE = "tp_pairnorm_graph_mlp_sphys_logit_avg_valobj0"
TP_FUSION_LOWFI_CANDIDATE = "tp_pairnorm_sphys_pair_logitdiff_s1_valobj0p5"


def load_samples(root: Path) -> pd.DataFrame:
    samples = pd.read_parquet(root / "data/processed/main/tautobase_pref_samples.parquet")
    train_counts = samples[samples["split"].eq("train")]["scaffold"].value_counts().to_dict()
    samples = samples.copy()
    samples["train_scaffold_count"] = samples["scaffold"].map(train_counts).fillna(0).astype(int)
    samples["sample_cold_scaffold"] = samples["train_scaffold_count"].eq(0)
    samples["sample_long_tail_scaffold"] = samples["train_scaffold_count"].le(2)
    samples["sample_head_scaffold"] = samples["train_scaffold_count"].ge(5)
    return samples


def enrich_predictions(pred: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [
        "sample_id",
        "scaffold",
        "family_key",
        "train_scaffold_count",
        "sample_cold_scaffold",
        "sample_long_tail_scaffold",
        "sample_head_scaffold",
    ]
    out = pred.merge(samples[meta_cols], on="sample_id", how="left")
    if out["scaffold"].isna().any():
        raise ValueError("prediction rows missing sample metadata")
    return out


def pair_ids(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(lambda row: "::".join(sorted([str(row["sample_id"]), str(row.get("paired_id", ""))])), axis=1)


def pair_level_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    work = frame.copy()
    work["pair_id"] = pair_ids(work)
    pair_info = []
    for pair_id, group in work.groupby("pair_id"):
        if len(group) != 2:
            continue
        counts = group["train_scaffold_count"].to_numpy(dtype=int)
        pair_info.append(
            {
                "pair_id": pair_id,
                "pair_any_cold_scaffold": bool(np.any(counts == 0)),
                "pair_both_seen_scaffold": bool(np.all(counts > 0)),
                "pair_long_tail_scaffold": bool(np.min(counts) <= 2),
                "pair_head_scaffold": bool(np.min(counts) >= 5),
            }
        )
    info = pd.DataFrame(pair_info)
    if info.empty:
        return {}
    annotated = work[["sample_id", "pair_id"]].merge(info, on="pair_id", how="left")
    masks = {}
    for col in [
        "pair_any_cold_scaffold",
        "pair_both_seen_scaffold",
        "pair_long_tail_scaffold",
        "pair_head_scaffold",
    ]:
        keep_ids = set(annotated.loc[annotated[col].fillna(False), "sample_id"])
        masks[col] = frame["sample_id"].isin(keep_ids)
    return masks


def scenario_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    masks = {
        "all_test": pd.Series(True, index=frame.index),
        "sample_cold_start_scaffold": frame["sample_cold_scaffold"].astype(bool),
        "sample_seen_scaffold": ~frame["sample_cold_scaffold"].astype(bool),
        "sample_long_tail_scaffold": frame["sample_long_tail_scaffold"].astype(bool),
        "sample_head_scaffold": frame["sample_head_scaffold"].astype(bool),
    }
    masks.update(pair_level_masks(frame))
    return masks


def evaluate_subset(frame: pd.DataFrame, scenario: str, mask: pd.Series) -> dict[str, float | int | str]:
    subset = frame[mask].copy()
    row: dict[str, float | int | str] = {
        "scenario": scenario,
        "n_samples": int(len(subset)),
    }
    if subset.empty:
        row.update(
            {
                "test_accuracy": math.nan,
                "test_brier": math.nan,
                "test_nll": math.nan,
                "test_ece10": math.nan,
                "test_mce10": math.nan,
                "test_roc_auc": math.nan,
                "test_pr_auc": math.nan,
                "false_invariance_rate_test_pairs": math.nan,
                "false_invariance_n_pairs": 0,
                "pair_order_accuracy": math.nan,
                "pair_gap_mean": math.nan,
                "pair_gap_p10": math.nan,
            }
        )
        return row
    metrics = classification_metrics(subset["y"].to_numpy(dtype=float), subset["prediction"].to_numpy(dtype=float))
    pair = paired_stats(subset[["sample_id", "y", "paired_id", "prediction"]], "prediction")
    row.update({f"test_{k}": v for k, v in metrics.items()})
    row.update(pair)
    return row


def load_globalw500_prediction(root: Path, objective: str, seed: int) -> pd.DataFrame:
    temp_path = root / "reports/tables/main_sphysnet_numconf10_pair_logit_global_weight_selection_boltzmann_denom500_fixeds12_tempcal_seed_raw.csv"
    temps = pd.read_csv(temp_path)
    candidate = f"{GLOBALW500_CANDIDATE_PREFIX}_val{objective}tempcal"
    selected = temps[temps["candidate"].eq(candidate) & temps["seed"].eq(seed)]
    if selected.empty:
        raise ValueError(f"missing temperature row for {candidate} seed={seed}")
    temperature = float(selected.iloc[0]["selected_calibration_temperature"])
    spec = next(spec for spec in SPECS if spec.label == "graph_mlp_select01_sphys_boltzmann")
    sphys = load_sphys_predictions(root, 10, "boltzmann_rt_298k")
    merged = merge_for_spec(root, sphys, spec, seed, "test")
    columns = component_columns(spec)
    scores = GLOBALW500_WEIGHTS[None, :] @ safe_logit(merged[columns].to_numpy(dtype=float)).T
    projected = project_logit_difference(merged[["sample_id", "y", "paired_id"]], scores, GLOBALW500_SCALE)[0]
    calibrated = calibrate_probabilities(projected, temperature)
    out = merged[["sample_id", "y", "paired_id"]].copy()
    out["prediction"] = calibrated
    return out


def load_graph_mlp_no_external_prediction(root: Path, seed: int) -> pd.DataFrame:
    weights = pd.read_csv(root / "reports/tables/main_prediction_ensemble_val_selected_seed_raw.csv")
    row = weights[
        weights["ensemble"].eq("graph_main_plus_mlp_select01_viewpred")
        & weights["seed"].eq(seed)
        & np.isclose(weights["val_selection_false_inv_weight"].astype(float), 0.2)
    ]
    if row.empty:
        raise ValueError(f"missing graph+mlp seedwise weight seed={seed}")
    first_weight = float(row.iloc[0]["selected_first_weight"])
    graph = load_predictions(root, GRAPH_MAIN, seed, "test")
    mlp_comp = Component("mlp_select01_viewpred_mechanistic", "main_concat_select01_viewpred_5seed", "concat", "mechanistic_gate")
    mlp = load_predictions(root, mlp_comp, seed, "test")
    merged = graph.merge(mlp, on=["sample_id", "y", "paired_id"], how="inner")
    if len(merged) != len(graph):
        raise ValueError("graph+MLP row mismatch")
    out = merged[["sample_id", "y", "paired_id"]].copy()
    out["prediction"] = first_weight * merged[GRAPH_MAIN.label].to_numpy(dtype=float) + (1.0 - first_weight) * merged[
        mlp_comp.label
    ].to_numpy(dtype=float)
    return out


def load_component_prediction(root: Path, label: str, component: Component, seed: int) -> pd.DataFrame:
    out = load_predictions(root, component, seed, "test")
    out = out.rename(columns={component.label: "prediction"})
    out["model_label"] = label
    return out


def load_sphysnet_prediction(root: Path) -> pd.DataFrame:
    sphys = load_sphys_predictions(root, 10, "boltzmann_rt_298k")
    out = sphys[sphys["split"].eq("test")][["sample_id", "y", "paired_id", "sphysnet_prediction"]].copy()
    return out.rename(columns={"sphysnet_prediction": "prediction"})


def load_tautomer_predictor_prediction(root: Path, method: str, seed: int) -> pd.DataFrame | None:
    path = root / "runs/main_tautomer_predictor_baseline" / method / "tautobase_pref" / "identity" / f"seed_{seed}" / "test_predictions.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df[["sample_id", "y", "paired_id", "prediction"]].copy()


def load_tautomer_predictor_fusion_prediction(root: Path, candidate: str, seed: int) -> pd.DataFrame | None:
    path = (
        root
        / "runs/main_tautomer_predictor_fusion"
        / candidate
        / "tautobase_pref"
        / "identity"
        / f"seed_{seed}"
        / "test_predictions.parquet"
    )
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df[["sample_id", "y", "paired_id", "prediction"]].copy()


def load_classical_prediction(root: Path, method: str, seed: int) -> pd.DataFrame | None:
    path = root / "runs/main_classical_external_baselines" / method / "tautobase_pref" / "identity" / f"seed_{seed}" / "test_predictions.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df[["sample_id", "y", "paired_id", "prediction"]].copy()


def prediction_sources(root: Path, include_classical: bool) -> list[dict]:
    sources: list[dict] = [
        {
            "model_label": "CondChemSymm-globalw500-ficap10",
            "baseline_group": "proposed_external_fusion_pair_projected",
            "modality_condition": "full",
            "source_result_type": "rerun_our_split_posthoc_from_seed_predictions",
            "loader": lambda seed: load_globalw500_prediction(root, "nll_ficap10", seed),
            "seeds": SEEDS,
        },
        {
            "model_label": "CondChemSymm-globalw500-ficap15",
            "baseline_group": "proposed_external_fusion_pair_projected",
            "modality_condition": "full",
            "source_result_type": "rerun_our_split_posthoc_from_seed_predictions",
            "loader": lambda seed: load_globalw500_prediction(root, "nll_ficap15", seed),
            "seeds": SEEDS,
        },
        {
            "model_label": "Graph+MLP-select01-no-sPhysNet",
            "baseline_group": "missing_external_energy_ablation",
            "modality_condition": "missing_3d_energy_expert",
            "source_result_type": "rerun_our_split_posthoc_from_seed_predictions",
            "loader": lambda seed: load_graph_mlp_no_external_prediction(root, seed),
            "seeds": SEEDS,
        },
        {
            "model_label": "Graph-DP-pairm005-select02",
            "baseline_group": "internal_graph_baseline",
            "modality_condition": "2d_graph_only",
            "source_result_type": "rerun_our_split",
            "loader": lambda seed: load_component_prediction(root, "Graph-DP-pairm005-select02", GRAPH_MAIN, seed),
            "seeds": SEEDS,
        },
        {
            "model_label": "Chemprop-v2.3.1-canonical-positive-aug",
            "baseline_group": "modern_dpmpnn_external",
            "modality_condition": "2d_graph_only",
            "source_result_type": "rerun_our_split",
            "loader": lambda seed: load_component_prediction(
                root,
                "Chemprop-v2.3.1-canonical-positive-aug",
                Component("chemprop_canonical_positive_aug", "main_chemprop_v231_tautobase_5seed", "chemprop_v231", "canonical_positive_aug"),
                seed,
            ),
            "seeds": SEEDS,
        },
        {
            "model_label": "sPhysNet-Taut-numconf10-Boltzmann",
            "baseline_group": "tautomer_energy_external",
            "modality_condition": "3d_energy_expert_only",
            "source_result_type": "released_deterministic_repeated_over_seed_table",
            "loader": lambda seed: load_sphysnet_prediction(root),
            "seeds": SEEDS,
        },
        {
            "model_label": "Tautomer-Predictor-AttentiveFP-raw",
            "baseline_group": "recent_same_task_2d_gnn",
            "modality_condition": "2d_graph_only",
            "source_result_type": "released_checkpoint_rerun_our_split_repeated_over_seed_table",
            "loader": lambda seed: load_tautomer_predictor_prediction(root, "tautomer_predictor_attentivefp_raw", seed),
            "seeds": SEEDS,
        },
        {
            "model_label": "Tautomer-Predictor-AttentiveFP-pairnorm",
            "baseline_group": "direct_mechanism_pair_normalized_2d_gnn",
            "modality_condition": "2d_graph_pair_normalized",
            "source_result_type": "released_checkpoint_rerun_our_split_repeated_over_seed_table",
            "loader": lambda seed: load_tautomer_predictor_prediction(root, "tautomer_predictor_attentivefp_pairnorm", seed),
            "seeds": SEEDS,
        },
        {
            "model_label": "CondChemSymm+TautPred-fusion-ROC",
            "baseline_group": "proposed_external_checkpoint_fusion",
            "modality_condition": "full_plus_external_tautomer_checkpoint",
            "source_result_type": "rerun_our_split_posthoc_val_selected_from_seed_predictions",
            "loader": lambda seed: load_tautomer_predictor_fusion_prediction(root, TP_FUSION_ROC_CANDIDATE, seed),
            "seeds": SEEDS,
        },
        {
            "model_label": "CondChemSymm+TautPred-fusion-Brier",
            "baseline_group": "proposed_external_checkpoint_fusion",
            "modality_condition": "full_plus_external_tautomer_checkpoint",
            "source_result_type": "rerun_our_split_posthoc_val_selected_from_seed_predictions",
            "loader": lambda seed: load_tautomer_predictor_fusion_prediction(root, TP_FUSION_BRIER_CANDIDATE, seed),
            "seeds": SEEDS,
        },
        {
            "model_label": "TautPred+sPhysNet-lowFI",
            "baseline_group": "direct_mechanism_pair_projected_external_fusion",
            "modality_condition": "external_tautomer_checkpoint_plus_3d_energy",
            "source_result_type": "rerun_our_split_posthoc_val_selected_from_seed_predictions",
            "loader": lambda seed: load_tautomer_predictor_fusion_prediction(root, TP_FUSION_LOWFI_CANDIDATE, seed),
            "seeds": SEEDS,
        },
    ]
    if include_classical:
        methods = [
            "ecfp_logreg",
            "ecfp_linear_svm",
            "ecfp_rf",
            "ecfp_extratrees",
            "ecfp_histgb",
            "rdkit_desc_logreg",
            "rdkit_desc_rf",
            "rdkit_desc_extratrees",
            "rdkit_desc_histgb",
            "tanimoto_knn",
            "rdkit_tautomer_canonical_prior",
            "inchi_mobile_h_invariant_prior",
        ]
        for method in methods:
            sources.append(
                {
                    "model_label": method,
                    "baseline_group": "classical_or_direct_mechanism",
                    "modality_condition": "2d_or_rule_only",
                    "source_result_type": "rerun_our_split",
                    "loader": lambda seed, m=method: load_classical_prediction(root, m, seed),
                    "seeds": SEEDS,
                }
            )
    return sources


def summarize(rows: list[dict], root: Path) -> pd.DataFrame:
    raw = pd.DataFrame(rows)
    raw_path = root / "reports/tables/main_special_scenario_comparison_seed_raw.csv"
    summary_path = root / "reports/tables/main_special_scenario_comparison.csv"
    md_path = root / "reports/tables/main_special_scenario_comparison.md"
    raw.to_csv(raw_path, index=False)
    keys = ["model_label", "baseline_group", "modality_condition", "source_result_type", "scenario"]
    metric_cols = [
        "n_samples",
        "test_accuracy",
        "test_brier",
        "test_nll",
        "test_ece10",
        "test_mce10",
        "test_roc_auc",
        "test_pr_auc",
        "false_invariance_rate_test_pairs",
        "false_invariance_n_pairs",
        "pair_order_accuracy",
        "pair_gap_mean",
        "pair_gap_p10",
    ]
    out_rows = []
    for key_values, group in raw.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, key_values))
        row["seed_count"] = int(group["seed"].nunique())
        for col in metric_cols:
            vals = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else math.nan
            row[f"{col}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
            row[f"{col}_count"] = int(vals.notna().sum())
        out_rows.append(row)
    summary = pd.DataFrame(out_rows).sort_values(["scenario", "baseline_group", "model_label"]).reset_index(drop=True)
    summary.to_csv(summary_path, index=False)

    focus = summary[
        summary["scenario"].isin(
            [
                "all_test",
                "pair_any_cold_scaffold",
                "pair_long_tail_scaffold",
                "sample_cold_start_scaffold",
                "sample_long_tail_scaffold",
            ]
        )
    ].copy()
    cols = [
        "scenario",
        "model_label",
        "modality_condition",
        "seed_count",
        "n_samples_mean",
        "false_invariance_n_pairs_mean",
        "test_roc_auc_mean",
        "test_roc_auc_std",
        "test_brier_mean",
        "false_invariance_rate_test_pairs_mean",
        "pair_order_accuracy_mean",
        "source_result_type",
    ]
    present = [c for c in cols if c in focus.columns]
    md = [
        "# Phase 1 Special Scenario Comparison",
        "",
        "Scenarios are computed on the frozen duplicate-aware Tautobase test split. Pair scenarios keep both members of each selected tautomer pair so pair false-invariance is well-defined.",
        "",
        focus[present].to_markdown(index=False),
        "",
        "Scenario definitions:",
        "",
        "- `all_test`: all duplicate-aware Tautobase test samples.",
        "- `sample_cold_start_scaffold`: test samples whose Murcko scaffold is absent from training.",
        "- `pair_any_cold_scaffold`: tautomer pairs where at least one member has a training-unseen scaffold.",
        "- `sample_long_tail_scaffold`: test samples with training scaffold count <= 2, including unseen.",
        "- `pair_long_tail_scaffold`: tautomer pairs whose rarer member has training scaffold count <= 2.",
        "- `missing_3d_energy_expert`: model condition with no sPhysNet/3D energy expert.",
        "",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-classical", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    samples = load_samples(root)
    rows = []
    for source in prediction_sources(root, args.include_classical):
        for seed in source["seeds"]:
            pred = source["loader"](seed)
            if pred is None:
                print(f"SKIP missing prediction model={source['model_label']} seed={seed}", flush=True)
                continue
            pred = enrich_predictions(pred, samples)
            for scenario, mask in scenario_masks(pred).items():
                metric_row = evaluate_subset(pred, scenario, mask)
                rows.append(
                    {
                        "model_label": source["model_label"],
                        "baseline_group": source["baseline_group"],
                        "modality_condition": source["modality_condition"],
                        "source_result_type": source["source_result_type"],
                        "seed": int(seed),
                        **metric_row,
                    }
                )
    summary = summarize(rows, root)
    print(f"WROTE {len(summary)} summary rows")


if __name__ == "__main__":
    main()
