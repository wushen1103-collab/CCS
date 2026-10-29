from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from condchemsymm.metrics import classification_metrics


SEEDS = [0, 1, 2, 3, 4]
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20261027

METHODS = {
    "A": {
        "label": "Tautomer-Predictor pairnorm",
        "run": "main_tautomer_predictor_baseline",
        "candidate": "tautomer_predictor_attentivefp_pairnorm",
        "components": "TP",
        "policy": "pair normalization",
    },
    "B": {
        "label": "TP + sPhysNet ordinary fusion",
        "run": "main_tautomer_predictor_fusion",
        "candidate": "tp_pairnorm_sphys_logit_avg_valobj0",
        "components": "TP+sPhysNet",
        "policy": "ordinary logit fusion",
    },
    "C": {
        "label": "TP + sPhysNet + Graph/MLP ordinary fusion",
        "run": "main_tautomer_predictor_fusion",
        "candidate": "tp_pairnorm_graph_mlp_sphys_logit_avg_valobj0",
        "components": "TP+sPhysNet+Graph+MLP",
        "policy": "ordinary logit fusion",
    },
    "D": {
        "label": "TP + sPhysNet pair-logit certificate",
        "run": "main_tautomer_predictor_fusion",
        "candidate": "tp_pairnorm_sphys_pair_logitdiff_s1_valobj0p5",
        "components": "TP+sPhysNet",
        "policy": "pair-logit certificate, scale=1.0",
    },
    "E": {
        "label": "Full CondChemSymm",
        "run": "main_tautomer_predictor_fusion",
        "candidate": "tp_pairnorm_graph_mlp_sphys_pair_logitdiff_s1p2_valobj0p5",
        "components": "TP+sPhysNet+Graph+MLP",
        "policy": "pair-logit certificate, scale=1.2",
    },
}


def safe_logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-values))


def pair_indices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    by_id = {sample_id: idx for idx, sample_id in enumerate(frame["sample_id"])}
    left, right, seen = [], [], set()
    for row in frame.itertuples(index=False):
        sid, pid = str(row.sample_id), str(row.paired_id)
        if not pid or sid in seen or pid in seen or pid not in by_id:
            continue
        seen.update((sid, pid))
        left.append(by_id[sid])
        right.append(by_id[pid])
    return np.asarray(left, dtype=int), np.asarray(right, dtype=int)


def project_pair_logit(frame: pd.DataFrame, score: np.ndarray, scale: float) -> np.ndarray:
    out = sigmoid(score)
    left, right = pair_indices(frame)
    for i, j in zip(left, right):
        diff = float(scale) * (float(score[i]) - float(score[j]))
        out[i], out[j] = float(sigmoid(np.asarray([diff]))[0]), float(sigmoid(np.asarray([-diff]))[0])
    return out


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    pred = frame["prediction"].to_numpy(dtype=float)
    labels = frame["y"].to_numpy(dtype=float)
    base = classification_metrics(labels, pred)
    left, right = pair_indices(frame)
    gaps = np.abs(pred[left] - pred[right])
    order = []
    for i, j in zip(left, right):
        order.append((pred[i] > pred[j]) if labels[i] > labels[j] else (pred[j] > pred[i]))
    return {
        "roc_auc": float(base["roc_auc"]),
        "brier": float(base["brier"]),
        "nll": float(base["nll"]),
        "false_invariance": float(np.mean(gaps < 0.25)),
        "pair_order_accuracy": float(np.mean(order)),
        "n_pairs": int(len(left)),
    }


def main_prediction_path(root: Path, method: str, seed: int) -> Path:
    spec = METHODS[method]
    if method == "A":
        return (
            root
            / "runs"
            / spec["run"]
            / spec["candidate"]
            / "tautobase_pref/identity"
            / f"seed_{seed}/test_predictions.parquet"
        )
    return (
        root
        / "runs"
        / spec["run"]
        / spec["candidate"]
        / "tautobase_pref/identity"
        / f"seed_{seed}/test_predictions.parquet"
    )


def summarize_seed_rows(rows: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    metric_columns = ["roc_auc", "brier", "nll", "false_invariance", "pair_order_accuracy", "n_pairs"]
    summary_rows = []
    for keys, group in rows.groupby(group_columns, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["seed_count"] = int(group["seed"].nunique())
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def build_main_ablation(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(root / "reports/tables/evaluation_clean_tautobase_family_manifest.csv")
    clean_ids = set(manifest.loc[manifest["checkpoint_clean"], "left_sample_id"]) | set(
        manifest.loc[manifest["checkpoint_clean"], "right_sample_id"]
    )
    rows = []
    for method, spec in METHODS.items():
        for seed in SEEDS:
            prediction = pd.read_parquet(main_prediction_path(root, method, seed))
            for scope, frame in [
                ("full_frozen_tautobase_test", prediction),
                ("checkpoint_clean_tautobase_subset", prediction[prediction["sample_id"].isin(clean_ids)].copy()),
            ]:
                rows.append(
                    {
                        "scope": scope,
                        "method": method,
                        "label": spec["label"],
                        "components": spec["components"],
                        "policy": spec["policy"],
                        "seed": seed,
                        **metrics(frame),
                    }
                )
    raw = pd.DataFrame(rows)
    summary = summarize_seed_rows(raw, ["scope", "method", "label", "components", "policy"])
    return raw, summary


def load_external_component(root: Path, component: str, seed: int, dataset: str) -> pd.DataFrame:
    if component == "tp_pairnorm":
        path = root / f"runs/evaluation_external_released_baselines/tautomer_predictor_pairnorm/{dataset}/test_predictions.parquet"
    elif component == "sphys_boltzmann":
        source = "sphysnet_published" if dataset == "sphysnet_100_aqueous" else "sphysnet_checkpoint"
        path = root / f"runs/evaluation_external_released_baselines/{source}/{dataset}/test_predictions.parquet"
    elif component == "graph_main":
        path = root / f"runs/evaluation_external_internal_models/graph_main/{dataset}/seed_{seed}/test_predictions.parquet"
    elif component == "mlp_select01":
        path = root / f"runs/evaluation_external_internal_models/mlp_select01/{dataset}/seed_{seed}/test_predictions.parquet"
    else:
        raise ValueError(component)
    frame = pd.read_parquet(path)
    return frame[["sample_id", "y", "paired_id", "family_key", "prediction"]].rename(
        columns={"prediction": component}
    )


def selected_weights(root: Path, candidate: str, seed: int) -> dict[str, float]:
    selection = pd.read_csv(root / "reports/tables/main_tautomer_predictor_fusion_seed_raw.csv")
    row = selection[selection["candidate"].eq(candidate) & selection["seed"].eq(seed)]
    if len(row) != 1:
        raise ValueError(f"expected one frozen weight row for {candidate} seed={seed}, got {len(row)}")
    row = row.iloc[0]
    mapping = {
        "tp_pairnorm": "selected_tp_pairnorm_weight",
        "sphys_boltzmann": "selected_sphys_boltzmann_weight",
        "graph_main": "selected_graph_main_weight",
        "mlp_select01": "selected_mlp_select01_weight",
    }
    return {component: float(row[column]) for component, column in mapping.items() if pd.notna(row[column])}


def external_fusion_prediction(root: Path, method: str, seed: int, dataset: str) -> pd.DataFrame:
    if method == "A":
        frame = load_external_component(root, "tp_pairnorm", seed, dataset)
        support = load_external_component(root, "sphys_boltzmann", seed, dataset)[["sample_id"]]
        frame = frame.merge(support, on="sample_id", how="inner")
        return frame.rename(columns={"tp_pairnorm": "prediction"})
    candidate = METHODS[method]["candidate"]
    weights = selected_weights(root, candidate, seed)
    components = list(weights)
    frame = load_external_component(root, components[0], seed, dataset)
    for component in components[1:]:
        frame = frame.merge(
            load_external_component(root, component, seed, dataset),
            on=["sample_id", "y", "paired_id", "family_key"],
            how="inner",
        )
    scores = sum(weights[component] * safe_logit(frame[component].to_numpy(dtype=float)) for component in components)
    policy = METHODS[method]["policy"]
    if policy == "ordinary logit fusion":
        prediction = sigmoid(scores)
    elif "scale=1.0" in policy:
        prediction = project_pair_logit(frame, scores, 1.0)
    elif "scale=1.2" in policy:
        prediction = project_pair_logit(frame, scores, 1.2)
    else:
        raise ValueError(policy)
    out = frame[["sample_id", "y", "paired_id", "family_key"]].copy()
    out["prediction"] = prediction
    out_path = (
        root
        / "runs/evaluation_external_fusion"
        / METHODS[method]["candidate"]
        / dataset
        / f"seed_{seed}/test_predictions.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out


def build_external_ablation(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for dataset in ["sphysnet_100_aqueous", "nci_aqueous_zero_overlap"]:
        for method, spec in METHODS.items():
            for seed in SEEDS:
                frame = external_fusion_prediction(root, method, seed, dataset)
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "label": spec["label"],
                        "components": spec["components"],
                        "policy": spec["policy"],
                        "seed": seed,
                        **metrics(frame),
                    }
                )
    raw = pd.DataFrame(rows)
    summary = summarize_seed_rows(raw, ["dataset", "method", "label", "components", "policy"])
    return raw, summary


def prediction_by_family(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in frame.groupby("family_key", sort=True):
        if len(group) != 2 or set(group["y"].astype(float)) != {0.0, 1.0}:
            continue
        positive = group[group["y"].eq(1.0)].iloc[0]
        negative = group[group["y"].eq(0.0)].iloc[0]
        rows.append(
            {
                "family_key": family,
                "positive_prediction": float(positive["prediction"]),
                "negative_prediction": float(negative["prediction"]),
            }
        )
    return pd.DataFrame(rows).sort_values("family_key").reset_index(drop=True)


def bootstrap_metric(positive: np.ndarray, negative: np.ndarray, indices: np.ndarray, metric: str) -> float:
    pos, neg = positive[indices], negative[indices]
    if metric == "roc_auc":
        y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
        p = np.concatenate((pos, neg))
        return float(roc_auc_score(y, p))
    if metric == "pair_order_accuracy":
        return float(np.mean(pos > neg))
    if metric == "false_invariance":
        return float(np.mean(np.abs(pos - neg) < 0.25))
    raise ValueError(metric)


def bootstrap_metric_replicates(
    positive: np.ndarray,
    negative: np.ndarray,
    draws: np.ndarray,
    metric: str,
) -> np.ndarray:
    pos = positive[draws]
    neg = negative[draws]
    if metric == "pair_order_accuracy":
        return np.mean(pos > neg, axis=1)
    if metric == "false_invariance":
        return np.mean(np.abs(pos - neg) < 0.25, axis=1)
    if metric == "roc_auc":
        result = np.empty(len(draws), dtype=float)
        chunk_size = 128
        for start in range(0, len(draws), chunk_size):
            stop = min(start + chunk_size, len(draws))
            difference = pos[start:stop, :, None] - neg[start:stop, None, :]
            result[start:stop] = np.mean(difference > 0.0, axis=(1, 2)) + 0.5 * np.mean(
                difference == 0.0, axis=(1, 2)
            )
        return result
    raise ValueError(metric)


def family_bootstrap(
    frames: dict[str, dict[int, pd.DataFrame]], dataset: str, comparisons: list[tuple[str, str]]
) -> pd.DataFrame:
    family_predictions: dict[str, dict[int, pd.DataFrame]] = {
        method: {seed: prediction_by_family(frame) for seed, frame in seed_frames.items()}
        for method, seed_frames in frames.items()
    }
    families = None
    for seed_frames in family_predictions.values():
        for frame in seed_frames.values():
            current = set(frame["family_key"])
            families = current if families is None else families & current
    ordered_families = sorted(families or [])
    if not ordered_families:
        raise ValueError(f"no common families for bootstrap dataset={dataset}")

    arrays: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for method, seed_frames in family_predictions.items():
        arrays[method] = {}
        for seed, frame in seed_frames.items():
            aligned = frame.set_index("family_key").loc[ordered_families]
            arrays[method][seed] = (
                aligned["positive_prediction"].to_numpy(dtype=float),
                aligned["negative_prediction"].to_numpy(dtype=float),
            )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(ordered_families), size=(BOOTSTRAP_REPLICATES, len(ordered_families)))
    rows = []
    for candidate, baseline in comparisons:
        for metric in ["roc_auc", "pair_order_accuracy", "false_invariance"]:
            observed_by_seed, boot_by_seed = [], []
            identity = np.arange(len(ordered_families))
            for seed in SEEDS:
                cand_pos, cand_neg = arrays[candidate][seed]
                base_pos, base_neg = arrays[baseline][seed]
                observed_by_seed.append(
                    bootstrap_metric(cand_pos, cand_neg, identity, metric)
                    - bootstrap_metric(base_pos, base_neg, identity, metric)
                )
                seed_boot = bootstrap_metric_replicates(
                    cand_pos, cand_neg, draws, metric
                ) - bootstrap_metric_replicates(base_pos, base_neg, draws, metric)
                boot_by_seed.append(seed_boot)
            bootstrap_delta = np.mean(np.vstack(boot_by_seed), axis=0)
            rows.append(
                {
                    "dataset": dataset,
                    "candidate": candidate,
                    "baseline": baseline,
                    "metric": metric,
                    "delta_definition": "candidate_minus_baseline",
                    "n_families": len(ordered_families),
                    "n_seeds": len(SEEDS),
                    "replicates": BOOTSTRAP_REPLICATES,
                    "observed_delta": float(np.mean(observed_by_seed)),
                    "ci95_low": float(np.percentile(bootstrap_delta, 2.5)),
                    "ci95_high": float(np.percentile(bootstrap_delta, 97.5)),
                    "bootstrap_probability_delta_gt_0": float(np.mean(bootstrap_delta > 0.0)),
                    "favorable_direction": "negative" if metric == "false_invariance" else "positive",
                    "bootstrap_probability_candidate_better": float(
                        np.mean(bootstrap_delta < 0.0)
                        if metric == "false_invariance"
                        else np.mean(bootstrap_delta > 0.0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def load_bootstrap_frames(
    root: Path,
) -> tuple[dict[str, dict[int, pd.DataFrame]], dict[str, dict[str, dict[int, pd.DataFrame]]]]:
    manifest = pd.read_csv(root / "reports/tables/evaluation_clean_tautobase_family_manifest.csv")
    clean_ids = set(manifest.loc[manifest["checkpoint_clean"], "left_sample_id"]) | set(
        manifest.loc[manifest["checkpoint_clean"], "right_sample_id"]
    )
    samples = pd.read_parquet(root / "data/processed/main/tautobase_pref_samples.parquet")[
        ["sample_id", "family_key"]
    ]
    main_frames: dict[str, dict[int, pd.DataFrame]] = {}
    external_frames: dict[str, dict[str, dict[int, pd.DataFrame]]] = {
        "sphysnet_100_aqueous": {},
        "nci_aqueous_zero_overlap": {},
    }
    for method in ["A", "C", "D", "E"]:
        main_frames[method] = {}
        for dataset in external_frames:
            external_frames[dataset][method] = {}
        for seed in SEEDS:
            main = pd.read_parquet(main_prediction_path(root, method, seed))
            main = main[main["sample_id"].isin(clean_ids)].copy()
            if "family_key" not in main.columns:
                main = main.merge(samples, on="sample_id", how="left")
            main_frames[method][seed] = main
            for dataset in external_frames:
                external_frames[dataset][method][seed] = external_fusion_prediction(root, method, seed, dataset)
    return main_frames, external_frames


def paired_seed_deltas(raw: pd.DataFrame, scope_column: str) -> pd.DataFrame:
    rows = []
    comparisons = [("E", "A"), ("E", "C"), ("E", "D")]
    for scope, group in raw.groupby(scope_column, sort=False):
        for candidate, baseline in comparisons:
            left = group[group["method"].eq(candidate)].set_index("seed")
            right = group[group["method"].eq(baseline)].set_index("seed")
            for metric in ["roc_auc", "pair_order_accuracy", "false_invariance", "brier", "nll"]:
                delta = left[metric] - right[metric]
                rows.append(
                    {
                        "evaluation_scope": scope,
                        "candidate": candidate,
                        "baseline": baseline,
                        "metric": metric,
                        "delta_definition": "candidate_minus_baseline",
                        "mean_delta": float(delta.mean()),
                        "std_delta": float(delta.std(ddof=1)),
                        "candidate_wins": int((delta > 0).sum())
                        if metric in {"roc_auc", "pair_order_accuracy"}
                        else int((delta < 0).sum()),
                        "seed_count": int(len(delta)),
                    }
                )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False, floatfmt=".6f")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    table_dir = root / "reports/tables"
    main_raw, main_summary = build_main_ablation(root)
    external_raw, external_summary = build_external_ablation(root)
    main_frames, external_frames = load_bootstrap_frames(root)
    deltas = pd.concat(
        [paired_seed_deltas(main_raw, "scope"), paired_seed_deltas(external_raw, "dataset")],
        ignore_index=True,
    )
    bootstrap = pd.concat(
        [
            family_bootstrap(
                main_frames,
                "tautobase_checkpoint_clean",
                [("E", "A"), ("E", "C"), ("E", "D")],
            ),
            family_bootstrap(
                external_frames["sphysnet_100_aqueous"],
                "sphysnet_100_aqueous",
                [("E", "A"), ("E", "C"), ("E", "D")],
            ),
            family_bootstrap(
                external_frames["nci_aqueous_zero_overlap"],
                "nci_aqueous_zero_overlap",
                [("E", "A"), ("E", "C"), ("E", "D")],
            ),
        ],
        ignore_index=True,
    )

    main_raw.to_csv(table_dir / "evaluation_ablation_ABCDE_seed_raw.csv", index=False)
    main_summary.to_csv(table_dir / "evaluation_ablation_ABCDE.csv", index=False)
    external_raw.to_csv(table_dir / "evaluation_external_ABCDE_seed_raw.csv", index=False)
    external_summary.to_csv(table_dir / "evaluation_external_ABCDE.csv", index=False)
    deltas.to_csv(table_dir / "evaluation_paired_seed_deltas.csv", index=False)
    bootstrap.to_csv(table_dir / "evaluation_family_bootstrap_5000.csv", index=False)

    display = [
        "scope",
        "method",
        "label",
        "roc_auc_mean",
        "roc_auc_std",
        "pair_order_accuracy_mean",
        "false_invariance_mean",
        "brier_mean",
        "nll_mean",
        "n_pairs_mean",
    ]
    external_display = [
        "dataset",
        "method",
        "label",
        "roc_auc_mean",
        "roc_auc_std",
        "pair_order_accuracy_mean",
        "false_invariance_mean",
        "brier_mean",
        "nll_mean",
        "n_pairs_mean",
    ]
    report = [
        "# Evaluation Required Experiments",
        "",
        "## A-E ablation on the frozen Tautobase protocol",
        "",
        markdown_table(main_summary[display]),
        "",
        "The full-split scale=1.2 row remains retrospective because the earlier exploratory campaign inspected test behavior. The checkpoint-clean subset is a contamination sensitivity analysis, not a fresh untouched test. Confirmatory interpretation is therefore reserved for the frozen external evaluation below.",
        "",
        "## Frozen external evaluation",
        "",
        markdown_table(external_summary[external_display]),
        "",
        "All fusion weights, policies, scales, and internal-model hyperparameters were frozen from Tautobase before external labels were evaluated. The sPhysNet-100 set is training/validation clean for the released artifacts available locally, but it is a known published test set for both same-task checkpoint papers and is labeled accordingly in the provenance audit. NCI results use the 22-family common support left after six families with elements unsupported by the released sPhysNet checkpoint were excluded.",
        "",
        "## Paired five-seed deltas",
        "",
        markdown_table(deltas),
        "",
        "## Family-level paired bootstrap (5,000 replicates)",
        "",
        markdown_table(bootstrap),
        "",
        "For false-invariance, a negative candidate-minus-baseline delta favors the candidate. Bootstrap draws resample complete tautomer families and use the same draw for both compared methods and all five seeds.",
        "",
    ]
    (table_dir / "evaluation_required_experiments.md").write_text("\n".join(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "main_rows": len(main_raw),
                "external_rows": len(external_raw),
                "bootstrap_rows": len(bootstrap),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
