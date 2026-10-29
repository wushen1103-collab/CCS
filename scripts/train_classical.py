from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR

try:
    from sklearn.calibration import CalibratedClassifierCV
except Exception:  # pragma: no cover
    CalibratedClassifierCV = None

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None

from condchemsymm.metrics import classification_metrics, regression_metrics


FEATURE_CACHE: dict[tuple[str, str], FeatureBundle] = {}


DEFAULT_METHODS = [
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


CLASSIFICATION_METHODS = {
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
}


REGRESSION_METHODS = {
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
}


METHOD_GROUPS = {
    "ecfp_logreg": "classic_fingerprint_linear",
    "ecfp_linear_svm": "classic_fingerprint_linear",
    "ecfp_rf": "classic_fingerprint_tree",
    "ecfp_extratrees": "classic_fingerprint_tree",
    "ecfp_histgb": "classic_fingerprint_tree",
    "rdkit_desc_logreg": "classic_descriptor_linear",
    "rdkit_desc_rf": "classic_descriptor_tree",
    "rdkit_desc_extratrees": "classic_descriptor_tree",
    "rdkit_desc_histgb": "classic_descriptor_tree",
    "tanimoto_knn": "classic_similarity_nonparametric",
    "rdkit_tautomer_canonical_prior": "direct_mechanism_tautomer_rule",
    "inchi_mobile_h_invariant_prior": "direct_mechanism_identity_rule",
}


@dataclass(frozen=True)
class FeatureBundle:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    positive_views: dict[str, tuple[pd.DataFrame, np.ndarray]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def mol_from_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    return mol


def canonical_smiles(smiles: str, isomeric: bool = True) -> str:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def morgan_matrix(smiles: list[str], n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    rows = []
    for smi in smiles:
        mol = mol_from_smiles(smi)
        arr = np.zeros((n_bits,), dtype=np.float32)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius,
                nBits=n_bits,
                useChirality=True,
            )
            DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    return np.stack(rows).astype(np.float32)


DESC_LIST = list(Descriptors._descList)


def descriptor_vector(smiles: str) -> list[float]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return [math.nan] * len(DESC_LIST)
    values = []
    for _, fn in DESC_LIST:
        try:
            value = float(fn(mol))
        except Exception:
            value = math.nan
        if not math.isfinite(value):
            value = math.nan
        values.append(value)
    return values


def descriptor_matrix(smiles: list[str]) -> np.ndarray:
    return np.asarray([descriptor_vector(smi) for smi in smiles], dtype=np.float32)


def positive_view_smiles(samples: pd.DataFrame, edges: pd.DataFrame, transform_type: str) -> pd.DataFrame:
    view = edges[(edges["z"] == 1) & (edges["transform_type"] == transform_type)][["sample_id", "view_smiles"]]
    subset = samples.merge(view, on="sample_id", how="inner")
    subset = subset.copy()
    subset["eval_smiles"] = subset["view_smiles"]
    return subset


def make_feature_bundle(samples: pd.DataFrame, edges: pd.DataFrame, feature_kind: str) -> FeatureBundle:
    train = samples[samples["split"] == "train"].copy()
    val = samples[samples["split"] == "val"].copy()
    test = samples[samples["split"] == "test"].copy()

    def featurize(smiles: list[str]) -> np.ndarray:
        if feature_kind == "ecfp":
            return morgan_matrix(smiles)
        if feature_kind == "rdkit_desc":
            return descriptor_matrix(smiles)
        raise ValueError(feature_kind)

    positive_views = {}
    for transform_type in sorted(edges[edges["z"] == 1]["transform_type"].unique()):
        subset = positive_view_smiles(test, edges, str(transform_type))
        if len(subset):
            positive_views[str(transform_type)] = (subset, featurize(list(subset["eval_smiles"])))

    return FeatureBundle(
        train=featurize(list(train["smiles"])),
        val=featurize(list(val["smiles"])),
        test=featurize(list(test["smiles"])),
        positive_views=positive_views,
    )


def make_estimator(method: str, task_type: str, seed: int, n_jobs: int):
    is_classification = task_type == "classification"
    if method in {"ecfp_logreg", "rdkit_desc_logreg"}:
        if is_classification:
            return make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(with_mean=False),
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    class_weight="balanced",
                    n_jobs=1,
                    random_state=seed,
                ),
            )
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0, random_state=seed))

    if method in {"ecfp_linear_svm", "rdkit_desc_linear_svm"}:
        if is_classification:
            if CalibratedClassifierCV is None:
                raise RuntimeError("CalibratedClassifierCV is unavailable")
            base = LinearSVC(C=1.0, class_weight="balanced", random_state=seed, max_iter=5000)
            try:
                return make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(with_mean=False),
                    CalibratedClassifierCV(estimator=base, cv=3),
                )
            except TypeError:
                return make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(with_mean=False),
                    CalibratedClassifierCV(base_estimator=base, cv=3),
                )
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearSVR(C=1.0, random_state=seed, max_iter=5000))

    if method in {"ecfp_rf", "rdkit_desc_rf"}:
        cls = RandomForestClassifier if is_classification else RandomForestRegressor
        kwargs = {
            "n_estimators": 400,
            "min_samples_leaf": 1,
            "n_jobs": n_jobs,
            "random_state": seed,
        }
        if is_classification:
            kwargs["class_weight"] = "balanced_subsample"
        return make_pipeline(SimpleImputer(strategy="median"), cls(**kwargs))

    if method in {"ecfp_extratrees", "rdkit_desc_extratrees"}:
        cls = ExtraTreesClassifier if is_classification else ExtraTreesRegressor
        kwargs = {
            "n_estimators": 400,
            "min_samples_leaf": 1,
            "n_jobs": n_jobs,
            "random_state": seed,
        }
        if is_classification:
            kwargs["class_weight"] = "balanced"
        return make_pipeline(SimpleImputer(strategy="median"), cls(**kwargs))

    if method in {"ecfp_histgb", "rdkit_desc_histgb"}:
        cls = HistGradientBoostingClassifier if is_classification else HistGradientBoostingRegressor
        return make_pipeline(
            SimpleImputer(strategy="median"),
            cls(
                max_iter=300,
                learning_rate=0.05,
                l2_regularization=0.01,
                early_stopping=True,
                random_state=seed,
            ),
        )

    raise ValueError(method)


def predict_estimator(estimator, x: np.ndarray, task_type: str) -> np.ndarray:
    if task_type == "classification":
        if hasattr(estimator, "predict_proba"):
            return estimator.predict_proba(x)[:, 1].astype(float)
        if hasattr(estimator, "decision_function"):
            raw = estimator.decision_function(x)
            return (1.0 / (1.0 + np.exp(-raw))).astype(float)
        raise RuntimeError(f"Estimator lacks probability output: {type(estimator)}")
    return estimator.predict(x).astype(float)


def tanimoto_knn_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, task_type: str, k: int = 11) -> np.ndarray:
    x_train_bool = x_train.astype(bool)
    x_eval_bool = x_eval.astype(bool)
    preds = []
    for row in x_eval_bool:
        inter = np.logical_and(x_train_bool, row).sum(axis=1).astype(float)
        union = np.logical_or(x_train_bool, row).sum(axis=1).astype(float)
        sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        top = np.argsort(sim)[-min(k, len(sim)) :]
        weights = sim[top] + 1e-6
        pred = float(np.average(y_train[top], weights=weights))
        if task_type == "classification":
            pred = float(np.clip(pred, 1e-4, 1 - 1e-4))
        preds.append(pred)
    return np.asarray(preds, dtype=float)


def fit_predict_method(method: str, task_type: str, seed: int, features: FeatureBundle, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, n_jobs: int):
    y_train = train["y"].to_numpy(dtype=float)
    if method == "tanimoto_knn":
        val_pred = tanimoto_knn_predict(features.train, y_train, features.val, task_type)
        test_pred = tanimoto_knn_predict(features.train, y_train, features.test, task_type)
        view_pred = {
            transform: tanimoto_knn_predict(features.train, y_train, x_view, task_type)
            for transform, (_, x_view) in features.positive_views.items()
        }
        return val_pred, test_pred, view_pred

    estimator = make_estimator(method, task_type, seed, n_jobs)
    estimator.fit(features.train, y_train)
    val_pred = predict_estimator(estimator, features.val, task_type)
    test_pred = predict_estimator(estimator, features.test, task_type)
    view_pred = {
        transform: predict_estimator(estimator, x_view, task_type)
        for transform, (_, x_view) in features.positive_views.items()
    }
    if task_type == "classification":
        val_pred = np.clip(val_pred, 1e-4, 1 - 1e-4)
        test_pred = np.clip(test_pred, 1e-4, 1 - 1e-4)
        view_pred = {k: np.clip(v, 1e-4, 1 - 1e-4) for k, v in view_pred.items()}
    return val_pred, test_pred, view_pred


def get_feature_bundle(dataset: str, feature_kind: str, samples: pd.DataFrame, edges: pd.DataFrame) -> FeatureBundle:
    key = (dataset, feature_kind)
    if key not in FEATURE_CACHE:
        FEATURE_CACHE[key] = make_feature_bundle(samples, edges, feature_kind)
    return FEATURE_CACHE[key]


def tautomer_prior_prob(samples_subset: pd.DataFrame, all_samples: pd.DataFrame) -> np.ndarray:
    enum = rdMolStandardize.TautomerEnumerator()
    by_id = {row["sample_id"]: row for _, row in all_samples.iterrows()}
    cache = {}

    def canonical_tautomer(smiles: str) -> str:
        if smiles not in cache:
            mol = mol_from_smiles(smiles)
            if mol is None:
                cache[smiles] = ""
            else:
                try:
                    taut = enum.Canonicalize(mol)
                    cache[smiles] = Chem.MolToSmiles(taut, canonical=True, isomericSmiles=True)
                except Exception:
                    cache[smiles] = ""
        return cache[smiles]

    pair_scores = {}
    for _, row in all_samples.iterrows():
        sid = row["sample_id"]
        pid = row.get("paired_id", "")
        if not pid or sid in pair_scores or pid not in by_id:
            continue
        a = canonical_smiles(str(row["smiles"]))
        b = canonical_smiles(str(by_id[pid]["smiles"]))
        c = canonical_tautomer(str(row["smiles"])) or canonical_tautomer(str(by_id[pid]["smiles"]))
        if c and a == c and b != c:
            pair_scores[sid] = 0.75
            pair_scores[pid] = 0.25
        elif c and b == c and a != c:
            pair_scores[sid] = 0.25
            pair_scores[pid] = 0.75
        else:
            pair_scores[sid] = 0.5
            pair_scores[pid] = 0.5
    return np.asarray([pair_scores.get(sid, 0.5) for sid in samples_subset["sample_id"]], dtype=float)


def inchi_mobile_h_prior_prob(samples_subset: pd.DataFrame, all_samples: pd.DataFrame) -> np.ndarray:
    by_id = {row["sample_id"]: row for _, row in all_samples.iterrows()}
    key_cache: dict[str, str] = {}

    def inchi_key14(smiles: str) -> str:
        if smiles not in key_cache:
            mol = mol_from_smiles(smiles)
            if mol is None:
                key_cache[smiles] = ""
            else:
                try:
                    key_cache[smiles] = Chem.MolToInchiKey(mol).split("-")[0]
                except Exception:
                    key_cache[smiles] = ""
        return key_cache[smiles]

    scores = {}
    fallback = tautomer_prior_prob(all_samples, all_samples)
    fallback_by_id = dict(zip(all_samples["sample_id"], fallback))
    for _, row in all_samples.iterrows():
        sid = row["sample_id"]
        pid = row.get("paired_id", "")
        if not pid or sid in scores or pid not in by_id:
            continue
        key_a = inchi_key14(str(row["smiles"]))
        key_b = inchi_key14(str(by_id[pid]["smiles"]))
        if key_a and key_a == key_b:
            scores[sid] = 0.5
            scores[pid] = 0.5
        else:
            scores[sid] = fallback_by_id.get(sid, 0.5)
            scores[pid] = fallback_by_id.get(pid, 0.5)
    return np.asarray([scores.get(sid, 0.5) for sid in samples_subset["sample_id"]], dtype=float)


def paired_false_invariance(samples_subset: pd.DataFrame, predictions: np.ndarray, threshold: float = 0.25) -> tuple[float, int]:
    pred_by_id = dict(zip(samples_subset["sample_id"], predictions))
    vals = []
    seen = set()
    for _, row in samples_subset.iterrows():
        sid = row["sample_id"]
        pid = row.get("paired_id", "")
        if not pid or pid not in pred_by_id or sid in seen or pid in seen:
            continue
        seen.add(sid)
        seen.add(pid)
        vals.append(abs(float(pred_by_id[sid]) - float(pred_by_id[pid])) < threshold)
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), len(vals)


def eval_predictions(task_type: str, y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if task_type == "classification":
        return classification_metrics(y_true, np.clip(pred, 1e-7, 1 - 1e-7))
    return regression_metrics(y_true, pred)


def positive_transform_eval(task_type: str, view_frames: dict[str, tuple[pd.DataFrame, np.ndarray]], view_predictions: dict[str, np.ndarray]) -> dict:
    out = {}
    for transform, (subset, _) in view_frames.items():
        metrics = eval_predictions(task_type, subset["y"].to_numpy(dtype=float), view_predictions[transform])
        out[transform] = metrics
    if task_type == "classification":
        aucs = [m["roc_auc"] for m in out.values() if "roc_auc" in m and not np.isnan(m["roc_auc"])]
        briers = [m["brier"] for m in out.values() if "brier" in m]
        out["worst_positive_transform_roc_auc"] = float(min(aucs)) if aucs else float("nan")
        out["worst_positive_transform_brier"] = float(max(briers)) if briers else float("nan")
    else:
        maes = [m["mae"] for m in out.values() if "mae" in m]
        out["worst_positive_transform_mae"] = float(max(maes)) if maes else float("nan")
    return out


def write_predictions(frame: pd.DataFrame, predictions: np.ndarray, path: Path) -> None:
    cols = [c for c in ["sample_id", "dataset", "split", "smiles", "y", "paired_id", "scaffold", "family_key"] if c in frame.columns]
    out = frame[cols].copy()
    out["prediction"] = predictions
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def flatten_metrics(metrics: dict) -> dict:
    row = {}
    for prefix in ["train", "val", "test"]:
        for key, value in metrics.get(prefix, {}).items():
            row[f"{prefix}_{key}"] = value
    row["val_false_invariance_rate_pairs"] = metrics.get("val_false_invariance_rate_pairs", math.nan)
    row["val_false_invariance_n_pairs"] = metrics.get("val_false_invariance_n_pairs", 0)
    row["false_invariance_rate_test_pairs"] = metrics.get("false_invariance_rate_test_pairs", math.nan)
    row["false_invariance_n_pairs"] = metrics.get("false_invariance_n_pairs", 0)
    pos = metrics.get("positive_transform_eval", {})
    row["pos_worst_positive_transform_roc_auc"] = pos.get("worst_positive_transform_roc_auc", math.nan)
    row["pos_worst_positive_transform_brier"] = pos.get("worst_positive_transform_brier", math.nan)
    row["pos_worst_positive_transform_mae"] = pos.get("worst_positive_transform_mae", math.nan)
    return row


def summarize(seed_rows: list[dict], out_prefix: Path) -> None:
    raw = pd.DataFrame(seed_rows)
    raw_path = out_prefix.with_name(out_prefix.name + "_seed_raw.csv")
    summary_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")
    raw.to_csv(raw_path, index=False)

    id_cols = [
        "method",
        "baseline_group",
        "dataset",
        "task_type",
        "feature_family",
        "source_result_type",
        "split_protocol",
    ]
    metric_cols = [
        c
        for c in raw.columns
        if c.startswith("test_")
        or c.startswith("val_")
        or c.startswith("pos_worst")
        or c in {"false_invariance_rate_test_pairs", "false_invariance_n_pairs"}
    ]
    rows = []
    for keys, group in raw.groupby(id_cols, dropna=False):
        row = dict(zip(id_cols, keys))
        row["seed_count"] = int(group["seed"].nunique())
        for col in metric_cols:
            vals = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else math.nan
            row[f"{col}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
            row[f"{col}_count"] = int(vals.notna().sum())
        rows.append(row)
    summary = pd.DataFrame(rows)
    sort_cols = ["dataset", "task_type", "baseline_group", "method"]
    summary = summary.sort_values(sort_cols).reset_index(drop=True)
    summary.to_csv(summary_path, index=False)

    display_cols = [
        "method",
        "baseline_group",
        "dataset",
        "seed_count",
        "test_roc_auc_mean",
        "test_roc_auc_std",
        "test_brier_mean",
        "test_brier_std",
        "false_invariance_rate_test_pairs_mean",
        "pos_worst_positive_transform_roc_auc_mean",
        "test_mae_mean",
        "test_mae_std",
        "pos_worst_positive_transform_mae_mean",
        "source_result_type",
    ]
    present = [c for c in display_cols if c in summary.columns]
    md = [
        "# Phase 1 Classical and Direct-Mechanism Baselines",
        "",
        "All rows are rerun on the frozen Phase 1 splits. Deterministic rule rows are repeated over the seed list with identical predictions and marked by their method name/source type.",
        "",
        summary[present].to_markdown(index=False),
        "",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")


def run_dataset_method(root: Path, dataset: str, method: str, seed: int, args) -> dict | None:
    samples_path = root / f"data/processed/{args.processed_scope}/{dataset}_samples.parquet"
    if not samples_path.exists():
        print(f"SKIP missing samples: {samples_path}", flush=True)
        return None
    samples = pd.read_parquet(samples_path)
    edges = pd.read_parquet(root / f"data/processed/{args.processed_scope}/transform_edges.parquet")
    edges = edges[edges["dataset"] == dataset].copy()
    task_type = str(samples["task_type"].iloc[0])

    if task_type == "classification" and method not in CLASSIFICATION_METHODS:
        return None
    if task_type == "regression" and method not in REGRESSION_METHODS:
        return None
    if method in {"rdkit_tautomer_canonical_prior", "inchi_mobile_h_invariant_prior"} and dataset != "tautobase_pref":
        return None

    train = samples[samples["split"] == "train"].copy()
    val = samples[samples["split"] == "val"].copy()
    test = samples[samples["split"] == "test"].copy()
    out_dir = root / "runs" / args.run_name / method / dataset / "identity" / f"seed_{seed}"
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        return {**{k: metrics[k] for k in ["method", "baseline_group", "dataset", "task_type", "seed", "feature_family", "source_result_type", "split_protocol"]}, **flatten_metrics(metrics)}

    start = time.time()
    set_seed(seed)
    if method in {"rdkit_tautomer_canonical_prior", "inchi_mobile_h_invariant_prior"}:
        direct_prob = tautomer_prior_prob if method == "rdkit_tautomer_canonical_prior" else inchi_mobile_h_prior_prob
        val_pred = direct_prob(val, samples)
        test_pred = direct_prob(test, samples)
        view_predictions = {}
        view_frames = {}
        for transform_type in sorted(edges[edges["z"] == 1]["transform_type"].unique()):
            subset = positive_view_smiles(test, edges, str(transform_type))
            if len(subset):
                view_frames[str(transform_type)] = (subset, np.empty((len(subset), 0), dtype=float))
                view_predictions[str(transform_type)] = direct_prob(subset, samples)
    else:
        feature_kind = "rdkit_desc" if method.startswith("rdkit_desc") else "ecfp"
        features = get_feature_bundle(dataset, feature_kind, samples, edges)
        val_pred, test_pred, view_predictions = fit_predict_method(method, task_type, seed, features, train, val, test, args.n_jobs)
        view_frames = features.positive_views

    train_pred = np.full(len(train), float(np.mean(train["y"])), dtype=float)
    if method not in {"rdkit_tautomer_canonical_prior", "inchi_mobile_h_invariant_prior"}:
        if method == "tanimoto_knn":
            feature_kind = "ecfp"
            features = get_feature_bundle(dataset, feature_kind, samples, edges)
            train_pred = tanimoto_knn_predict(features.train, train["y"].to_numpy(dtype=float), features.train, task_type)
        else:
            # Keep train metrics diagnostic only. Refit prediction is not needed for model selection.
            train_pred = np.full(len(train), float(np.mean(train["y"])), dtype=float)

    val_metrics = eval_predictions(task_type, val["y"].to_numpy(dtype=float), val_pred)
    test_metrics = eval_predictions(task_type, test["y"].to_numpy(dtype=float), test_pred)
    train_metrics = eval_predictions(task_type, train["y"].to_numpy(dtype=float), train_pred)
    val_fi, val_fi_n = paired_false_invariance(val, val_pred) if task_type == "classification" else (math.nan, 0)
    test_fi, test_fi_n = paired_false_invariance(test, test_pred) if task_type == "classification" else (math.nan, 0)
    pos_metrics = positive_transform_eval(task_type, view_frames, view_predictions)

    metrics = {
        "method": method,
        "baseline_group": METHOD_GROUPS[method],
        "dataset": dataset,
        "task_type": task_type,
        "feature_family": "rdkit_desc"
        if method.startswith("rdkit_desc")
        else ("direct_identity_rule" if method == "inchi_mobile_h_invariant_prior" else ("direct_tautomer_rule" if method == "rdkit_tautomer_canonical_prior" else "ecfp")),
        "source_result_type": "rerun_our_split",
        "split_protocol": f"{args.processed_scope}_frozen_duplicate_aware",
        "seed": int(seed),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "seconds": float(time.time() - start),
        "n_jobs": int(args.n_jobs),
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "val_false_invariance_rate_pairs": val_fi,
        "val_false_invariance_n_pairs": int(val_fi_n),
        "false_invariance_rate_test_pairs": test_fi,
        "false_invariance_n_pairs": int(test_fi_n),
        "positive_transform_eval": pos_metrics,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_predictions(val, val_pred, out_dir / "val_predictions.parquet")
    write_predictions(test, test_pred, out_dir / "test_predictions.parquet")
    for transform, pred in view_predictions.items():
        subset = view_frames[transform][0]
        write_predictions(subset, pred, out_dir / "view_predictions" / f"{transform}.parquet")

    row = {
        "method": method,
        "baseline_group": METHOD_GROUPS[method],
        "dataset": dataset,
        "task_type": task_type,
        "seed": int(seed),
        "feature_family": metrics["feature_family"],
        "source_result_type": metrics["source_result_type"],
        "split_protocol": metrics["split_protocol"],
    }
    row.update(flatten_metrics(metrics))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["tautobase_pref", "bace", "esol", "freesolv", "lipophilicity"])
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--run-name", default="main_classical_external_baselines")
    parser.add_argument("--processed-scope", default="main")
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.n_jobs))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.n_jobs))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.n_jobs))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(args.n_jobs))

    root = Path(__file__).resolve().parents[1]
    rows = []
    context = threadpool_limits(limits=args.n_jobs) if threadpool_limits is not None else None
    if context is None:
        iterator = None
    else:
        iterator = context.__enter__()
    try:
        for dataset in args.datasets:
            for method in args.methods:
                for seed in args.seeds:
                    print(f"RUN dataset={dataset} method={method} seed={seed}", flush=True)
                    row = run_dataset_method(root, dataset, method, seed, args)
                    if row is not None:
                        rows.append(row)
    finally:
        if context is not None:
            context.__exit__(None, None, None)

    out_prefix = root / "reports/tables/main_classical_external_baselines"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summarize(rows, out_prefix)
    print(f"WROTE {out_prefix.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
