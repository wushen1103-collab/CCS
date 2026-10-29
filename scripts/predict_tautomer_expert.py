from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from condchemsymm.metrics import classification_metrics


METHODS = [
    "tautomer_predictor_attentivefp_raw",
    "tautomer_predictor_attentivefp_pairnorm",
]
METHOD_GROUPS = {
    "tautomer_predictor_attentivefp_raw": "recent_same_task_2d_gnn_released_checkpoint",
    "tautomer_predictor_attentivefp_pairnorm": "direct_mechanism_pair_normalized_2d_gnn",
}


def configure_external_imports(root: Path) -> None:
    deps = root / ".deps/tautpred-pyg"
    external = root / "data/raw/external/Tautomer-Predictor-src"
    for path in [str(deps), str(external)]:
        if path not in sys.path:
            sys.path.insert(0, path)


def load_external_model(root: Path):
    configure_external_imports(root)
    from src.tautomer_model import load_model_from_state_dict

    checkpoint = root / "data/raw/external/Tautomer-Predictor-src/checkpoints/best_model_state_dict.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return load_model_from_state_dict(str(checkpoint))


def predict_external(smiles: list[str], model) -> np.ndarray:
    from src.tautomer_model import predict_smiles

    probs = np.asarray(predict_smiles(smiles, model), dtype=float)
    return np.clip(probs, 1e-5, 1.0 - 1e-5)


def positive_view_smiles(samples: pd.DataFrame, edges: pd.DataFrame, transform_type: str) -> pd.DataFrame:
    view = edges[(edges["z"] == 1) & (edges["transform_type"] == transform_type)][["sample_id", "view_smiles"]]
    subset = samples.merge(view, on="sample_id", how="inner")
    subset = subset.copy()
    subset["eval_smiles"] = subset["view_smiles"]
    return subset


def logit(prob: float) -> float:
    p = min(max(float(prob), 1e-7), 1.0 - 1e-7)
    return math.log(p / (1.0 - p))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def pair_normalize_predictions(frame: pd.DataFrame, raw_predictions: np.ndarray) -> np.ndarray:
    out = np.asarray(raw_predictions, dtype=float).copy()
    index_by_id = {sid: idx for idx, sid in enumerate(frame["sample_id"])}
    seen: set[str] = set()
    for _, row in frame.iterrows():
        sid = str(row["sample_id"])
        pid = str(row.get("paired_id", ""))
        if not pid or sid in seen or pid in seen or pid not in index_by_id:
            continue
        i = index_by_id[sid]
        j = index_by_id[pid]
        li = logit(raw_predictions[i])
        lj = logit(raw_predictions[j])
        out[i] = sigmoid(li - lj)
        out[j] = sigmoid(lj - li)
        seen.add(sid)
        seen.add(pid)
    return np.clip(out, 1e-5, 1.0 - 1e-5)


def paired_stats(frame: pd.DataFrame, predictions: np.ndarray, threshold: float = 0.25) -> dict[str, float]:
    pred_by_id = dict(zip(frame["sample_id"], predictions))
    y_by_id = dict(zip(frame["sample_id"], frame["y"]))
    false_flags = []
    order_flags = []
    gaps = []
    seen: set[str] = set()
    for _, row in frame.iterrows():
        sid = str(row["sample_id"])
        pid = str(row.get("paired_id", ""))
        if not pid or sid in seen or pid in seen or pid not in pred_by_id:
            continue
        seen.add(sid)
        seen.add(pid)
        pa = float(pred_by_id[sid])
        pb = float(pred_by_id[pid])
        ya = float(y_by_id[sid])
        yb = float(y_by_id[pid])
        gap = abs(pa - pb)
        gaps.append(gap)
        false_flags.append(gap < threshold)
        if ya != yb:
            pos = pa if ya > yb else pb
            neg = pb if ya > yb else pa
            order_flags.append(pos > neg)
    return {
        "false_invariance_rate_test_pairs": float(np.mean(false_flags)) if false_flags else math.nan,
        "false_invariance_n_pairs": int(len(false_flags)),
        "pair_order_accuracy": float(np.mean(order_flags)) if order_flags else math.nan,
        "pair_gap_mean": float(np.mean(gaps)) if gaps else math.nan,
        "pair_gap_p10": float(np.percentile(gaps, 10)) if gaps else math.nan,
    }


def eval_predictions(frame: pd.DataFrame, predictions: np.ndarray) -> dict[str, float]:
    out = classification_metrics(frame["y"].to_numpy(dtype=float), predictions)
    out.update(paired_stats(frame, predictions))
    return out


def positive_transform_eval(test: pd.DataFrame, edges: pd.DataFrame, model, method: str) -> tuple[dict, dict[str, pd.DataFrame]]:
    rows = {}
    pred_frames = {}
    for transform_type in sorted(edges[edges["z"] == 1]["transform_type"].unique()):
        subset = positive_view_smiles(test, edges, str(transform_type))
        if subset.empty:
            continue
        raw = predict_external(list(subset["eval_smiles"]), model)
        pred = pair_normalize_predictions(subset, raw) if method.endswith("_pairnorm") else raw
        metrics = eval_predictions(subset, pred)
        rows[str(transform_type)] = metrics
        pred_frame = subset[[c for c in ["sample_id", "dataset", "split", "smiles", "y", "paired_id", "scaffold", "family_key"] if c in subset.columns]].copy()
        pred_frame["prediction"] = pred
        pred_frames[str(transform_type)] = pred_frame
    aucs = [m["roc_auc"] for m in rows.values() if not np.isnan(m.get("roc_auc", np.nan))]
    briers = [m["brier"] for m in rows.values() if not np.isnan(m.get("brier", np.nan))]
    rows["worst_positive_transform_roc_auc"] = float(min(aucs)) if aucs else math.nan
    rows["worst_positive_transform_brier"] = float(max(briers)) if briers else math.nan
    return rows, pred_frames


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
    pos = metrics.get("positive_transform_eval", {})
    row["pos_worst_positive_transform_roc_auc"] = pos.get("worst_positive_transform_roc_auc", math.nan)
    row["pos_worst_positive_transform_brier"] = pos.get("worst_positive_transform_brier", math.nan)
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
    metric_cols = [c for c in raw.columns if c.startswith(("train_", "val_", "test_", "pos_worst"))]
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
    summary = pd.DataFrame(rows).sort_values(["baseline_group", "method"]).reset_index(drop=True)
    summary.to_csv(summary_path, index=False)

    display_cols = [
        "method",
        "baseline_group",
        "seed_count",
        "test_roc_auc_mean",
        "test_roc_auc_std",
        "test_brier_mean",
        "test_brier_std",
        "test_nll_mean",
        "test_nll_std",
        "test_false_invariance_rate_test_pairs_mean",
        "test_pair_order_accuracy_mean",
        "pos_worst_positive_transform_roc_auc_mean",
        "source_result_type",
    ]
    md = [
        "# Phase 1 Tautomer-Predictor Baseline",
        "",
        "Released AttentiveFP checkpoint from `xiaolinpan/Tautomer-Predictor`, rerun on the frozen duplicate-aware Tautobase split. The checkpoint is deterministic; five seed rows intentionally repeat the same predictions so downstream mean/std tables remain protocol-aligned.",
        "",
        summary[[c for c in display_cols if c in summary.columns]].to_markdown(index=False),
        "",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")


def run_method(root: Path, model, method: str, seed: int) -> dict:
    samples = pd.read_parquet(root / "data/processed/main/tautobase_pref_samples.parquet")
    edges = pd.read_parquet(root / "data/processed/main/transform_edges.parquet")
    edges = edges[edges["dataset"] == "tautobase_pref"].copy()
    train = samples[samples["split"].eq("train")].copy()
    val = samples[samples["split"].eq("val")].copy()
    test = samples[samples["split"].eq("test")].copy()

    start = time.time()
    raw_train = predict_external(list(train["smiles"]), model)
    raw_val = predict_external(list(val["smiles"]), model)
    raw_test = predict_external(list(test["smiles"]), model)
    if method.endswith("_pairnorm"):
        train_pred = pair_normalize_predictions(train, raw_train)
        val_pred = pair_normalize_predictions(val, raw_val)
        test_pred = pair_normalize_predictions(test, raw_test)
    else:
        train_pred = raw_train
        val_pred = raw_val
        test_pred = raw_test

    pos_eval, view_predictions = positive_transform_eval(test, edges, model, method)
    metrics = {
        "method": method,
        "baseline_group": METHOD_GROUPS[method],
        "dataset": "tautobase_pref",
        "task_type": "classification",
        "feature_family": "attentivefp_2d_graph",
        "source_result_type": "released_checkpoint_rerun_our_split_repeated_over_seed_table",
        "split_protocol": "main_frozen_duplicate_aware",
        "seed": int(seed),
        "seconds": float(time.time() - start),
        "train": eval_predictions(train, train_pred),
        "val": eval_predictions(val, val_pred),
        "test": eval_predictions(test, test_pred),
        "positive_transform_eval": pos_eval,
    }

    out_dir = root / "runs/main_tautomer_predictor_baseline" / method / "tautobase_pref" / "identity" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_predictions(train, train_pred, out_dir / "train_predictions.parquet")
    write_predictions(val, val_pred, out_dir / "val_predictions.parquet")
    write_predictions(test, test_pred, out_dir / "test_predictions.parquet")
    for transform, frame in view_predictions.items():
        path = out_dir / "view_predictions" / f"{transform}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    row = {
        "method": method,
        "baseline_group": METHOD_GROUPS[method],
        "dataset": "tautobase_pref",
        "task_type": "classification",
        "seed": int(seed),
        "feature_family": "attentivefp_2d_graph",
        "source_result_type": metrics["source_result_type"],
        "split_protocol": metrics["split_protocol"],
    }
    row.update(flatten_metrics(metrics))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    model = load_external_model(root)
    rows = []
    for method in args.methods:
        for seed in args.seeds:
            print(f"RUN method={method} seed={seed}", flush=True)
            rows.append(run_method(root, model, method, seed))
    out_prefix = root / "reports/tables/main_tautomer_predictor_baseline"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summarize(rows, out_prefix)
    print(f"WROTE {out_prefix.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
