from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from condchemsymm.metrics import classification_metrics, regression_metrics


POSITIVE_TRANSFORMS = ["atom_renumber", "identity", "random_smiles"]
AUGMENT_TRANSFORMS = ["atom_renumber", "random_smiles"]
MODES = ["no_aug", "positive_aug", "canonical_no_aug", "canonical_positive_aug"]


def run(cmd: list[str], log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("RUN " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)


def task_metrics(task_type: str, y_true, pred) -> dict[str, float]:
    if task_type == "regression":
        return regression_metrics(y_true, pred)
    return classification_metrics(y_true, pred)


def primary_score(task_type: str) -> tuple[list[str], str]:
    if task_type == "regression":
        return ["rmse", "mae"], "rmse"
    return ["roc", "prc", "accuracy"], "roc"


def mode_uses_canonical_smiles(mode: str) -> bool:
    return mode.startswith("canonical_")


def mode_uses_positive_aug(mode: str) -> bool:
    return mode.endswith("positive_aug")


def canonical_smiles(smiles: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES for Chemprop canonicalization: {smiles}")
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def maybe_canonicalize(df: pd.DataFrame, enabled: bool) -> pd.DataFrame:
    if not enabled:
        return df
    out = df.copy()
    out["smiles"] = [canonical_smiles(smi) for smi in out["smiles"]]
    return out


def prepare_training_input(data_dir: Path, out_dir: Path, mode: str) -> tuple[Path, pd.DataFrame, pd.DataFrame, list[str]]:
    identity = pd.read_csv(data_dir / "identity.csv")
    frames = [identity.assign(aug_view="identity")]
    aug_views = ["identity"]
    if mode_uses_positive_aug(mode):
        for transform in AUGMENT_TRANSFORMS:
            path = data_dir / f"view_{transform}_train.csv"
            if not path.exists():
                continue
            frames.append(pd.read_csv(path).assign(aug_view=transform))
            aug_views.append(transform)
    train_input = pd.concat(frames, ignore_index=True)
    train_input = maybe_canonicalize(train_input, mode_uses_canonical_smiles(mode))
    train_path = out_dir / "chemprop_train_input.csv"
    train_input.to_csv(train_path, index=False)
    return train_path, identity, train_input, aug_views


def paired_false_invariance(pred: pd.DataFrame, threshold: float = 0.25) -> tuple[float, int]:
    if "paired_id" not in pred.columns:
        return float("nan"), 0
    pred_by_id = pred.set_index("sample_id")["prediction"].to_dict()
    vals = []
    seen = set()
    for _, row in pred.iterrows():
        sid = row["sample_id"]
        pid = row.get("paired_id", "")
        if not pid or pid not in pred_by_id or sid in seen or pid in seen:
            continue
        vals.append(abs(float(pred_by_id[sid]) - float(pred_by_id[pid])) < threshold)
        seen.add(sid)
        seen.add(pid)
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), len(vals)


def predict_view(
    chemprop_bin: str,
    model_path: Path,
    input_path: Path,
    output_path: Path,
    accelerator: str,
    devices: str,
    batch_size: int,
    num_workers: int,
    log_path: Path,
    canonicalize_input: bool,
) -> pd.DataFrame:
    source = pd.read_csv(input_path)
    model_input = maybe_canonicalize(source, canonicalize_input)
    if canonicalize_input:
        model_input_path = output_path.with_name(f"{output_path.stem}.input.csv")
        model_input.to_csv(model_input_path, index=False)
    else:
        model_input_path = input_path
    cmd = [
        chemprop_bin,
        "predict",
        "-i",
        str(model_input_path),
        "-o",
        str(output_path),
        "--smiles-columns",
        "smiles",
        "--model-paths",
        str(model_path),
        "--accelerator",
        accelerator,
        "--devices",
        devices,
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
    ]
    run(cmd, log_path)
    pred_out = pd.read_csv(output_path)
    if len(source) != len(pred_out):
        raise RuntimeError(f"Prediction length mismatch for {input_path}: source={len(source)} output={len(pred_out)}")
    pred = source.copy()
    if "sample_id" in source.columns and "sample_id" in pred_out.columns:
        aligned = source[["sample_id"]].merge(
            pred_out[["sample_id", "target"]],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        if aligned["target"].isna().any():
            missing = aligned.loc[aligned["target"].isna(), "sample_id"].head().tolist()
            raise RuntimeError(f"Missing Chemprop predictions for sample_id(s): {missing}")
        pred["prediction"] = aligned["target"].to_numpy(dtype=float)
    else:
        pred["prediction"] = pred_out["target"].to_numpy(dtype=float)
    if "target" in pred.columns and "y" not in pred.columns:
        pred["y"] = pred["target"].to_numpy(dtype=float)
    if canonicalize_input:
        pred["chemprop_smiles"] = model_input["smiles"].to_numpy()
    return pred


def positive_transform_eval(task_type: str, view_predictions: dict[str, pd.DataFrame]) -> dict:
    out = {}
    for transform, pred in view_predictions.items():
        out[transform] = task_metrics(task_type, pred["target"].to_numpy(), pred["prediction"].to_numpy())
    if task_type == "regression":
        vals = [m["mae"] for m in out.values() if "mae" in m]
        out["worst_positive_transform_mae"] = float(max(vals)) if vals else float("nan")
    else:
        aucs = [m["roc_auc"] for m in out.values() if "roc_auc" in m and not np.isnan(m["roc_auc"])]
        briers = [m["brier"] for m in out.values() if "brier" in m]
        out["worst_positive_transform_roc_auc"] = float(min(aucs)) if aucs else float("nan")
        out["worst_positive_transform_brier"] = float(max(briers)) if briers else float("nan")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-name", default="base_chemprop_v231_5seed")
    parser.add_argument("--mode", default="no_aug", choices=MODES)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--devices", default="1")
    parser.add_argument("--chemprop-bin", default=".venv_chemprop/bin/chemprop")
    parser.add_argument("--data-dir", default="data/chemprop_base")
    parser.add_argument("--processed-scope", default="base")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = root / args.data_dir / args.dataset
    identity_csv = data_dir / "identity.csv"
    identity_test_csv = data_dir / "identity_test.csv"
    if not identity_csv.exists() or not identity_test_csv.exists():
        raise SystemExit(f"Missing Chemprop export for {args.dataset}; run scripts/export_chemprop_data.py first")

    samples = pd.read_csv(identity_csv)
    task_type = "classification" if set(samples["target"].dropna().astype(float).unique()).issubset({0.0, 1.0}) else "regression"
    out_dir = root / "runs" / args.run_name / "chemprop_v231" / args.dataset / args.mode / f"seed_{args.seed}"
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        print(metrics_path.read_text())
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = root / "logs" / f"{args.run_name}_{args.dataset}_{args.mode}_seed_{args.seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    train_csv, identity_samples, train_input, aug_views = prepare_training_input(data_dir, out_dir, args.mode)
    metrics, tracking_metric = primary_score(task_type)
    warmup_epochs = max(0, min(2, args.epochs - 1))
    train_cmd = [
        args.chemprop_bin,
        "train",
        "-i",
        str(train_csv),
        "-o",
        str(out_dir),
        "--smiles-columns",
        "smiles",
        "--target-columns",
        "target",
        "--ignore-columns",
        "sample_id",
        "dataset",
        "split",
        "paired_id",
        "aug_view",
        "--splits-column",
        "split",
        "-t",
        task_type,
        "--metrics",
        *metrics,
        "--tracking-metric",
        tracking_metric,
        "--epochs",
        str(args.epochs),
        "--warmup-epochs",
        str(warmup_epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--accelerator",
        args.accelerator,
        "--devices",
        args.devices,
        "--pytorch-seed",
        str(args.seed),
        "--data-seed",
        str(args.seed),
    ]

    start = time.time()
    run(train_cmd, log_path)
    model_paths = sorted(out_dir.glob("model_*/best.pt"))
    if not model_paths:
        raise RuntimeError(f"No Chemprop best.pt found under {out_dir}")
    model_path = model_paths[0]

    val_metrics = {}
    val_false_inv = float("nan")
    val_n_pairs = 0
    identity_val_csv = data_dir / "identity_val.csv"
    if identity_val_csv.exists():
        val_pred = predict_view(
            args.chemprop_bin,
            model_path,
            identity_val_csv,
            out_dir / "val_predictions.csv",
            args.accelerator,
            args.devices,
            args.batch_size,
            args.num_workers,
            log_path,
            mode_uses_canonical_smiles(args.mode),
        )
        val_metrics = task_metrics(task_type, val_pred["target"].to_numpy(), val_pred["prediction"].to_numpy())
        val_false_inv, val_n_pairs = paired_false_invariance(val_pred)
        val_pred.to_parquet(out_dir / "val_predictions.parquet", index=False)

    view_predictions = {}
    pred_dir = out_dir / "view_predictions"
    pred_dir.mkdir(exist_ok=True)
    for transform in POSITIVE_TRANSFORMS:
        view_csv = data_dir / f"view_{transform}_test.csv"
        if not view_csv.exists():
            continue
        pred = predict_view(
            args.chemprop_bin,
            model_path,
            view_csv,
            pred_dir / f"{transform}.csv",
            args.accelerator,
            args.devices,
            args.batch_size,
            args.num_workers,
            log_path,
            mode_uses_canonical_smiles(args.mode),
        )
        pred.to_parquet(pred_dir / f"{transform}.parquet", index=False)
        view_predictions[transform] = pred

    test_pred = view_predictions.get("identity")
    if test_pred is None:
        test_pred = predict_view(
            args.chemprop_bin,
            model_path,
            identity_test_csv,
            pred_dir / "identity.csv",
            args.accelerator,
            args.devices,
            args.batch_size,
            args.num_workers,
            log_path,
            mode_uses_canonical_smiles(args.mode),
        )
    test_metrics = task_metrics(task_type, test_pred["target"].to_numpy(), test_pred["prediction"].to_numpy())
    false_inv, n_pairs = paired_false_invariance(test_pred)
    metrics_record = {
        "dataset": args.dataset,
        "mode": args.mode,
        "feature_kind": "chemprop_v231",
        "run_name": args.run_name,
        "processed_scope": args.processed_scope,
        "seed": args.seed,
        "task_type": task_type,
        "n_train": int((identity_samples["split"] == "train").sum()),
        "n_val": int((identity_samples["split"] == "val").sum()),
        "n_test": int((identity_samples["split"] == "test").sum()),
        "n_train_rows": int(train_input["split"].eq("train").sum()),
        "train_aug_views": aug_views,
        "canonicalize_smiles": mode_uses_canonical_smiles(args.mode),
        "epochs_run": int(args.epochs),
        "seconds": float(time.time() - start),
        "device": args.accelerator,
        "cuda_device_name": "",
        "pair_margin_lambda": 0.0,
        "pair_margin": 0.0,
        "pair_margin_n_pairs": 0,
        "selection_false_inv_weight": 0.0,
        "selection_best_score": float("nan"),
        "train": {},
        "val": val_metrics,
        "val_false_invariance_rate_pairs": val_false_inv,
        "val_false_invariance_n_pairs": int(val_n_pairs),
        "test": test_metrics,
        "false_invariance_rate_test_pairs": false_inv,
        "false_invariance_n_pairs": int(n_pairs),
        "positive_transform_eval": positive_transform_eval(task_type, view_predictions),
        "chemprop_model_path": str(model_path.relative_to(root)),
    }
    metrics_path.write_text(json.dumps(metrics_record, indent=2), encoding="utf-8")
    test_pred.to_parquet(out_dir / "test_predictions.parquet", index=False)
    print(json.dumps(metrics_record, indent=2))


if __name__ == "__main__":
    main()
