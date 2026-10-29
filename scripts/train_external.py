from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from condchemsymm.metrics import classification_metrics
from train_graph import (
    GraphPilotModel,
    collate_batch,
    evaluate as evaluate_graph,
    false_invariance,
    graph_pair_margin_loss,
    make_loader,
    set_seed,
)
from train_fingerprint import (
    MLP,
    build_features,
    evaluate as evaluate_mlp,
    paired_false_invariance,
)


EXTERNAL_DATASETS = ["sphysnet_100_aqueous", "nci_aqueous_zero_overlap"]


def selection_score(roc_auc: float, false_invariance: float, penalty: float) -> float:
    return float(roc_auc - penalty * false_invariance)


def pair_metrics(frame: pd.DataFrame, prediction: np.ndarray, threshold: float = 0.25) -> dict[str, float]:
    pred = dict(zip(frame["sample_id"], prediction))
    labels = dict(zip(frame["sample_id"], frame["y"]))
    seen: set[str] = set()
    false_invariant, order_correct, gaps = [], [], []
    for row in frame.itertuples(index=False):
        sid, pid = str(row.sample_id), str(row.paired_id)
        if not pid or sid in seen or pid in seen or pid not in pred:
            continue
        seen.update((sid, pid))
        pa, pb = float(pred[sid]), float(pred[pid])
        ya, yb = float(labels[sid]), float(labels[pid])
        gap = abs(pa - pb)
        gaps.append(gap)
        false_invariant.append(gap < threshold)
        order_correct.append((pa > pb) if ya > yb else (pb > pa))
    return {
        "false_invariance": float(np.mean(false_invariant)),
        "pair_order_accuracy": float(np.mean(order_correct)),
        "pair_gap_mean": float(np.mean(gaps)),
        "n_pairs": len(gaps),
    }


def save_predictions(frame: pd.DataFrame, prediction: np.ndarray, path: Path) -> None:
    columns = [
        column
        for column in ["sample_id", "dataset", "split", "smiles", "y", "paired_id", "family_key"]
        if column in frame.columns
    ]
    out = frame[columns].copy()
    out["prediction"] = np.asarray(prediction, dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def train_graph(root: Path, seed: int, device: torch.device) -> tuple[GraphPilotModel, dict]:
    samples = pd.read_parquet(root / "data/processed/main/tautobase_pref_samples.parquet")
    edges = pd.read_parquet(root / "data/processed/main/transform_edges.parquet")
    edges = edges[edges["dataset"].eq("tautobase_pref")].copy()
    train = samples[samples["split"].eq("train")].copy()
    val = samples[samples["split"].eq("val")].copy()
    train_loader = make_loader(train, edges[edges["sample_id"].isin(train["sample_id"])], "mechanistic_gate", 256, device, shuffle=True)
    pair_loader = make_loader(train, edges[edges["sample_id"].isin(train["sample_id"])], "mechanistic_gate", 256, device, shuffle=False)
    val_loader = make_loader(val, edges[edges["sample_id"].isin(val["sample_id"])], "mechanistic_gate", 256, device)

    model = GraphPilotModel(hidden=256, depth=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    best_score, best_state, bad = -np.inf, None, 0
    start = time.time()
    epochs_run = 0
    for epoch in range(80):
        epochs_run = epoch + 1
        model.train()
        for items in train_loader:
            graph_batch, pos_map, neg_map, y, _ = collate_batch(items, device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(graph_batch, pos_map, neg_map), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        margin_loss, _ = graph_pair_margin_loss(model, pair_loader, device, margin=1.0)
        if margin_loss is not None:
            (0.05 * margin_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        val_metrics, val_prediction, val_ids, val_paired = evaluate_graph(model, val_loader, "classification")
        val_fi, _ = false_invariance(val_ids, val_paired, val_prediction)
        score = selection_score(float(val_metrics["roc_auc"]), float(val_fi), 0.2)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= 20:
            break
    if best_state is None:
        raise RuntimeError("graph model failed to produce a validation checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "epochs_run": epochs_run,
        "selection_best_score": best_score,
        "seconds": time.time() - start,
        "config": {
            "hidden": 256,
            "depth": 4,
            "batch_size": 256,
            "lr": 3e-4,
            "pair_margin_lambda": 0.05,
            "pair_margin": 1.0,
            "selection_false_inv_weight": 0.2,
            "mode": "mechanistic_gate",
        },
    }


def train_mlp(root: Path, seed: int, device: torch.device) -> tuple[MLP, dict]:
    samples = pd.read_parquet(root / "data/processed/main/tautobase_pref_samples.parquet")
    edges = pd.read_parquet(root / "data/processed/main/transform_edges.parquet")
    edges = edges[edges["dataset"].eq("tautobase_pref")].copy()
    features = build_features(samples, edges, "mechanistic_gate", "concat")
    labels = samples["y"].to_numpy(dtype=np.float32)
    train_idx = np.flatnonzero(samples["split"].to_numpy() == "train")
    val_idx = np.flatnonzero(samples["split"].to_numpy() == "val")
    x_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(labels, dtype=torch.float32, device=device)
    model = MLP(features.shape[1], hidden=512).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    best_score, best_state, bad = -np.inf, None, 0
    start = time.time()
    epochs_run = 0
    for epoch in range(80):
        epochs_run = epoch + 1
        model.train()
        order = np.random.permutation(train_idx)
        for offset in range(0, len(order), 256):
            idx = torch.as_tensor(order[offset : offset + 256], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_tensor[idx]), y_tensor[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        val_metrics, val_prediction = evaluate_mlp(model, features[val_idx], labels[val_idx], "classification")
        val_fi, _ = paired_false_invariance(samples.iloc[val_idx].copy(), val_prediction)
        score = selection_score(float(val_metrics["roc_auc"]), float(val_fi), 0.1)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= 20:
            break
    if best_state is None:
        raise RuntimeError("MLP model failed to produce a validation checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "epochs_run": epochs_run,
        "selection_best_score": best_score,
        "seconds": time.time() - start,
        "config": {
            "hidden": 512,
            "batch_size": 256,
            "lr": 3e-4,
            "pair_margin_lambda": 0.0,
            "selection_false_inv_weight": 0.1,
            "mode": "mechanistic_gate",
            "feature_kind": "concat",
        },
    }


def evaluate_external(root: Path, seed: int, graph: GraphPilotModel, mlp: MLP, device: torch.device) -> list[dict]:
    rows = []
    output_root = root / "runs/evaluation_external_internal_models"
    for dataset in EXTERNAL_DATASETS:
        samples = pd.read_parquet(root / f"data/processed/evaluation_required/{dataset}_samples.parquet")
        edges = pd.read_parquet(root / f"data/processed/evaluation_required/{dataset}_edges.parquet")

        graph_loader = make_loader(samples, edges, "mechanistic_gate", 256, device, shuffle=False)
        graph_metrics, graph_prediction, graph_ids, _ = evaluate_graph(graph, graph_loader, "classification")
        graph_map = dict(zip(graph_ids, graph_prediction))
        graph_prediction = np.asarray([graph_map[sample_id] for sample_id in samples["sample_id"]], dtype=float)
        graph_pair = pair_metrics(samples, graph_prediction)
        graph_path = output_root / "graph_main" / dataset / f"seed_{seed}" / "test_predictions.parquet"
        save_predictions(samples, graph_prediction, graph_path)

        mlp_features = build_features(samples, edges, "mechanistic_gate", "concat")
        mlp_metrics, mlp_prediction = evaluate_mlp(
            mlp, mlp_features, samples["y"].to_numpy(dtype=np.float32), "classification"
        )
        mlp_pair = pair_metrics(samples, mlp_prediction)
        mlp_path = output_root / "mlp_select01" / dataset / f"seed_{seed}" / "test_predictions.parquet"
        save_predictions(samples, mlp_prediction, mlp_path)

        for component, metrics, paired in [
            ("graph_main", graph_metrics, graph_pair),
            ("mlp_select01", mlp_metrics, mlp_pair),
        ]:
            rows.append(
                {
                    "component": component,
                    "dataset": dataset,
                    "seed": seed,
                    **{key: float(value) for key, value in metrics.items()},
                    **paired,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph, graph_meta = train_graph(root, args.seed, device)
    mlp, mlp_meta = train_mlp(root, args.seed, device)
    rows = evaluate_external(root, args.seed, graph, mlp, device)

    out_dir = root / "runs/evaluation_external_internal_models/checkpoints" / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": graph.state_dict(), "metadata": graph_meta}, out_dir / "graph_main.pt")
    torch.save({"state_dict": mlp.state_dict(), "metadata": mlp_meta}, out_dir / "mlp_select01.pt")
    payload = {
        "seed": args.seed,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "graph": graph_meta,
        "mlp": mlp_meta,
        "external": rows,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
