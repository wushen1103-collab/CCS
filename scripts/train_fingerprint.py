from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from condchemsymm.chem import atom_renumber_smiles, morgan_fp, random_smiles, smiles_hash_fp
from condchemsymm.gating import mechanistic_accepts_edge
from condchemsymm.metrics import classification_metrics, regression_metrics


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_features(samples, edges, mode: str, feature_kind: str, n_bits: int = 2048, override_transform: str | None = None):
    edge_groups = {sid: g for sid, g in edges.groupby("sample_id")}
    fp_cache = {}

    def fp(smi):
        key = (feature_kind, smi)
        if key not in fp_cache:
            if feature_kind == "ecfp":
                fp_cache[key] = morgan_fp(smi, n_bits=n_bits)
            elif feature_kind == "smiles_hash":
                fp_cache[key] = smiles_hash_fp(smi, n_bits=n_bits)
            elif feature_kind == "concat":
                fp_cache[key] = np.concatenate([morgan_fp(smi, n_bits=n_bits), smiles_hash_fp(smi, n_bits=n_bits)]).astype(np.float32)
            else:
                raise ValueError(f"Unknown feature_kind: {feature_kind}")
        return fp_cache[key]

    def unique_mean(vecs):
        unique = {}
        for vec in vecs:
            unique[vec.tobytes()] = vec
        if not unique:
            return np.zeros_like(next(iter(fp_cache.values()))) if fp_cache else np.zeros(n_bits, dtype=np.float32)
        return np.stack(list(unique.values())).mean(axis=0)

    def representation_closure(smiles_list):
        out = []
        for smi in smiles_list:
            if not isinstance(smi, str) or not smi:
                continue
            out.append(smi)
            for fn in (atom_renumber_smiles, random_smiles):
                view = fn(smi, seed=17)
                if view:
                    out.append(view)
        return out

    def chemical_state_smiles(group, z_value=None):
        states = group[group["transform_type"].isin(["identity", "mirror_chirality", "tautomer_shift"])].copy()
        if z_value is not None:
            states = states[states["z"] == z_value]
        return list(states["view_smiles"])

    xs = []
    for _, row in samples.iterrows():
        sid = row["sample_id"]
        group = edge_groups[sid]
        identity = group[group["transform_type"] == "identity"].iloc[0]
        selected = identity
        if override_transform is not None:
            cand = group[group["transform_type"] == override_transform]
            if len(cand) > 0:
                selected = cand.iloc[0]
        f0 = fp(selected["view_smiles"])
        if mode == "no_aug":
            inv = f0
            residual = np.zeros_like(f0)
        elif mode == "always_invariant":
            # First choose chemical states, then close each state under
            # representation-preserving views. Do not treat atom/random edges
            # themselves as new chemical states; doing so creates orientation
            # leakage in SMILES-hash pilots.
            inv = unique_mean([fp(s) for s in representation_closure(chemical_state_smiles(group))])
            residual = np.zeros_like(f0)
        elif mode == "always_sensitive":
            diffs = [np.abs(f0 - fp(s)) for s in group[group["transform_type"] != "identity"]["view_smiles"]]
            inv = f0
            residual = unique_mean(diffs) if diffs else np.zeros_like(f0)
        elif mode == "oracle_gate":
            pos_state_smiles = chemical_state_smiles(group, z_value=1)
            neg = group[group["z"] == 0]
            inv = unique_mean([fp(s) for s in representation_closure(pos_state_smiles)])
            diffs = [np.abs(f0 - fp(s)) for s in neg["view_smiles"]]
            residual = unique_mean(diffs) if diffs else np.zeros_like(f0)
        elif mode == "mechanistic_gate":
            states = group[group["transform_type"].isin(["identity", "mirror_chirality", "tautomer_shift"])].copy()
            accept = states.apply(mechanistic_accepts_edge, axis=1)
            pos_state_smiles = list(states[accept]["view_smiles"])
            neg = states[~accept]
            inv = unique_mean([fp(s) for s in representation_closure(pos_state_smiles)])
            diffs = [np.abs(f0 - fp(s)) for s in neg["view_smiles"]]
            residual = unique_mean(diffs) if diffs else np.zeros_like(f0)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        xs.append(np.concatenate([inv, residual]).astype(np.float32))
    return np.stack(xs)


def primary_metric(task_type, y, pred):
    if task_type == "regression":
        return -regression_metrics(y, pred)["mae"]
    return classification_metrics(y, pred)["roc_auc"]


def selection_score(primary: float, false_inv: float, weight: float) -> float:
    if weight <= 0 or np.isnan(false_inv):
        return float(primary)
    return float(primary - weight * false_inv)


def evaluate(model, x, y, task_type, y_mean=0.0, y_std=1.0):
    model.eval()
    with torch.no_grad():
        raw = model(torch.as_tensor(x, dtype=torch.float32, device=next(model.parameters()).device)).detach().cpu().numpy()
    if task_type == "regression":
        pred = raw * y_std + y_mean
        return regression_metrics(y, pred), pred
    prob = 1.0 / (1.0 + np.exp(-raw))
    return classification_metrics(y, prob), prob


def paired_false_invariance(samples, predictions, threshold=0.25):
    pred_by_id = dict(zip(samples["sample_id"], predictions))
    vals = []
    seen = set()
    for _, row in samples.iterrows():
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


def paired_margin_indices(samples: pd.DataFrame, allowed_indices):
    allowed = set(int(i) for i in allowed_indices)
    id_to_idx = {samples.iloc[i]["sample_id"]: int(i) for i in allowed}
    pairs = []
    seen = set()
    for idx in allowed:
        row = samples.iloc[idx]
        sid = row["sample_id"]
        pid = row.get("paired_id", "")
        if not pid or pid not in id_to_idx or sid in seen or pid in seen:
            continue
        j = id_to_idx[pid]
        y_i = float(row["y"])
        y_j = float(samples.iloc[j]["y"])
        if y_i == y_j:
            continue
        pos = idx if y_i > y_j else j
        neg = j if y_i > y_j else idx
        pairs.append((pos, neg))
        seen.add(sid)
        seen.add(pid)
    return pairs


def pair_margin_step(model, opt, x_t, pairs, margin: float, weight: float):
    if weight <= 0 or not pairs:
        return 0.0
    device = next(model.parameters()).device
    pos_idx = torch.as_tensor([p for p, _ in pairs], dtype=torch.long, device=device)
    neg_idx = torch.as_tensor([n for _, n in pairs], dtype=torch.long, device=device)
    opt.zero_grad(set_to_none=True)
    raw = model(x_t)
    loss = torch.relu(float(margin) - (raw[pos_idx] - raw[neg_idx])).mean()
    (float(weight) * loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    return float(loss.detach().cpu())


def write_split_predictions(samples_subset: pd.DataFrame, predictions, path: Path):
    cols = [c for c in ["sample_id", "dataset", "split", "smiles", "y", "paired_id"] if c in samples_subset.columns]
    pred_df = samples_subset[cols].copy()
    pred_df["prediction"] = predictions
    pred_df.to_parquet(path, index=False)


def positive_transform_metrics(model, samples, edges, mode, feature_kind, task_type, y_mean, y_std, save_dir: Path | None = None):
    test_samples = samples[samples["split"] == "test"].copy()
    y = test_samples["y"].to_numpy(dtype=np.float32)
    out = {}
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    for transform_type in sorted(edges[edges["z"] == 1]["transform_type"].unique()):
        available = set(edges[(edges["z"] == 1) & (edges["transform_type"] == transform_type)]["sample_id"])
        subset = test_samples[test_samples["sample_id"].isin(available)].copy()
        if len(subset) == 0:
            continue
        x_view = build_features(subset, edges[edges["sample_id"].isin(subset["sample_id"])], mode, feature_kind, override_transform=transform_type)
        metrics, pred = evaluate(model, x_view, subset["y"].to_numpy(dtype=np.float32), task_type, y_mean, y_std)
        out[transform_type] = metrics
        if save_dir is not None:
            write_split_predictions(subset, pred, save_dir / f"{transform_type}.parquet")
    if task_type == "regression":
        maes = [m["mae"] for m in out.values() if "mae" in m]
        out["worst_positive_transform_mae"] = float(max(maes)) if maes else float("nan")
    else:
        aucs = [m["roc_auc"] for m in out.values() if "roc_auc" in m and not np.isnan(m["roc_auc"])]
        briers = [m["brier"] for m in out.values() if "brier" in m]
        out["worst_positive_transform_roc_auc"] = float(min(aucs)) if aucs else float("nan")
        out["worst_positive_transform_brier"] = float(max(briers)) if briers else float("nan")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", required=True, choices=["no_aug", "always_invariant", "always_sensitive", "oracle_gate", "mechanistic_gate"])
    parser.add_argument("--feature-kind", default="ecfp", choices=["ecfp", "smiles_hash", "concat"])
    parser.add_argument("--run-name", default="base")
    parser.add_argument("--processed-scope", default="base")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pair-margin-lambda", type=float, default=0.0)
    parser.add_argument("--pair-margin", type=float, default=1.0)
    parser.add_argument("--selection-false-inv-weight", type=float, default=0.0)
    parser.add_argument("--save-positive-view-predictions", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    root = Path(__file__).resolve().parents[1]
    samples = pd.read_parquet(root / f"data/processed/{args.processed_scope}/{args.dataset}_samples.parquet")
    edges = pd.read_parquet(root / f"data/processed/{args.processed_scope}/transform_edges.parquet")
    edges = edges[edges["dataset"] == args.dataset].copy()
    task_type = str(samples["task_type"].iloc[0])

    x = build_features(samples, edges, args.mode, args.feature_kind)
    y = samples["y"].to_numpy(dtype=np.float32)
    train_idx = np.where(samples["split"].to_numpy() == "train")[0]
    val_idx = np.where(samples["split"].to_numpy() == "val")[0]
    test_idx = np.where(samples["split"].to_numpy() == "test")[0]

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise SystemExit(f"Bad split sizes for {args.dataset}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    y_train = y[train_idx]
    y_mean = float(y_train.mean()) if task_type == "regression" else 0.0
    y_std = float(y_train.std() + 1e-8) if task_type == "regression" else 1.0
    y_fit = (y - y_mean) / y_std if task_type == "regression" else y

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(x.shape[1], hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    if task_type == "regression":
        loss_fn = nn.HuberLoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_fit, dtype=torch.float32, device=device)
    margin_pairs = paired_margin_indices(samples.reset_index(drop=True), train_idx) if task_type == "classification" else []
    best = -1e18
    best_state = None
    bad = 0
    history = []
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(train_idx)
        losses = []
        for off in range(0, len(order), args.batch_size):
            idx = torch.as_tensor(order[off : off + args.batch_size], dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            raw = model(x_t[idx])
            loss = loss_fn(raw, y_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        pair_loss = pair_margin_step(model, opt, x_t, margin_pairs, args.pair_margin, args.pair_margin_lambda)

        val_metrics, val_pred = evaluate(model, x[val_idx], y[val_idx], task_type, y_mean, y_std)
        primary = primary_metric(task_type, y[val_idx], val_pred)
        val_false_inv, val_false_inv_n = paired_false_invariance(samples.iloc[val_idx].copy(), val_pred) if task_type == "classification" else (float("nan"), 0)
        score = selection_score(primary, val_false_inv, args.selection_false_inv_weight)
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "pair_margin_loss": pair_loss,
            "val_score": float(primary),
            "selection_score": float(score),
            "val_false_invariance_rate_pairs": val_false_inv,
            "val_false_invariance_n_pairs": int(val_false_inv_n),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })
        if score > best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics, train_pred = evaluate(model, x[train_idx], y[train_idx], task_type, y_mean, y_std)
    val_metrics, val_pred = evaluate(model, x[val_idx], y[val_idx], task_type, y_mean, y_std)
    test_metrics, test_pred = evaluate(model, x[test_idx], y[test_idx], task_type, y_mean, y_std)
    val_false_inv, val_false_inv_n = paired_false_invariance(samples.iloc[val_idx].copy(), val_pred) if task_type == "classification" else (float("nan"), 0)
    false_inv, n_pairs = paired_false_invariance(samples.iloc[test_idx].copy(), test_pred)
    out_dir = root / "runs" / args.run_name / args.feature_kind / args.dataset / args.mode / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    transform_save_dir = out_dir / "view_predictions" if args.save_positive_view_predictions else None
    transform_metrics = positive_transform_metrics(
        model,
        samples,
        edges,
        args.mode,
        args.feature_kind,
        task_type,
        y_mean,
        y_std,
        transform_save_dir,
    )
    metrics = {
        "dataset": args.dataset,
        "mode": args.mode,
        "feature_kind": args.feature_kind,
        "run_name": args.run_name,
        "processed_scope": args.processed_scope,
        "seed": args.seed,
        "task_type": task_type,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "epochs_run": int(len(history)),
        "seconds": float(time.time() - start),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "pair_margin_lambda": float(args.pair_margin_lambda),
        "pair_margin": float(args.pair_margin),
        "pair_margin_n_pairs": int(len(margin_pairs)),
        "selection_false_inv_weight": float(args.selection_false_inv_weight),
        "selection_best_score": float(best),
        "train": train_metrics,
        "val": val_metrics,
        "val_false_invariance_rate_pairs": val_false_inv,
        "val_false_invariance_n_pairs": int(val_false_inv_n),
        "test": test_metrics,
        "false_invariance_rate_test_pairs": false_inv,
        "false_invariance_n_pairs": int(n_pairs),
        "positive_transform_eval": transform_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    write_split_predictions(samples.iloc[val_idx].copy(), val_pred, out_dir / "val_predictions.parquet")
    write_split_predictions(samples.iloc[test_idx].copy(), test_pred, out_dir / "test_predictions.parquet")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
