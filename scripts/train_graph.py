from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from condchemsymm.graph import ATOM_FDIM, BOND_FDIM, collate_mol_graphs, mol_graph_from_smiles
from condchemsymm.gating import mechanistic_accepts_edge
from condchemsymm.metrics import classification_metrics, regression_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unique_list(values):
    out = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def graph_state_views(group: pd.DataFrame, mode: str, override_transform: str | None = None):
    identity = group[group["transform_type"] == "identity"].iloc[0]
    identity_smiles = str(identity["view_smiles"])
    if override_transform is not None:
        cand = group[group["transform_type"] == override_transform]
        if len(cand) > 0:
            identity_smiles = str(cand.iloc[0]["view_smiles"])

    chemical = group[group["transform_type"].isin(["identity", "mirror_chirality", "tautomer_shift"])].copy()
    if mode == "no_aug":
        return [identity_smiles], []
    if mode == "always_invariant":
        return unique_list(list(chemical["view_smiles"])), []
    if mode == "always_sensitive":
        neg = [s for s in chemical["view_smiles"] if s != identity["view_smiles"]]
        return [identity_smiles], unique_list(neg)
    if mode == "oracle_gate":
        pos = unique_list(list(chemical[chemical["z"] == 1]["view_smiles"]))
        neg = unique_list(list(chemical[chemical["z"] == 0]["view_smiles"]))
        if override_transform is not None and identity_smiles not in pos and not neg:
            pos = [identity_smiles]
        return pos or [identity_smiles], neg
    if mode == "mechanistic_gate":
        accept = chemical.apply(mechanistic_accepts_edge, axis=1)
        pos = unique_list(list(chemical[accept]["view_smiles"]))
        neg = unique_list(list(chemical[~accept]["view_smiles"]))
        if override_transform is not None and identity_smiles not in pos and not neg:
            pos = [identity_smiles]
        return pos or [identity_smiles], neg
    raise ValueError(mode)


class GraphPilotDataset(Dataset):
    def __init__(self, samples: pd.DataFrame, edges: pd.DataFrame, mode: str, override_transform: str | None = None):
        self.samples = samples.reset_index(drop=True)
        self.edge_groups = {sid: g for sid, g in edges.groupby("sample_id")}
        self.mode = mode
        self.override_transform = override_transform
        self.graph_cache = {}

    def __len__(self):
        return len(self.samples)

    def graph(self, smiles: str):
        if smiles not in self.graph_cache:
            self.graph_cache[smiles] = mol_graph_from_smiles(smiles)
        return self.graph_cache[smiles]

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        group = self.edge_groups[row["sample_id"]]
        pos, neg = graph_state_views(group, self.mode, self.override_transform)
        return {
            "sample_id": row["sample_id"],
            "y": float(row["y"]),
            "pos": [self.graph(s) for s in pos],
            "neg": [self.graph(s) for s in neg],
            "paired_id": row.get("paired_id", ""),
        }


def collate_batch(items, device):
    graphs = []
    pos_map = []
    neg_map = []
    for item_idx, item in enumerate(items):
        pidx = []
        for g in item["pos"]:
            pidx.append(len(graphs))
            graphs.append(g)
        nidx = []
        for g in item["neg"]:
            nidx.append(len(graphs))
            graphs.append(g)
        pos_map.append(pidx)
        neg_map.append(nidx)
    batch_graph = collate_mol_graphs(graphs, device)
    y = torch.as_tensor([item["y"] for item in items], dtype=torch.float32, device=device)
    return batch_graph, pos_map, neg_map, y, items


class DMPNNEncoder(nn.Module):
    def __init__(self, hidden: int = 256, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.depth = depth
        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden)
        self.W_h = nn.Linear(hidden, hidden)
        self.W_o = nn.Sequential(
            nn.Linear(ATOM_FDIM + hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, graph_batch):
        atom_x = graph_batch["atom_x"]
        bond_x = graph_batch["bond_x"]
        src = graph_batch["src"]
        dst = graph_batch["dst"]
        rev = graph_batch["rev"]
        atom_scope = graph_batch["atom_scope"]
        n_atoms = atom_x.shape[0]
        if bond_x.shape[0] == 0:
            atom_msg = torch.zeros((n_atoms, self.hidden), device=atom_x.device)
        else:
            msg = torch.relu(self.W_i(torch.cat([atom_x[src], bond_x], dim=1)))
            for _ in range(self.depth - 1):
                incoming = torch.zeros((n_atoms, self.hidden), device=atom_x.device)
                incoming.index_add_(0, dst, msg)
                rev_msg = msg[rev]
                msg = torch.relu(self.W_h(incoming[src] - rev_msg) + msg)
            atom_msg = torch.zeros((n_atoms, self.hidden), device=atom_x.device)
            atom_msg.index_add_(0, dst, msg)
        atom_h = self.W_o(torch.cat([atom_x, atom_msg], dim=1))
        mol_h = []
        for start, length in atom_scope:
            mol_h.append(atom_h[start : start + length].mean(dim=0))
        return torch.stack(mol_h, dim=0)


class GraphPilotModel(nn.Module):
    def __init__(self, hidden: int = 256, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = DMPNNEncoder(hidden=hidden, depth=depth, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, graph_batch, pos_map, neg_map):
        graph_h = self.encoder(graph_batch)
        reps = []
        for pos_idxs, neg_idxs in zip(pos_map, neg_map):
            pos_h = graph_h[pos_idxs].mean(dim=0)
            if neg_idxs:
                neg_h = torch.stack([torch.abs(pos_h - graph_h[i]) for i in neg_idxs], dim=0).mean(dim=0)
            else:
                neg_h = torch.zeros_like(pos_h)
            reps.append(torch.cat([pos_h, neg_h], dim=0))
        return self.head(torch.stack(reps, dim=0)).squeeze(-1)


def primary_score(task_type, y_true, pred):
    if task_type == "regression":
        return -regression_metrics(y_true, pred)["mae"]
    return classification_metrics(y_true, pred)["roc_auc"]


def selection_score(primary: float, false_inv: float, weight: float) -> float:
    if weight <= 0 or np.isnan(false_inv):
        return float(primary)
    return float(primary - weight * false_inv)


def evaluate(model, loader, task_type, y_mean=0.0, y_std=1.0):
    device = next(model.parameters()).device
    ys = []
    outs = []
    sample_ids = []
    paired_ids = []
    model.eval()
    with torch.no_grad():
        for items in loader:
            graph_batch, pos_map, neg_map, y, raw_items = collate_batch(items, device)
            raw = model(graph_batch, pos_map, neg_map).detach().cpu().numpy()
            ys.extend(y.detach().cpu().numpy().tolist())
            outs.extend(raw.tolist())
            sample_ids.extend([item["sample_id"] for item in raw_items])
            paired_ids.extend([item.get("paired_id", "") for item in raw_items])
    y = np.asarray(ys, dtype=float)
    raw = np.asarray(outs, dtype=float)
    if task_type == "regression":
        pred = raw * y_std + y_mean
        metrics = regression_metrics(y, pred)
    else:
        pred = 1.0 / (1.0 + np.exp(-raw))
        metrics = classification_metrics(y, pred)
    return metrics, pred, sample_ids, paired_ids


def false_invariance(sample_ids, paired_ids, predictions, threshold=0.25):
    pred_by_id = dict(zip(sample_ids, predictions))
    vals = []
    seen = set()
    for sid, pid in zip(sample_ids, paired_ids):
        if not pid or pid not in pred_by_id or sid in seen or pid in seen:
            continue
        seen.add(sid)
        seen.add(pid)
        vals.append(abs(float(pred_by_id[sid]) - float(pred_by_id[pid])) < threshold)
    return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)


def write_split_predictions(samples_subset: pd.DataFrame, predictions, path: Path):
    cols = [c for c in ["sample_id", "dataset", "split", "smiles", "y", "paired_id"] if c in samples_subset.columns]
    pred_df = samples_subset[cols].copy()
    pred_df["prediction"] = predictions
    pred_df.to_parquet(path, index=False)


def graph_pair_margin_loss(model, loader, device, margin: float):
    raws = []
    ys = []
    sample_ids = []
    paired_ids = []
    for items in loader:
        graph_batch, pos_map, neg_map, y, raw_items = collate_batch(items, device)
        raws.append(model(graph_batch, pos_map, neg_map))
        ys.extend(y.detach().cpu().numpy().tolist())
        sample_ids.extend([item["sample_id"] for item in raw_items])
        paired_ids.extend([item.get("paired_id", "") for item in raw_items])
    if not raws:
        return None, 0
    raw = torch.cat(raws, dim=0)
    id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
    losses = []
    seen = set()
    for i, (sid, pid) in enumerate(zip(sample_ids, paired_ids)):
        if not pid or pid not in id_to_idx or sid in seen or pid in seen:
            continue
        j = id_to_idx[pid]
        if ys[i] == ys[j]:
            continue
        pos = i if ys[i] > ys[j] else j
        neg = j if ys[i] > ys[j] else i
        losses.append(torch.relu(float(margin) - (raw[pos] - raw[neg])))
        seen.add(sid)
        seen.add(pid)
    if not losses:
        return None, 0
    return torch.stack(losses).mean(), len(losses)


def make_loader(samples, edges, mode, batch_size, device, shuffle=False, override_transform=None):
    ds = GraphPilotDataset(samples, edges, mode, override_transform=override_transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: x,
        num_workers=0,
    )


def positive_transform_eval(model, samples, edges, mode, task_type, batch_size, device, y_mean, y_std):
    out = {}
    test_samples = samples[samples["split"] == "test"].copy()
    for transform_type in sorted(edges[edges["z"] == 1]["transform_type"].unique()):
        available = set(edges[(edges["z"] == 1) & (edges["transform_type"] == transform_type)]["sample_id"])
        subset = test_samples[test_samples["sample_id"].isin(available)].copy()
        if len(subset) == 0:
            continue
        loader = make_loader(subset, edges[edges["sample_id"].isin(subset["sample_id"])], mode, batch_size, device, override_transform=transform_type)
        metrics, _, _, _ = evaluate(model, loader, task_type, y_mean, y_std)
        out[transform_type] = metrics
    if task_type == "regression":
        vals = [m["mae"] for m in out.values() if "mae" in m]
        out["worst_positive_transform_mae"] = float(max(vals)) if vals else float("nan")
    else:
        vals = [m["roc_auc"] for m in out.values() if "roc_auc" in m and not np.isnan(m["roc_auc"])]
        out["worst_positive_transform_roc_auc"] = float(min(vals)) if vals else float("nan")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", required=True, choices=["no_aug", "always_invariant", "always_sensitive", "oracle_gate", "mechanistic_gate"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-name", default="base_graph")
    parser.add_argument("--processed-scope", default="base")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pair-margin-lambda", type=float, default=0.0)
    parser.add_argument("--pair-margin", type=float, default=1.0)
    parser.add_argument("--selection-false-inv-weight", type=float, default=0.0)
    args = parser.parse_args()

    set_seed(args.seed)
    root = Path(__file__).resolve().parents[1]
    samples = pd.read_parquet(root / f"data/processed/{args.processed_scope}/{args.dataset}_samples.parquet")
    edges = pd.read_parquet(root / f"data/processed/{args.processed_scope}/transform_edges.parquet")
    edges = edges[edges["dataset"] == args.dataset].copy()
    task_type = str(samples["task_type"].iloc[0])
    train = samples[samples["split"] == "train"].copy()
    val = samples[samples["split"] == "val"].copy()
    test = samples[samples["split"] == "test"].copy()

    y_train = train["y"].to_numpy(dtype=float)
    y_mean = float(y_train.mean()) if task_type == "regression" else 0.0
    y_std = float(y_train.std() + 1e-8) if task_type == "regression" else 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphPilotModel(hidden=args.hidden, depth=args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.HuberLoss() if task_type == "regression" else nn.BCEWithLogitsLoss()

    train_loader = make_loader(train, edges[edges["sample_id"].isin(train["sample_id"])], args.mode, args.batch_size, device, shuffle=True)
    pair_train_loader = make_loader(train, edges[edges["sample_id"].isin(train["sample_id"])], args.mode, args.batch_size, device, shuffle=False)
    val_loader = make_loader(val, edges[edges["sample_id"].isin(val["sample_id"])], args.mode, args.batch_size, device)
    test_loader = make_loader(test, edges[edges["sample_id"].isin(test["sample_id"])], args.mode, args.batch_size, device)

    best = -1e18
    best_state = None
    bad = 0
    history = []
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for items in train_loader:
            graph_batch, pos_map, neg_map, y, _ = collate_batch(items, device)
            target = (y - y_mean) / y_std if task_type == "regression" else y
            opt.zero_grad(set_to_none=True)
            raw = model(graph_batch, pos_map, neg_map)
            loss = loss_fn(raw, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        pair_loss_value = 0.0
        pair_loss_n = 0
        if task_type == "classification" and args.pair_margin_lambda > 0:
            model.train()
            opt.zero_grad(set_to_none=True)
            pair_loss, pair_loss_n = graph_pair_margin_loss(model, pair_train_loader, device, args.pair_margin)
            if pair_loss is not None:
                (args.pair_margin_lambda * pair_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                pair_loss_value = float(pair_loss.detach().cpu())
        val_metrics, val_pred, val_ids, val_paired_ids = evaluate(model, val_loader, task_type, y_mean, y_std)
        primary = primary_score(task_type, val["y"].to_numpy(dtype=float), val_pred)
        val_false_inv, val_false_inv_n = false_invariance(val_ids, val_paired_ids, val_pred) if task_type == "classification" else (float("nan"), 0)
        score = selection_score(primary, val_false_inv, args.selection_false_inv_weight)
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "pair_margin_loss": pair_loss_value,
            "pair_margin_n_pairs": pair_loss_n,
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

    train_metrics, _, _, _ = evaluate(model, train_loader, task_type, y_mean, y_std)
    val_metrics, val_pred, val_ids, val_paired_ids = evaluate(model, val_loader, task_type, y_mean, y_std)
    test_metrics, test_pred, test_ids, paired_ids = evaluate(model, test_loader, task_type, y_mean, y_std)
    val_false_inv, val_false_inv_n = false_invariance(val_ids, val_paired_ids, val_pred) if task_type == "classification" else (float("nan"), 0)
    false_inv, n_pairs = false_invariance(test_ids, paired_ids, test_pred)
    transform_metrics = positive_transform_eval(model, samples, edges, args.mode, task_type, args.batch_size, device, y_mean, y_std)

    out_dir = root / "runs" / args.run_name / "graph_dp" / args.dataset / args.mode / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "dataset": args.dataset,
        "mode": args.mode,
        "feature_kind": "graph_dp",
        "run_name": args.run_name,
        "processed_scope": args.processed_scope,
        "seed": args.seed,
        "task_type": task_type,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "epochs_run": int(len(history)),
        "seconds": float(time.time() - start),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "pair_margin_lambda": float(args.pair_margin_lambda),
        "pair_margin": float(args.pair_margin),
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
    write_split_predictions(val, val_pred, out_dir / "val_predictions.parquet")
    write_split_predictions(test, test_pred, out_dir / "test_predictions.parquet")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
