from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from condchemsymm.splits import scaffold_group_split
from prepare_base_data import (
    build_tautobase_pref,
    load_bace,
    load_esol,
    make_base_dataset,
    make_edges,
    pick_col,
)


def duplicate_pair_component_split(frame: pd.DataFrame, seed: int, ratios=(0.7, 0.15, 0.15)) -> pd.DataFrame:
    out = frame.reset_index(drop=True).copy()
    parent = list(range(len(out)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for col in ["family_key", "smiles"]:
        groups = defaultdict(list)
        for i, value in enumerate(out[col].astype(str)):
            groups[value].append(i)
        for idxs in groups.values():
            head = idxs[0]
            for idx in idxs[1:]:
                union(head, idx)

    id_to_idx = {str(sid): i for i, sid in enumerate(out["sample_id"])}
    for i, pid in enumerate(out.get("paired_id", [])):
        pid = str(pid)
        if pid and pid in id_to_idx:
            union(i, id_to_idx[pid])

    components = defaultdict(list)
    for i in range(len(out)):
        components[find(i)].append(i)
    ordered = sorted(components.values(), key=lambda idxs: min(str(out.loc[i, "sample_id"]) for i in idxs))
    split_group = {}
    for comp_id, idxs in enumerate(ordered):
        key = f"main_tautobase_component_{comp_id:05d}"
        for idx in idxs:
            split_group[idx] = key
    out["main_split_group"] = [split_group[i] for i in range(len(out))]
    assignment = scaffold_group_split(out["main_split_group"], seed=seed, ratios=ratios)
    out["split"] = out["main_split_group"].map(assignment)
    return out


def load_freesolv(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "data/raw/moleculenet/SAMPL.csv")
    smiles_col = pick_col(df, ["smiles", "SMILES"])
    label_col = pick_col(df, ["expt", "y"])
    return make_base_dataset(df, "freesolv", smiles_col, label_col, "regression")


def load_lipophilicity(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "data/raw/moleculenet/Lipophilicity.csv")
    smiles_col = pick_col(df, ["smiles", "SMILES"])
    label_col = pick_col(df, ["exp", "y"])
    return make_base_dataset(df, "lipophilicity", smiles_col, label_col, "regression")


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "data/processed/main"
    out_dir.mkdir(parents=True, exist_ok=True)
    (root / "manifests/transforms").mkdir(parents=True, exist_ok=True)

    frames = [
        load_esol(root),
        load_bace(root),
        load_freesolv(root),
        load_lipophilicity(root),
    ]
    tautobase = build_tautobase_pref(root)
    if len(tautobase):
        tautobase = duplicate_pair_component_split(tautobase, seed=12)
        frames.append(tautobase)

    all_samples = pd.concat(frames, ignore_index=True)
    raw_edges = make_edges(all_samples)
    unsupported_scalar_mirror = raw_edges["evidence"].eq("pilot_scalar_mirror_assumption_not_main_evidence")
    all_edges = raw_edges.loc[~unsupported_scalar_mirror].reset_index(drop=True)

    for name, frame in all_samples.groupby("dataset"):
        frame.to_parquet(out_dir / f"{name}_samples.parquet", index=False)
    all_samples.to_parquet(out_dir / "all_samples.parquet", index=False)
    all_edges.to_parquet(out_dir / "transform_edges.parquet", index=False)

    summary = {
        "scope": "main",
        "datasets": {
            name: {
                "n": int(len(frame)),
                "split_counts": {k: int(v) for k, v in frame["split"].value_counts().to_dict().items()},
                "task_type": str(frame["task_type"].iloc[0]),
                "pilot_only": bool(frame["is_pilot_only"].all()),
            }
            for name, frame in all_samples.groupby("dataset")
        },
        "edges": {
            "n": int(len(all_edges)),
            "by_transform": {k: int(v) for k, v in all_edges["transform_type"].value_counts().to_dict().items()},
            "by_z": {str(k): int(v) for k, v in all_edges["z"].value_counts().to_dict().items()},
        },
        "dropped_edges": {
            "unsupported_scalar_mirror_assumption": int(unsupported_scalar_mirror.sum()),
        },
    }
    (root / "manifests/transforms/main_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
