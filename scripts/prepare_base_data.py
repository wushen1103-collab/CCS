from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from condchemsymm.chem import (
    atom_renumber_smiles,
    canonical_smiles,
    canonical_unmapped_smiles,
    connectivity_key,
    invert_chirality_smiles,
    random_smiles,
    scaffold_smiles,
    single_rs_label,
)
from condchemsymm.splits import scaffold_group_split


def pick_col(df: pd.DataFrame, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise KeyError(f"None of {candidates} found in columns {list(df.columns)}")


def load_esol(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "data/raw/moleculenet/delaney-processed.csv")
    smiles_col = pick_col(df, ["smiles", "SMILES"])
    label_col = pick_col(df, ["measured log solubility in mols per litre", "measured log solubility", "y"])
    return make_base_dataset(df, "esol", smiles_col, label_col, "regression")


def load_bace(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "data/raw/moleculenet/bace.csv")
    smiles_col = pick_col(df, ["mol", "smiles", "SMILES"])
    label_col = pick_col(df, ["Class", "class", "label", "Label"])
    return make_base_dataset(df, "bace", smiles_col, label_col, "classification")


def make_base_dataset(df, name, smiles_col, label_col, task_type):
    rows = []
    for i, row in df.iterrows():
        smi = canonical_smiles(str(row[smiles_col]), isomeric=True)
        if smi is None:
            continue
        y = row[label_col]
        if pd.isna(y):
            continue
        rows.append(
            {
                "sample_id": f"{name}_{len(rows)}",
                "dataset": name,
                "task_type": task_type,
                "smiles": smi,
                "y": float(y),
                "scaffold": scaffold_smiles(smi),
                "family_key": connectivity_key(smi) or smi,
                "source_row": int(i),
                "is_pilot_only": False,
                "paired_id": "",
            }
        )
    out = pd.DataFrame(rows)
    assignment = scaffold_group_split(out["scaffold"], seed=0)
    out["split"] = out["scaffold"].map(assignment)
    return out


def build_synthetic_rs(base_frames):
    candidates = []
    seen = set()
    for base in base_frames:
        for _, row in base.iterrows():
            smi = row["smiles"]
            key = connectivity_key(smi)
            if not key or key in seen:
                continue
            label = single_rs_label(smi)
            mirror = invert_chirality_smiles(smi)
            mirror_label = single_rs_label(mirror) if mirror else None
            if label is None or mirror is None or mirror_label is None or mirror_label == label:
                continue
            seen.add(key)
            candidates.append((smi, label, mirror, mirror_label, scaffold_smiles(smi), key))

    rows = []
    for i, (smi, label, mirror, mirror_label, scaf, key) in enumerate(candidates):
        a = f"synthetic_rs_{i}_a"
        b = f"synthetic_rs_{i}_b"
        rows.append(
            {
                "sample_id": a,
                "dataset": "synthetic_rs",
                "task_type": "classification",
                "smiles": smi,
                "y": float(label),
                "scaffold": scaf,
                "family_key": key,
                "source_row": i,
                "is_pilot_only": True,
                "paired_id": b,
            }
        )
        rows.append(
            {
                "sample_id": b,
                "dataset": "synthetic_rs",
                "task_type": "classification",
                "smiles": mirror,
                "y": float(mirror_label),
                "scaffold": scaf,
                "family_key": key,
                "source_row": i,
                "is_pilot_only": True,
                "paired_id": a,
            }
        )
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    assignment = scaffold_group_split(out["family_key"], seed=1)
    out["split"] = out["family_key"].map(assignment)
    return out


def parse_logk(value):
    if pd.isna(value):
        return None
    text = str(value).replace("<NL>", "").strip()
    if not text:
        return None
    sign = 1.0
    if text.startswith(">"):
        text = text[1:].strip()
    elif text.startswith("<"):
        sign = -1.0 if text.startswith("<-") else 1.0
        text = text[1:].strip()
    try:
        val = float(text)
    except ValueError:
        return None
    if str(value).strip().startswith("<-"):
        val = -abs(val)
    return val


def parse_percent(value):
    if pd.isna(value):
        return None, ""
    text = str(value).replace("<NL>", "").replace("%", "").replace("~", "").strip()
    if not text:
        return None, ""
    op = ""
    if text[0] in "<>":
        op = text[0]
        text = text[1:].strip()
    try:
        return float(text), op
    except ValueError:
        return None, op


def preferred_from_row(row):
    pref = row.get("Preferred")
    if not pd.isna(pref):
        text = str(pref).replace("*", "").strip()
        if text.startswith("1"):
            return 1, "preferred_column_explicit"
        if text.startswith("2"):
            return 2, "preferred_column_explicit"

    val, op = parse_percent(row.get("% (Tautomer1)"))
    if val is not None:
        if val >= 70 and op != "<":
            return 1, "percent_tautomer1_ge70"
        if val <= 30 and op != ">":
            return 2, "percent_tautomer1_le30"
    return None, "insufficient_preference_evidence"


def build_tautobase_pref(root: Path) -> pd.DataFrame:
    path = root / "data/raw/tautobase/Tautobase/Tautobase_SMIRKS.txt"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t")
    rows = []
    seen = set()
    for _, row in df.iterrows():
        pref, evidence = preferred_from_row(row)
        if pref is None:
            continue
        smirks = row.get("Smirks")
        if not isinstance(smirks, str) or ">>" not in smirks:
            continue
        left, right = smirks.split(">>", 1)
        s1 = canonical_unmapped_smiles(left, isomeric=True)
        s2 = canonical_unmapped_smiles(right, isomeric=True)
        if s1 is None or s2 is None or s1 == s2:
            continue
        key = tuple(sorted([s1, s2]))
        if key in seen:
            continue
        seen.add(key)
        family = f"tautobase_{int(row.get('Row Nr', len(rows)))}"
        y1 = 1.0 if pref == 1 else 0.0
        y2 = 1.0 - y1
        sid1 = f"{family}_t1"
        sid2 = f"{family}_t2"
        for sid, smi, y, paired in [(sid1, s1, y1, sid2), (sid2, s2, y2, sid1)]:
            rows.append(
                {
                    "sample_id": sid,
                    "dataset": "tautobase_pref",
                    "task_type": "classification",
                    "smiles": smi,
                    "y": y,
                    "scaffold": scaffold_smiles(smi),
                    "family_key": family,
                    "source_row": int(row.get("Row Nr", len(rows))),
                    "is_pilot_only": False,
                    "paired_id": paired,
                    "preference_evidence": evidence,
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        assignment = scaffold_group_split(out["family_key"], seed=2, ratios=(0.7, 0.15, 0.15))
        out["split"] = out["family_key"].map(assignment)
    return out


def make_edges(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in tqdm(samples.iterrows(), total=len(samples), desc="edges"):
        sid = row["sample_id"]
        smi = row["smiles"]
        dataset = row["dataset"]
        y = float(row["y"])
        base = {
            "sample_id": sid,
            "dataset": dataset,
            "base_smiles": smi,
            "target_y": y,
        }
        rows.append({**base, "view_id": f"{sid}:identity", "transform_type": "identity", "view_smiles": smi, "z": 1, "evidence": "identity"})
        for transform_type, fn in [
            ("atom_renumber", atom_renumber_smiles),
            ("random_smiles", random_smiles),
        ]:
            view = fn(smi, seed=17)
            if view:
                rows.append(
                    {
                        **base,
                        "view_id": f"{sid}:{transform_type}",
                        "transform_type": transform_type,
                        "view_smiles": view,
                        "z": 1,
                        "evidence": "representation_preserving",
                    }
                )
        mirror = invert_chirality_smiles(smi)
        if mirror:
            if dataset == "synthetic_rs":
                z = 0
                target_y = 1.0 - y
                evidence = "rdkit_single_center_cip_inversion_pilot"
            else:
                z = 1
                target_y = y
                evidence = "pilot_scalar_mirror_assumption_not_main_evidence"
            rows.append(
                {
                    **base,
                    "view_id": f"{sid}:mirror_chirality",
                    "transform_type": "mirror_chirality",
                    "view_smiles": mirror,
                    "z": z,
                    "target_y": target_y,
                    "evidence": evidence,
                }
            )
        if dataset == "tautobase_pref" and row.get("paired_id", ""):
            paired = samples[samples["sample_id"] == row["paired_id"]]
            if len(paired):
                paired_row = paired.iloc[0]
                rows.append(
                    {
                        **base,
                        "view_id": f"{sid}:tautomer_shift",
                        "transform_type": "tautomer_shift",
                        "view_smiles": paired_row["smiles"],
                        "z": 0,
                        "target_y": float(paired_row["y"]),
                        "evidence": row.get("preference_evidence", "tautobase_preference"),
                    }
                )
    return pd.DataFrame(rows)


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "data/processed/base"
    out_dir.mkdir(parents=True, exist_ok=True)
    (root / "manifests/splits").mkdir(parents=True, exist_ok=True)
    (root / "manifests/transforms").mkdir(parents=True, exist_ok=True)

    esol = load_esol(root)
    bace = load_bace(root)
    rs = build_synthetic_rs([esol, bace])
    tautobase = build_tautobase_pref(root)

    frames = [esol, bace]
    if len(rs):
        frames.append(rs)
    if len(tautobase):
        frames.append(tautobase)
    all_samples = pd.concat(frames, ignore_index=True)
    all_edges = make_edges(all_samples)

    for name, frame in all_samples.groupby("dataset"):
        frame.to_parquet(out_dir / f"{name}_samples.parquet", index=False)
    all_samples.to_parquet(out_dir / "all_samples.parquet", index=False)
    all_edges.to_parquet(out_dir / "transform_edges.parquet", index=False)

    summary = {
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
    }
    (root / "manifests/transforms/base_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
