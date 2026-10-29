from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_DATASETS = ["synthetic_rs", "tautobase_pref", "bace", "esol"]
META_COLUMNS = ["sample_id", "dataset", "split", "smiles", "target", "paired_id"]
SPLITS = ["train", "val", "test"]


def write_with_split_files(df: pd.DataFrame, base_path: Path, cols: list[str]) -> list[Path]:
    df[cols].to_csv(base_path, index=False)
    paths = [base_path]
    if "split" not in df.columns:
        return paths
    for split in SPLITS:
        split_path = base_path.with_name(f"{base_path.stem}_{split}{base_path.suffix}")
        df[df["split"] == split][cols].to_csv(split_path, index=False)
        paths.append(split_path)
    return paths


def export_identity(samples: pd.DataFrame, out_dir: Path) -> list[Path]:
    out = samples.rename(columns={"y": "target"}).copy()
    cols = [c for c in META_COLUMNS if c in out.columns]
    path = out_dir / "identity.csv"
    return write_with_split_files(out, path, cols)


def export_transform_views(samples: pd.DataFrame, edges: pd.DataFrame, out_dir: Path) -> list[Path]:
    meta = samples[["sample_id", "split", "y", "paired_id"]].rename(columns={"y": "target"})
    exported = []
    for transform_type in sorted(edges[edges["z"] == 1]["transform_type"].unique()):
        view = edges[edges["transform_type"] == transform_type][["sample_id", "dataset", "view_smiles"]].copy()
        view = view.merge(meta, on="sample_id", how="inner")
        view = view.rename(columns={"view_smiles": "smiles"})
        cols = [c for c in META_COLUMNS if c in view.columns]
        path = out_dir / f"view_{transform_type}.csv"
        exported.extend(write_with_split_files(view, path, cols))
    return exported


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--processed-scope", default="base")
    parser.add_argument("--out-dir")
    args = parser.parse_args()
    datasets = args.datasets or DEFAULT_DATASETS
    root = Path(__file__).resolve().parents[1]
    processed_dir = root / "data" / "processed" / args.processed_scope
    if args.out_dir is None:
        args.out_dir = f"data/chemprop_{args.processed_scope}"
    out_root = root / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    edges_all = pd.read_parquet(processed_dir / "transform_edges.parquet")
    manifest = []
    for dataset in datasets:
        samples = pd.read_parquet(processed_dir / f"{dataset}_samples.parquet")
        task_type = str(samples["task_type"].iloc[0])
        edges = edges_all[edges_all["dataset"] == dataset].copy()
        out_dir = out_root / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        identity_paths = export_identity(samples, out_dir)
        view_paths = export_transform_views(samples, edges, out_dir)
        manifest.append(
            {
                "dataset": dataset,
                "processed_scope": args.processed_scope,
                "task_type": task_type,
                "n_samples": int(len(samples)),
                "identity_csv": str(identity_paths[0].relative_to(root)),
                "identity_split_csvs": [str(p.relative_to(root)) for p in identity_paths[1:]],
                "view_csvs": [str(p.relative_to(root)) for p in view_paths],
                "split_counts": {k: int(v) for k, v in samples["split"].value_counts().items()},
            }
        )
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
