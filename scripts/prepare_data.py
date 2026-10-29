"""Reconstruct the Tautobase training data using the fixed split manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from prepare_base_data import build_tautobase_pref, make_edges


def main():
    samples = build_tautobase_pref(ROOT)
    if samples.empty:
        raise SystemExit("Download the source first: python scripts/fetch_sources.py --tautobase")
    manifest = pd.read_csv(ROOT / "data/splits/split_manifest.csv")
    if set(samples.sample_id) != set(manifest.sample_id):
        raise ValueError("Source/RDKit preprocessing changed retained sample identities")
    expected = manifest.set_index("sample_id")
    for column in ("y", "paired_id", "family_key"):
        if not samples[column].equals(samples.sample_id.map(expected[column])):
            raise ValueError(f"Preprocessing differs from the frozen {column} values")
    samples["split"] = samples.sample_id.map(expected.split)
    samples["main_split_group"] = samples.family_key
    edges = make_edges(samples)
    edges = edges[~edges.evidence.eq("pilot_scalar_mirror_assumption_not_main_evidence")].reset_index(drop=True)
    output = ROOT / "data/processed/main"
    output.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(output/"tautobase_pref_samples.parquet",index=False)
    samples.to_parquet(output/"all_samples.parquet",index=False)
    edges.to_parquet(output/"transform_edges.parquet",index=False)
    print(json.dumps({"states":len(samples),"families":samples.family_key.nunique(),
                      "split_counts":samples.split.value_counts().to_dict(),"edges":len(edges)},indent=2))


if __name__ == "__main__":
    main()
