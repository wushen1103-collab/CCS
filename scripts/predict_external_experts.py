from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict_tautomer_expert import (
    load_external_model,
    pair_normalize_predictions,
    predict_external,
)


RT_298_KCAL_MOL = 0.592


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-value))


def write_predictions(samples: pd.DataFrame, prediction: np.ndarray, path: Path) -> None:
    columns = [
        column
        for column in ["sample_id", "dataset", "split", "smiles", "y", "paired_id", "family_key"]
        if column in samples.columns
    ]
    out = samples[columns].copy()
    out["prediction"] = np.asarray(prediction, dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def sphys_published_predictions(samples: pd.DataFrame) -> np.ndarray:
    values = []
    for row in samples.itertuples(index=False):
        ddg = float(row.sphysnet_published_ddg)
        is_left = str(row.sample_id).endswith("_t1")
        logit = ddg / RT_298_KCAL_MOL if is_left else -ddg / RT_298_KCAL_MOL
        values.append(float(sigmoid(np.asarray([logit]))[0]))
    return np.asarray(values, dtype=float)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed = root / "data/processed/evaluation_required"
    out_root = root / "runs/evaluation_external_released_baselines"
    model = load_external_model(root)
    summary = []
    for dataset in ["sphysnet_100_aqueous", "nci_aqueous_zero_overlap"]:
        samples = pd.read_parquet(processed / f"{dataset}_samples.parquet")
        raw = predict_external(samples["smiles"].astype(str).tolist(), model)
        pairnorm = pair_normalize_predictions(samples, raw)
        write_predictions(samples, raw, out_root / "tautomer_predictor_raw" / dataset / "test_predictions.parquet")
        write_predictions(
            samples,
            pairnorm,
            out_root / "tautomer_predictor_pairnorm" / dataset / "test_predictions.parquet",
        )
        summary.append({"dataset": dataset, "baseline": "tautomer_predictor_pairnorm", "states": len(samples)})
        if dataset == "sphysnet_100_aqueous":
            sphys = sphys_published_predictions(samples)
            write_predictions(samples, sphys, out_root / "sphysnet_published" / dataset / "test_predictions.parquet")
            summary.append(
                {
                    "dataset": dataset,
                    "baseline": "sphysnet_published_boltzmann_rt298",
                    "states": len(samples),
                }
            )
    manifest = {
        "tautomer_predictor": "released AttentiveFP checkpoint, deterministic rerun",
        "sphysnet": "published sphysnet-pre pair dG converted with fixed RT=0.592 kcal/mol",
        "datasets": summary,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
