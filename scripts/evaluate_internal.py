from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_fusion import (  # noqa: E402
    SEEDS,
    family_bootstrap,
    metrics,
    project_pair_logit,
    safe_logit,
    sigmoid,
    summarize_seed_rows,
)
from ensemble_components import GRAPH_MAIN, load_predictions  # noqa: E402
from fusion_components import MLP_SELECT01  # noqa: E402


WEIGHT_GRID = np.linspace(0.0, 1.0, 26)
FALSE_INVARIANCE_WEIGHT = 0.5
CERTIFICATE_SCALE = 1.0


def load_components(root: Path, seed: int, split: str) -> pd.DataFrame:
    graph = load_predictions(root, GRAPH_MAIN, seed, split).rename(columns={GRAPH_MAIN.label: "graph"})
    mlp = load_predictions(root, MLP_SELECT01, seed, split).rename(columns={MLP_SELECT01.label: "mlp"})
    frame = graph.merge(mlp, on=["sample_id", "y", "paired_id"], how="inner")
    if len(frame) != len(graph) or len(frame) != len(mlp):
        raise ValueError(f"component row mismatch seed={seed} split={split}")
    return frame


def ordinary_prediction(frame: pd.DataFrame, graph_weight: float) -> np.ndarray:
    score = graph_weight * safe_logit(frame["graph"].to_numpy(dtype=float))
    score += (1.0 - graph_weight) * safe_logit(frame["mlp"].to_numpy(dtype=float))
    return sigmoid(score)


def choose_shared_weight(validation: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    rows = []
    for graph_weight in WEIGHT_GRID:
        frame = validation[["sample_id", "y", "paired_id"]].copy()
        frame["prediction"] = ordinary_prediction(validation, float(graph_weight))
        result = metrics(frame)
        rows.append(
            {
                "graph_weight": float(graph_weight),
                "mlp_weight": float(1.0 - graph_weight),
                "objective": float(result["nll"] + FALSE_INVARIANCE_WEIGHT * result["false_invariance"]),
                **result,
            }
        )
    grid = pd.DataFrame(rows)
    best = grid.sort_values(["objective", "nll", "brier", "graph_weight"]).iloc[0]
    return float(best["graph_weight"]), grid


def main() -> None:
    root = PROJECT_ROOT
    samples = pd.read_parquet(root / "data/processed/main/tautobase_pref_samples.parquet")[[
        "sample_id",
        "family_key",
    ]]
    raw_rows = []
    selection_rows = []
    bootstrap_frames: dict[str, dict[int, pd.DataFrame]] = {"ordinary": {}, "certificate": {}}
    for seed in SEEDS:
        validation = load_components(root, seed, "val")
        test = load_components(root, seed, "test")
        graph_weight, grid = choose_shared_weight(validation)
        selected = grid[grid["graph_weight"].eq(graph_weight)].iloc[0].to_dict()
        selection_rows.append({"seed": seed, **selected})

        score = graph_weight * safe_logit(test["graph"].to_numpy(dtype=float))
        score += (1.0 - graph_weight) * safe_logit(test["mlp"].to_numpy(dtype=float))
        predictions = {
            "ordinary": sigmoid(score),
            "certificate": project_pair_logit(test, score, CERTIFICATE_SCALE),
        }
        for method, prediction in predictions.items():
            frame = test[["sample_id", "y", "paired_id"]].copy()
            frame["prediction"] = prediction
            frame = frame.merge(samples, on="sample_id", how="left")
            out_dir = root / f"runs/evaluation_internal_only_certificate/{method}/seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(out_dir / "test_predictions.parquet", index=False)
            bootstrap_frames[method][seed] = frame
            raw_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "graph_weight": graph_weight,
                    "mlp_weight": 1.0 - graph_weight,
                    "certificate_scale": CERTIFICATE_SCALE if method == "certificate" else np.nan,
                    **metrics(frame),
                }
            )

    raw = pd.DataFrame(raw_rows)
    summary = summarize_seed_rows(raw, ["method"])
    bootstrap = family_bootstrap(
        bootstrap_frames,
        "tautobase_internal_only_family_split",
        [("certificate", "ordinary")],
    )
    table_dir = root / "reports/tables"
    pd.DataFrame(selection_rows).to_csv(
        table_dir / "evaluation_internal_only_validation_selection.csv", index=False
    )
    raw.to_csv(table_dir / "evaluation_internal_only_certificate_seed_raw.csv", index=False)
    summary.to_csv(table_dir / "evaluation_internal_only_certificate.csv", index=False)
    bootstrap.to_csv(table_dir / "evaluation_internal_only_certificate_bootstrap_5000.csv", index=False)
    print(
        json.dumps(
            {
                "seed_count": len(SEEDS),
                "selected_graph_weights": [float(row["graph_weight"]) for row in selection_rows],
                "bootstrap_replicates": 5000,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
