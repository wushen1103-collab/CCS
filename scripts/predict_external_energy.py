from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from predict_energy_expert import (  # noqa: E402
    PairRecord,
    add_sphysnet_to_path,
    configure_runtime,
    prediction_rows_for_pairs,
    samples_from_pair_predictions,
)


DATASET = "nci_aqueous_zero_overlap"
NUM_CONFS = 10
TEMPERATURE = 0.592


def load_pairs(root: Path) -> tuple[list[PairRecord], dict[str, str]]:
    path = root / f"data/processed/evaluation_required/{DATASET}_samples.parquet"
    samples = pd.read_parquet(path)
    by_id = samples.set_index("sample_id")
    family_by_id = dict(zip(samples["sample_id"].astype(str), samples["family_key"].astype(str)))
    pairs: list[PairRecord] = []
    seen: set[str] = set()
    for row in samples.sort_values("sample_id").itertuples(index=False):
        sample_id = str(row.sample_id)
        paired_id = str(row.paired_id)
        if sample_id in seen or paired_id in seen:
            continue
        other = by_id.loc[paired_id]
        pairs.append(
            PairRecord(
                pair_id="::".join(sorted((sample_id, paired_id))),
                split="external_test",
                left_sample_id=sample_id,
                right_sample_id=paired_id,
                left_smiles=str(row.smiles),
                right_smiles=str(other["smiles"]),
                left_y=float(row.y),
                right_y=float(other["y"]),
            )
        )
        seen.update((sample_id, paired_id))
    return pairs, family_by_id


def main() -> None:
    root = PROJECT_ROOT
    add_sphysnet_to_path(root)
    configure_runtime(torch_threads=4, rdkit_threads=8, cluster_method="fast_rms_matrix")
    pairs, family_by_id = load_pairs(root)
    rows = prediction_rows_for_pairs(
        pairs,
        num_confs=NUM_CONFS,
        progress_every=1,
        cluster_method="fast_rms_matrix",
    )
    pair_frame = pd.DataFrame(rows)
    run_dir = root / "runs/evaluation_external_released_baselines/sphysnet_checkpoint" / DATASET
    run_dir.mkdir(parents=True, exist_ok=True)
    pair_frame.to_parquet(run_dir / "pair_predictions.parquet", index=False)

    predictions = samples_from_pair_predictions(pair_frame, TEMPERATURE)
    predictions["family_key"] = predictions["sample_id"].astype(str).map(family_by_id)
    predictions.to_parquet(run_dir / "test_predictions.parquet", index=False)
    status = pair_frame["status"].value_counts(dropna=False).to_dict()
    result = {
        "dataset": DATASET,
        "num_confs": NUM_CONFS,
        "temperature": TEMPERATURE,
        "total_pairs": len(pair_frame),
        "prediction_pairs": int(len(predictions) // 2),
        "status": {str(key): int(value) for key, value in status.items()},
    }
    (run_dir / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
