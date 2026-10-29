from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.ML.Cluster import Butina

from condchemsymm.metrics import classification_metrics


ALLOWED_ATOMIC_NUMS = {1, 6, 7, 8, 9, 15, 16, 17}
FALSE_INV_THRESHOLD = 0.25
DEFAULT_TEMPERATURE_GRID = np.geomspace(0.1, 100.0, 301)


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    split: str
    left_sample_id: str
    right_sample_id: str
    left_smiles: str
    right_smiles: str
    left_y: float
    right_y: float


def molecule_stats(smiles: str) -> dict[str, float | int | str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "n_heavy_atoms": 0,
            "n_rotatable_bonds": 0,
            "canonical_smiles": "",
        }
    return {
        "n_heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "n_rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
    }


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x))


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sphysnet_root(root: Path) -> Path:
    return root / "data" / "raw" / "external" / "sPhysNet-Taut-src"


def add_sphysnet_to_path(root: Path) -> None:
    path = sphysnet_root(root)
    if not path.exists():
        raise FileNotFoundError(f"sPhysNet-Taut source tree not found: {path}")
    sys.path.insert(0, str(path))


def configure_runtime(torch_threads: int, rdkit_threads: int, cluster_method: str) -> None:
    if torch_threads > 0:
        try:
            import torch

            torch.set_num_threads(torch_threads)
        except ImportError:
            pass
    if rdkit_threads > 0:
        import taut_src.gen_confs as gen_confs

        if cluster_method == "fast_rms_matrix":

            def cluster_conformers_fast(mol: Chem.Mol, rmsd_cutoff: float = 0.5) -> tuple:
                n_conf = mol.GetNumConformers()
                if n_conf < 2:
                    return tuple([[conf.GetId() for conf in mol.GetConformers()]])
                dists = list(AllChem.GetConformerRMSMatrix(mol, prealigned=False))
                return Butina.ClusterData(dists, n_conf, rmsd_cutoff, isDistData=True, reordering=True)

            gen_confs.cluster_conformers = cluster_conformers_fast
        elif cluster_method != "author_best_rms":
            raise ValueError(f"unsupported cluster method: {cluster_method}")

        def generate_confs_limited(smi: str, numConfs: int = 1) -> list:
            mol = Chem.MolFromSmiles(smi)
            mol = AllChem.AddHs(mol)
            ps = AllChem.ETKDG()
            ps.randomSeed = 1
            ps.pruneRmsThresh = 0.1
            ps.numThreads = rdkit_threads
            AllChem.EmbedMultipleConfs(mol, numConfs, ps)
            clusters = gen_confs.cluster_conformers(mol, rmsd_cutoff=0.5)
            confs = []
            for cluster in clusters:
                cid = cluster[0]
                confs.append(gen_confs.extract_mol_by_confId(mol, cid))
            return confs

        gen_confs.generate_confs = generate_confs_limited


def unsupported_elements(smiles: str) -> list[int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [-1]
    nums = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
    return sorted(nums - ALLOWED_ATOMIC_NUMS)


def load_pairs(root: Path, splits: set[str]) -> list[PairRecord]:
    samples = pd.read_parquet(root / "data" / "processed" / "main" / "tautobase_pref_samples.parquet")
    samples = samples[samples["split"].isin(splits)].copy()
    by_id = samples.set_index("sample_id")
    pairs = []
    seen = set()
    for row in samples.itertuples(index=False):
        sid = str(row.sample_id)
        pid = str(row.paired_id)
        if not pid or sid in seen or pid in seen or pid not in by_id.index:
            continue
        other = by_id.loc[pid]
        pair_id = "::".join(sorted([sid, pid]))
        if row.split != other["split"]:
            raise ValueError(f"pair crosses splits unexpectedly: {pair_id}")
        pairs.append(
            PairRecord(
                pair_id=pair_id,
                split=str(row.split),
                left_sample_id=sid,
                right_sample_id=pid,
                left_smiles=str(row.smiles),
                right_smiles=str(other["smiles"]),
                left_y=float(row.y),
                right_y=float(other["y"]),
            )
        )
        seen.add(sid)
        seen.add(pid)
    return sorted(pairs, key=lambda item: (item.split, item.pair_id))


def prediction_rows_for_pairs(
    pairs: list[PairRecord],
    num_confs: int,
    progress_every: int,
    cluster_method: str,
) -> list[dict]:
    from taut_src.rank_tautomer import predict_by_smis

    rows = []
    for idx, pair in enumerate(pairs, start=1):
        start = time.time()
        left_stats = molecule_stats(pair.left_smiles)
        right_stats = molecule_stats(pair.right_smiles)
        print(
            "start "
            f"{idx}/{len(pairs)} pair_id={pair.pair_id} split={pair.split} "
            f"left_heavy={left_stats['n_heavy_atoms']} right_heavy={right_stats['n_heavy_atoms']} "
            f"left_rot={left_stats['n_rotatable_bonds']} right_rot={right_stats['n_rotatable_bonds']} "
            f"num_confs={num_confs} cluster_method={cluster_method}",
            flush=True,
        )
        bad_left = unsupported_elements(pair.left_smiles)
        bad_right = unsupported_elements(pair.right_smiles)
        status = "ok"
        d_g_right_minus_left = np.nan
        error = ""
        if bad_left or bad_right:
            status = "unsupported_elements"
            error = f"left={bad_left};right={bad_right}"
        else:
            try:
                pred = predict_by_smis([pair.left_smiles, pair.right_smiles], num_confs=num_confs)
                d_g_right_minus_left = float(pred[1][2])
            except Exception:  # pragma: no cover - external model failures are recorded as rows.
                status = "error"
                error = traceback.format_exc(limit=4).replace("\n", " | ")
        elapsed = time.time() - start
        rows.append(
            {
                "pair_id": pair.pair_id,
                "split": pair.split,
                "left_sample_id": pair.left_sample_id,
                "right_sample_id": pair.right_sample_id,
                "left_smiles": pair.left_smiles,
                "right_smiles": pair.right_smiles,
                "left_y": pair.left_y,
                "right_y": pair.right_y,
                "left_heavy_atoms": left_stats["n_heavy_atoms"],
                "right_heavy_atoms": right_stats["n_heavy_atoms"],
                "left_rotatable_bonds": left_stats["n_rotatable_bonds"],
                "right_rotatable_bonds": right_stats["n_rotatable_bonds"],
                "num_confs": num_confs,
                "cluster_method": cluster_method,
                "d_g_right_minus_left": d_g_right_minus_left,
                "status": status,
                "elapsed_sec": elapsed,
                "error": error,
            }
        )
        if idx % progress_every == 0 or idx == len(pairs):
            print(f"processed {idx}/{len(pairs)} pairs", flush=True)
    return rows


def samples_from_pair_predictions(pair_df: pd.DataFrame, temperature: float) -> pd.DataFrame:
    rows = []
    for row in pair_df.itertuples(index=False):
        if row.status != "ok" or pd.isna(row.d_g_right_minus_left):
            continue
        left_logit = float(row.d_g_right_minus_left) / temperature
        right_logit = -left_logit
        left_prediction = float(sigmoid(np.asarray([left_logit]))[0])
        right_prediction = float(sigmoid(np.asarray([right_logit]))[0])
        rows.append(
            {
                "sample_id": row.left_sample_id,
                "paired_id": row.right_sample_id,
                "pair_id": row.pair_id,
                "split": row.split,
                "y": float(row.left_y),
                "prediction": left_prediction,
                "energy_score": float(row.d_g_right_minus_left),
            }
        )
        rows.append(
            {
                "sample_id": row.right_sample_id,
                "paired_id": row.left_sample_id,
                "pair_id": row.pair_id,
                "split": row.split,
                "y": float(row.right_y),
                "prediction": right_prediction,
                "energy_score": -float(row.d_g_right_minus_left),
            }
        )
    return pd.DataFrame(rows)


def pair_metrics(sample_df: pd.DataFrame) -> dict[str, float]:
    if sample_df.empty:
        return {
            "false_invariance_rate_test_pairs": np.nan,
            "pair_order_accuracy": np.nan,
            "pair_gap_mean": np.nan,
            "pair_gap_p10": np.nan,
            "n_pairs": 0,
        }
    pair_rows = []
    for _, group in sample_df.groupby("pair_id", sort=True):
        if len(group) != 2:
            continue
        left, right = group.iloc[0], group.iloc[1]
        gap = abs(float(left["prediction"]) - float(right["prediction"]))
        if float(left["y"]) == float(right["y"]):
            continue
        order_correct = (
            float(left["prediction"]) > float(right["prediction"])
            if float(left["y"]) > float(right["y"])
            else float(right["prediction"]) > float(left["prediction"])
        )
        pair_rows.append(
            {
                "gap": gap,
                "false_invariant": gap < FALSE_INV_THRESHOLD,
                "order_correct": bool(order_correct),
            }
        )
    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        return {
            "false_invariance_rate_test_pairs": np.nan,
            "pair_order_accuracy": np.nan,
            "pair_gap_mean": np.nan,
            "pair_gap_p10": np.nan,
            "n_pairs": 0,
        }
    return {
        "false_invariance_rate_test_pairs": float(pairs["false_invariant"].mean()),
        "pair_order_accuracy": float(pairs["order_correct"].mean()),
        "pair_gap_mean": float(pairs["gap"].mean()),
        "pair_gap_p10": float(pairs["gap"].quantile(0.10)),
        "n_pairs": int(len(pairs)),
    }


def evaluate_split(pair_df: pd.DataFrame, split: str, temperature: float) -> dict[str, float | str]:
    split_pairs = pair_df[pair_df["split"].eq(split)].copy()
    sample_df = samples_from_pair_predictions(split_pairs, temperature)
    if sample_df.empty:
        metrics = {
            "accuracy": np.nan,
            "brier": np.nan,
            "nll": np.nan,
            "ece10": np.nan,
            "mce10": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
        }
    else:
        metrics = classification_metrics(sample_df["y"].to_numpy(), sample_df["prediction"].to_numpy())
    pair = pair_metrics(sample_df)
    return {
        "split": split,
        "temperature": float(temperature),
        "n_samples": int(len(sample_df)),
        "n_ok_pairs": int(pair["n_pairs"]),
        "n_total_pairs": int(len(split_pairs)),
        "coverage_pairs": float(pair["n_pairs"] / len(split_pairs)) if len(split_pairs) else np.nan,
        **{f"{split}_{key}": value for key, value in metrics.items()},
        **{f"{split}_{key}": value for key, value in pair.items() if key != "n_pairs"},
    }


def choose_temperature(pair_df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    rows = []
    val_pairs = pair_df[pair_df["split"].eq("val")].copy()
    for temperature in DEFAULT_TEMPERATURE_GRID:
        sample_df = samples_from_pair_predictions(val_pairs, float(temperature))
        if sample_df.empty:
            continue
        metrics = classification_metrics(sample_df["y"].to_numpy(), sample_df["prediction"].to_numpy())
        pair = pair_metrics(sample_df)
        rows.append(
            {
                "temperature": float(temperature),
                "val_nll": metrics["nll"],
                "val_brier": metrics["brier"],
                "val_roc_auc": metrics["roc_auc"],
                "val_false_invariance_rate": pair["false_invariance_rate_test_pairs"],
                "val_pair_order_accuracy": pair["pair_order_accuracy"],
            }
        )
    grid = pd.DataFrame(rows)
    if grid.empty:
        return 0.592, grid
    best = grid.sort_values(["val_nll", "val_brier", "temperature"], ascending=[True, True, True]).iloc[0]
    return float(best["temperature"]), grid


def summarize(run_dir: Path, table_dir: Path, num_confs: int) -> None:
    shard_paths = sorted(run_dir.glob(f"pair_predictions_numconf{num_confs}_shard*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"no shard outputs found under {run_dir}")
    pair_df = pd.concat([pd.read_parquet(path) for path in shard_paths], ignore_index=True)
    pair_df = pair_df.drop_duplicates(["pair_id", "split"], keep="last").sort_values(["split", "pair_id"])
    ok_df = pair_df[pair_df["status"].eq("ok")].copy()
    temperature, grid = choose_temperature(ok_df)
    eval_rows = []
    for mode, temp in [("boltzmann_rt_298k", 0.592), ("val_nll_temperature", temperature)]:
        row = {
            "method": "sPhysNet-Taut",
            "mode": mode,
            "num_confs": num_confs,
            "temperature": float(temp),
            "n_pair_predictions": int(len(pair_df)),
            "n_ok_pair_predictions": int(len(ok_df)),
            "n_error_pair_predictions": int((pair_df["status"] == "error").sum()),
            "n_unsupported_pair_predictions": int((pair_df["status"] == "unsupported_elements").sum()),
        }
        for split in ["val", "test"]:
            row.update(evaluate_split(ok_df, split, float(temp)))
        eval_rows.append(row)
    result = pd.DataFrame(eval_rows)
    table_dir.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(table_dir / f"main_sphysnet_tautobase_numconf{num_confs}_pair_predictions.csv", index=False)
    result.to_csv(table_dir / f"main_sphysnet_tautobase_numconf{num_confs}_baseline.csv", index=False)
    if not grid.empty:
        grid.to_csv(table_dir / f"main_sphysnet_tautobase_numconf{num_confs}_temperature_grid.csv", index=False)
    write_markdown(table_dir / f"main_sphysnet_tautobase_numconf{num_confs}_baseline.md", result)
    print(f"wrote sPhysNet summary rows={len(result)} pairs={len(pair_df)} ok={len(ok_df)} temperature={temperature:.4f}")


def fmt(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def write_markdown(path: Path, result: pd.DataFrame) -> None:
    cols = [
        "method",
        "mode",
        "num_confs",
        "temperature",
        "n_ok_pair_predictions",
        "test_roc_auc",
        "test_brier",
        "test_nll",
        "test_ece10",
        "test_false_invariance_rate_test_pairs",
        "test_pair_order_accuracy",
        "test_pair_gap_mean",
        "coverage_pairs",
    ]
    view = result[[col for col in cols if col in result.columns]].copy()
    for col in view.columns:
        view[col] = view[col].map(fmt)
    path.write_text(
        "# Phase 1 sPhysNet-Taut Tautobase Baseline\n\n"
        "External deterministic tautomer-pair baseline using released sPhysNet-Taut weights. "
        "`boltzmann_rt_298k` maps predicted pair energy gaps to probabilities using RT=0.592 kcal/mol; "
        "`val_nll_temperature` fits one scalar temperature on Phase 1 validation pairs before test evaluation. "
        "This is an external baseline/provenance check, not a trained five-seed CondChemSymm run.\n\n"
        + view.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--num-confs", type=int, default=3)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--run-name", default="main_sphysnet_tautobase")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--rdkit-threads", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--cluster-method",
        choices=["author_best_rms", "fast_rms_matrix"],
        default="author_best_rms",
        help="Conformer RMSD clustering implementation inside the sPhysNet conformer generator.",
    )
    args = parser.parse_args()

    root = project_root()
    run_dir = root / "runs" / args.run_name
    table_dir = root / "reports" / "tables"
    if args.summarize:
        summarize(run_dir, table_dir, args.num_confs)
        return

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    add_sphysnet_to_path(root)
    configure_runtime(args.torch_threads, args.rdkit_threads, args.cluster_method)
    pairs = load_pairs(root, set(args.splits))
    pairs = [pair for idx, pair in enumerate(pairs) if idx % args.num_shards == args.shard_index]
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"pair_predictions_numconf{args.num_confs}_shard{args.shard_index}_of_{args.num_shards}.parquet"
    rows = prediction_rows_for_pairs(pairs, args.num_confs, max(1, args.progress_every), args.cluster_method)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"wrote {out_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
