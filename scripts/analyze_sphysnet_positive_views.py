from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from condchemsymm.metrics import classification_metrics
from predict_energy_expert import samples_from_pair_predictions
from ensemble_components import load_component_view_predictions, paired_stats
from fusion_components import SPECS, component_columns, load_sphys_temperature


TRANSFORMS = ["identity", "atom_renumber", "random_smiles"]
MODES = ["boltzmann_rt_298k", "val_nll_temperature"]


def fmt(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def objective_label(value: float) -> str:
    return f"{float(value):g}"


def load_pair_predictions(root: Path, num_confs: int, transform_type: str) -> pd.DataFrame:
    table_dir = root / "reports" / "tables"
    if transform_type == "identity":
        df = pd.read_csv(table_dir / f"main_sphysnet_tautobase_numconf{num_confs}_pair_predictions.csv")
        return df[df["split"].eq("test")].copy()
    run_dir = root / "runs" / f"main_sphysnet_tautobase_numconf{num_confs}_views"
    paths = sorted(run_dir.glob(f"view_pair_predictions_{transform_type}_numconf{num_confs}_shard*.parquet"))
    if not paths:
        raise FileNotFoundError(f"missing sPhysNet view predictions for {transform_type} under {run_dir}")
    df = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    return df.drop_duplicates(["pair_id", "split"], keep="last").sort_values(["split", "pair_id"])


def sphys_sample_predictions(root: Path, num_confs: int, transform_type: str, mode: str) -> pd.DataFrame:
    pair_df = load_pair_predictions(root, num_confs, transform_type)
    pair_df = pair_df[pair_df["status"].eq("ok")].copy()
    temperature = load_sphys_temperature(root / "reports" / "tables", num_confs, mode)
    sample = samples_from_pair_predictions(pair_df, temperature)
    return sample[["sample_id", "y", "paired_id", "prediction"]].rename(columns={"prediction": "sphysnet_prediction"})


def evaluate_sample_predictions(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    pred = df[pred_col].to_numpy(dtype=float)
    metrics = classification_metrics(df["y"].to_numpy(dtype=float), pred)
    pair = paired_stats(df[["sample_id", "y", "paired_id", pred_col]].rename(columns={pred_col: "prediction"}), "prediction")
    return {
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "brier": metrics["brier"],
        "nll": metrics["nll"],
        "ece10": metrics["ece10"],
        "mce10": metrics["mce10"],
        **pair,
    }


def sphys_baseline_rows(root: Path, num_confs: int) -> list[dict]:
    rows = []
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    for mode in MODES:
        for transform in TRANSFORMS:
            cache[(mode, transform)] = sphys_sample_predictions(root, num_confs, transform, mode)
            metrics = evaluate_sample_predictions(cache[(mode, transform)], "sphysnet_prediction")
            for seed in range(5):
                rows.append(
                    {
                        "candidate": f"sphysnet_{mode}_n{num_confs}",
                        "model_group": "sphysnet_baseline_replicated",
                        "seed": seed,
                        "transform_type": transform,
                        "sphys_mode": mode,
                        "sphys_num_confs": num_confs,
                        "val_selection_false_inv_weight": np.nan,
                        "selected_graph_weight": 0.0,
                        "selected_mlp_weight": 0.0,
                        "selected_sphysnet_weight": 1.0,
                        **metrics,
                    }
                )
    return rows


def fusion_rows(root: Path, num_confs: int) -> list[dict]:
    table_dir = root / "reports" / "tables"
    raw_path = table_dir / f"main_sphysnet_numconf{num_confs}_fusion_val_selected_seed_raw.csv"
    raw = pd.read_csv(raw_path)
    specs = {spec.label: spec for spec in SPECS}
    sphys_cache: dict[tuple[str, str], pd.DataFrame] = {}
    rows = []
    for _, selected in raw.iterrows():
        spec = specs[str(selected["fusion"])]
        seed = int(selected["seed"])
        objective = float(selected["val_selection_false_inv_weight"])
        for transform in TRANSFORMS:
            key = (str(selected["sphys_mode"]), transform)
            if key not in sphys_cache:
                sphys_cache[key] = sphys_sample_predictions(root, num_confs, transform, str(selected["sphys_mode"]))
            sphys = sphys_cache[key]
            graph = load_component_view_predictions(root, spec.graph, seed, transform)
            merged = graph.merge(sphys, on=["sample_id", "y", "paired_id"], how="inner")
            if len(merged) != len(graph) or len(merged) != len(sphys):
                raise ValueError(f"view row mismatch for {spec.label} seed={seed} transform={transform}")
            if spec.mlp is not None:
                mlp = load_component_view_predictions(root, spec.mlp, seed, transform)
                merged = merged.merge(mlp, on=["sample_id", "y", "paired_id"], how="inner")
                if len(merged) != len(graph) or len(merged) != len(mlp):
                    raise ValueError(f"MLP view row mismatch for {spec.label} seed={seed} transform={transform}")
            columns = component_columns(spec)
            weights = []
            for column in columns:
                if column == spec.graph.label:
                    weights.append(float(selected["selected_graph_weight"]))
                elif spec.mlp is not None and column == spec.mlp.label:
                    weights.append(float(selected["selected_mlp_weight"]))
                elif column == "sphysnet_prediction":
                    weights.append(float(selected["selected_sphysnet_weight"]))
                else:
                    raise ValueError(f"unrecognized component column {column}")
            merged["prediction"] = merged[columns].to_numpy(dtype=float) @ np.asarray(weights, dtype=float)
            metrics = evaluate_sample_predictions(merged, "prediction")
            rows.append(
                {
                    "candidate": f"{selected['fusion']}_n{num_confs}_valobj{objective_label(objective)}",
                    "model_group": "sphysnet_fusion",
                    "seed": seed,
                    "transform_type": transform,
                    "sphys_mode": selected["sphys_mode"],
                    "sphys_num_confs": num_confs,
                    "val_selection_false_inv_weight": objective,
                    "selected_graph_weight": float(selected["selected_graph_weight"]),
                    "selected_mlp_weight": float(selected["selected_mlp_weight"]) if not pd.isna(selected["selected_mlp_weight"]) else 0.0,
                    "selected_sphysnet_weight": float(selected["selected_sphysnet_weight"]),
                    **metrics,
                }
            )
    return rows


def by_transform_summary(raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "candidate",
        "model_group",
        "sphys_num_confs",
        "sphys_mode",
        "val_selection_false_inv_weight",
        "transform_type",
    ]
    metric_cols = [
        "selected_graph_weight",
        "selected_mlp_weight",
        "selected_sphysnet_weight",
        "roc_auc",
        "pr_auc",
        "brier",
        "nll",
        "ece10",
        "mce10",
        "false_invariance_rate_test_pairs",
        "pair_order_accuracy",
        "pair_gap_mean",
    ]
    out = raw.groupby(group_cols, dropna=False, sort=True)[metric_cols].agg(["mean", "std", "count"])
    out.columns = [f"{name}_{stat}" for name, stat in out.columns]
    return out.reset_index()


def worst_summary(raw: pd.DataFrame) -> pd.DataFrame:
    per_seed = (
        raw.groupby(
            [
                "candidate",
                "model_group",
                "sphys_num_confs",
                "sphys_mode",
                "val_selection_false_inv_weight",
                "seed",
            ],
            dropna=False,
            sort=True,
        )
        .agg(
            selected_graph_weight=("selected_graph_weight", "mean"),
            selected_mlp_weight=("selected_mlp_weight", "mean"),
            selected_sphysnet_weight=("selected_sphysnet_weight", "mean"),
            worst_positive_transform_roc_auc=("roc_auc", "min"),
            worst_positive_transform_brier=("brier", "max"),
            worst_positive_transform_nll=("nll", "max"),
            worst_positive_transform_ece10=("ece10", "max"),
            worst_positive_transform_false_invariance=("false_invariance_rate_test_pairs", "max"),
            worst_positive_transform_pair_order_accuracy=("pair_order_accuracy", "min"),
        )
        .reset_index()
    )
    metric_cols = [col for col in per_seed.columns if col not in ["candidate", "model_group", "sphys_num_confs", "sphys_mode", "val_selection_false_inv_weight", "seed"]]
    out = per_seed.groupby(
        ["candidate", "model_group", "sphys_num_confs", "sphys_mode", "val_selection_false_inv_weight"],
        dropna=False,
        sort=True,
    )[metric_cols].agg(["mean", "std", "count"])
    out.columns = [f"{name}_{stat}" for name, stat in out.columns]
    return out.reset_index()


def make_markdown(worst: pd.DataFrame) -> str:
    focus = [
        "sphysnet_boltzmann_rt_298k_n10",
        "sphysnet_val_nll_temperature_n10",
        "graph_mlp_select01_sphys_valtemp_n10_valobj0",
        "graph_mlp_select01_sphys_valtemp_n10_valobj0.2",
        "graph_sphys_boltzmann_n10_valobj0.2",
        "graph_mlp_select03_sphys_boltzmann_n10_valobj0.1",
    ]
    view = worst[worst["candidate"].isin(focus)].copy()
    rank = {candidate: idx for idx, candidate in enumerate(focus)}
    view["_rank"] = view["candidate"].map(rank)
    view = view.sort_values("_rank")
    cols = [
        "candidate",
        "selected_graph_weight_mean",
        "selected_mlp_weight_mean",
        "selected_sphysnet_weight_mean",
        "worst_positive_transform_roc_auc_mean",
        "worst_positive_transform_brier_mean",
        "worst_positive_transform_nll_mean",
        "worst_positive_transform_false_invariance_mean",
        "worst_positive_transform_pair_order_accuracy_mean",
    ]
    view = view[[col for col in cols if col in view.columns]]
    for col in view.columns:
        view[col] = view[col].map(fmt)
    return (
        "# Phase 1 sPhysNet Positive-View Evaluation\n\n"
        "sPhysNet and sPhysNet-fusion predictions evaluated on identity, atom-renumber, and random-SMILES Tautobase test views. "
        "Worst-positive metrics take the worst value across those three representation-preserving views within each seed before averaging.\n\n"
        + view.to_markdown(index=False)
        + "\n"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-confs", type=int, default=10)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    table_dir = root / "reports" / "tables"
    rows = sphys_baseline_rows(root, args.num_confs)
    rows.extend(fusion_rows(root, args.num_confs))
    raw = pd.DataFrame(rows)
    by_transform = by_transform_summary(raw)
    worst = worst_summary(raw)
    raw.to_csv(table_dir / f"main_sphysnet_numconf{args.num_confs}_positive_view_eval_seed_raw.csv", index=False)
    by_transform.to_csv(table_dir / f"main_sphysnet_numconf{args.num_confs}_positive_view_eval_by_transform.csv", index=False)
    worst.to_csv(table_dir / f"main_sphysnet_numconf{args.num_confs}_positive_view_eval_worst.csv", index=False)
    (table_dir / f"main_sphysnet_numconf{args.num_confs}_positive_view_eval_worst.md").write_text(
        make_markdown(worst),
        encoding="utf-8",
    )
    print(f"wrote sPhysNet positive-view eval num_confs={args.num_confs} raw={len(raw)} worst={len(worst)}")


if __name__ == "__main__":
    main()
