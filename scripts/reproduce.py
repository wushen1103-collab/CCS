"""Recompute the frozen preference, baseline, scaffold and bootstrap results."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from evaluate_fusion import family_bootstrap, metrics

METHOD_LABELS = {"A":"A", "D":"D", "E":"E"}


def check_inventory():
    records = json.loads((ROOT / "data/inventory.json").read_text())
    for record in records:
        path = ROOT / record["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Changed frozen input: {record['path']}")
    return len(records)


def check_metrics(state, expected):
    checks = []
    for (scope, method), group in state.groupby(["scope", "method"]):
        if sorted(group.seed.unique()) != list(range(5)):
            raise ValueError(f"Missing seed records: {scope}/{method}")
        target = expected[(expected.scope == scope) & (expected.method == method)].iloc[0]
        per_seed = [metrics(frame) for _, frame in group.groupby("seed")]
        for endpoint in ("roc_auc", "pair_order_accuracy", "false_invariance", "brier", "nll"):
            values = [row[endpoint] for row in per_seed]
            for name, value in (("mean", np.mean(values)), ("std", np.std(values, ddof=1))):
                error = abs(float(value) - float(target[endpoint + "_" + name]))
                if error > 1e-6:
                    raise ValueError((scope, method, endpoint, name, value, target[endpoint + "_" + name]))
                checks.append({"scope":scope,"method":method,"metric":endpoint,
                               "statistic":name,"value":float(value),"absolute_error":error})
    return pd.DataFrame(checks)


def check_baselines(predictions, reference):
    checks = 0
    mapping = {"roc_auc":"test_roc_auc", "brier":"test_brier", "nll":"test_nll",
               "false_invariance":"false_invariance_rate_test_pairs", "pair_order_accuracy":"pair_order_accuracy"}
    reference = reference[reference.scenario == "all_test"]
    for method, group in predictions.groupby("method"):
        target = reference[reference.model_label == method].iloc[0]
        seed_rows = [metrics(frame) for _, frame in group.groupby("seed")]
        for endpoint, column in mapping.items():
            values = [row[endpoint] for row in seed_rows]
            for statistic, value in (("mean", np.mean(values)), ("std", np.std(values, ddof=1))):
                np.testing.assert_allclose(value, target[column + "_" + statistic], atol=1e-6, rtol=0)
                checks += 1
    return checks


def check_scaffolds(predictions, reference):
    checks = 0
    columns = {"roc_auc":"test_roc_auc", "false_invariance":"false_invariance_rate_test_pairs", "pair_order_accuracy":"pair_order_accuracy"}
    for method in METHOD_LABELS.values():
        group = predictions[predictions.method == method]
        collected = {}
        for seed, frame in group.groupby("seed"):
            count = frame.train_scaffold_count
            pair_count = frame.family_key.map(frame.groupby("family_key").train_scaffold_count.min())
            masks = {"all_test": np.ones(len(frame), dtype=bool),
                     "sample_cold_start_scaffold":count == 0, "sample_seen_scaffold":count > 0,
                     "sample_long_tail_scaffold":count <= 2, "sample_head_scaffold":count >= 5,
                     "pair_any_cold_scaffold":pair_count == 0, "pair_both_seen_scaffold":pair_count > 0,
                     "pair_long_tail_scaffold":pair_count <= 2, "pair_head_scaffold":pair_count >= 5}
            for scenario, mask in masks.items():
                collected.setdefault(scenario, []).append(metrics(frame[mask]))
        for scenario, rows in collected.items():
            target = reference[(reference.model_label == method) & (reference.scenario == scenario)].iloc[0]
            for metric, column in columns.items():
                values = [row[metric] for row in rows]
                for statistic, value in (("mean", np.mean(values)), ("std", np.std(values, ddof=1))):
                    np.testing.assert_allclose(value, target[column + "_" + statistic], atol=1e-6, rtol=0)
                    checks += 1
    return checks


def check_selection(pairs):
    expected = pd.read_csv(ROOT / "data/reference/selective_risk_seed_raw.csv")
    rows = []
    for (scope, method, seed), frame in pairs.groupby(["scope", "method", "seed"]):
        if scope not in ("tautobase_full", "external100") or method not in ("A", "C", "D", "E"):
            continue
        ordered = frame.sort_values(["pair_confidence", "family_key"], ascending=[False, True])
        for coverage in (0.4, 0.6, 0.8, 1.0):
            retained = ordered.head(max(1, int(np.ceil(len(frame)*coverage))))
            rows.append({"scope":scope,"method":method,"seed":seed,"coverage":coverage,
                         "n_pairs":len(retained),"false_invariance":retained.false_invariance.mean(),
                         "pair_order_error":1-retained.pair_order_correct.mean()})
    rebuilt = pd.DataFrame(rows)
    joined = rebuilt.merge(expected, on=["scope","method","seed","coverage"], suffixes=("_new","_reference"), validate="one_to_one")
    if len(joined) != len(expected):
        raise ValueError("Missing selective comparisons")
    for column in ("n_pairs", "false_invariance", "pair_order_error"):
        np.testing.assert_allclose(joined[column+"_new"], joined[column+"_reference"], atol=1e-12, rtol=0)
    return len(joined)


def check_bootstrap(state):
    internal = pd.read_csv(ROOT / "data/frozen/internal_only_state_predictions.csv.gz")
    expected = pd.read_csv(ROOT / "data/reference/bootstrap_effects.csv")
    rows = []
    for scope, source in (("tautobase_clean",state),("external100",state),("nci22",state),("internal_only",internal)):
        comparisons = [("certificate","ordinary")] if scope == "internal_only" else [("E","A"),("E","C"),("E","D")]
        subset = source[source.scope == scope]
        names = {name for pair in comparisons for name in pair}
        frames = {name:{int(seed):frame.copy() for seed,frame in subset[subset.method == name].groupby("seed")} for name in names}
        result = family_bootstrap(frames, scope, comparisons)
        for row in result.to_dict("records"):
            target = expected[(expected.dataset == scope) & (expected.candidate == row["candidate"]) &
                              (expected.baseline == row["baseline"]) & (expected.metric == row["metric"])].iloc[0]
            for column in ("observed_delta","ci95_low","ci95_high"):
                np.testing.assert_allclose(row[column], target[column], atol=1e-9, rtol=0)
            rows.append(row)
        print(f"Verified {scope}: {len(result)} paired metric contrasts", flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", action="store_true", help="Recompute 5,000 paired family draws")
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproduction")
    args = parser.parse_args()
    hashes = check_inventory()
    args.output.mkdir(parents=True, exist_ok=True)
    state = pd.read_csv(ROOT / "data/frozen/state_predictions.csv.gz")
    metric_checks = check_metrics(state, pd.read_csv(ROOT / "data/reference/main_results.csv"))
    metric_checks.to_csv(args.output / "metrics.csv", index=False)
    baseline = pd.read_csv(ROOT / "data/frozen/baseline_predictions.csv.gz")
    scenario = pd.read_csv(ROOT / "data/reference/scenario_results.csv")
    baseline_checks = check_baselines(baseline, scenario)
    scaffold_checks = check_scaffolds(baseline, scenario)
    selection_checks = check_selection(pd.read_csv(ROOT / "data/frozen/pair_predictions.csv.gz"))
    bootstrap_count = 0
    if args.bootstrap:
        bootstrap = check_bootstrap(state)
        bootstrap.to_csv(args.output / "bootstrap.csv", index=False)
        bootstrap_count = len(bootstrap)
    report = {"frozen_files_verified":hashes,"metric_statistics_checked":len(metric_checks),
              "baseline_statistics_checked":baseline_checks,"scaffold_statistics_checked":scaffold_checks,
              "selective_seed_records_checked":selection_checks,
              "bootstrap_metric_contrasts_checked":bootstrap_count,"all_passed":True}
    (args.output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
