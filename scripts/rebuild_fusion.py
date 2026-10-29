"""Rebuild frozen A--E predictions from component scores and fusion weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from condchemsymm.fusion import choose_candidate, policy_predictions, simplex_weights

SPEC = {
    "B":(["tp_pairnorm","sphys_boltzmann"],"logit_avg",0.0),
    "C":(["tp_pairnorm","graph_main","mlp_select01","sphys_boltzmann"],"logit_avg",0.0),
    "D":(["tp_pairnorm","sphys_boltzmann"],"pair_logitdiff_s1",0.5),
    "E":(["tp_pairnorm","graph_main","mlp_select01","sphys_boltzmann"],"pair_logitdiff_s1p2",0.5),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refit-weights", action="store_true", help="Refit validation weights within the four fixed policies")
    args = parser.parse_args()
    main_components = pd.read_csv(ROOT / "data/frozen/main_components.csv.gz")
    external = pd.read_csv(ROOT / "data/frozen/external_components.csv.gz")
    frozen_weights = pd.read_csv(ROOT / "data/frozen/frozen_weights.csv")
    expected = pd.read_csv(ROOT / "data/frozen/state_predictions.csv.gz")
    output = ROOT / "results/rebuilt_fusion"
    output.mkdir(parents=True, exist_ok=True)
    weights = {}
    for (method,seed), group in frozen_weights.groupby(["method","seed"]):
        columns, policy, penalty = SPEC[method]
        selected = group.iloc[0][columns].to_numpy(dtype=float)
        if args.refit_weights:
            validation = main_components[(main_components.seed == seed) & (main_components.split == "val")]
            denom = 50 if len(columns) <= 3 else 25
            selected, _, _ = choose_candidate(validation, columns, simplex_weights(len(columns),denom), policy, penalty)
        weights[(method,int(seed))] = selected
    components = main_components[main_components.split == "test"].copy()
    components["scope"] = "tautobase_full"
    external["scope"] = external.scope.replace({"sphysnet_100_aqueous":"external100","nci_aqueous_zero_overlap":"nci22"})
    all_components = pd.concat([components,external],ignore_index=True)
    outputs = []
    for (scope,seed), frame in all_components.groupby(["scope","seed"]):
        for method in ("A","B","C","D","E"):
            result = frame[["sample_id","paired_id","family_key","y"]].copy()
            if method == "A":
                prediction = frame.tp_pairnorm.to_numpy(dtype=float)
            else:
                columns,policy,_ = SPEC[method]
                prediction = policy_predictions(frame,columns,weights[(method,int(seed))][None,:],policy)[0]
            result["scope"],result["method"],result["seed"],result["prediction"] = scope,method,seed,prediction
            outputs.append(result)
    rebuilt = pd.concat(outputs,ignore_index=True)
    clean_ids = set(expected[expected.scope == "tautobase_clean"].sample_id)
    clean = rebuilt[(rebuilt.scope == "tautobase_full") & rebuilt.sample_id.isin(clean_ids)].copy()
    clean["scope"] = "tautobase_clean"
    rebuilt = pd.concat([rebuilt,clean],ignore_index=True)
    joined = rebuilt.merge(expected,on=["scope","method","seed","sample_id"],suffixes=("_rebuilt","_frozen"),validate="one_to_one")
    if len(joined) != len(expected):
        raise ValueError("Missing frozen prediction rows")
    error = float(np.max(np.abs(joined.prediction_rebuilt-joined.prediction_frozen)))
    np.testing.assert_allclose(joined.prediction_rebuilt,joined.prediction_frozen,atol=1e-6,rtol=0)
    rebuilt.to_csv(output/"state_predictions.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    report = {"prediction_rows_checked":len(joined),"maximum_error":error,"validation_weights_refitted":args.refit_weights,"all_passed":True}
    (output/"verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
