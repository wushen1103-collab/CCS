"""Run the fixed Graph-DP and fingerprint-MLP training recipes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds",nargs="+",type=int,default=[0,1,2,3,4])
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--threads",type=int,default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT/"src") + os.pathsep + environment.get("PYTHONPATH", "")
    for variable in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        environment[variable] = str(args.threads)
    if args.device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    for seed in args.seeds:
        common = ["--dataset","tautobase_pref","--processed-scope","main","--mode","mechanistic_gate",
                  "--seed",str(seed),"--epochs","80","--patience","20","--batch-size","256","--lr","0.0003"]
        graph = [sys.executable,str(ROOT/"scripts/train_graph.py"),*common,
                 "--run-name","main_graph_b256_pairm005_select02_5seed","--hidden","256","--depth","4",
                 "--pair-margin-lambda","0.05","--pair-margin","1.0","--selection-false-inv-weight","0.2"]
        mlp = [sys.executable,str(ROOT/"scripts/train_fingerprint.py"),*common,
               "--run-name","main_concat_select01_viewpred_5seed","--feature-kind","concat",
               "--hidden","512","--selection-false-inv-weight","0.1","--save-positive-view-predictions"]
        for command in (graph,mlp):
            subprocess.run(command,cwd=ROOT,env=environment,check=True)


if __name__ == "__main__":
    main()
