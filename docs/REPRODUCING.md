# Reproduction commands

All paths are relative to the repository root. Installation needs no server
account or private filesystem. Statistical reproduction runs on a CPU with the
bundled CSV inputs. Full model training additionally needs the upstream chemical
structures and the training dependencies.

## Frozen numerical checks

```bash
python scripts/reproduce.py --bootstrap
python scripts/rebuild_fusion.py
python scripts/rebuild_fusion.py --refit-weights
```

`reproduce.py` reconstructs 200 A--E mean/SD statistics, the displayed same-split
baseline endpoints, 162 scaffold subgroup statistics, 160 selective seed records,
and 30 paired bootstrap metric contrasts. Bootstrap seed is 20261027, with 5,000
complete-family draws. Paired methods and seeds use shared draw indices. FI
differences are stored as candidate-minus-reference; a reported FI reduction
reverses both the difference and its interval bounds.

`rebuild_fusion.py` rebuilds A--E from individual component scores. The default
uses frozen weights. `--refit-weights` fits weights on validation predictions
within the four fixed policies, using the original simplex grid, objective and
tie rules. No external labels enter weight fitting. The 76-family support is a
fixed subset of the full Tautobase test.

## Internal training

```bash
python scripts/fetch_sources.py --tautobase
python scripts/prepare_data.py
python scripts/train_internal.py --seeds 0 1 2 3 4 --device cpu --threads 4
```

The preparation command verifies sample identities, labels and pair links against
the frozen manifest, then restores its split assignments. Graph-DP uses width
256, message depth 4, dropout 0.1, a complete-pair margin update of weight 0.05,
and validation AUC minus 0.2 FI for checkpoint selection. The MLP uses the
concatenated 2,048-bit chiral Morgan and 2,048-dimensional character-hash inputs,
accepted/residual view aggregation and validation AUC minus 0.1 FI. Both use
AdamW at 0.0003, batch size 256, up to 80 epochs and patience 20. The MLP state
loss has no pair-margin term. Original internal runs initialize each branch
separately per seed. External refits initialize once before Graph-DP and continue
that random state for the MLP.

The reference training stack is Python 3.10, NumPy 1.26.4, pandas 2.2.2,
scikit-learn 1.5.1, PyTorch 2.4.1 and RDKit 2022.9.5. CPU and GPU kernels can
produce different fitted values; the frozen-score path preserves the reported
numbers independently of these differences.

## Classical and directed-message-passing baselines

```bash
python scripts/train_classical.py --datasets tautobase_pref --seeds 0 1 2 3 4 --processed-scope main --n-jobs 4
python scripts/export_chemprop_data.py --dataset tautobase_pref --processed-scope main
python scripts/train_chemprop.py --dataset tautobase_pref --processed-scope main --data-dir data/chemprop_main --mode canonical_positive_aug --seed 0 --run-name main_chemprop_v231_tautobase_5seed --chemprop-bin chemprop
```

The Chemprop adapter targets version 2.3.1. Use a separate environment following
Chemprop's installation instructions. Its reference stack uses Python 3.11,
PyTorch 2.6.0, and RDKit 2026.03.5. The adapter exposes the processed scope,
CLI path, accelerator, seeds, canonical augmentation and fixed split files.
Repeat the training command for seeds 1--4 in the separate Chemprop environment.
Use `--accelerator cpu` when no compatible GPU is available.

## Released experts and external evaluation

```bash
python scripts/fetch_sources.py --experts --nci
python scripts/predict_tautomer_expert.py --seeds 0 1 2 3 4
python scripts/predict_energy_expert.py --num-confs 10 --run-name main_sphysnet_tautobase_numconf10 --summarize
```

Install each expert's own dependencies in a suitable environment. These include
graph-library extensions for the 2D expert and the upstream geometry/runtime
dependencies for the energy expert. The adapters import their code from
`data/raw/external/Tautomer-Predictor-src` and
`data/raw/external/sPhysNet-Taut-src`.

After the same-split expert predictions and internal model runs are available:

```bash
python scripts/fit_fusion.py
python scripts/prepare_evaluation_data.py
python scripts/predict_external_experts.py
python scripts/predict_external_energy.py
python scripts/train_external.py --seed 0
python scripts/evaluate_internal.py
python scripts/evaluate_fusion.py
```

Repeat external internal-model refitting for seeds 1--4 before evaluating the
five-seed external summary. Released-expert adapters preserve deterministic
inference; repeating their records over seed indices does not create independent
expert fits. Energy conversion uses RT = 0.592 kcal/mol, with the published
aqueous predictions and local ten-conformer inference kept distinct.

`fit_fusion.py` fits the fixed B--E specifications by default. Its optional
`--development-sweep` retains the broader development grid and needs the extra
model variants referenced in that grid. The numerical reproduction path and
`rebuild_fusion.py --refit-weights` use only the fixed specifications and make no
policy selection on external data.

## Output locations

The training adapters write relative paths beneath `runs/` and `reports/`.
Download, processed data, model outputs, environment folders and logs are ignored
by Git. Frozen verification writes only beneath `results/`.
