# CCS: CondChemSymm

Code and compact frozen inputs for provenance-aware pair certification in
tautomer preference prediction. CondChemSymm combines molecular evidence and
projects the fused state-logit difference to complementary preference scores.

## Reproduce the numerical results

Python 3.10 is recommended. Run the following from the repository root:

```bash
python -m venv .venv
# Activate the environment using the command for your operating system.
python -m pip install -r requirements-evaluation.txt
python -m pip install . --no-deps
python scripts/reproduce.py
python scripts/reproduce.py --bootstrap
python scripts/rebuild_fusion.py
```

The first evaluation verifies input hashes and reconstructs A--E metrics,
same-split baselines, scaffold subgroup metrics and selective predictions. The
optional bootstrap recomputes 5,000 matched family draws. Fusion reconstruction
starts from individual component scores and frozen weights. To repeat validation
weight fitting within the fixed B--E policies:

```bash
python scripts/rebuild_fusion.py --refit-weights
```

Results are written under `results/`. Frozen inputs contain labels, paired record
identifiers and numerical predictions; molecular structures and large expert
checkpoints are obtained separately from their original sources.

## Train and evaluate the models

```bash
python -m pip install -r requirements-training.txt
python scripts/fetch_sources.py --tautobase
python scripts/prepare_data.py
python scripts/train_internal.py --seeds 0 1 2 3 4 --device cpu --threads 4
```

The split manifest fixes 616 training, 132 validation and 132 test families.
Graph-DP and fingerprint/SMILES-hash MLP training scripts retain the experiment
recipes. Classical models, the Chemprop baseline, released experts, overlap
screening, external inference and fusion are described in
[docs/REPRODUCING.md](docs/REPRODUCING.md).

Exact frozen-score checks are independent of retraining. New training runs can
vary with dependency versions and numerical kernels. The recorded software
versions and fixed seeds make those runs comparable.

## Configurations and evaluation

| ID | Evidence and readout |
|---|---|
| A | Tautomer-Predictor pair normalization |
| B | TP + sPhysNet-Taut, ordinary state-logit fusion |
| C | TP + sPhysNet-Taut + Graph-DP + MLP, ordinary fusion |
| D | Compact two-expert pair-logit certificate, scale 1.0 |
| E | Full four-source pair-logit certificate, scale 1.2 |

The four evaluation scopes contain 132 Tautobase families, a nested 76-family
known-source-excluded subset, 100 aqueous comparisons, and 22 NCI common-support
families. A family is one labeled two-state record. AUC uses both state labels;
pair ordering requires the preferred score to exceed its alternative; false
invariance is the fraction with absolute score gap below 0.25.

The fixed-weight internal comparison isolates the readout. C/E and D/E compare
complete policies, including their selected weights and scales. The Tautobase
comparison was reviewed during development. Policies were frozen before aqueous
and NCI evaluation. The aqueous data had prior published-expert evaluation
history, and the 76-family screen covers accessible released artifacts.

## Repository layout

```text
src/condchemsymm/     Chemical features, graph encoder, metrics and split helpers
scripts/             Training, expert adapters, fusion and reproduction commands
data/frozen/         Structure-free prediction and component tables
data/reference/      Locked metric, bootstrap and audit summaries
data/splits/         Fixed sample/family split assignments
data/sources.json    Public upstream sources and critical file checksums
docs/                Protocol and dependency instructions
```

## Sources and licenses

Repository contributions are provided under the [MIT License](LICENSE).
Third-party source code, datasets and expert weights retain their upstream
licenses. Source links, checksums and the distinction between evaluation targets
and predictor inputs are documented in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).
