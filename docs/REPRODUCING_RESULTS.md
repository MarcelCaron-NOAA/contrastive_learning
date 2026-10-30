# Reproducing the final experiments

This is the supported URSA workflow for the final SYN-3, SYN-8, and NFV study.
Historical development experiments are preserved in Git history and the
`research-snapshot-2026-10-30` tag.

## 0. Initialize the project

```bash
cp site_config.example.sh site_config.sh   # first clone only; edit for account
source load_env.sh
```

See `docs/DATA.md` to restore the frozen data snapshot or rebuild individual
data products.

## 1. Train the final learned metrics

```bash
./scripts/submit_syn.sh syn3 full
./scripts/submit_syn.sh syn8 full
./scripts/submit_nfv.sh
```

Short structural tests:

```bash
./scripts/submit_syn.sh syn3 smoke
./scripts/submit_syn.sh syn8 smoke
```

Canonical engines:

```text
src/training/synthetic.py  -> SYN-3 and SYN-8
src/training/nfv.py        -> NFV
```

NFV retains the historical on-disk experiment ID `hard_ranking_v3_full_v1` so
existing validated checkpoints remain usable.

## 2. Primary diagnostics

### Morphology

```bash
./scripts/submit_diagnostic.sh morphology
```

```text
scripts/evaluate/morphology.py
scripts/plot/morphology.py
slurm/evaluate_morphology.slurm
```

### Stage IV percentile galleries

```bash
./scripts/submit_diagnostic.sh stageiv
```

```text
scripts/evaluate/stageiv.py
scripts/plot/stageiv_buckets.py
slurm/evaluate_stageiv.slurm
```

### Forecast-hour / die-off curves

The Stage-IV forecast-hour archive must exist and pass its preflight first.
Once it does:

```bash
./scripts/submit_diagnostic.sh forecast-hour
```

```text
scripts/evaluate/forecast_hour_traditional.py
scripts/evaluate/forecast_hour.py
scripts/plot/forecast_hour.py
slurm/evaluate_forecast_hour.slurm
```

### Feature-space inversion

```bash
CASE_INDEX=299 N_STEPS=1200 ./scripts/submit_diagnostic.sh inversion
```

```text
src/eval/feature_inversion.py
scripts/evaluate/select_inversion_case.py
scripts/evaluate/inversion.py
scripts/plot/inversion.py
slurm/evaluate_inversion.slurm
```

## 3. NFV training-design / time-lag diagnostics

```text
scripts/evaluate/timelag.py
scripts/plot/timelag.py
scripts/plot/timelag_cases.py
scripts/plot/timelag_negatives.py
scripts/submit_timelag_experiment.sh
slurm/evaluate_timelag.slurm
```

## 4. Traditional thresholds

The final reproducible diagnostics use inch-based precipitation thresholds:

```text
0.1 in = 2.54 mm
0.5 in = 12.70 mm
```

Morphology ETS/POD/SR/FSS use 0.1 inch. Stage-IV FSS and wet-case filtering use
0.1 inch; Stage-IV ETS/FBIAS/POD/SR are retained at both 0.1 and 0.5 inch.
Forecast-hour traditional metrics use 0.1 inch.

The earlier CPU-only `recompute_*_inches.py` scripts are development history;
the canonical evaluators now compute the final inch definitions directly.

## 5. Historical workflows

BatchNorm investigations, FP32 debugging, superseded V1/V2 comparisons, pilot
feature-inversion paths, and older diagnostic generations are not part of the
supported reproduction path. They remain recoverable from the research snapshot
instead of being maintained as parallel live workflows.
