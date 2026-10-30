# Reproducing the final SYN-3 / SYN-8 / NFV study

This is the supported URSA workflow. Historical development experiments are
preserved by Git history and `research-snapshot-2026-10-30`.

## 0. Setup

```bash
cp site_config.example.sh site_config.sh   # first clone only
# edit WORK_DIR, ENV_DIR, and optionally HPSS_DATA_ROOT
source load_env.sh
```

For NOAA/EMC restoration details, see `docs/DATA.md`.

The compact front door is:

```bash
./scripts/reproduce.sh plan
```

It only dispatches to the canonical commands below; it does not hide or chain
Slurm jobs.

## 1. Validate the repository and local data

Repository code/syntax checks:

```bash
./scripts/check_repo.sh
```

Final diagnostic products/checkpoints:

```bash
./scripts/reproduce.sh preflight diagnostics
```

Full training archive:

```bash
./scripts/reproduce.sh preflight training --verify-stores
```

## 2. Train the final learned metrics

```bash
./scripts/reproduce.sh train syn3 full
./scripts/reproduce.sh train syn8 full
./scripts/reproduce.sh train nfv full
```

Short non-destructive smoke jobs use unique output directories:

```bash
./scripts/reproduce.sh train syn3 smoke
./scripts/reproduce.sh train syn8 smoke
./scripts/reproduce.sh train nfv smoke
```

Canonical engines:

```text
src/training/synthetic.py  -> SYN-3 and SYN-8
src/training/nfv.py        -> NFV
```

NFV retains the historical on-disk experiment ID `hard_ranking_v3_full_v1` so
the validated final checkpoint remains directly usable.

## 3. Primary diagnostics

### Controlled morphology response

```bash
./scripts/reproduce.sh diagnostic morphology
```

Outputs:

```text
$DATA_DIR/derived/evaluation/final_syn_nfv_morphology_v1/
  morphology_per_case_inches.csv.gz
  morphology_summary_inches.csv
  plots/
```

### Stage IV percentile galleries

```bash
./scripts/reproduce.sh diagnostic stageiv
```

Outputs:

```text
$DATA_DIR/derived/evaluation/final_syn_nfv_stageiv_v1/
  stageiv_per_case_metrics_inches.csv.gz
  galleries_inches/
```

### Forecast-hour / die-off curves

The processed archive must have a `PASS` preflight report. Then:

```bash
./scripts/reproduce.sh diagnostic forecast-hour
```

Outputs:

```text
$DATA_DIR/derived/evaluation/stageiv_forecast_hour_v1/
  traditional_metrics_by_forecast_hour.csv
  learned_similarity_per_case.csv.gz
  verification_scores_by_forecast_hour.csv
  plots/
```

### Feature-space inversion

```bash
CASE_INDEX=299 N_STEPS=1200 ./scripts/reproduce.sh diagnostic inversion
```

The canonical targets are:

```text
+0.75, +0.50, +0.25, 0.00, -0.25, -0.50, -0.75, -1.00
```

Each case is isolated under:

```text
$DATA_DIR/derived/evaluation/feature_inversion_v1/case_XXXX/
```

### NFV time-lag design analysis

```bash
./scripts/reproduce.sh diagnostic timelag
```

This remains an array-job research diagnostic because it evaluates many
Stage-IV/HRRR relationships and supports the NFV training design rather than the
routine final-model verification scores.

## 4. Replot existing evaluation output

The main non-inversion figures can be regenerated without rerunning model
inference:

```bash
./scripts/reproduce.sh plot morphology
./scripts/reproduce.sh plot stageiv
./scripts/reproduce.sh plot forecast-hour
```

Inversion plotting remains case-specific; use `scripts/plot/inversion.py` with
the desired case directory.

## 5. Final traditional thresholds

Canonical verification thresholds are encoded directly in the final evaluators:

```text
0.1 in = 2.54 mm
0.5 in = 12.70 mm
```

Morphology ETS/POD/SR/FSS use 0.1 inch. Stage-IV FSS and non-dry filtering use
0.1 inch; Stage-IV categorical metrics retain both 0.1 and 0.5 inch. The
forecast-hour traditional metrics use 0.1 inch.

## 6. Expected separation of responsibilities

```text
restore/build data
      ↓
preflight
      ↓
train -> checkpoints/history
      ↓
evaluate -> score tables / diagnostic data
      ↓
plot -> figures / galleries
```

The Slurm diagnostic wrappers often run evaluate + plot consecutively for
convenience, but the Python entry points remain separate so figures can be
regenerated from existing score products.
