# Contrastive Learning for Spatial Precipitation Verification

Research code for developing and evaluating learned spatial-similarity metrics
for precipitation verification. The final study compares three GroupNorm
ResNet-18 representations:

- **SYN-3** — synthetic supervision using displacement, intensity, and area-size
  perturbations adapted from Wang et al. (2026).
- **SYN-8** — SYN-3 plus five additional morphology perturbation families.
- **NFV** — natural forecast variability supervision using time-lagged and
  forecast-evolution relationships.

The supported workflow is currently targeted to **NOAA RDHPCS URSA**. The goal
of this repository is reproducibility of the research experiment, not a general
cross-platform software package.

## What this repository reproduces

The final workflow covers training plus five diagnostic families:

1. **Controlled morphology response** — learned and traditional metric response
   to controlled precipitation perturbations.
2. **Stage IV real-case verification** — score comparisons plus percentile-rank
   alignment/disagreement galleries.
3. **Forecast-hour response** — full-test score behavior with HRRR lead time.
4. **Feature-space inversion** — constrained gradient-based input optimization
   used to probe the learned representations.
5. **NFV time-lag diagnostics** — supporting analysis of the natural-variability
   training design.

Historical BatchNorm, FP32, V1/V2, and superseded exploratory workflows are
preserved by Git history and the `research-snapshot-2026-10-30` tag rather than
maintained as parallel live code.

## Quick start on URSA

### 1. Clone and configure one local file

```bash
git clone <repository-url>
cd contrastive_learning
cp site_config.example.sh site_config.sh
```

Edit **only `site_config.sh`** for local paths. The normal URSA values are:

```bash
export WORK_DIR="/scratch4/NCEPDEV/stmp/${USER}/mach_learn/contrastive_learning"
export ENV_DIR="/scratch4/NCEPDEV/ovp/${USER}/mach_learn/envs/contrastive_learning"
```

For NOAA/EMC HPSS restoration, also set `HPSS_DATA_ROOT` to the shared snapshot
path documented in [`docs/DATA.md`](docs/DATA.md). `site_config.sh` is ignored
by Git; tracked code contains no user-specific runtime paths.

### 2. Create the environment

```bash
source site_config.sh
module load rdhpcs-conda/25.11.0
conda env create -p "$ENV_DIR" -f environment.yml
source load_env.sh
```

`environment.yml` is the supported URSA environment definition. The larger
`environment-lock.yml` and `requirements-lock.txt` files preserve the historical
development environment for reference only.

### 3. Restore data

The repository contains no large data. For NOAA/EMC users, the smallest useful
restore is the ~37 GB project-support snapshot containing metadata, final
checkpoints, and processed evaluation datasets:

```bash
./scripts/reproduce.sh restore support
```

Retraining SYN-3/SYN-8/NFV additionally requires the multi-year native HRRR
archive (~1.1 TB):

```bash
./scripts/reproduce.sh restore native
```

The smaller extended-synoptic archive is only needed when rebuilding the
forecast-hour archive from source products:

```bash
./scripts/reproduce.sh restore extended
```

See [`docs/DATA.md`](docs/DATA.md) for archive contents and selective year
restore examples.

### 4. Run preflight

```bash
source load_env.sh
./scripts/reproduce.sh preflight diagnostics
```

For a full manifest-vs-disk check before retraining:

```bash
./scripts/reproduce.sh preflight training --verify-stores
```

### 5. Use the reproduction front door

Show the recommended order without launching anything:

```bash
./scripts/reproduce.sh plan
```

Training:

```bash
./scripts/reproduce.sh train syn3 full
./scripts/reproduce.sh train syn8 full
./scripts/reproduce.sh train nfv full
```

Short structural smoke runs:

```bash
./scripts/reproduce.sh train syn3 smoke
./scripts/reproduce.sh train syn8 smoke
./scripts/reproduce.sh train nfv smoke
```

Primary diagnostics:

```bash
./scripts/reproduce.sh diagnostic morphology
./scripts/reproduce.sh diagnostic stageiv
./scripts/reproduce.sh diagnostic forecast-hour
CASE_INDEX=299 N_STEPS=1200 ./scripts/reproduce.sh diagnostic inversion
```

The driver is intentionally a thin dispatcher. It does **not** hide Slurm jobs
or silently chain long runs; each expensive stage stays explicit and auditable.
Detailed commands and outputs are documented in
[`docs/REPRODUCING_RESULTS.md`](docs/REPRODUCING_RESULTS.md).

## Repository layout

```text
configs/       Tracked evaluation/data configuration

docs/          Data, methods, diagnostics, and reproduction notes
scripts/       User-facing setup, training, evaluation, plotting, and utilities
scripts/data/  Derived evaluation-archive construction
scripts/evaluate/  Canonical score/diagnostic calculations
scripts/plot/      Canonical result figures/galleries
slurm/         URSA batch launchers
src/data/      HRRR archive readers, preprocessing, and sampling
src/eval/      Transforms, traditional metrics, model registry, inversion helper
src/models/    GroupNorm precipitation encoders and checkpoint loading
src/training/  Shared SYN/NFV training engines and losses
tests/         Lightweight deterministic/unit tests
```

Generated data, model checkpoints, logs, and figures live under `$WORK_DIR`,
not in Git. Canonical outputs are rooted under:

```text
$DATA_DIR/derived/experiments/
$DATA_DIR/derived/evaluation/
```

## Repository integrity check

After a clean clone and environment setup:

```bash
./scripts/check_repo.sh
```

This runs the unit tests, Python compilation, shell/Slurm syntax checks, and a
scan for personal absolute runtime paths in live code.

## Reference

The synthetic-supervision experiments build on the contrastive precipitation
verification framework introduced by Wang et al. (2026), while modifying the
architecture, input data, normalization, and training design for this study.
