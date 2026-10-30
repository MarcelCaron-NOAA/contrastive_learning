# Contrastive Learning for Spatial Precipitation Verification

Research code for developing and evaluating learned spatial-similarity metrics
for precipitation verification. The current study compares three GroupNorm
ResNet-18 contrastive representations:

- **SYN-3** — synthetic supervision using displacement, intensity, and area-size
  perturbations adapted from Wang et al. (2026).
- **SYN-8** — SYN-3 plus five additional morphology perturbation families.
- **NFV** — natural forecast variability supervision using time-lagged and
  forecast-evolution relationships.

The project is currently targeted to **NOAA RDHPCS URSA**. Portability to other
systems is a secondary goal; machine- and user-specific paths are deliberately
kept out of tracked code.

## What this repository reproduces

The cleaned workflow is organized around four primary result/interpretability diagnostic suites:

1. **Controlled morphology response** — learned similarity and traditional
   metrics as precipitation structure is perturbed.
2. **Stage IV real-case verification** — score comparisons plus percentile-based
   alignment/disagreement case galleries.
3. **Forecast-hour response** — full-test score behavior as a function of lead.
4. **Feature-space inversion** — gradient-based constrained perturbations used to
   interpret what changes move each learned representation through similarity
   space.

Time-lag diagnostics used to motivate the NFV training design are also retained.
Historical BatchNorm, FP32, and superseded V1/V2 experiments remain available in
Git history and the `research-snapshot-2026-10-30` tag rather than defining the
final reproduction path.

## Quick start on URSA

### 1. Clone and configure local paths

```bash
git clone <repository-url>
cd contrastive_learning
cp site_config.example.sh site_config.sh
```

Edit **only `site_config.sh`** for your account. At minimum, set:

```bash
export WORK_DIR="/scratch4/NCEPDEV/stmp/<user>/mach_learn/contrastive_learning"
export ENV_DIR="/scratch4/NCEPDEV/ovp/<user>/mach_learn/envs/contrastive_learning"
```

`site_config.sh` is ignored by Git. `PROJECT_DIR` is detected automatically.

### 2. Create the environment

Load the path values you just configured, then create the environment:

```bash
source site_config.sh
module load rdhpcs-conda/25.11.0
conda env create -p "$ENV_DIR" -f environment.yml
```

Then initialize the project in any interactive shell or Slurm job with:

```bash
source load_env.sh
```

The historical development environment exports are retained in
`environment-lock.yml` and `requirements-lock.txt` for reference. They are not
the supported installation route and may include incidental development packages.
The smaller `environment.yml` is the supported setup file; after it is validated
on URSA we can refresh a true reproducibility lock from that clean environment.

### 3. Restore or prepare data

Large data are intentionally not stored in Git. The expected layout is:

```text
$DATA_DIR/
├── forecasts/
│   └── hrrr/
│       ├── qpf6_native/
│       └── qpf6_extended_synoptic/
├── observations/
├── metadata/
└── derived/
```

NOAA/EMC users can restore the frozen 2026 project snapshot from HPSS; see
[`docs/DATA.md`](docs/DATA.md). The repository also retains preprocessing code
for rebuilding the canonical HRRR archive when source data are available.

### 4. Reproduce experiments

Training has now been consolidated to three clear model entry points:

```bash
./scripts/submit_syn.sh syn3 full
./scripts/submit_syn.sh syn8 full
./scripts/submit_nfv.sh
```

Use `smoke` instead of `full` for a short SYN test.  The canonical evaluation
entry points are listed in [`docs/REPRODUCING_RESULTS.md`](docs/REPRODUCING_RESULTS.md).
Canonical diagnostics can now be submitted with `scripts/submit_diagnostic.sh`;
a final end-to-end reproduction driver is added in the next cleanup stage.

## Repository layout

```text
configs/       Experiment/evaluation configuration tracked by Git
docs/          Methods, data, diagnostics, and reproduction notes
scripts/       Training plus data/evaluate/plot workflow entry points
slurm/         URSA Slurm launchers for canonical workflows
src/data/      HRRR archive readers, preprocessing, and sampling
src/eval/      Controlled transforms and traditional verification metrics
src/models/    GroupNorm precipitation encoders and checkpoint loading
src/training/  Shared SYN/NFV training engines plus contrastive/ranking losses
tests/         Lightweight automated tests
```

See [`docs/REPOSITORY_CLEANUP.md`](docs/REPOSITORY_CLEANUP.md) for the current
canonical/supporting/historical classification used during the refactor.

## Data and generated results

Data, checkpoints, experiment outputs, logs, and figures belong under
`$WORK_DIR`, not in the Git repository. The default derived locations are:

```text
$DATA_DIR/derived/experiments/
$DATA_DIR/derived/evaluation/
```

A later cleanup stage will standardize new canonical outputs without rewriting
or deleting the archived historical results.

## Reference

The synthetic-supervision experiments build on the contrastive precipitation
verification framework introduced by Wang et al. (2026), while modifying the
architecture, input data, normalization, and training design for this study.
