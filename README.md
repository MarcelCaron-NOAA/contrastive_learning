# Contrastive Learning for Spatial Forecast Verification

Research project investigating learned representations of meteorological
forecast similarity, with initial emphasis on precipitation fields and
time-lagged forecast pairs.

## Environment

Load the project environment with:

```bash
source load_env.sh
```

## Project directory

`/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning`

## Working directories

Large data, logs, and scratch files are kept under:
`/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning`
with subdirectories:
- data/
- logs/
cat > README.md <<'EOF'
# Contrastive Learning for Spatial Forecast Verification

Research project investigating learned representations of meteorological
forecast similarity, with initial emphasis on precipitation fields and
time-lagged forecast pairs.

## Environment

Load the project environment with:

```bash
source load_env.sh
```

## Project directory

`/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning`

## Working directories

Large data, logs, and scratch files are kept under:
`/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning`
with subdirectories:
- data/
- logs/
- scratch/

## Project structure
- `configs/` - experiment and model configuration files
- `notebooks/` - exploratory notebooks
- `scripts/` - executable Python workflows
- `slurm/` - Slurm batch scripts
- `src/data/` - data loading and preprocessing
- `src/metrics/` - verification and similarity metrics
- `src/models/` - neural-network architectures
- `src/training/` - losses and training code
- `src/utils/` - shared utilities
- `tests/` - automated tests
