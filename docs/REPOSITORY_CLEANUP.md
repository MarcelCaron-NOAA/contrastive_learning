# Repository cleanup status

The complete development state is preserved by the
`research-snapshot-2026-10-30` tag. The clean branch therefore maintains only
workflows needed to reproduce the final SYN-3/SYN-8/NFV study plus a small set
of supporting method/data utilities.

## Completed

### Patch 1 — reproducibility foundation

- one local `site_config.sh` for machine/user paths
- URSA-focused environment specification
- data/reproduction documentation
- removal of personal paths from canonical configuration

### Patch 2 / 2a — final training

```text
SYN-3 / SYN-8
  src/training/synthetic.py
  scripts/train_syn.py
  scripts/submit_syn.sh
  slurm/train_syn.slurm

NFV
  src/training/nfv.py
  scripts/train_nfv.py
  scripts/submit_nfv.sh
  slurm/train_nfv.slurm
```

SYN-3 and SYN-8 now share one training engine while retaining their validated
sampling and transform definitions.

### Patch 3 — final diagnostics

Canonical diagnostic code is organized by purpose:

```text
scripts/data/       derived-data/archive preparation
scripts/evaluate/   score calculation / diagnostic experiments
scripts/plot/       presentation-ready figures and galleries
```

Primary workflows:

```text
morphology      scripts/evaluate/morphology.py
                scripts/plot/morphology.py
                slurm/evaluate_morphology.slurm

Stage IV        scripts/evaluate/stageiv.py
                scripts/plot/stageiv_buckets.py
                slurm/evaluate_stageiv.slurm

forecast hour   scripts/evaluate/forecast_hour_traditional.py
                scripts/evaluate/forecast_hour.py
                scripts/plot/forecast_hour.py
                slurm/evaluate_forecast_hour.slurm

inversion       src/eval/feature_inversion.py
                scripts/evaluate/inversion.py
                scripts/plot/inversion.py
                slurm/evaluate_inversion.slurm

time lag        scripts/evaluate/timelag.py
                scripts/plot/timelag*.py
                slurm/evaluate_timelag.slurm
```

Final traditional verification defaults are now encoded directly in the
canonical evaluators: 0.1 inch (2.54 mm), with 0.5 inch retained as the second
Stage-IV categorical threshold. The temporary post-processing scripts that
recomputed inch-based metrics are therefore historical.

## Patch 3b — historical workflow pruning

The canonical Patch-3 diagnostics were validated on URSA before pruning. The
following development-only families were removed from the visible branch:

- BatchNorm and FP32 investigations
- V1/V2 learned-metric comparisons
- old Wang/V2 diagnostic generations
- pilot/smoke feature-space inversion workflows
- superseded Stage-IV and morphology generations
- temporary inch-threshold recomputation scripts
- superseded pre-final training workflows and launchers

All of these remain recoverable from `research-snapshot-2026-10-30`.

## Supporting workflows retained

The clean branch still retains utilities needed to build or inspect the canonical
data products, including HRRR/Stage-IV preprocessing, archive validation,
normalization/split construction, transform schematics, and targeted audits.

## Remaining cleanup

1. Add the simple end-to-end `reproduce.sh` front door.
2. Consolidate the remaining data-preparation entry points and documentation.
3. Run final clean-clone smoke tests.
4. Promote the validated branch through `develop` to `main`.
