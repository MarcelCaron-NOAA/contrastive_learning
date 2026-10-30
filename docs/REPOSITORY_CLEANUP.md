# Repository cleanup status

The complete exploratory state is preserved by the
`research-snapshot-2026-10-30` tag. The live repository now maintains only code
needed to reproduce the final SYN-3/SYN-8/NFV study plus supporting data-build
and method utilities.

## Completed refactor

### Reproducibility foundation

- one untracked `site_config.sh` for machine/user paths
- URSA-focused environment definition
- frozen-data/HPSS documentation
- removal of personal runtime paths from tracked live code

### Final training

```text
SYN-3 / SYN-8 -> src/training/synthetic.py
NFV           -> src/training/nfv.py
```

SYN-3 and SYN-8 share one validated training engine while retaining their final
sampling, transform, and objective definitions. All three models have symmetric
full/smoke launch paths.

### Final diagnostics

```text
scripts/data/       derived evaluation-archive construction
scripts/evaluate/   score calculation / diagnostic experiments
scripts/plot/       final figures and galleries
```

Primary maintained diagnostics are morphology, Stage IV percentile galleries,
forecast-hour curves, feature-space inversion, and NFV time-lag design analysis.

### Historical workflow pruning

BatchNorm/FP32 investigations, V1/V2 comparisons, old Wang diagnostic
generations, pilot inversion paths, superseded Stage-IV/morphology pipelines,
and temporary repair scripts were removed from the visible branch after the
canonical replacements were validated on URSA.

### Reproduction interface

The final cleanup adds:

```text
scripts/reproduce.sh               small documented dispatcher
scripts/preflight.py               setup/data/checkpoint validation
scripts/restore_hpss_snapshot.sh   selective frozen-data restore
scripts/check_repo.sh              clean-clone code/syntax checks
```

The dispatcher intentionally does not auto-chain expensive Slurm jobs.

## Supporting workflows retained

Utilities remain for HRRR/Stage-IV preprocessing, archive validation,
normalization/split construction, transform schematics, and targeted audits.
These support rebuilding or inspecting canonical inputs without cluttering the
primary training/evaluation path.

## Final promotion checklist

1. Apply and commit the reproduction-interface patch.
2. Run `./scripts/check_repo.sh` on URSA.
3. Run `./scripts/reproduce.sh preflight diagnostics` against the frozen data.
4. Optionally validate a fresh SYN/NFV smoke job from the front door.
5. Push `refactor/reproducible-v1`, merge to `develop`, then promote the tested
   state to `main`.

Figure aesthetics and presentation-specific refinements can continue after this
structural reproducibility milestone without reopening the repository cleanup.
