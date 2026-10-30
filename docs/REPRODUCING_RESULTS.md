# Reproducing the final experiments

This document identifies the current canonical workflows while the repository is
being consolidated. File names will become shorter in later refactor stages, but
the scientific experiment definitions should not change.

## 0. Initialize URSA environment

```bash
cp site_config.example.sh site_config.sh   # first clone only; then edit paths
source load_env.sh
```

## 1. Data integrity / preprocessing

Useful canonical data tools:

```text
scripts/audit_project_data.py
scripts/validate_hrrr_archive.py
scripts/preprocess_hrrr_day.py
scripts/preprocess_hrrr_extended_synoptic_day.py
scripts/build_hrrr_manifest.py
scripts/build_dataset_index.py
scripts/compute_precip_normalization_stats.py
```

The frozen HPSS snapshot can be restored instead of rebuilding the full HRRR
archive. See `docs/DATA.md`.

## 2. Train the final learned metrics

Current canonical training entry points:

```text
SYN-3  scripts/train_syn3_wang_adapted.py
       scripts/submit_syn3_wang_full.sh
       slurm/train_syn3_wang.slurm

SYN-8  scripts/train_syn8_wang_adapted.py
       scripts/submit_syn8_wang_full.sh
       slurm/train_syn8_wang.slurm

NFV    scripts/train_hard_ranking_v3_full.py
       scripts/submit_hard_ranking_v3_full.sh
       slurm/train_hard_ranking_v3_full.slurm
```

SYN-3 and SYN-8 will be consolidated into a shared synthetic trainer in the next
cleanup stage; this document deliberately records the current working entry
points first.

## 3. Primary diagnostics

### Controlled morphology response

```text
scripts/evaluate_final_morphology_models.py
scripts/recompute_final_morphology_traditional_inches.py
scripts/plot_final_morphology_models.py
slurm/run_final_morphology_models.slurm
```

### Stage IV comparison and percentile galleries

```text
scripts/evaluate_final_stageiv_models.py
scripts/recompute_final_stageiv_metrics_inches.py
scripts/plot_final_stageiv_buckets.py
slurm/run_final_stageiv_buckets.slurm
```

### Forecast-hour / die-off behavior

```text
scripts/make_stageiv_forecast_hour_manifest.py
scripts/build_stageiv_forecast_hour_archive.py
scripts/compute_stageiv_forecast_hour_traditional_metrics.py
scripts/plot_stageiv_scores_by_forecast_hour.py
slurm/build_stageiv_forecast_hour_archive.slurm
slurm/run_stageiv_scores_by_forecast_hour.slurm
```

### Feature-space inversion

Feature-space inversion is a **primary interpretability diagnostic** in this
study, not merely a development test.

```text
scripts/feature_inversion_common.py
scripts/select_feature_inversion_case.py
scripts/run_feature_inversion_continuation.py
scripts/plot_feature_inversion_continuation.py
scripts/submit_feature_inversion_continuation.sh
slurm/run_feature_inversion_continuation.slurm
```

The continuation experiment optimizes the precipitation input while holding the
trained network fixed and targets prescribed cosine-similarity levels.

## 4. NFV training-design / time-lag diagnostics

These are retained because they document the empirical motivation for the NFV
positive/time-lag construction:

```text
scripts/prepare_timelag_grid_mapping.py
scripts/run_timelag_experiment.py
scripts/plot_timelag_experiment.py
scripts/plot_timelag_case_examples.py
scripts/plot_timelag_negative_examples.py
slurm/timelag_experiment.slurm
```

## 5. Historical experiments

BatchNorm investigations, FP32 debugging, superseded V1/V2 comparisons, and
other exploratory diagnostics are valuable research history but are not part of
the final reproduction path. They are preserved by Git history and the
`research-snapshot-2026-10-30` tag while being removed from the visible workflow
in later cleanup patches.
