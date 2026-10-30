# Repository cleanup map

This is the working classification for the reproducibility refactor. It is intentionally conservative: no file marked `HISTORY` should be deleted until its canonical replacement has been exercised on URSA. The `research-snapshot-2026-10-30` tag preserves the complete development state.

## Meaning

- **CANONICAL** — part of the final data/training/evaluation reproduction path. Keep, rename, or consolidate.
- **SUPPORT** — useful scientific/methodological utility that should remain available but does not need to be a front-door command.
- **HISTORY** — development/debugging/superseded workflow. Remove from the clean branch after replacements are validated; recoverable from Git history/tag.

## Scripts

| File | Disposition | Planned action |
|---|---|---|
| `scripts/analyze_precip_transforms.py` | SUPPORT | Keep as a secondary utility/method figure; remove hard-coded paths. |
| `scripts/audit_area_size_response.py` | SUPPORT | Keep as a secondary utility/method figure; remove hard-coded paths. |
| `scripts/audit_project_data.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/audit_stage4_raw.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/audit_training_cues.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/benchmark_hrrr_store.py` | SUPPORT | Keep as a secondary utility/method figure; remove hard-coded paths. |
| `scripts/build_dataset_index.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/build_hrrr_manifest.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/build_stageiv_forecast_hour_archive.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/build_stageiv_reference_archive.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/build_wang_transform_archive.py` | SUPPORT | Keep as a secondary utility/method figure; remove hard-coded paths. |
| `scripts/compare_hard_ranking_margins.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/compute_precip_normalization_stats.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/compute_stageiv_forecast_hour_traditional_metrics.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/dataloader_io_benchmark.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/diagnose_batchnorm_swap.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/diagnose_bn_batch_stats.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/diagnose_stageiv_domain_transfer.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/evaluate_bnrecal_failure_tail.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/evaluate_bnrecal_timelag_fp32.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/evaluate_failure_tail.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/evaluate_final_morphology_models.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/evaluate_final_stageiv_models.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/evaluate_fixed_validation_diagnostics.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/evaluate_fp32_failure_tail.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/evaluate_full_epoch_failure_tail.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/evaluate_full_epoch_timelag_fp32.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/evaluate_layerwise_transform_sensitivity.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/evaluate_morphology_perturbations.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/evaluate_morphology_synthetic_vs_v3.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/evaluate_morphology_v2_v3.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/evaluate_signed_morphology_v2.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/evaluate_stageiv_reference.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/evaluate_stageiv_synthetic_vs_v3.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/evaluate_stageiv_v1_v2.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/evaluate_stageiv_v1_v2_v3.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/evaluate_timelag_fp32.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/evaluate_timelag_gradation.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/evaluate_timelag_gradation_fp32.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/evaluate_validation_similarity_pairs.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/evaluate_wang_transformations.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/evaluate_wang_v2.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/feature_inversion_common.py` | CANONICAL | Preserve as primary interpretability diagnostic; consolidate pilot/smoke history. |
| `scripts/finalize_hrrr_extended_synoptic_archive.sh` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/finish_timelag_gradation.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/inspect_hrrr_extended_store.py` | SUPPORT | Keep as a secondary utility/method figure; remove hard-coded paths. |
| `scripts/make_hrrr_date_list.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/make_similarity_case_gallery.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/make_stageiv_forecast_hour_manifest.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/make_stageiv_pair_manifest.py` | SUPPORT | Keep as a secondary utility/method figure; remove hard-coded paths. |
| `scripts/minimal_model_loss_test.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/plot_bn_vs_groupnorm.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/plot_feature_inversion_continuation.py` | CANONICAL | Preserve as primary interpretability diagnostic; consolidate pilot/smoke history. |
| `scripts/plot_feature_inversion_pilot.py` | HISTORY | Superseded pilot/smoke path; retain only in snapshot. |
| `scripts/plot_final_morphology_models.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/plot_final_stageiv_buckets.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/plot_full_epoch_history.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/plot_hard_ranking_finetune.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/plot_hard_ranking_v2_full.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/plot_hard_ranking_v3_full.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/plot_lead_consistency_experiment.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/plot_morphology_syn3_traditional.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/plot_morphology_syn3_vs_syn8.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/plot_morphology_synthetic_vs_v3.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/plot_morphology_v2_v3_wang_style.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/plot_quadrant_transform_schematic_panels.py` | SUPPORT | Keep as a secondary utility/method figure; remove hard-coded paths. |
| `scripts/plot_signed_morphology_wang_style.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/plot_similarity_decomposition.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/plot_stageiv_alignment_check.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/plot_stageiv_case_examples_v2.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/plot_stageiv_case_examples_v3.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/plot_stageiv_case_triptychs.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/plot_stageiv_galleries.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/plot_stageiv_scores_by_forecast_hour.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/plot_stageiv_syn3_galleries.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/plot_stageiv_synthetic_galleries.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/plot_syn3_wang_training.py` | SUPPORT | Keep as a secondary utility/method figure; remove hard-coded paths. |
| `scripts/plot_syn8_wang_training.py` | SUPPORT | Keep as a secondary utility/method figure; remove hard-coded paths. |
| `scripts/plot_synthetic_perturbation_training.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/plot_timelag_case_examples.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/plot_timelag_experiment.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/plot_timelag_negative_examples.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/plot_train_validation_diagnostics.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/plot_transform_schematic_panels.py` | SUPPORT | Keep as a secondary utility/method figure; remove hard-coded paths. |
| `scripts/plot_validation_by_group.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/plot_wang_raw_scatter.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/plot_wang_style.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/plot_wang_v2.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/preflight_first_real.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/preflight_hrrr_extended_synoptic.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/preflight_stageiv_forecast_hour_archive.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/prepare_stageiv_forecast_hour_manifest.sh` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/prepare_timelag_grid_mapping.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/preprocess_hrrr_day.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/preprocess_hrrr_extended_synoptic_day.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/recalibrate_batchnorm.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/recompute_final_morphology_traditional_inches.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/recompute_final_stageiv_metrics_inches.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/run_feature_inversion.py` | HISTORY | Superseded pilot/smoke path; retain only in snapshot. |
| `scripts/run_feature_inversion_continuation.py` | CANONICAL | Preserve as primary interpretability diagnostic; consolidate pilot/smoke history. |
| `scripts/run_lead_consistency_experiment.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/run_stageiv_case_panels.sh` | SUPPORT | Keep as a secondary utility/method figure; remove hard-coded paths. |
| `scripts/run_timelag_experiment.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/select_feature_inversion_case.py` | CANONICAL | Preserve as primary interpretability diagnostic; consolidate pilot/smoke history. |
| `scripts/select_stageiv_case_buckets.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/smoke_test_sampler.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/stage4_extract_day.sh` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/submit_feature_inversion_continuation.sh` | CANONICAL | Preserve as primary interpretability diagnostic; consolidate pilot/smoke history. |
| `scripts/submit_feature_inversion_pilot.sh` | HISTORY | Superseded pilot/smoke path; retain only in snapshot. |
| `scripts/submit_feature_inversion_smoke.sh` | HISTORY | Superseded pilot/smoke path; retain only in snapshot. |
| `scripts/submit_feature_inversion_smoke_v2.sh` | HISTORY | Superseded pilot/smoke path; retain only in snapshot. |
| `scripts/submit_final_diagnostics.sh` | SUPPORT | Keep as a secondary utility/method figure; remove hard-coded paths. |
| `scripts/submit_hard_ranking_v3_full.sh` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/submit_hrrr_extended_synoptic_preprocessing.sh` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/submit_hrrr_extended_synoptic_smoke.sh` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/submit_hrrr_preprocessing.sh` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/submit_lead_consistency_experiment.sh` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/submit_representation_characterization.sh` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/submit_stage4_download.sh` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/submit_syn3_wang_full.sh` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/submit_syn3_wang_smoke.sh` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/submit_syn8_wang_full.sh` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/submit_syn8_wang_smoke.sh` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/submit_synthetic_perturbation.sh` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/submit_timelag_experiment.sh` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/submit_v2_external_diagnostics.sh` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/train_first_real.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/train_full_epoch.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/train_groupnorm.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/train_groupnorm_pilot.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/train_hard_ranking_finetune.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/train_hard_ranking_v3_full.py` | CANONICAL | Rename/clean as NFV trainer; preserve objective. |
| `scripts/train_syn3_wang_adapted.py` | CANONICAL | Consolidate into shared SYN trainer + SYN-3/SYN-8 configs. |
| `scripts/train_syn8_wang_adapted.py` | CANONICAL | Consolidate into shared SYN trainer + SYN-3/SYN-8 configs. |
| `scripts/train_synthetic_perturbation.py` | HISTORY | Remove from clean branch after canonical workflow passes smoke tests. |
| `scripts/validate_hrrr_archive.py` | CANONICAL | Move into a clear data/train/evaluate/plot entry-point group and remove hard-coded paths. |
| `scripts/validate_hrrr_zarr_vs_grib.py` | SUPPORT | Keep as a secondary utility/method figure; remove hard-coded paths. |

## Slurm launchers

| File | Disposition | Planned action |
|---|---|---|
| `slurm/build_stageiv_forecast_hour_archive.slurm` | CANONICAL | Keep functionality, simplify naming, source central environment/config. |
| `slurm/compute_precip_normalization_stats.slurm` | CANONICAL | Keep functionality, simplify naming, source central environment/config. |
| `slurm/continue_groupnorm_full_epoch.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/dataloader_io_benchmark.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/download_stage4.slurm` | CANONICAL | Keep functionality, simplify naming, source central environment/config. |
| `slurm/lead_consistency.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/minimal_model_loss_test.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/preprocess_hrrr_day.slurm` | CANONICAL | Keep functionality, simplify naming, source central environment/config. |
| `slurm/preprocess_hrrr_extended_synoptic_day.slurm` | CANONICAL | Keep functionality, simplify naming, source central environment/config. |
| `slurm/recompute_final_inches.slurm` | CANONICAL | Keep functionality, simplify naming, source central environment/config. |
| `slurm/resume_hard_ranking_finetune.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/resume_hard_ranking_v3_full.slurm` | CANONICAL | Keep functionality, simplify naming, source central environment/config. |
| `slurm/run_all_synthetic_diagnostics.slurm` | SUPPORT | Retain only if still useful after final diagnostic wrappers exist. |
| `slurm/run_bn_batchstat_diagnostic.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/run_bn_recalibration_test.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/run_bnrecal_physical_diagnostics.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/run_feature_inversion_continuation.slurm` | CANONICAL | Keep continuation launcher; retire pilot/smoke launchers. |
| `slurm/run_feature_inversion_pilot.slurm` | HISTORY | Retire after continuation workflow is validated. |
| `slurm/run_feature_inversion_smoke.slurm` | HISTORY | Retire after continuation workflow is validated. |
| `slurm/run_final_morphology_models.slurm` | CANONICAL | Keep functionality, simplify naming, source central environment/config. |
| `slurm/run_final_stageiv_buckets.slurm` | CANONICAL | Keep functionality, simplify naming, source central environment/config. |
| `slurm/run_fp32_failure_diagnostics.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/run_full_epoch_core_diagnostics.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/run_groupnorm_full_diagnostics.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/run_last_checkpoint_comparison.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/run_layerwise_transform_sensitivity.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/run_morphology_perturbations.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/run_morphology_synthetic_vs_v3.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/run_morphology_v2_v3.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/run_next_similarity_diagnostics.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/run_ranking_margin_v1_v2.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/run_signed_morphology_v2.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/run_stageiv_case_examples_v2.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/run_stageiv_domain_transfer_diagnostics.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/run_stageiv_reference_build.slurm` | CANONICAL | Keep functionality, simplify naming, source central environment/config. |
| `slurm/run_stageiv_reference_eval.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/run_stageiv_scores_by_forecast_hour.slurm` | CANONICAL | Keep functionality, simplify naming, source central environment/config. |
| `slurm/run_stageiv_synthetic_galleries.slurm` | SUPPORT | Retain only if still useful after final diagnostic wrappers exist. |
| `slurm/run_stageiv_v1_v2.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/run_stageiv_v3_galleries.slurm` | SUPPORT | Retain only if still useful after final diagnostic wrappers exist. |
| `slurm/run_syn3_diagnostics.slurm` | SUPPORT | Retain only if still useful after final diagnostic wrappers exist. |
| `slurm/run_training_cue_audit.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/run_validation_diagnostics.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/run_wang_transform_eval.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/run_wang_v2.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/test_gpu.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/timelag_experiment.slurm` | CANONICAL | Keep functionality, simplify naming, source central environment/config. |
| `slurm/train_first_real.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/train_full_epoch.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/train_groupnorm_pilot.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/train_hard_ranking_finetune.slurm` | HISTORY | Remove from clean branch after replacement validation. |
| `slurm/train_hard_ranking_v3_full.slurm` | CANONICAL | Rename/clean as NFV launcher. |
| `slurm/train_syn3_wang.slurm` | CANONICAL | Consolidate around the shared SYN trainer. |
| `slurm/train_syn8_wang.slurm` | CANONICAL | Consolidate around the shared SYN trainer. |
| `slurm/train_synthetic_perturbation.slurm` | HISTORY | Remove from clean branch after replacement validation. |

## `src/` guidance

The core library is comparatively small and should not be aggressively rewritten during structural cleanup.

- Keep `src/data/` readers, preprocessing, the hard-ranking sampler, and the canonical index/sampling utilities.
- Keep `src/eval/` transforms and verification metrics; later give the final eight morphology transforms one documented API.
- Keep `src/models/precip_resnet_gn.py` and `checkpoint_loader.py`.
- Treat the older BatchNorm `src/models/precip_resnet.py` as historical once no canonical workflow imports it.
- Keep the contrastive/ranking losses under `src/training/`; add the shared SYN trainer here rather than duplicating experiment scripts.

## Refactor order

1. Foundation: site configuration, environment, documentation, repo hygiene.
2. Training: shared SYN trainer/configs; rename hard-ranking V3 as NFV without changing science.
3. Diagnostics: consolidate morphology, Stage IV, forecast-hour, inversion, and TL workflows.
4. Reproduction front door and URSA smoke tests.
5. Only then remove `HISTORY` files from the clean branch.
