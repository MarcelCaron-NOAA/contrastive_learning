# Contrastive-learning diagnostics v1

Copy into the existing repository:

```text
scripts/plot_train_validation_diagnostics.py
scripts/evaluate_fixed_validation_diagnostics.py
scripts/plot_validation_by_group.py
slurm/run_validation_diagnostics.slurm
```

## A. Training/validation trajectory plots

No GPU needed:

```bash
source load_env.sh
python scripts/plot_train_validation_diagnostics.py
```

Outputs to:

```text
$DATA_DIR/derived/experiments/first_real_train_validation_v1/diagnostic_plots/
```

## B. Per-example fixed-validation diagnostics

This reruns the fixed 512 validation recipes through `checkpoint_best.pt`
(default) and writes one CSV row per validation example.

Recommended GPU job:

```bash
sbatch slurm/run_validation_diagnostics.slurm
```

The evaluation CSV contains:
- z and h per-example InfoNCE loss
- cos(A,P)
- mean and hardest-negative cosine
- A/P margins
- top-1 retrieval
- forecast lead
- season
- valid UTC hour
- wet/dry status
- anchor precipitation mean/max/p95/p99
- fraction of pixels >= 0.1 mm/6h
- P90 intensity among wet pixels
- a precipitation-intensity bin

Wet/dry is defined here as whether the anchor patch contains at least one grid
point >= 0.1 mm/6h. The threshold is configurable.

Intensity binning uses the 90th percentile among anchor pixels >= 0.1 mm/6h:
dry, 0.1-1, 1-5, 5-10, 10-25, 25-50, >=50 mm/6h.

Stratified plots show mean +/- approximate 95% normal confidence intervals.
With only 512 fixed validation examples, narrow groups (especially valid hour)
are exploratory diagnostics, not final uncertainty estimates.

# Next similarity diagnostics

These diagnostics are deliberately run **before** any I/O redesign or longer
training experiment.

Copy these files into the existing project:

```text
scripts/evaluate_validation_similarity_pairs.py
scripts/plot_similarity_decomposition.py
scripts/evaluate_timelag_gradation.py
scripts/make_similarity_case_gallery.py
slurm/run_next_similarity_diagnostics.slurm
```

Then submit:

```bash
source load_env.sh
sbatch slurm/run_next_similarity_diagnostics.slurm
```

The job uses the existing `checkpoint_best.pt` by default (currently the
step-2000 checkpoint) and does not touch the sealed test split.

## 1. Similarity decomposition by intensity

The fixed 512 validation recipes are rerun and every learned similarity is
saved separately:

- A/P positive similarity
- every valid A/N similarity
- mean A/N similarity
- hardest A/N similarity
- A/P minus hardest-N margin

Outputs in the experiment directory include:

```text
validation_similarity_pairs_stepXXXX.csv
validation_similarity_anchor_summary_stepXXXX.csv
```

Plots go under:

```text
diagnostic_plots/similarity_decomposition/
```

The most useful two projection-space plots are:

```text
z_positive_negative_boxplot_by_intensity.png
z_similarity_components_by_intensity.png
```

The first directly compares positive and negative score distributions. The
second shows mean A/P, mean A/N, and hardest A/N similarity by precipitation
intensity. This should distinguish "all dry fields are sensibly similar" from
"even A/P similarity is poor for dry fields."

## 2. 1 h / 3 h / 6 h learned-similarity gradation

This uses held-out validation F12 anchors and same-valid-time fields:

```text
lag 1 h: F13 / F11
lag 3 h: F15 / F09
lag 6 h: F18 / F06
```

The model was trained only with +/-1 h positives. It was never told that
3-hour or 6-hour pairs should be progressively less similar.

For each anchor, both initialization directions are evaluated on the exact
same 256x256 crop, then averaged by lag for the primary paired diagnostic.

Defaults:
- 1,000 validation F12 anchors
- deterministic random crop per anchor
- checkpoint_best.pt
- projection z is primary; normalized backbone h is retained

Outputs include:

```text
timelag_similarity_long_stepXXXX.csv
timelag_similarity_anchor_stepXXXX.csv
timelag_similarity_summary_stepXXXX.csv
diagnostic_plots/timelag_gradation/timelag_similarity_z_boxplot_stepXXXX.png
```

The log prints:
- mean/median similarity at 1, 3, and 6 h
- fraction with S1 > S3
- fraction with S3 > S6
- fraction with full S1 > S3 > S6 ordering

We do **not** expect every individual 1-h pair to exceed every 6-h pair.
The question is whether the distributions and paired tendencies are graded.

## 3. Small physical case gallery

Three cases each are selected for:

```text
best positive similarity
worst positive similarity
hardest negative similarity
```

Each case shows A, P, and the hardest valid N on the identical crop with a
common `log1p(QPF6)` display scale.

Outputs:

```text
diagnostic_plots/case_gallery/
    best_positive_01.png ... 03.png
    worst_positive_01.png ... 03.png
    hard_negative_01.png ... 03.png
    gallery_selections.csv
```

This is intentionally small: the point is to see whether the learned scores
make meteorological/spatial sense, not to build a giant case atlas.

# Full-epoch diagnostics

Copy these files into the existing contrastive-learning project:

```text
scripts/plot_full_epoch_history.py
scripts/diagnose_batchnorm_swap.py
scripts/evaluate_full_epoch_failure_tail.py
scripts/evaluate_full_epoch_timelag_fp32.py
slurm/run_full_epoch_core_diagnostics.slurm
slurm/run_last_checkpoint_comparison.slurm
```

## Recommended first run

```bash
source load_env.sh
sbatch slurm/run_full_epoch_core_diagnostics.slurm
```

This performs four diagnostics.

### 1. Full training/validation history

Produces separate figures under:

```text
$DATA_DIR/derived/experiments/full_epoch_train_validation_v1/
  diagnostic_plots/full_epoch_history/
```

including:
- train + validation loss
- learning-rate schedule
- validation A/P, A/N, and margin
- validation retrieval
- late-run validation loss
- input wait vs GPU compute

### 2. BatchNorm buffer-swap diagnostic

The fixed validation set oscillates very strongly even when the optimizer LR is
tiny. PyTorch BatchNorm running means/variances keep updating during training
independently of optimizer learning rate.

The diagnostic evaluates:

```text
best_native
last_native
last_weights_best_bn
best_weights_last_bn
```

Only `running_mean`, `running_var`, and `num_batches_tracked` are swapped.
BatchNorm affine weights remain attached to the learned-parameter checkpoint.

Interpretation:

- If `last_weights_best_bn` becomes much better than `last_native`, BN running
  state is a major cause of the late validation deterioration.
- If `best_weights_last_bn` becomes much worse than `best_native`, that is the
  reciprocal evidence.
- If swapping BN state has little effect, the oscillation is mainly in learned
  parameters / representation geometry instead.

Results are written to:

```text
diagnostic_plots/batchnorm_swap/batchnorm_swap_results.csv
```

### 3. Best-checkpoint FP32 failure tail

Runs the same failure-tail analysis used on the pilot:
- retrieval failure rate by intensity
- hardest-N season
- MAE, RMSE, correlation
- multi-threshold / multi-scale FSS
- fixed-scale failure gallery

The full-epoch best checkpoint should be step 36000 according to the completed
run.

### 4. Best-checkpoint FP32 1/3/6-h time-lag gradation

Repeats the same held-out F12 same-valid-time diagnostic:
- +/-1 h
- +/-3 h
- +/-6 h
- means / medians
- paired S1>S3, S3>S6, and S1>S3>S6 ordering fractions

## Optional: compare the final checkpoint

The run ended at step 45646 with much poorer fixed-validation loss than the
best checkpoint. To determine whether the final representation also degraded
on the physically meaningful diagnostics, submit:

```bash
sbatch slurm/run_last_checkpoint_comparison.slurm
```

Outputs are checkpoint-step-specific, so they do not overwrite the best-run
diagnostics.

## Why the BatchNorm diagnostic matters

The training loss/margin trajectory becomes smooth and strong late in the
epoch, while fixed-validation performance continues to jump between very good
and very poor states. This persists at learning rates around 1e-5 to 1e-6.

BatchNorm running statistics are updated by forward passes in train mode and
are not scaled by the optimizer learning rate. With heterogeneous precipitation
batches, those buffers can continue to move even when model parameters barely
change. The buffer-swap experiment tests that hypothesis directly without
retraining anything.

# Hard-ranking V2 external diagnostics

This bundle evaluates the **best hard-ranking fine-tune checkpoint** against the same validation-only diagnostics that exposed the V1 representation's strengths and weaknesses. It never uses the sealed test split.

Default V2 checkpoint:

```text
$DATA_DIR/derived/experiments/hard_ranking_finetune_v1/checkpoint_best.pt
```

Default V1 baseline:

```text
$DATA_DIR/derived/experiments/groupnorm_pilot_v1/checkpoint_best.pt
```

All new output stays under:

```text
$DATA_DIR/derived/experiments/hard_ranking_finetune_v1/evaluation/
```

so none of the original V1 evaluation directories are overwritten.

## Install

Copy the bundle contents into the corresponding paths under `$PROJECT_DIR`:

```text
scripts/evaluate_wang_v2.py
scripts/plot_wang_v2.py
scripts/evaluate_signed_morphology_v2.py
scripts/plot_signed_morphology_wang_style.py
scripts/evaluate_stageiv_v1_v2.py
scripts/compare_hard_ranking_margins.py
scripts/submit_v2_external_diagnostics.sh
src/eval/morphology_signed_transforms.py
src/eval/wang_plot_style.py
slurm/run_wang_v2.slurm
slurm/run_signed_morphology_v2.slurm
slurm/run_stageiv_v1_v2.slurm
slurm/run_ranking_margin_v1_v2.slurm
```

The existing project files `src/eval/wang_transforms.py`, `src/eval/verification_metrics.py`, `src/models/checkpoint_loader.py`, `src/data/hard_ranking_sampler.py`, and the already-built Wang/Stage-IV archives are reused.

## Run everything

```bash
cd $PROJECT_DIR
source load_env.sh
bash scripts/submit_v2_external_diagnostics.sh
```

The four jobs are independent and may run simultaneously if GPUs are available.

## 1. Wang-style controlled transformations

The V2 evaluator uses the exact same existing displacement, intensity, and area-size transforms as V1 and the same 2.54-mm traditional-metric threshold. It computes both V1 and V2 learned similarities on the same 2,000 archived cases.

Main outputs:

```text
.../evaluation/wang_v1_vs_v2/wang_style_v2.png
.../evaluation/wang_v1_vs_v2/wang_learned_v1_vs_v2.png
.../evaluation/wang_v1_vs_v2/summary_by_bias.csv
```

`wang_style_v2.png` deliberately preserves the visual grammar of the original Wang-style figure: TS/POD/normalized-FAR in greens, FSS in blues, learned similarity in red, black dotted reference parabola, the same score axis, and three vertically stacked panels.

## 2. Signed expanded morphology suite

The same 500 archived validation patches and case seed (`20261007`) are used for comparability with the first morphology diagnostic. The new axes are signed and centered at identity:

- **Sharpening / smoothing:** negative = unsharp high-frequency enhancement, positive = Gaussian smoothing. These are diagnostic counterparts, not mathematical inverses.
- **Rotation:** negative and positive degrees.
- **Principal-axis deformation:** plotted bias is `log2(scale along the original precipitation major axis)`. Negative compresses that axis; positive elongates it. The perpendicular axis uses the reciprocal scale, so area is preserved by the affine operator before edge clipping/interpolation.
- **Tile fragmentation:** negative and positive signs reverse the deterministic jitter vectors. Both sides are equal-magnitude fragmentation; negative does **not** mean "less fragmented."
- **Object organization:** negative moves connected objects inward toward the precipitation centroid (concentration); positive moves them outward (dispersion).

Domain-total precipitation is renormalized after each geometry transform.

The morphology figures use the same metric colors, score axis, line widths, and general layout as the Wang plots. A Wang-style **reference parabola is intentionally not drawn** for these new axes because there is no established ideal response curve for these transformations.

Main outputs:

```text
.../evaluation/signed_morphology_v1_vs_v2/signed_morphology_wang_style_all.png
.../evaluation/signed_morphology_v1_vs_v2/signed_morphology_learned_v1_vs_v2.png
.../evaluation/signed_morphology_v1_vs_v2/*_wang_style.png
.../evaluation/signed_morphology_v1_vs_v2/signed_morphology_summary.csv
```

## 3. Stage-IV F/O V1-vs-V2 validation

This reuses the existing 1,000-case `stageiv_reference_v1` archive and the already-selected BUDGET remap. Earlier diagnostics showed BUDGET-vs-BILIN differences were negligible, so this comparison does not repeat the regridding study.

It calculates V1 and V2 learned F/O similarities on exactly the same HRRR/Stage-IV pairs, plus correlation, RMSE, MAE, and representative FSS (default 1 mm / width 15). Outputs include all-case and both-wet score distributions, V1-vs-V2 score scatter, V2-vs-FSS/correlation, and the cases with the largest V2 score increases/decreases.

Main outputs:

```text
.../evaluation/stageiv_v1_vs_v2/score_summary.csv
.../evaluation/stageiv_v1_vs_v2/per_case_metrics.csv.gz
.../evaluation/stageiv_v1_vs_v2/plots/
```

## 4. Per-case ranking-margin diagnostic

A ranking percentage only tells whether a margin is positive. It cannot distinguish a near tie (`+0.001`) from a confident ordering (`+0.20`). This diagnostic reuses the **exact fixed 512 validation recipes saved by the fine-tuning run** and evaluates both checkpoints on them.

For example:

```text
P − TL3
TL3 − TL6
E1 − E2
E2 − E3
E3 − E6
```

A positive margin means the desired ordering is satisfied. Comparing the full V1 and V2 distributions tells us whether V2 shifted the distribution rightward, merely enlarged already-correct cases, or changed the negative tail. That is why this adds information beyond `full_order_pct` in `validation_history.csv`.

Main outputs:

```text
.../evaluation/ranking_margin_v1_vs_v2/adjacent_margin_distributions_v1_vs_v2.png
.../evaluation/ranking_margin_v1_vs_v2/margin_summary.csv
.../evaluation/ranking_margin_v1_vs_v2/per_case_pair_margins.csv.gz
```

## What to send back

The most useful morning review set is:

```text
wang_style_v2.png
wang_learned_v1_vs_v2.png
signed_morphology_wang_style_all.png
signed_morphology_learned_v1_vs_v2.png
stageiv_v1_vs_v2/score_summary.csv
stageiv_v1_vs_v2/plots/learned_similarity_v1_vs_v2_histogram.png
stageiv_v1_vs_v2/plots/stageiv_v1_vs_v2_scatter.png
ranking_margin_v1_vs_v2/margin_summary.csv
ranking_margin_v1_vs_v2/adjacent_margin_distributions_v1_vs_v2.png
```

Stage-IV forecast-hour validation pipeline (v2)
================================================

Files are intended to be copied into the root of the existing
contrastive_learning repository, preserving scripts/ and slurm/.

EXPERIMENT
----------
Validation valid times: 2024-10-01 through 2025-09-20
HRRR cycles:            00, 06, 12, 18 UTC
Forecast leads:         F06, F12, F18, F24, F30, F36, F42, F48
Reference:              Stage IV 6-h precipitation
Patch size:             256 x 256

One valid-time patch is shared across all eight forecast leads. A valid time is
retained only when all eight HRRR fields exist. Stage IV is BUDGET-remapped
once per valid time. The archive builder tries deterministic crop locations
until it finds one with complete finite Stage-IV and HRRR coverage, and fails
rather than silently unbalancing the lead sample.

Traditional metrics are pooled by forecast hour:
  ETS 0.1 mm
  POD 0.1 mm
  normalized FAR / success ratio 0.1 mm
  FSS-5 0.1 mm
  FSS-10 0.1 mm

The H100 stage then computes mean V2 and V3 learned cosine similarity by lead.

STEP 1: CHEAP LOGIN-NODE PREFLIGHT
---------------------------------
From the repository root:

  source load_env.sh
  ./scripts/prepare_stageiv_forecast_hour_manifest.sh

By default Stage IV is expected under the canonical project location:

  $DATA_DIR/observations/stage4/raw/YYYY/YYYYMM/st4_conus.YYYYMMDDHH.06h.grb2

If yours is elsewhere:

  export STAGEIV_ROOT=/your/stage4/raw
  ./scripts/prepare_stageiv_forecast_hour_manifest.sh

This does not use a GPU. It must pass before continuing.

STEP 2: CPU ARCHIVE BUILD + PREFLIGHT + TRADITIONAL METRICS
-----------------------------------------------------------
Set the same MET-readable native-HRRR grid template used in the older Stage-IV
reference experiment:

  export HRRR_GRID_TEMPLATE=/path/to/one/native/hrrr/grid/file.grib2

Optional if Stage IV is not in the canonical project path:

  export STAGEIV_ROOT=/your/stage4/raw

Submit:

  sbatch --export=ALL,HRRR_GRID_TEMPLATE="$HRRR_GRID_TEMPLATE",STAGEIV_ROOT="${STAGEIV_ROOT:-$DATA_DIR/observations/stage4/raw}" \
    slurm/build_stageiv_forecast_hour_archive.slurm

The log is:

  $LOG_DIR/stageiv_forecast_hour_build_<JOBID>.log

When it finishes, inspect the images in:

  $DATA_DIR/derived/evaluation/stageiv_forecast_hour_v1/alignment_check_forecast_hour/

Do NOT submit the GPU step until the HRRR and Stage-IV fields have the same
orientation and plausible spatial alignment.

STEP 3: H100 V2/V3 INFERENCE + FINAL PLOT
-----------------------------------------
After the CPU job passes and you inspect the alignment figures:

  sbatch slurm/run_stageiv_scores_by_forecast_hour.slurm

Log:

  $LOG_DIR/stageiv_scores_by_forecast_hour_<JOBID>.log

Final figure:

  $DATA_DIR/derived/experiments/hard_ranking_v3_full_v1/evaluation/stageiv_by_forecast_hour_v1/plots/hrrr_stageiv_verification_scores_by_forecast_hour.png

Final table:

  $DATA_DIR/derived/experiments/hard_ranking_v3_full_v1/evaluation/stageiv_by_forecast_hour_v1/forecast_hour_summary.csv

IMPORTANT
---------
This is a new dedicated archive. It does NOT read stageiv_reference_v1, which
was restricted to the old F07-F17 Stage-IV diagnostic experiment.

WANG-STYLE 8-TRANSFORM / V3 DIAGNOSTICS
========================================

Files
-----
scripts/evaluate_stageiv_synthetic_vs_v3.py
scripts/plot_stageiv_synthetic_galleries.py
scripts/evaluate_morphology_synthetic_vs_v3.py
scripts/plot_morphology_synthetic_vs_v3.py
slurm/run_stageiv_synthetic_galleries.slurm
slurm/run_morphology_synthetic_vs_v3.slurm
slurm/run_all_synthetic_diagnostics.slurm

Defaults
--------
Synthetic checkpoint:
  $DATA_DIR/derived/experiments/synthetic_perturb_all8_v1/checkpoint_best.pt

Natural V3 checkpoint:
  $DATA_DIR/derived/experiments/hard_ranking_v3_full_v1/checkpoint_best.pt

Stage-IV galleries reuse the same old diagnostic archive as prior galleries:
  $DATA_DIR/derived/evaluation/stageiv_reference_v1

Morphology reuses the same held-out transform archive and deterministic 500-case
selection used by prior morphology work:
  $DATA_DIR/derived/evaluation/wang_transform_v1
  case_count=500
  case_seed=20261007

Traditional morphology metrics use threshold 2.54 mm (0.10 inch), matching the
prior signed-morphology run.  Success Ratio is hits/(hits+false alarms) = 1-FAR.

Recommended run
---------------
If you want to wait in the H100 queue only once:

  sbatch slurm/run_all_synthetic_diagnostics.slurm

Or run separately:

  sbatch slurm/run_stageiv_synthetic_galleries.slurm
  sbatch slurm/run_morphology_synthetic_vs_v3.slurm

Outputs
-------
Stage-IV:
  $DATA_DIR/derived/experiments/synthetic_perturb_all8_v1/evaluation/stageiv_vs_v3/
    per_case_metrics.csv.gz
    galleries/
      selected_gallery_cases.csv
      bucket_A/*.png
      bucket_B/*.png
      bucket_C/*.png
      bucket_D/*.png

Morphology:
  $DATA_DIR/derived/experiments/synthetic_perturb_all8_v1/evaluation/morphology_vs_v3/
    selected_cases.csv
    morphology_per_case.csv.gz
    morphology_summary.csv
    morphology_traditional_plus_synthetic.png
    morphology_v3_vs_synthetic.png
    run_metadata.json

Gallery definitions
-------------------
A: highest non-dry synthetic scores
B: best rank agreement with mean(FSS-5,FSS-10), preferring high joint rank
C: largest absolute synthetic-vs-V3 cosine-similarity departures
D: largest rank departures from mean(FSS-5,FSS-10)

All four buckets are restricted to cases with at least 16 pixels >=0.1 mm in
both HRRR and Stage IV.  Cases are unique across A-D.

Notes
-----
The learned scores are raw cosine similarities, not normalized skill scores.
The plot label "SYN-8" is intentionally configurable.  Once final manuscript
naming is chosen, simply pass different --synthetic-label/--natural-label values
or edit the Slurm wrapper labels; no scientific rerun is needed.

The evaluator also accepts a different synthetic checkpoint.  When the 3-transform
run is ready, the same scripts can be rerun using:

  --synthetic-checkpoint $DATA_DIR/derived/experiments/<WANG3_RUN>/checkpoint_best.pt

and a different --output-dir, so the plotting/evaluation code remains identical.

FINAL SYN-3 / SYN-8 / NFV DIAGNOSTICS V1
=========================================

Purpose
-------
This bundle evaluates the *revised* models:

  SYN-3: $DATA_DIR/derived/experiments/synthetic_perturb_syn3_v2/checkpoint_best.pt
  SYN-8: $DATA_DIR/derived/experiments/synthetic_perturb_syn8_v2/checkpoint_best.pt
  NFV:   $DATA_DIR/derived/experiments/hard_ranking_v3_full_v1/checkpoint_best.pt

Do not submit these jobs until both revised SYN runs have written checkpoint_best.pt.
All learned scores are raw cosine similarities of normalized projection-z embeddings.

Files
-----
scripts/evaluate_final_stageiv_models.py
scripts/plot_final_stageiv_buckets.py
scripts/evaluate_final_morphology_models.py
scripts/plot_final_morphology_models.py
scripts/submit_final_diagnostics.sh
slurm/run_final_stageiv_buckets.slurm
slurm/run_final_morphology_models.slurm

Install
-------
Unzip at the repository root so scripts/ and slurm/ merge into the existing tree.

  cd $PROJECT_DIR
  unzip -o /path/to/final_syn_nfv_diagnostics_v1.zip
  chmod +x scripts/submit_final_diagnostics.sh

Run
---
Submit both independent H100 jobs:

  ./scripts/submit_final_diagnostics.sh

or separately:

  sbatch slurm/run_final_stageiv_buckets.slurm
  sbatch slurm/run_final_morphology_models.slurm

Logs:

  $LOG_DIR/final_stageiv_buckets_<JOBID>.log
  $LOG_DIR/final_morphology_<JOBID>.log

The plotting scripts themselves do NOT require a GPU.  After the evaluation CSVs
exist, they can be rerun on a login node while iterating on figure appearance:

  python scripts/plot_final_stageiv_buckets.py
  python scripts/plot_final_morphology_models.py

Add --show-ci to the morphology plotter if 95% CI shading is desired.

Morphology experiment
---------------------
Uses 500 deterministic held-out HRRR patches (seed 20261007) and the same transformed
field for SYN-3, SYN-8, NFV, and traditional metrics.

Eight families:
  1 displacement
  2 intensity
  3 area size
  4 smoothing ONLY (sigma 0,0.6,...,6.0; no sharpening)
  5 object concentration / dispersion
  6 principal-axis compression / elongation
  7 rotation
  8 tile fragmentation

Traditional threshold: 0.1 mm.
FSS uses 5- and 10-grid-cell neighborhoods.
Success Ratio = hits / (hits + false alarms) = 1 - FAR.

The dashed gray target is the revised Wang-adapted training target expressed as
cosine similarity:

  s_target = 1 - 0.9*u^2

where u is normalized transformation severity.  For smoothing this is intentionally
one-sided.  For intensity and area, u follows the Wang discrete ladders on each side
of identity rather than log2 scaling.

Outputs:
  $DATA_DIR/derived/evaluation/final_syn_nfv_morphology_v1/
      morphology_per_case.csv.gz
      morphology_summary.csv
      run_metadata.json
      plots/
          morphology_syn3_vs_traditional.png
          morphology_syn8_vs_traditional.png
          morphology_nfv_vs_traditional.png
          morphology_nfv_vs_syn3_syn8.png

Requested plotting colors:
  learned focal model: black
  ETS-0.1: dark green
  POD-0.1: green
  SR-0.1: light green
  FSS-5: light blue
  FSS-10: dark blue
  Wang target: gray dashed

The 3-model plot uses NFV black, SYN-3 solid red, SYN-8 dashed red.

Stage-IV galleries
------------------
Uses the existing stageiv_reference_v1 archive (same 1000 cases as earlier galleries).
Traditional statistics written on every case:
  FSS-5 and FSS-10 at 0.1 mm
  ETS at 0.1 and 0.5 mm
  FBIAS at 0.1 and 0.5 mm

Non-dry default:
  BOTH HRRR and Stage IV must contain at least 16 pixels >= 0.1 mm.

The 11 requested buckets are selected independently (overlap is allowed):
  01 Top 5 highest SYN-3
  02 Top 5 SYN-3 / FSS-5 alignment
  03 Top 5 SYN-3 / ETS-0.1 alignment
  04 Bottom 5 SYN-3 / FSS-5 alignment
  05 Bottom 5 SYN-3 / ETS-0.1 alignment
  06 Bottom 5 SYN-3 / SYN-8 alignment
  07 Bottom 5 SYN-8 / FSS-5 alignment
  08 Bottom 5 SYN-8 / ETS-0.1 alignment
  09 Bottom 5 SYN-3 / NFV alignment
  10 Bottom 5 NFV / FSS-5 alignment
  11 Bottom 5 NFV / ETS-0.1 alignment

Alignment definition:
  learned vs FSS/ETS = absolute percentile-rank difference.
  This avoids pretending cosine similarity and traditional scores are calibrated to
  the same numerical scale.

  learned vs learned = absolute raw cosine-similarity difference, because SYN-3,
  SYN-8, and NFV use the same final score definition.

Outputs:
  $DATA_DIR/derived/evaluation/final_syn_nfv_stageiv_v1/
      stageiv_per_case_metrics.csv.gz
      run_metadata.json
      galleries/
          01_highest_syn3/
          ...
          11_worst_nfv_ets01_alignment/
          all_bucket_selections.csv

Notes
-----
- These diagnostics use validation/reference data only; they do not retrain models.
- Revised SYN-8 smoothing is one-sided in both training and these primary morphology
  plots.  Sharpening can still be retained later as a separate out-of-training probe.
- Figure appearance is intentionally a first pass; the plotters are separated from
  the GPU evaluation so presentation styling can be iterated cheaply.

Final diagnostics inch-threshold/display patch
==============================================

This patch DOES NOT rerun any SYN-3/SYN-8/NFV model inference.

It reuses the existing learned cosine similarities and recalculates only the
traditional verification quantities that were accidentally evaluated at mm
thresholds.

Stage-IV galleries
------------------
* ETS/FBIAS 0.1 -> 0.1 inch = 2.54 mm internal threshold
* ETS/FBIAS 0.5 -> 0.5 inch = 12.70 mm internal threshold
* FSS-5/FSS-10 -> 0.1 inch threshold
* non-dry -> BOTH fields have >=16 pixels with precipitation >=0.1 inch
* HRRR/Stage-IV/difference displays converted from archived mm to inches
* supplied weather-style precipitation palette
* pcolormesh is used for ALL three panels, including the signed difference field
* horizontal precipitation colorbar spans panels 1+2 exactly
* horizontal difference colorbar spans panel 3 exactly
* "Worst" gallery language replaced by "Bottom"
* valid-date format: valid 12Z Feb 02, 2025 (F15)

Morphology
----------
* Learned SYN-3/SYN-8/NFV scores are REUSED exactly.
* Synthetic transforms are deterministically regenerated on CPU.
* ETS/POD/SR/FSS-5/FSS-10 are recalculated at 0.1 inch = 2.54 mm.
* Smoothing remains the one-sided revised family.
* Pixel-space diagnostics are added to the new morphology summary to help audit
  the local SYN-8 area-size non-monotonicity.

Recommended run
---------------
From the project root after copying/unzipping these files:

  source load_env.sh
  sbatch slurm/recompute_final_inches.slurm

No GPU is requested.

Outputs
-------
Stage-IV:
  $DATA_DIR/derived/evaluation/final_syn_nfv_stageiv_v1/stageiv_per_case_metrics_inches.csv.gz
  $DATA_DIR/derived/evaluation/final_syn_nfv_stageiv_v1/galleries_inches/

Morphology:
  $DATA_DIR/derived/evaluation/final_syn_nfv_morphology_v1/morphology_per_case_inches.csv.gz
  $DATA_DIR/derived/evaluation/final_syn_nfv_morphology_v1/morphology_summary_inches.csv
  $DATA_DIR/derived/evaluation/final_syn_nfv_morphology_v1/plots/

After the CPU recalculation is done, all subsequent aesthetic plot changes are
login-node-safe because the revised CSVs already contain the correct metrics.

Final diagnostics plotting patch v3
===================================

This patch requires NO model inference and NO recomputation of traditional
metrics if morphology_summary_inches.csv and stageiv_per_case_metrics_inches.csv.gz
already exist from the inch-threshold patch.

Changes:

1. Morphology
   - 95% CI ribbons are shown on the three model-vs-traditional figures.
   - The learned-model-only NFV/SYN-3/SYN-8 comparison intentionally has no CIs.
   - CI values are read from the existing *_ci95_lo / *_ci95_hi columns.

2. Stage-IV galleries
   - panels are placed manually as equal-size squares with ~0.13-inch physical
     gaps between adjacent panels;
   - the vertical gap between each panel row and its colorbars is the same
     ~0.13-inch physical distance;
   - horizontal colorbars retain their exact spans (panels 1+2 and panel 3);
   - learned-vs-learned "Bottom alignment" buckets now also use absolute
     percentile-rank gap, consistent with learned-vs-FSS/ETS alignment.

Install from repository root:
    unzip -o final_diagnostics_plotting_patch_v3.zip

Then replot on the login node:
    source load_env.sh
    python scripts/plot_final_morphology_models.py
    python scripts/plot_final_stageiv_buckets.py

No GPU is required.

FEATURE-SPACE INVERSION PILOT
=============================

Purpose
-------
Visualize what NFV, SYN-3, and SYN-8 consider progressively less similar to a
fixed Stage-IV precipitation field by optimizing a constrained precipitation
input against a frozen model.

Important interpretation
------------------------
This is feature inversion, not decoding. The model is an encoder and has no
unique inverse. Each optimized image is ONE constrained local solution whose
embedding approaches the requested cosine similarity. It should not be called
"the physical opposite" of the original precipitation field.

The inversion operates in the same projection z space used by the learned
verification metric. The original Stage-IV field is preprocessed with the same
log1p + HRRR-training z-score transform used in evaluation.

Constraints
-----------
- precipitation >= 0
- precipitation <= effective cap (default 10 in, automatically raised if the
  original Stage-IV maximum requires it)
- weak total-variation regularization to discourage adversarial pixel noise
- weak domain-mean precipitation penalty
- very weak proximity-to-original penalty

Defaults are intentionally weak. The similarity target remains the dominant
objective.

Case selection
--------------
Before spending GPU time, inspect candidate Stage-IV cases on the login node:

  source load_env.sh
  python scripts/select_feature_inversion_case.py

Outputs:

  $DATA_DIR/derived/evaluation/feature_inversion_pilot_v1/case_selection/
      candidate_gallery.png
      ranked_stageiv_candidates.csv
      auto_selected_case.json

If CASE_INDEX is not supplied, both GPU jobs deterministically use the top
heuristic candidate. To select your own case from the candidate gallery:

  CASE_INDEX=244 ./scripts/submit_feature_inversion_smoke.sh

Smoke test
----------
One Stage-IV case, revised SYN-3, targets +0.5 and -0.5, 300 iterations each:

  ./scripts/submit_feature_inversion_smoke.sh

or:

  CASE_INDEX=244 N_STEPS=400 ./scripts/submit_feature_inversion_smoke.sh

Log:

  $LOG_DIR/feature_inversion_smoke_<JOBID>.log

Default output:

  $DATA_DIR/derived/evaluation/feature_inversion_pilot_v1/smoke/

Full pilot
----------
Same Stage-IV case, NFV + revised SYN-3 + revised SYN-8, targets:

  +0.5, 0.0, -0.5, -1.0

Each model uses continuation: the +0.5 solution initializes 0.0, which
initializes -0.5, which initializes -1.0.

Run:

  CASE_INDEX=244 ./scripts/submit_feature_inversion_pilot.sh

or omit CASE_INDEX to use the deterministic auto-selection.

Log:

  $LOG_DIR/feature_inversion_pilot_<JOBID>.log

Default output:

  $DATA_DIR/derived/evaluation/feature_inversion_pilot_v1/pilot/

Key figures:

  gallery_nfv.png
  gallery_syn3.png
  gallery_syn8.png
  gallery_all_models.png

Each target directory also contains optimized_precip_mm.npy,
optimization_history.csv.gz, and summary.json. Always discuss TARGET and
ACHIEVED similarity separately, especially for negative targets. Failure to
reach -1 is scientifically informative and does not invalidate the experiment.

Checkpoint defaults
-------------------
NFV:
  $DATA_DIR/derived/experiments/hard_ranking_v3_full_v1/checkpoint_best.pt
SYN-3:
  $DATA_DIR/derived/experiments/synthetic_perturb_syn3_v2/checkpoint_best.pt
SYN-8:
  $DATA_DIR/derived/experiments/synthetic_perturb_syn8_v2/checkpoint_best.pt

If the revised SYN checkpoints are still training, wait until checkpoint_best.pt
exists before submitting the corresponding inversion job.

Feature-inversion smoke-test patch v2
=====================================

Purpose
-------
The first smoke test used a sigmoid-bounded image parameterization.  Exact-zero
precipitation pixels had to be clipped to a small positive value before taking
an inverse sigmoid, so the optimizer did NOT actually start from the original
Stage-IV image.  With a 10-inch cap and eps=1e-4, zeros became 0.001 inch.

This patch replaces that parameterization with direct optimization in log1p
precipitation space:

    q0 = log1p(original_mm)
    precip_mm = expm1(clamp(q, 0, log1p(cap_mm)))

Therefore original zero pixels initialize to exactly zero and the initial field
reconstructs the original to floating-point precision.

A mandatory pre-optimization sanity check now prints:

    Exact-init check: max |Δprecip|=... mm, S=...

and aborts unless the field is reconstructed within 1e-4 mm and its learned
similarity to the original is at least 0.999999.

The patch also prints and writes simple image-change diagnostics (precip MAE,
RMSE, and the fraction of pixels changed by >=0.01 and >=0.10 inch), so if the
new inversion still finds visually subtle/adversarial solutions we can quantify
how much of the image actually changed.

Install
-------
From the project root:

    unzip -o feature_inversion_exact_init_patch_v2.zip
    source load_env.sh

Retry the same Stage-IV case used before (case 299):

    CASE_INDEX=299 ./scripts/submit_feature_inversion_smoke_v2.sh

The retry defaults to 500 optimization steps per target (still targets +0.50
and -0.50 for SYN-3).  Override if desired, e.g.:

    CASE_INDEX=299 N_STEPS=800 ./scripts/submit_feature_inversion_smoke_v2.sh

Default log:

    $LOG_DIR/feature_inversion_smoke_v2_<JOBID>.log

Default output:

    $DATA_DIR/derived/evaluation/feature_inversion_pilot_v1/smoke_exact_init_v2

Important interpretation
------------------------
This remains unrestricted pixel-level feature inversion subject to the existing
weak smoothness/mass/proximity penalties and precipitation bounds.  If it still
finds visually small changes that strongly alter learned similarity, that is a
real property of this inversion problem/model rather than the previous
zero-initialization artifact and is worth diagnosing separately.

Dense continuation feature-space inversion (v3)
================================================

Purpose
-------
Follow a constrained path away from one Stage-IV reference image through
progressively lower cosine-similarity targets for NFV, SYN-3, and SYN-8.

Targets
-------
1.00 original, then:
  +0.75, +0.50, +0.25, 0.00, -0.25, -0.50, -0.75, -1.00

Each selected solution initializes the next target.

Adaptive restarts
-----------------
The primary run starts exactly from the previous target's selected solution.
If that run misses the requested similarity by >0.03, two additional
low-frequency deterministic restarts are tried in log1p precipitation space
(default scales 0.03 and 0.08). The best restart is selected.

Precipitation can be removed
----------------------------
Yes. Positive precipitation may erode all the way to exactly zero, and dry
pixels may become wet. For this exploratory continuation experiment the net
mass penalty defaults to ZERO, specifically so the optimizer is not forced to
replace precipitation it removes somewhere else. The only hard physical
constraints are nonnegative precipitation and the upper precipitation cap.
Weak TV and proximity penalties remain to discourage pixel-noise solutions.

Gradient interpretation
-----------------------
The network weights are frozen. PyTorch backpropagates the inversion loss all
the way to the optimizable log1p precipitation field. Adam therefore receives,
for every input pixel, a gradient describing how an infinitesimal increase or
decrease would change the target-matching objective plus regularization.
The optimizer can naturally erode one side of a feature and build precipitation
on another side if that path lowers the objective; no explicit displacement
operator is imposed.

Run
---
  cd $PROJECT_DIR
  source load_env.sh
  CASE_INDEX=299 ./scripts/submit_feature_inversion_continuation.sh

Optional longer maximum per restart:
  CASE_INDEX=299 N_STEPS=1800 ./scripts/submit_feature_inversion_continuation.sh

Outputs
-------
  $DATA_DIR/derived/evaluation/feature_inversion_pilot_v1/continuation_v3/

Per model:
  continuation_gallery_<model>.png     (3x3: original + eight targets)

Overall:
  continuation_summary.csv
  continuation_target_vs_achieved.png

Each target directory contains the selected field/summary and all attempted
restart histories.

Asterisk on a gallery title means the requested target was not reached within
the configured 0.01 similarity tolerance.
