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

