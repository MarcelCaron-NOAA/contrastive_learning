REVISED SYN-3 / WANG-ADAPTED V2
===============================

Purpose
-------
Replace the first SYN-3 formulation with a more faithful adaptation of the
released Wang/CLPFV synthetic supervision while retaining this project's
GroupNorm ResNet-18, projection head, HRRR patches, log1p-zscore preprocessing,
and optimizer/LR infrastructure.

Core objective
--------------
Views per underlying HRRR case:
  original + displacement + intensity + area-size = 4

Batch default:
  8 underlying cases x 4 views = 32 physical model inputs / step

Discrete severity:
  rank r = 1..10, random direction for each transform independently.

Wang target:
  cosine distance target = 0.009 * r^2
  equivalent cosine similarity = 1 - 0.009 * r^2

Loss:
  total = SupCon + 1000 * mean_abs_error(cosine_distance, target_distance)

SupCon:
  all 4 views of the same underlying HRRR case share a class label;
  all views of the other underlying cases in the minibatch are negatives.

FIRST RUN A 10-STEP SMOKE TEST
------------------------------
From the repo root:

  chmod +x scripts/submit_syn3_wang_smoke.sh scripts/submit_syn3_wang_full.sh
  ./scripts/submit_syn3_wang_smoke.sh

Expected header includes:
  Target cosine distance    : 0.009 * rank^2
  Equivalent similarity     : 1 - 0.009 * rank^2 (0.991 -> 0.100)
  Underlying cases / step   : 8
  Views / case              : 4
  Physical fields / step    : 32
  Global objective          : SupCon, temperature=0.1
  Loss weights              : SupCon=1, penalty=1000
  Full deterministic epoch  : ~45,646 steps
  Target end step           : 10

The smoke log should report, e.g.:
  SupCon=<raw>
  P=<raw penalty MAE> x1000=<weighted contribution>
  d=<mean cosine distance>
  target_d=<mean Wang target distance>
  retrieval=<same-case nearest-neighbor top1>

It also reports per-transform rank, similarity, distance, target distance, and
MAE at validation.

FULL RUN
--------
Only after the smoke test looks numerically sane:

  ./scripts/submit_syn3_wang_full.sh

Default run directory:
  $DATA_DIR/derived/experiments/synthetic_perturb_syn3_v2

Default walltime:
  05:00:00

To request a different walltime without editing the Slurm file:

  WALLTIME=06:00:00 ./scripts/submit_syn3_wang_full.sh

RESUME
------
The full trainer writes checkpoint_best.pt, checkpoint_last.pt, and periodic
checkpoint_step*.pt files. To resume into the same run directory:

  RUN_NAME=synthetic_perturb_syn3_v2 \
  RESUME_CHECKPOINT="$DATA_DIR/derived/experiments/synthetic_perturb_syn3_v2/checkpoint_last.pt" \
  ./scripts/submit_syn3_wang_full.sh

NOTES
-----
- This intentionally does NOT reproduce Wang's pretrained BatchNorm ResNet18,
  raw-precipitation input, unused projection head, tiny 69x73 grids, or hundreds
  of short epochs.
- It DOES reproduce the key released-code synthetic supervision we decided to
  adopt: SupCon, discrete ranks 1..10, cosine-distance quadratic 0.009*r^2,
  and nominal x1000 penalty weighting.
- The custom penalty is a proper mean over valid original/perturbation pairs
  before x1000, rather than reproducing the released notebook's peculiar
  group-size-only divisor.
- Dry sampled patches are excluded from SupCon/penalty through the existing wet
  mask; this prevents identical all-zero transforms from becoming contradictory
  supervision.

REVISED SYN-8 / WANG-ADAPTED PIPELINE
=====================================

Purpose
-------
This is the eight-transformation extension of revised SYN-3.  It keeps the
same GroupNorm ResNet-18, projection head, HRRR preprocessing, optimizer,
SupCon global loss, Wang cosine-distance target, x1000 penalty weight,
training/validation split, and one-full-anchor-epoch schedule.

The only scientific extension is adding five perturbation families:
  smoothing
  object_dispersion_cells
  principal_axis_log2_scale
  rotation_deg
  tile_fragmentation_cells

All eight families use discrete severity ranks 1..10 and the SAME target:
  target cosine distance = 0.009 * rank^2
  target similarity      = 1 - 0.009 * rank^2

Physical rank mappings
----------------------
Wang three (exact released-code ladders):
  displacement: +/- rank cells
  intensity: lower 1-0.05*rank, upper 1+0.10*rank
  area size: same ladder as intensity

Five added families (rank/10 of the previous diagnostic maximum magnitude):
  smoothing:             +0.6*rank sigma         -> +6 at rank 10
                         (one-sided in training; sharpening remains diagnostic-only)
  object dispersion:    +/-round(1.6*rank) cells -> +/-16 at rank 10
  principal axis:       +/-0.1*rank log2 scale -> +/-1 at rank 10
  rotation:             +/-9*rank degrees      -> +/-90 at rank 10
  tile fragmentation:   +/-round(1.2*rank) cells -> +/-12 at rank 10

Why smoothing is one-sided
--------------------------
Sharpening and Gaussian smoothing are diagnostic counterparts, not mathematical
inverses. Previous morphology tests showed a strong, repeatable asymmetry across
traditional metrics, NFV, SYN-3, and SYN-8-v1: sharpening was tolerated much
more than smoothing. Therefore revised SYN-8 does NOT force equal quadratic
penalties onto +/- sigma. Only positive-sigma Gaussian smoothing is supervised.
Sharpening should still be evaluated later on the full signed morphology axis as
an out-of-sample response.

Batching
--------
8 underlying HRRR cases per step by default.
Each case has 9 views: original + one view from each of 8 transformations.
Thus 72 physical fields per optimizer step.
All nine views of a case share a SupCon class; views of the other seven cases
serve as negatives.  The Wang penalty is only original-to-perturbed.

For the three transformations shared with SYN-3, the sampler intentionally
uses the same seed and per-family random stream as revised SYN-3.  Therefore,
for the same anchor/epoch, crop, rank, and direction are matched between
SYN-3 and SYN-8.  This makes their comparison cleaner.

Smoke test
----------
  ./scripts/submit_syn8_wang_smoke.sh

Defaults:
  RUN_NAME=synthetic_perturb_syn8_v2_smoke
  END_STEP=10
  WALLTIME=00:45:00

Full run
--------
  ./scripts/submit_syn8_wang_full.sh

Defaults:
  RUN_NAME=synthetic_perturb_syn8_v2
  WALLTIME=12:00:00

Because SYN-8 has 72 model inputs/step and five more CPU transforms than SYN-3,
it is strongly recommended to use the smoke-test timing to right-size the full
walltime before submission. Override, e.g.:

  WALLTIME=10:00:00 ./scripts/submit_syn8_wang_full.sh

Output
------
  $DATA_DIR/derived/experiments/synthetic_perturb_syn8_v2/

Key files include checkpoint_best.pt, checkpoint_last.pt,
training_history.csv, validation_history.csv, config.json, run_summary.json,
and plots/.

SYN-8 v2 smoothing-only patch
=============================

Changes from the first SYN-8-v2 bundle:

1. The training family `smoothing_sharpening` is renamed `smoothing`.
2. Smoothing is one-sided in training: rank r -> sigma = +0.6*r, so ranks
   1..10 correspond to sigma 0.6..6.0.
3. No sharpening examples are generated during SYN-8 training.
4. The Wang target remains identical: target cosine distance = 0.009*r^2.
5. The other seven families are unchanged.
6. Sharpening remains part of the later signed morphology evaluation and is
   intentionally treated as out-of-sample behavior rather than forced onto a
   symmetric training curve.

The run names remain synthetic_perturb_syn8_v2[_smoke]. If a previous run was
started with the old symmetric smoothing/sharpening bundle, do not resume that
checkpoint into this patched trainer. Use a fresh output directory or remove /
rename the partial old experiment directory first.

