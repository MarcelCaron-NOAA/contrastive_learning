# BatchNorm recalibration experiment

This is the next surgical experiment for the full-epoch HRRR contrastive model.

## Copy into the current project

```text
scripts/recalibrate_batchnorm.py
scripts/evaluate_bnrecal_failure_tail.py
scripts/evaluate_bnrecal_timelag_fp32.py
slurm/run_bn_recalibration_test.slurm
slurm/run_bnrecal_physical_diagnostics.slurm
```

## Phase 1 — test whether post-hoc BN recalibration works

Submit:

```bash
source load_env.sh
sbatch slurm/run_bn_recalibration_test.slurm
```

The script recalibrates both:

```text
checkpoint_best.pt
checkpoint_last.pt
```

using the same deterministic 10,000-anchor sample from the TRAIN split only.

Calibration uses the original v1 training data definition:

```text
positive: +/-1 h same-valid-time HRRR
negative: climo45
K: 8
patch: 256 x 256
transform: log1p_zscore
```

With logical batch size 8, every BN update sees 80 physical fields:

```text
8 * (A + P + 8 N) = 80
```

### What changes

Only BatchNorm running buffers are recomputed:

```text
running_mean
running_var
num_batches_tracked
```

The script:
- resets these buffers,
- puts only BN modules into train mode,
- uses `momentum=None` for cumulative moving averaging,
- forwards the calibration data in FP32,
- uses no gradients and no optimizer,
- verifies with a SHA-256 digest that model parameters remain bitwise unchanged.

### Outputs

```text
$DATA_DIR/derived/experiments/full_epoch_train_validation_v1/bn_recalibration/
```

contains:

```text
checkpoint_best_bnrecal.pt
checkpoint_last_bnrecal.pt
bn_recalibration_comparison.csv
bn_recalibration_metadata.json
bn_recalibration_recipes_10000_seed20261002.jsonl
```

The log prints native-versus-recalibrated fixed-validation:

```text
z loss
cos(A,P)
mean cos(A,N)
margin
top-1 retrieval
```

for both checkpoints.

## How to interpret Phase 1

Most interesting outcomes:

### A. Final checkpoint improves dramatically after recalibration

For example, if last goes from ~1.65 loss to something much lower while its
margin and retrieval recover, the final weights retain useful information and
their native BN state is a large part of the problem.

### B. Best checkpoint stays good after recalibration

That would be ideal. It would indicate that a training-only population estimate
can replace the accidentally favorable step-36000 running statistics without
using validation information.

### C. Best gets much worse but last improves

This would say the learned weights are highly coupled to their contemporaneous
BN state; recalibration is not a simple universal fix.

### D. Neither improves

Then post-hoc BN recalibration is insufficient, and a training run with a
batch-independent normalization (GroupNorm is the first candidate) becomes
much more attractive.

## Phase 2 — only if Phase 1 looks useful

Submit:

```bash
sbatch slurm/run_bnrecal_physical_diagnostics.slurm
```

This runs, for BOTH recalibrated checkpoints:

- FP32 failure-tail diagnostics by precipitation intensity
- fixed-scale failure galleries
- 1/3/6-h unseen time-lag gradation

Outputs use the checkpoint filename in their names/directories, so they do not
overwrite the native checkpoint diagnostics.

## Why 10,000 calibration anchors?

This is ~2.7% of the 365,167-anchor training set and gives 1,250 BN updates.
Each update contains 80 physical precipitation fields, or 100,000 physical
fields total per checkpoint. It is large enough to estimate stable population
statistics without making the test another full-epoch I/O job.

If results are promising but sensitive, a later robustness check can compare
2,500 / 10,000 / 40,000 calibration anchors.

