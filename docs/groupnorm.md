# BN batch-stat diagnostic + GroupNorm pilot

This bundle contains two separate next steps.

## Files to copy

```text
src/models/precip_resnet_gn.py
scripts/diagnose_bn_batch_stats.py
scripts/train_groupnorm_pilot.py
scripts/plot_bn_vs_groupnorm.py
slurm/run_bn_batchstat_diagnostic.slurm
slurm/train_groupnorm_pilot.slurm
```

---

# Phase 1: last BatchNorm diagnostic

Submit:

```bash
source load_env.sh
sbatch slurm/run_bn_batchstat_diagnostic.slurm
```

This evaluates the existing BatchNorm `checkpoint_best.pt` and
`checkpoint_last.pt` on the exact fixed 512 validation recipes in:

```text
ordinary eval mode
    -> BN running statistics

instantaneous batch-stat mode
    logical B=4  -> BN sees 40 physical images
    logical B=8  -> BN sees 80 physical images
    logical B=16 -> BN sees 160 physical images
```

No gradients are used. The BN running buffers are not modified.

The result is written to:

```text
$DATA_DIR/derived/experiments/full_epoch_train_validation_v1/
  diagnostic_plots/bn_batch_statistics/bn_batch_stat_diagnostic.csv
```

## Interpretation

If the final checkpoint improves dramatically in batch-stat mode, especially
at B=8 or B=16, then the learned weights still function much better under the
normalization environment used during optimization.

If B=16 is materially more stable/better than B=4 or B=8, larger training
batches might help BatchNorm. If all batch-stat sizes are similar, merely
increasing logical batch size is unlikely to solve the underlying train/eval
normalization mismatch.

This is diagnostic only. Batch-stat inference makes a sample's score depend on
what other samples happen to be in its minibatch, which is undesirable for a
standalone verification metric.

---

# Phase 2: controlled GroupNorm pilot

The model is still ResNet-18. The only architectural change is:

```text
BatchNorm2d -> GroupNorm(32 groups)
```

The stem, residual layout, projection head, loss, K=8 negative definition,
normalization of the input precipitation, crop size, optimizer, learning-rate
schedule, training sampler seed, and validation sampler seed remain unchanged.

Submit:

```bash
sbatch slurm/train_groupnorm_pilot.slurm
```

## Why 20,000 steps?

This covers:

```text
20,000 * 8 = 160,000 anchor visits
```

or about 44% of the 365,167 training anchors.

The existing BN run was already showing strong validation oscillation well
before 20,000 steps, so this is long enough to test the normalization
hypothesis without paying for another full epoch immediately.

## Important control: the LR schedule is NOT compressed

The GroupNorm pilot uses the same LR that the full 45,646-step BatchNorm run
would have at each corresponding step.

Thus:
- warmup still ends around step 2,282
- step 10k uses the same LR as BN step 10k
- step 20k uses the same LR as BN step 20k

This makes the first 20k steps an apples-to-apples normalization comparison.

## Fixed validation

The script uses the same deterministic validation construction and checks that
the newly generated `fixed_validation_recipes.jsonl` is byte-for-byte
identical to the BatchNorm baseline file. It aborts if they differ.

## Validation cadence

GroupNorm is evaluated every 500 steps so we get a denser stability curve.
The existing BN checkpoints at 1,000-step intervals remain directly
comparable at their matching steps.

## Output

```text
$DATA_DIR/derived/experiments/groupnorm_pilot_v1/
```

including:
- `training_history.csv`
- `validation_history.csv`
- `checkpoint_best.pt`
- `checkpoint_last.pt`
- periodic 2,500-step checkpoints
- `summary.json`

At the end of the GPU job, `plot_bn_vs_groupnorm.py` automatically compares
the GroupNorm trajectory with the existing BatchNorm baseline and writes:

```text
diagnostic_plots/bn_vs_gn/
```

with:
- validation loss
- A-P similarity
- mean A-N similarity
- margin
- top-1 retrieval
- GroupNorm train-vs-validation loss
- a CSV summarizing validation-loss variability and checkpoint-to-checkpoint
  jumps

## What would count as a promising GroupNorm result?

Not simply a lower minimum loss.

The useful result would be:
- A-P stays stably high,
- A-N separates progressively,
- margin grows with much smaller checkpoint-to-checkpoint collapse/recovery,
- positive top-1 remains high,
- validation loss has much smaller variance / jump size than BatchNorm,
- and the absolute loss/geometry is still competitive.

If that happens, the next step is a complete GroupNorm epoch followed by the
same FP32 failure-tail and 1/3/6-h tests.

# GroupNorm v1 — full-epoch continuation

This bundle turns the successful 20,000-step GroupNorm pilot into a
reproducible training workflow that can start fresh, stop at an arbitrary
absolute step, or resume from a saved checkpoint.

The current experiment directory remains:

```text
$DATA_DIR/derived/experiments/groupnorm_pilot_v1
```

It is okay that the directory still says `pilot_v1`; keeping the same directory
preserves the exact histories/checkpoints from the first 20,000 steps. It can be
renamed after the experiment is complete if desired.

## Files to copy into the repo

```text
src/models/precip_resnet_gn.py
src/models/checkpoint_loader.py

scripts/train_groupnorm.py
scripts/plot_bn_vs_groupnorm.py
scripts/evaluate_failure_tail.py
scripts/evaluate_timelag_fp32.py

slurm/continue_groupnorm_full_epoch.slurm
slurm/run_groupnorm_full_diagnostics.slurm
```

---

# 1. Continue the completed 20k pilot to the full epoch

Use the **last** checkpoint from the pilot:

```text
checkpoint_last.pt        step 20000
```

Do not resume from `checkpoint_best.pt` (step 19500) for this continuation.
`checkpoint_best.pt` is retained for model selection; `checkpoint_last.pt`
contains the model + AdamW state after all 20,000 already-completed updates.

Submit:

```bash
source load_env.sh
sbatch slurm/continue_groupnorm_full_epoch.slurm
```

No `--end-step` is supplied by the Slurm script, so `train_groupnorm.py`
derives the complete epoch length from the deterministic sampler and trains
through that point (45,646 steps for the current v1 index).

The continuation therefore performs only:

```text
steps 20001 ... 45646
```

rather than replaying the first 20,000 optimization steps.

## Why this is a real continuation

The v1 `ContrastiveBatchSampler` is deterministic for a fixed:

```text
seed
epoch
anchor index
batch size
positive rule
negative rule
patch size
K
```

The training driver rebuilds epoch 0 with the original seed and then skips
exactly the batches already represented by the checkpoint. Because the pilot
used the full-epoch LR schedule rather than a 20k-compressed schedule, the LR
at step 20,001 is exactly the LR the one-shot full-epoch experiment would use
at step 20,001.

The checkpoint restores both:

```text
model_state_dict
optimizer_state_dict
```

so AdamW momentum/state is continued rather than reinitialized.

The current model has no dropout or other stochastic neural-network layer.
The scientific data sequence is therefore reproducible from the deterministic
sampler plus the checkpoint. Exact bit-for-bit floating-point identity across
different GPU/software environments is not promised.

---

# 2. Generic training interface

`train_groupnorm.py` is not hard-coded to step 20,000.

### Fresh complete epoch

```bash
python scripts/train_groupnorm.py
```

### Fresh partial run through absolute step 20000

```bash
python scripts/train_groupnorm.py --end-step 20000
```

### Continue any saved step through an absolute later step

```bash
python scripts/train_groupnorm.py \
  --resume-checkpoint /path/to/checkpoint_step25000.pt \
  --end-step 30000
```

### Continue any saved step through the end of the epoch

```bash
python scripts/train_groupnorm.py \
  --resume-checkpoint /path/to/checkpoint.pt
```

`--max-steps` remains accepted as a backward-compatible alias for
`--end-step`, but `--end-step` is clearer because the value is an absolute
optimization step.

---

# 3. Resume-safety / provenance behavior

The driver is deliberately conservative.

It checks the existing experiment `config.json` against scientifically
meaningful settings such as:

```text
batch size
K
patch size
negative rules
projection dimensions
temperature
LR schedule
weight decay
training seed
validation definition
input/statistics paths
```

It also checks compatible metadata stored inside the checkpoint.

The fixed validation set is regenerated deterministically at startup. If an
existing `fixed_validation_recipes.jsonl` does not match byte-for-byte, the job
stops rather than silently changing the validation problem.

For every invocation it writes a step-specific file such as:

```text
invocation_step20000_to45646.json
```

containing the resolved start/target steps and source checkpoint SHA-256.

Before `checkpoint_last.pt` can be overwritten at the end of the continuation,
the source checkpoint is archived as:

```text
checkpoint_resume_source_step20000.pt
```

The original pilot `summary.json` is similarly preserved as:

```text
summary_through_step20000.json
```

Future checkpoints use schema v2 and also store the step at which the best
validation loss occurred.

## Recovery after an interrupted continuation

Periodic checkpoints are saved every 2,500 steps.

For example:

```text
checkpoint_step22500.pt
checkpoint_step25000.pt
...
```

If a job fails after CSV history was written beyond the most recent checkpoint,
resume from the most recent saved checkpoint. The script backs up the longer
history and rolls the active history back to the checkpoint step before
continuing. This prevents log rows from representing updates that are no longer
present in the restored model.

The Slurm script allows the source checkpoint to be overridden:

```bash
sbatch \
  --export=ALL,RESUME_CHECKPOINT="$EXP/checkpoint_step30000.pt" \
  slurm/continue_groupnorm_full_epoch.slurm
```

---

# 4. What to inspect when the full epoch finishes

The training job automatically refreshes the BN-vs-GN trajectory plots.

First inspect:

```text
summary.json
validation_history.csv
diagnostic_plots/bn_vs_gn/
```

We care about whether the stable GroupNorm behavior persists from 20k through
the low-LR tail of the epoch, rather than merely whether a new minimum loss is
found.

In particular:

```text
cos(A,P) remains reasonably stable
mean cos(A,N) remains well separated
hardest-negative margin remains large
top-1 retrieval remains high
validation loss does not develop BN-like collapse/recovery oscillations
best and final checkpoints remain reasonably similar
```

---

# 5. Post-training physical diagnostics

Once the complete epoch is confirmed, submit:

```bash
sbatch slurm/run_groupnorm_full_diagnostics.slurm
```

This evaluates **both**:

```text
checkpoint_best.pt
checkpoint_last.pt
```

with FP32 inference.

For each checkpoint it runs:

1. fixed-validation failure-tail diagnostics by precipitation intensity,
   including conventional MAE/RMSE/correlation/FSS comparisons and failure
   galleries;

2. the held-out same-valid-time 1/3/6-h time-lag gradation diagnostic.

The diagnostic scripts use `src/models/checkpoint_loader.py`, which selects the
correct ResNet implementation from checkpoint metadata. This keeps the same
diagnostic code usable for both the older BatchNorm checkpoint family and the
new GroupNorm checkpoint family.

These remain validation diagnostics. Do **not** open the sealed test split yet.

---

# 6. Scientific decision after the full epoch

A complete GroupNorm epoch is useful only if the pilot behavior survives.

The strongest outcome would be:

```text
stable fixed-validation geometry through 45,646
low wet-case retrieval failure rate
no egregious high-intensity failure tail
graded 1 h > 3 h > 6 h similarity
best and final checkpoints both physically sensible
```

That would justify freezing the representation architecture/training recipe and
moving on to forecast-vs-observation evaluation, controlled Wang-style
transformations, and comparison with traditional verification metrics.

It would still not mean the learned verification metric is proven useful;
that question belongs to the forecast/observation evaluation.

