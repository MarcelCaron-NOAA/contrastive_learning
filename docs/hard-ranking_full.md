# Hard-ranking V3 full run

This is the clean **from-scratch** joint-training experiment requested after the V2
fine-tuning diagnostics.

It does **not** load V1 or V2 weights. It reuses the V2 hard-ranking sampler and
ordered-ranking loss, but initializes a new GroupNorm ResNet-18 and trains all three
objectives together from optimization step 1.

## Output isolation

Everything goes under:

```text
$DATA_DIR/derived/experiments/hard_ranking_v3_full_v1
```

so the previous GroupNorm V1 and hard-ranking V2 experiments are untouched.

## Per-anchor training tuple

```text
A, P/TL1, TL3, TL6, E1, E2, E3, E6, N1, N2, N3, N4
```

All fields in a logical sample use the same 256x256 crop.

Relations:

```text
P > TL3 > TL6
E1 > E2 > E3 > E6
P should outrank the four unrelated N fields
```

There is still no imposed ordering between the TL family and E family.

## Objective

All three normalized loss families have equal weight:

```text
L_total =
    1.0 * [L_global / log(K+1)]
  + 1.0 * [L_TL / log(2)]
  + 1.0 * [L_evolution / log(2)]
```

with `K=4`, `tau_global=0.10`, and `tau_rank=0.10`.

The TL and evolution losses are the existing all-pairs smooth ranking losses and
average over their valid pair constraints internally. Exact-identical raw fields
remain masked from impossible strict-ranking constraints.

## Optimization

This deliberately returns to the full GroupNorm schedule used before V2:

```text
model                  GroupNorm ResNet-18
optimizer              AdamW
weight decay           1e-4
batch size             8 logical anchors
physical fields/step   8 * (8 + 4) = 96
forward                bf16 autocast on H100
similarity/loss math   FP32

warmup start LR        3e-5
peak LR                3e-4
warmup fraction        0.05 of the COMPLETE hard-ranking epoch
minimum LR             3e-6
post-warmup schedule   cosine decay, no restarts
```

Crucially, the script does **not** hard-code 45,646 steps. It asks
`HardRankingBatchSampler` for the number of hard-eligible training anchors and
derives the complete epoch length from that. This matters because the TL/evolution
branch requirements can exclude anchors that were legal in the older A/P/N-only
sampler.

The LR schedule is always referenced to the complete hard-ranking epoch. An
optional partial `--end-step` therefore does not compress or distort the schedule.

## Validation and checkpointing

```text
fixed validation cases     512
validation cadence         500 steps
periodic checkpoint        2500 steps
training negatives         climo45
validation negatives       calendar_any
best checkpoint criterion  normalized z-space composite objective
```

Step 0 is evaluated before training, so the random-initialization geometry is
recorded.

Outputs include:

```text
config.json
fixed_validation_recipes.jsonl
training_history.csv
validation_history.csv
checkpoint_best.pt
checkpoint_step*.pt
checkpoint_last.pt
summary.json
plots/
```

## Existing project files reused

These should already be present from V2 and are intentionally not replaced:

```text
src/data/hard_ranking_sampler.py
src/training/ordered_ranking_loss.py
src/models/precip_resnet_gn.py
src/training/contrastive_loss.py
```

## Install

From the downloaded bundle, copy:

```text
scripts/train_hard_ranking_v3_full.py
scripts/plot_hard_ranking_v3_full.py
scripts/submit_hard_ranking_v3_full.sh
slurm/train_hard_ranking_v3_full.slurm
slurm/resume_hard_ranking_v3_full.slurm
```

into the matching directories under `$PROJECT_DIR`.

For example:

```bash
cd $PROJECT_DIR

cp /path/to/bundle/scripts/train_hard_ranking_v3_full.py scripts/
cp /path/to/bundle/scripts/plot_hard_ranking_v3_full.py scripts/
cp /path/to/bundle/scripts/submit_hard_ranking_v3_full.sh scripts/
cp /path/to/bundle/slurm/train_hard_ranking_v3_full.slurm slurm/
cp /path/to/bundle/slurm/resume_hard_ranking_v3_full.slurm slurm/

chmod +x scripts/submit_hard_ranking_v3_full.sh
```

## Submit in the background

Either:

```bash
cd $PROJECT_DIR
source load_env.sh
sbatch slurm/train_hard_ranking_v3_full.slurm
```

or:

```bash
cd $PROJECT_DIR
source load_env.sh
bash scripts/submit_hard_ranking_v3_full.sh
```

Check:

```bash
squeue -u "$USER"

grep -hE \
'HARD-RANKING|Hard-eligible|Full V3 anchor epoch|Target end step|Warmup steps|TRAIN step|VAL step|NEW BEST|RUN COMPLETE|Traceback|ERROR' \
"$LOG_DIR"/hard_rank_v3_*.out \
"$LOG_DIR"/hard_rank_v3_*.err
```

## Optional short preflight

The training program already performs a 256-recipe metadata contract test before
the expensive run. If you also want a short GPU training preflight before committing
to the full epoch:

```bash
sbatch \
  --export=ALL,END_STEP=500 \
  --time=01:00:00 \
  slurm/train_hard_ranking_v3_full.slurm
```

That 500-step job is a valid partial V3 run, not a separate scientific experiment.
If it looks healthy, continue the same deterministic epoch with:

```bash
sbatch slurm/resume_hard_ranking_v3_full.slurm
```

If you want a truly disposable smoke test instead, invoke the Python training script
directly with a different `--output-dir`.

## Resume if walltime is insufficient

Periodic checkpoints are written every 2500 steps. The resume job automatically
finds the latest `checkpoint_step*.pt`:

```bash
sbatch slurm/resume_hard_ranking_v3_full.slurm
```

Or choose a checkpoint explicitly:

```bash
sbatch \
  --export=ALL,RESUME_CHECKPOINT="$DATA_DIR/derived/experiments/hard_ranking_v3_full_v1/checkpoint_step25000.pt" \
  slurm/resume_hard_ranking_v3_full.slurm
```

The resumed run reconstructs epoch 0, skips the already-completed deterministic
recipe batches, restores model + AdamW state, and continues on the original
full-epoch LR schedule.

## Walltime note

The supplied Slurm files request 16 h because this run processes 96 physical fields
per logical batch and traverses a complete hard-ranking epoch. If `gpuwf` on Ursa
rejects that walltime, override it at submission, for example:

```bash
sbatch --time=10:00:00 slurm/train_hard_ranking_v3_full.slurm
```

and continue with the resume script if needed.

