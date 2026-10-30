# Natural Forecast Variability (NFV) experiment

NFV is the final name for the from-scratch natural forecast-variability model
historically developed as **hard-ranking V3**.  Its checkpoint and on-disk
experiment identifier remains:

```text
hard_ranking_v3_full_v1
```

That legacy identifier is retained intentionally so the final trained
checkpoint remains resume-compatible and existing evaluation products do not
need to be rewritten.

Canonical entry points:

```text
src/training/nfv.py
scripts/train_nfv.py
scripts/submit_nfv.sh
slurm/train_nfv.slurm
```

## Per-anchor training tuple

```text
A, P/TL1, TL3, TL6, E1, E2, E3, E6, N1, N2, N3, N4
```

All fields in a logical sample use the same 256 x 256 crop.

The supervised order relations are:

```text
P > TL3 > TL6
E1 > E2 > E3 > E6
P should outrank unrelated negatives N1..N4
```

No ordering is imposed between the time-lag and forecast-evolution families.

## Objective

The final model uses equal weights on three normalized loss families:

```text
L_total =
    1.0 * [L_global / log(K+1)]
  + 1.0 * [L_TL / log(2)]
  + 1.0 * [L_evolution / log(2)]
```

with `K=4`, global temperature 0.10, and ranking temperature 0.10.

The global term is symmetric explicit-negative InfoNCE. The time-lag and
evolution terms are smooth all-pairs ordered-ranking losses. Exact-identical
raw pairs remain masked from impossible strict-ranking constraints.

## Optimization

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
warmup fraction        0.05 of the complete NFV epoch
minimum LR             3e-6
post-warmup schedule   cosine decay, no restarts
```

The complete epoch length is derived from the hard-ranking sampler at runtime.
The learning-rate schedule always references that full epoch even when an
optional `--end-step` is used for a partial run.

## Validation and checkpointing

```text
fixed validation cases     512
validation cadence         500 steps
periodic checkpoint        2500 steps
training negatives         climo45
validation negatives       calendar_any
best checkpoint criterion  normalized z-space composite objective
```

Step 0 is evaluated before training so the random-initialization geometry is
recorded.

## Submit

```bash
./scripts/submit_nfv.sh
```

Default output:

```text
$DATA_DIR/derived/experiments/hard_ranking_v3_full_v1
```

Optional short run:

```bash
END_STEP=500 WALLTIME=01:00:00 ./scripts/submit_nfv.sh
```

Resume explicitly from the last checkpoint:

```bash
RESUME_CHECKPOINT="$DATA_DIR/derived/experiments/hard_ranking_v3_full_v1/checkpoint_last.pt" \
./scripts/submit_nfv.sh
```

The same `slurm/train_nfv.slurm` launcher handles both fresh and resumed runs;
there is no separate resume Slurm script.
