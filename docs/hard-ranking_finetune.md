# Hard-ranking fine-tune v1

This is the first **hard-problem fine-tuning experiment** built on the best
GroupNorm checkpoint from the previous HRRR contrastive run.

It does **not** overwrite or continue the previous experiment directory. The
new default output is:

```text
$DATA_DIR/derived/experiments/hard_ranking_finetune_v1
```

The default source weights are:

```text
$DATA_DIR/derived/experiments/groupnorm_pilot_v1/checkpoint_best.pt
```

That checkpoint was the previous best GroupNorm model (step 43,500 in the run
we inspected). The new run loads **model weights only** from it. AdamW state is
fresh because the training objective has changed materially.

## Scientific design

For every logical anchor `A`, the sampler constructs:

```text
A, P, TL3, TL6, E1, E2, E3, E6, N1, N2, N3, N4
```

All twelve fields use the same 256x256 crop.

### Same-valid-time time-lag family

`P`, `TL3`, and `TL6` all verify the **same 6-h valid interval** as `A` but come
from increasingly displaced initialization times:

```text
S(A,P) > S(A,TL3) > S(A,TL6)
```

The sampler uses one branch per anchor. For F07-F11 it must use larger leads
(earlier initializations); for F13-F17 it must use smaller leads (later
initializations). At F12 the branch is chosen 50/50 when both complete branches
exist.

### Same-initialization forecast-evolution family

`E1`, `E2`, `E3`, and `E6` stay on the **same HRRR forecast trajectory** as `A`
but shift the 6-h accumulation interval by 1, 2, 3, and 6 h in the feasible
branch direction:

```text
S(A,E1) > S(A,E2) > S(A,E3) > S(A,E6)
```

This is intentionally called **forecast-evolution sensitivity**, not pure
translation sensitivity. The fields can move, intensify, weaken, grow, decay,
merge, split, or otherwise evolve naturally.

Evolution fields are prohibited from crossing the requested data split. A few
anchors near train/validation boundaries therefore become ineligible.

### Global retention family

Four unrelated negatives retain the old objective. Training uses the same
`climo45` negative definition; fixed validation uses `calendar_any` exactly as
in the previous experiments.

Their role is mostly a guardrail against catastrophic forgetting rather than
the main new learning problem.

## Loss

The optimized projection-space objective is

```text
0.25 * [L_global / log(K+1)]
+ 1.00 * [L_TL / log(2)]
+ 1.00 * [L_evolution / log(2)]
```

with `K=4` by default.

* `L_global` is the existing symmetric explicit-negative InfoNCE.
* `L_TL` is an all-pairs smooth ranking loss over `[P, TL3, TL6]`.
* `L_evolution` is an all-pairs smooth ranking loss over `[E1,E2,E3,E6]`.
* Ranking loss for an ordered pair `(better,worse)` is
  `softplus((S_worse - S_better) / tau_rank)`.
* `tau_global = tau_rank = 0.10` initially.
* Exact-identical raw field pairs are masked from strict ranking constraints,
  avoiding impossible labels for completely dry/identical patches.

The `log(K+1)` and `log(2)` factors are reference-scale normalizations. They
make the lambdas easier to read: at completely uninformative logits the three
normalized terms are roughly O(1). If a global N is masked as an exact
duplicate, its exact per-sample chance loss is slightly lower than `log(K+1)`;
this normalization is used as an objective-scale reference, not a calibrated
probability statement.

There is deliberately **no** imposed ordering between the TL family and the E
family, and no initial requirement that `P > E1`. Those are not relationships
we can defend universally.

## Optimization

Default first run:

```text
12,000 fine-tuning steps
batch size 8
96 physical fields / step
full network trainable
AdamW, weight decay 1e-4
500-step linear warmup: 1e-5 -> 5e-5
cosine decay after warmup: 5e-5 -> 5e-6
bf16 model forward on H100
FP32 cosine similarities and loss math
```

The FP32 loss calculation is deliberate because the hard ranking differences
can be very small near cosine similarity 1.0.

This first run samples each hard-eligible anchor at most once. It refuses a
requested step count longer than one deterministic hard-ranking epoch.

## Validation / checkpoint selection

A separate fixed 512-case **validation** cache is created and saved as
`fixed_validation_recipes.jsonl`. Test data are not used.

Validation reports both projection `z` and normalized backbone `h` geometry.
For `z`, the important outputs include:

```text
global top-1 and margin
P, TL3, TL6 mean similarities
P > TL3, P > TL6, TL3 > TL6 rates
full P > TL3 > TL6 rate
E1, E2, E3, E6 mean similarities
all evolution pairwise ranking rates
full E1 > E2 > E3 > E6 rate
separate normalized global / TL / evolution losses
weighted composite objective
```

Step 0 is evaluated **before any fine-tuning**. That is the direct baseline
from the old best checkpoint.

`checkpoint_best.pt` is selected by the fixed-validation weighted composite
objective. Regular 1,000-step checkpoints and `checkpoint_last.pt` are also
kept, so we do not have to trust one scalar checkpoint criterion if the
scientific diagnostics show a tradeoff.

## Files to install in the GitHub/project tree

Copy these files to the corresponding paths under `$PROJECT_DIR`:

```text
src/data/hard_ranking_sampler.py
src/training/ordered_ranking_loss.py
scripts/train_hard_ranking_finetune.py
scripts/plot_hard_ranking_finetune.py
slurm/train_hard_ranking_finetune.slurm
slurm/resume_hard_ranking_finetune.slurm
```

The existing files `src/models/precip_resnet_gn.py`,
`src/training/contrastive_loss.py`, and the v1 `src.data` modules are reused and
should not be replaced by this bundle.

## Run

From the project root:

```bash
source load_env.sh
sbatch slurm/train_hard_ranking_finetune.slurm
```

The job performs a 256-recipe metadata contract check before training. If the
same-valid-time or same-initialization relationships are wired incorrectly, it
should fail before the expensive run.

After training it automatically runs the plotting script. Output figures go
under:

```text
$DATA_DIR/derived/experiments/hard_ranking_finetune_v1/plots
```

## What to inspect first

The first scientific question is not simply whether total loss falls. Compare
step 0 with the first several validation points and ask:

1. Does `P > TL3 > TL6` improve materially?
2. Does `E1 > E2 > E3 > E6` improve materially?
3. Do the raw similarities spread into a sensible graded hierarchy rather than
   merely satisfying a tiny numerical ordering?
4. Does old global retrieval remain high?
5. Are the weighted training contributions actually concentrated on the new
   hard objectives, as intended?

If the new rankings do not move after ~1-2k steps, `5e-5` may be too cautious.
If global retrieval or the previously learned representation collapses early,
we should back off or increase the retention weight. The purpose of v1 is to
measure that tradeoff rather than assume the first hyperparameters are final.

