# Final synthetic-supervision experiments: SYN-3 and SYN-8

SYN-3 and SYN-8 share one training engine:

```text
src/training/synthetic.py
scripts/train_syn.py
scripts/submit_syn.sh
slurm/train_syn.slurm
```

The two experiments intentionally differ only in the perturbation families used
for supervision.  Architecture, HRRR patches, preprocessing, optimizer,
training schedule, SupCon formulation, target-distance penalty, seeds, and
validation design are shared.

## Shared model and objective

```text
encoder                 GroupNorm ResNet-18
projection head         512 hidden -> 128 output
input                    256 x 256 HRRR QPF6 patches
preprocessing            log1p z-score
optimizer                AdamW
batch                    8 underlying HRRR cases
SupCon temperature       0.10
SupCon weight            1.0
penalty weight           1000.0
peak learning rate       3e-4
warmup start             3e-5
warmup fraction          0.05 of the full deterministic anchor epoch
minimum learning rate    3e-6
weight decay             1e-4
```

For each active perturbation family, severity rank is sampled independently:

```text
r ~ Uniform{1,...,10}
```

The Wang-style target cosine distance is:

```text
d_target(r) = 0.009 * r^2
S_target(r) = 1 - 0.009 * r^2
```

so target similarity runs from 0.991 at rank 1 to 0.100 at rank 10.

The total objective is:

```text
L = L_SupCon + 1000 * mean(|d_cos - d_target|)
```

Other underlying HRRR cases in the minibatch provide the SupCon negatives; no
separate negative fields are sampled for these experiments.

## SYN-3

Four views per underlying case:

```text
original
+ displacement
+ intensity
+ area size
```

Physical rank mappings:

```text
displacement   +/- r grid cells
intensity      lower = 1 - 0.05*r; upper = 1 + 0.10*r
area size      same ladder as intensity
```

These are the three perturbation definitions adapted from Wang et al. (2026).
With the default batch size, SYN-3 processes 8 x 4 = 32 physical fields per
optimizer step.

### Submit

Smoke test:

```bash
./scripts/submit_syn.sh syn3 smoke
```

Full run:

```bash
./scripts/submit_syn.sh syn3 full
```

Default output:

```text
$DATA_DIR/derived/experiments/synthetic_perturb_syn3_v2
```

## SYN-8

Nine views per underlying case: the original plus eight perturbation families.
SYN-8 contains all three SYN-3 families and adds:

```text
smoothing
object concentration/dispersion
principal-axis deformation
rotation
tile fragmentation
```

Additional physical mappings:

```text
smoothing                    +0.6*r sigma (one-sided)
object dispersion            +/-round(1.6*r) cells
principal-axis deformation   +/-0.1*r log2 scale
rotation                     +/-9*r degrees
tile fragmentation           +/-round(1.2*r) cells
```

Smoothing is intentionally one-sided during training. Sharpening remains an
out-of-sample morphology diagnostic rather than being forced onto a symmetric
training target.

With the default batch size, SYN-8 processes 8 x 9 = 72 physical fields per
optimizer step.

### Submit

Smoke test:

```bash
./scripts/submit_syn.sh syn8 smoke
```

Full run:

```bash
./scripts/submit_syn.sh syn8 full
```

Default output:

```text
$DATA_DIR/derived/experiments/synthetic_perturb_syn8_v2
```

## Resume

Both experiments write `checkpoint_best.pt`, `checkpoint_last.pt`, and periodic
`checkpoint_step*.pt` files. Resume the same deterministic epoch by providing a
checkpoint explicitly:

```bash
RUN_NAME=synthetic_perturb_syn3_v2 \
RESUME_CHECKPOINT="$DATA_DIR/derived/experiments/synthetic_perturb_syn3_v2/checkpoint_last.pt" \
./scripts/submit_syn.sh syn3 full
```

The shared implementation preserves the validated final experiment definitions
and checkpoint metadata from the formerly separate SYN-3 and SYN-8 trainers.
