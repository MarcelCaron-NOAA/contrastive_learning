# HRRR contrastive sampler pipeline v1

This implements the sampling contract agreed on after the lag/negative diagnostics.

## Scientific sampling contract

For each training sample:

- **A (anchor)**: one eligible HRRR QPF6 field, F07-F17, from the training split.
- **P (positive)**: same valid time as A; randomly choose the HRRR forecast initialized 1 h earlier or 1 h later.
- **N (negative)**: K explicit HRRR fields satisfying the configured negative policy.
- **Crop**: one random 256x256 native-HRRR crop per sample. A, P, and all N for that sample use exactly the same coordinates. Different samples in a minibatch choose crops independently.
- **Epoch**: every eligible training anchor is visited exactly once, globally shuffled without replacement.
- **Minibatches**: not stratified by lead, UTC hour, season, or geography; those are mixed naturally by the global shuffle.
- **Negative uniqueness**: negative field IDs are sampled without replacement across a minibatch; they are also prevented from being an A or P field in that same minibatch.
- **Dry patches**: retained. A negative patch is masked only when its raw pixels are exactly identical to A or P.
- **Training domain**: full usable HRRR grid; no Stage-IV geographic restriction.

### `climo45` negative mode (default)

N must have:

- same split as A,
- same forecast lead as A,
- same valid UTC hour as A,
- absolute valid-time separation strictly greater than 90 days,
- a different calendar year,
- circular climatological-day distance <= 45 days.

### `calendar_any` negative mode

Retains the original comparison option:

- same split,
- same forecast lead,
- same valid UTC hour,
- absolute valid-time separation strictly greater than 90 days,
- no annual-cycle matching requirement.

## Files to add/replace in the repository

- `src/data/hrrr_archive.py`
- `src/data/contrastive_index.py`
- `src/data/contrastive_sampler.py`
- `src/data/contrastive_dataset.py`
- `src/data/__init__.py`
- `tests/test_contrastive_sampler.py`
- `scripts/smoke_test_sampler.py`

## First run on Ursa

From the repository root:

```bash
source load_env.sh

python -m pytest -q tests/test_contrastive_sampler.py
```

Then run the real-data smoke test:

```bash
python scripts/smoke_test_sampler.py \
  --negative-mode climo45 \
  --negative-count 8 \
  --batch-size 16 \
  --patch-size 256 \
  --metadata-recipes 10000 \
  --materialize-batches 2 \
  --transform log1p
```

The first smoke test deliberately uses `log1p` without z-scoring. The pipeline already supports `log1p_zscore`, but the final mean/std should be estimated from the training sampling distribution first rather than inserting provisional normalization constants.

The smoke test writes:

```text
$DATA_DIR/metadata/dataset_v1/sampler_smoke_test_summary.json
```

## What the smoke test proves

The metadata phase checks thousands of recipes without reading precipitation arrays:

- A is training F07-F17,
- P has identical valid time and the correct +/-1-h initialization relationship,
- N obeys all climo45 rules,
- N IDs are unique within each minibatch,
- no N is also an A/P field within the same minibatch,
- crop coordinates are valid,
- anchors are not repeated within the sampled epoch sequence.

The materialization phase then reads real ZipStore patches and checks:

- A shape `[B, 1, 256, 256]`,
- P shape `[B, 1, 256, 256]`,
- N shape `[B, K, 1, 256, 256]`,
- float32 and finite tensors,
- exact-duplicate negative masking,
- occurrence of dry A/P patches,
- approximate real I/O throughput.

It also compares epoch 0 and epoch 1 to verify that P choice, crop location, and negative sets actually change.

