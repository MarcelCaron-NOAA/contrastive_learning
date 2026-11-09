# Dataset v1 index + precipitation preprocessing diagnostic

## Locked split definition

All dates are inclusive and split by the END/valid time of the 6-h QPF field.

- TRAIN: 2020-12-02 through 2024-09-20
- GUARD 1: 2024-09-21 through 2024-09-30 (10 days)
- VALIDATION: 2024-10-01 through 2025-09-20
- GUARD 2: 2025-09-21 through 2025-09-30 (10 days)
- TEST: 2025-10-01 through 2026-09-22

## Pair definitions

Anchor A
- HRRR F07-F17

Positive P
- same valid time as A
- initialization offset -1 h or +1 h
- BOTH directions required in the eligible-anchor table

Negative N
- selected dynamically
- same split as A
- same forecast lead as A
- same UTC valid hour as A
- absolute valid-time separation strictly > 90 days
- no same-season requirement

## Build index

```bash
INDEX_DIR="$DATA_DIR/metadata/dataset_v1"

python scripts/build_dataset_index.py \
  --manifest "$DATA_DIR/forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz" \
  --output-dir "$INDEX_DIR"
```

## Raw vs log1p training-data diagnostic

This is descriptive only. The final transform should be chosen by a controlled
model ablation with identical architecture, sampler, optimizer, seeds, and
validation set.

```bash
PREP_DIR="$DATA_DIR/derived/experiments/preprocessing_v1"

python scripts/analyze_precip_transforms.py \
  --manifest "$DATA_DIR/forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz" \
  --anchor-index "$INDEX_DIR/eligible_anchors_v1.csv.gz" \
  --output-dir "$PREP_DIR" \
  --n-patches 5000
```

