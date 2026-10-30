# Final diagnostic suite

The reproducible study uses five diagnostic families. Historical BatchNorm,
FP32, V1/V2, and superseded plotting experiments are preserved by the research
snapshot tag but are not part of the supported workflow.

All commands below assume:

```bash
source load_env.sh
```

## 1. Controlled morphology response

Purpose: measure how SYN-3, SYN-8, NFV, and traditional metrics respond to the
same controlled precipitation perturbations.

Canonical entry points:

```text
scripts/evaluate/morphology.py
scripts/plot/morphology.py
slurm/evaluate_morphology.slurm
```

Submit:

```bash
./scripts/submit_diagnostic.sh morphology
```

The traditional metrics use a 0.1-inch threshold (2.54 mm internally). The
canonical summary includes per-case 95% confidence intervals for the learned
and traditional response curves.

## 2. Stage IV percentile galleries

Purpose: compare learned and traditional scores on real HRRR/Stage-IV pairs and
inspect cases where their percentile rankings agree or disagree most strongly.

```text
scripts/evaluate/stageiv.py
scripts/plot/stageiv_buckets.py
slurm/evaluate_stageiv.slurm
```

Submit:

```bash
./scripts/submit_diagnostic.sh stageiv
```

Traditional thresholds are 0.1 and 0.5 inch. FSS-5/FSS-10 and the non-dry
sample filter use 0.1 inch. Gallery alignment/disagreement is based on absolute
percentile-rank differences so learned and traditional metrics are compared by
ordering rather than raw calibration.

## 3. Forecast-hour / die-off curves

Purpose: evaluate how all final metrics change with HRRR forecast lead over the
full Stage-IV forecast-hour archive.

The archive is prepared once, then the canonical pipeline is:

```text
scripts/evaluate/forecast_hour_traditional.py  # CPU
scripts/evaluate/forecast_hour.py              # GPU, SYN-3/SYN-8/NFV
scripts/plot/forecast_hour.py
slurm/evaluate_forecast_hour.slurm
```

Submit the diagnostic after the archive/preflight exists:

```bash
./scripts/submit_diagnostic.sh forecast-hour
```

The final plot compares SYN-3, SYN-8, NFV, ETS, POD, success ratio, FSS-5, and
FSS-10 by forecast hour. Traditional metrics use the 0.1-inch threshold.

## 4. Feature-space inversion

Purpose: interpret the learned similarity space by optimizing the precipitation
input while holding the trained network fixed and targeting prescribed cosine
similarities.

```text
src/eval/feature_inversion.py
scripts/evaluate/select_inversion_case.py
scripts/evaluate/inversion.py
scripts/plot/inversion.py
slurm/evaluate_inversion.slurm
```

Example:

```bash
CASE_INDEX=299 N_STEPS=1200 ./scripts/submit_diagnostic.sh inversion
```

The canonical continuation targets are:

```text
+0.75, +0.50, +0.25, 0.00, -0.25, -0.50, -0.75, -1.00
```

These are constrained input inversions, not decoded precipitation fields. At
very low similarities they should be interpreted as representation stress tests
rather than realistic forecasts.

## 5. NFV time-lag design diagnostics

Purpose: document why time-lagged forecast relationships are useful natural
positive examples for NFV and compare them with unrelated negatives.

```text
scripts/evaluate/timelag.py
scripts/plot/timelag.py
scripts/plot/timelag_cases.py
scripts/plot/timelag_negatives.py
scripts/submit_timelag_experiment.sh
slurm/evaluate_timelag.slurm
```

The time-lag experiment retains its array-job workflow because it evaluates a
large number of Stage-IV/HRRR pairings and is primarily a training-design
analysis rather than a routine final-model score calculation.

## Supporting method figures

Useful but secondary scripts include:

```text
scripts/plot/transform_schematic.py
scripts/audit_area_size_response.py
```

They illustrate transform definitions or investigate a specific response and
are not required to reproduce the primary final-result figures/galleries.
