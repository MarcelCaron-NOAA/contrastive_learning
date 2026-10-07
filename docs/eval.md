# Stage-IV reference evaluation + Wang-style controlled transformations (v1)

This bundle starts the **evaluation** phase of the frozen GroupNorm v1
precipitation similarity model.

The two workstreams are intentionally separate:

1. **HRRR vs Stage IV on the validation period**
   - Does learned S(F,O) occupy a useful range?
   - Do visually good/bad forecast-observation pairs rank sensibly?
   - How does learned similarity relate to correlation and FSS?
   - What meteorological cases drive disagreements?
   - Does Stage-IV regridding choice materially change the learned score?

2. **Controlled Wang-style transformations on an independent validation subset**
   - displacement bias
   - intensity bias
   - precipitation-area-size bias
   - learned metric and traditional metrics as functions of known error size

The sealed 2025-10-01–2026-09-22 test period is **not used** here.

---

## Files to copy into the repo

```text
src/eval/wang_transforms.py
src/eval/verification_metrics.py
src/models/checkpoint_loader.py   # if not already present

scripts/make_stageiv_pair_manifest.py
scripts/build_stageiv_reference_archive.py
scripts/plot_stageiv_alignment_check.py
scripts/evaluate_stageiv_reference.py
scripts/plot_stageiv_galleries.py

scripts/build_wang_transform_archive.py
scripts/evaluate_wang_transformations.py
scripts/plot_wang_style.py
scripts/plot_wang_raw_scatter.py

configs/stageiv_reference_v1.json
configs/wang_transform_v1.json

slurm/run_stageiv_reference_build.slurm
slurm/run_stageiv_reference_eval.slurm
slurm/run_wang_transform_eval.slurm
```

---

# A. Stage-IV reference experiment

## Scientific design

Initial archive:

```text
split       validation only
HRRR leads  F07–F17
valid ends  00, 06, 12, 18 UTC
accum       exact 6-h interval
patch       256x256 native HRRR
target      1000 accepted F/O cases
```

Stage IV is regridded to the native HRRR grid in two ways:

```text
primary      MET BUDGET
sensitivity  MET BILIN
width         2
```

The model receives **exactly the same transform for HRRR and Stage IV**:

```text
raw mm/6h -> log1p -> HRRR-training mean/std
```

There is deliberately no Stage-IV-specific normalization.

## Why BUDGET primary?

Precipitation is an accumulated quantity, and METplus precipitation examples
use BUDGET regridding. BILIN is retained as a direct sensitivity experiment.
We do not assume BUDGET is harmless: the evaluation explicitly compares the
learned scores, raw remapped fields, correlation/RMSE, and rank ordering under
both choices.

## 1. Build candidate list

The manifest generator selects validation anchors at F07–F17 whose 6-h valid
end is 00/06/12/18 UTC. It generates deterministic crop coordinates.

It also writes:

```text
stageiv_files_needed.txt
stageiv_files_needed.csv
```

so only the required `st4_conus.YYYYMMDDHH.06h.grb2` files need to be staged.

You can run this before staging data:

```bash
python scripts/make_stageiv_pair_manifest.py --candidate-count 3000
```

Then stage the listed files from the archive source you use on Ursa/HPSS.

## 2. Build paired raw-field archive

You need two site-specific paths:

```bash
export STAGEIV_ROOT=/path/to/staged/stageiv_6h_files
export HRRR_GRID_TEMPLATE=/path/to/one/native/hrrr/grib2/file
```

`HRRR_GRID_TEMPLATE` may be any MET-readable gridded file defining exactly the
same native HRRR grid as the project's QPF6 Zarr archive.

Ensure `regrid_data_plane` is on PATH, then submit:

```bash
sbatch slurm/run_stageiv_reference_build.slurm
```

Outputs:

```text
$DATA_DIR/derived/evaluation/stageiv_reference_v1/
  candidate_pairs.csv
  stageiv_files_needed.*
  hrrr_qpf6_mm.npy
  stageiv_budget_qpf6_mm.npy
  stageiv_bilin_qpf6_mm.npy
  cases.csv
  archive_metadata.json
  alignment_check/*.png
```

The builder rejects Stage-IV patches containing missing/non-finite values.
Missing observations are **not converted to zero**.

### Mandatory alignment check

Before trusting model scores, inspect the `alignment_check` figures. Confirm:

- HRRR and Stage IV have the same geographic orientation;
- no reflection/translation is evident;
- precipitation magnitudes are in the same physical units/order;
- BUDGET-vs-BILIN differences look like interpolation, not grid misalignment.

If orientation is wrong, stop. The builder has explicit `--flip-y` and
`--flip-x` switches, but they should only be used after diagnosing a real
orientation mismatch.

## 3. Evaluate Stage IV

Once alignment is accepted:

```bash
sbatch slurm/run_stageiv_reference_eval.slurm
```

The frozen checkpoint defaults to:

```text
groupnorm_pilot_v1/checkpoint_best.pt
step 43500
```

Outputs include:

```text
results/per_case_metrics.csv.gz
results/score_summary.csv
results/regrid_and_association_summary.json

results/plots/
  learned_similarity_histogram.png
  budget_vs_bilin_learned_similarity.png
  learned_vs_correlation.png
  learned_vs_fss.png

results/galleries/
  highest/
  lowest/
  fss_disagreement/
  corr_disagreement/
```

Traditional diagnostics are calculated for:

```text
FSS thresholds: 0.1, 1, 5, 10 mm / 6 h
FSS windows:    1, 7, 15, 31 HRRR grid cells
```

The initial representative scatter uses FSS threshold 1 mm / window 15 cells;
this is a diagnostic choice, not yet a "best" FSS configuration.

## Initial go/no-go questions

We are looking for:

1. `S(F,O)` has non-degenerate spread rather than everything mapping near 0 or 1.
2. Highest/lowest-score galleries are visually credible.
3. Learned similarity has broad positive rank association with correlation/FSS,
   but not perfect redundancy.
4. Large learned-vs-FSS/correlation rank disagreements have meteorologically
   interpretable causes rather than obvious model nonsense.
5. BUDGET vs BILIN does not reorder the learned metric dramatically. If it does,
   Stage-IV preprocessing needs more work before forecast-hour experiments.

After this is settled, BUDGET can be frozen as the primary observation remap
(or changed if the evidence says otherwise), and forecast-hour curves can be
built on validation before the test period is opened.

---

# B. Wang-style controlled transformation experiment

## Critical distinction

The transformation definitions follow Wang et al. (2026):

```text
displacement  horizontal grid shift
intensity     multiplicative precipitation factor
area size     scaling around centroid of precipitating grids
```

For area-size enlargement, gaps are filled by bilinear interpolation.
For contraction, multiple source grids mapping to the same output grid are
averaged.

The bias grids are:

```text
displacement: -10 ... +10 grid cells
intensity:     0.50,0.55,...,0.95,1.00,1.10,...,2.00
area size:     0.50,0.55,...,0.95,1.00,1.10,...,2.00
```

The primary experiment intentionally preserves Wang's **grid-cell bias values**.
Wang used 0.125-degree IFS fields, while native HRRR is ~3 km, so ±10 cells is
not the same physical displacement. That physical-scale issue should be shown
as a separate sensitivity experiment later, not silently changed in the
paper-replication experiment.

## Independent validation archive

This experiment does **not** reuse the 512 fixed-validation cases used for
checkpoint selection.

It draws a separate deterministic 2000-case subset from the larger validation
split, with independent deterministic crops, and excludes completely dry
patches by default.

Build with:

```bash
python scripts/build_wang_transform_archive.py \
  --target-cases 2000
```

Outputs:

```text
$DATA_DIR/derived/evaluation/wang_transform_v1/
  original_patches.npz
  cases.csv
  transform_spec.json
```

Only originals are stored; transformed fields are generated on demand.

## Comparison metrics

Implemented now:

```text
learned z cosine similarity
TS
POD
normalized FAR = 1-FAR
FSS-5
FSS-10
RMSE
MAE
correlation
```

The plotting script creates a three-panel layout deliberately patterned after
Wang Fig. 9 and a Figure-8-like raw displacement scatter plot.

### Two items deliberately NOT guessed

Before calling the traditional curves an **exact** Wang replication, audit the
authors' released verification code for:

1. the precipitation threshold used for TS/POD/FAR/FSS;
2. the exact SAL implementation/object settings.

The paper specifies the transform equations, FSS window sizes, and normalized
FAR/SAL definitions, but the threshold/object details are not explicit enough
in the manuscript to justify guessing.

For that reason:

```text
WANG_THRESHOLD_MM
```

is required by the Slurm script, and normalized SAL is not fabricated.

Once the code audit determines the threshold, run:

```bash
export WANG_THRESHOLD_MM=<audited value>
sbatch slurm/run_wang_transform_eval.slurm
```

Outputs:

```text
results/per_case_metrics.csv.gz
results/summary_by_bias.csv
results/wang_style_comparison.png
results/wang_raw_displacement.png
```

Each Figure-9-style curve is the mean across cases with a 95% confidence band.

---

# What comes after these experiments?

If Stage IV passes the initial sanity/regridding checks and the transformation
experiment shows meaningful graded behavior:

1. freeze Stage-IV preprocessing;
2. build validation-period score vs forecast-hour graphics;
3. freeze all evaluation choices;
4. run the sealed test year;
5. later extend HRRR to the full 0–48 h range using 00/06/12/18 cycles.

Do not use the sealed test set to decide regridding, FSS settings, gallery
selection logic, or transformation methodology.


# Representation characterization v1

This bundle runs the three diagnostics discussed after the first HRRR–Stage-IV and Wang-style results.  It does **not** use the sealed test period and does not retrain the model.

## Files to install

Copy into the existing project tree:

- `scripts/audit_training_cues.py` -> `$PROJECT_DIR/scripts/audit_training_cues.py`
- `scripts/evaluate_layerwise_transform_sensitivity.py` -> `$PROJECT_DIR/scripts/evaluate_layerwise_transform_sensitivity.py`
- `scripts/evaluate_morphology_perturbations.py` -> `$PROJECT_DIR/scripts/evaluate_morphology_perturbations.py`
- `scripts/submit_representation_characterization.sh` -> `$PROJECT_DIR/scripts/submit_representation_characterization.sh`
- `src/eval/morphology_transforms.py` -> `$PROJECT_DIR/src/eval/morphology_transforms.py`
- `slurm/run_training_cue_audit.slurm` -> `$PROJECT_DIR/slurm/run_training_cue_audit.slurm`
- `slurm/run_layerwise_transform_sensitivity.slurm` -> `$PROJECT_DIR/slurm/run_layerwise_transform_sensitivity.slurm`
- `slurm/run_morphology_perturbations.slurm` -> `$PROJECT_DIR/slurm/run_morphology_perturbations.slurm`

The scripts rely on project files already present from the previous experiments: `src.data`, `src.eval.wang_transforms`, `src.eval.verification_metrics`, `src.models.checkpoint_loader`, and `src.models.precip_resnet_gn`.

## One-command overnight submission

From `$PROJECT_DIR`:

```bash
bash scripts/submit_representation_characterization.sh
```

This submits three independent Slurm jobs at once.  The cue audit is CPU/I/O based; the layerwise and morphology experiments each request one H100.  They may execute simultaneously if the scheduler has resources, but they do not depend on each other.

## 1. Training-cue audit

Uses the **actual training sampler distribution**: train split, sampler seed `20260930`, epoch 0, random +/-1-h positive direction, same 256x256 crop, and 8 `climo45` negatives.  It audits 3,000 anchors by default.

For each cue it asks how often P is more A-like than the sampled N values, plus the cue-only top-1 rate.  Cues include mean/intensity gaps, wet-coverage gaps, precipitation-centroid distance, RMSE, correlation, TS, and FSS.

Outputs:

`$DATA_DIR/derived/evaluation/representation_characterization_v1/training_cue_audit/`

Key morning files:

- `training_cue_summary.csv`
- `training_cue_pairwise_win_rate.png`
- `training_cue_top1_rate.png`
- `training_cue_recipes.jsonl` (exact audited recipes)

Interpretation: this diagnoses which physical cues were *available* to solve the contrastive task. It does not prove the network actually used a cue.

## 2. Layerwise + random-network transformation sensitivity

Uses 500 deterministic cases from the existing `wang_transform_v1` held-out validation archive.  It runs Wang displacement, intensity, and area-size transforms through:

`stem -> layer1 -> layer2 -> layer3 -> h -> z`

for both the trained GroupNorm checkpoint and an otherwise identical randomly initialized GroupNorm network.

Outputs:

`$DATA_DIR/derived/evaluation/representation_characterization_v1/layerwise_transform_sensitivity/`

Key morning files:

- `layerwise_transform_summary.csv`
- `trained_*_layerwise_similarity.png`
- `random_*_layerwise_similarity.png`
- matching `*_layerwise_distance.png` plots

Interpretation:

- similar invariance in random + trained networks -> preprocessing/architecture is implicated;
- invariance that develops mainly in deeper trained layers -> learned contrastive objective is implicated.

## 3. Expanded morphology perturbation suite

Uses 500 deterministic held-out HRRR validation patches and the frozen best GroupNorm checkpoint.  Families are:

- Gaussian smoothing (`sigma`)
- rotation about precipitation-weighted centroid (`degrees`, +/- variants)
- area-preserving anisotropic elongation (`ratio`, x/y variants)
- deterministic tile-jitter fragmentation (`cells`)
- radial connected-object dispersion (`cells`)

The last two are explicitly diagnostic proxies rather than claims of perfect meteorological realism.  Transform definitions are documented in `src/eval/morphology_transforms.py`.  Total precipitation is renormalized after geometry-changing operations so the probe is not dominated by accidental mass loss at crop boundaries.

Conventional TS/FSS use the current `2.54 mm` threshold by default.

Outputs:

`$DATA_DIR/derived/evaluation/representation_characterization_v1/morphology_perturbations/`

Key morning files:

- `morphology_summary.csv`
- `morphology_sensitivity_ranking.csv`
- `*_score_response.png`
- `morphology_per_case_metrics.csv.gz`

## Reproducibility notes

All case selections and synthetic perturbations use fixed seeds and write the selected cases/recipes.  Evaluation metadata records the checkpoint path/hash and settings.  No model weights are changed.

The morphology experiment is exploratory.  Do not interpret a larger response as automatically “better”; the first question is whether the learned representation has a coherent, meteorologically interpretable sensitivity profile that differs usefully from traditional metrics.

