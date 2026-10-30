#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../load_env.sh"

N_SHARDS="${N_SHARDS:-8}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"

PATCH_SIZE="${PATCH_SIZE:-256}"
PATCHES_PER_VALID="${PATCHES_PER_VALID:-1}"
NEGATIVE_COUNT="${NEGATIVE_COUNT:-8}"
NEGATIVE_MIN_DAYS="${NEGATIVE_MIN_DAYS:-90}"

NEGATIVE_MODE="${NEGATIVE_MODE:-climo45}"
CLIMO_WINDOW_DAYS="${CLIMO_WINDOW_DAYS:-45}"

WET_THRESHOLD_MM="${WET_THRESHOLD_MM:-1.0}"
MIN_WET_FRACTION="${MIN_WET_FRACTION:-0.01}"

FSS_THRESHOLD_MM="${FSS_THRESHOLD_MM:-5.0}"
FSS_WINDOWS="${FSS_WINDOWS:-1,3,7,15,31,61}"

MAX_VALID_TIMES="${MAX_VALID_TIMES:-}"
SEED="${SEED:-20260926}"

# New climo45 experiment lives separately from the old calendar_any run.
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$DATA_DIR/derived/experiments/timelag_lag_selection_${NEGATIVE_MODE}_v1}"

# Preferred fair-comparison behavior: reuse the exact valid times and crop
# coordinates from the already-completed calendar_any experiment.
SAMPLE_PLAN_FROM="${SAMPLE_PLAN_FROM:-$DATA_DIR/derived/experiments/timelag_lag_selection_v1}"

MAPPING="${MAPPING:-$DATA_DIR/derived/experiments/timelag_lag_selection_v1/grid_mapping/stage4_to_hrrr_nearest.npz}"
STAGE4_ROOT="${STAGE4_ROOT:-$DATA_DIR/observations/stage4/raw}"
MANIFEST="${MANIFEST:-$DATA_DIR/forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz}"

mkdir -p "$EXPERIMENT_DIR/shards" "$EXPERIMENT_DIR/figures" "$EXPERIMENT_DIR/summary"
mkdir -p "$EXPERIMENT_DIR/grid_mapping"

# Let the existing plotting scripts find the same mapping under the new
# experiment directory without making another copy.
ln -sfn "$MAPPING" "$EXPERIMENT_DIR/grid_mapping/stage4_to_hrrr_nearest.npz"

echo "Experiment: $EXPERIMENT_DIR"
echo "Array: 0-$((N_SHARDS - 1))%${MAX_CONCURRENT}"
echo "N_SHARDS=$N_SHARDS"
echo "MAX_CONCURRENT=$MAX_CONCURRENT"
echo "PATCH_SIZE=$PATCH_SIZE"
echo "PATCHES_PER_VALID=$PATCHES_PER_VALID"
echo "NEGATIVE_COUNT=$NEGATIVE_COUNT"
echo "NEGATIVE_MIN_DAYS=$NEGATIVE_MIN_DAYS"
echo "NEGATIVE_MODE=$NEGATIVE_MODE"
echo "CLIMO_WINDOW_DAYS=$CLIMO_WINDOW_DAYS"
echo "WET_THRESHOLD_MM=$WET_THRESHOLD_MM"
echo "MIN_WET_FRACTION=$MIN_WET_FRACTION"
echo "FSS_THRESHOLD_MM=$FSS_THRESHOLD_MM"
echo "FSS_WINDOWS=$FSS_WINDOWS"
echo "MAX_VALID_TIMES=${MAX_VALID_TIMES:-<unused when sample plan is reused>}"
echo "SAMPLE_PLAN_FROM=$SAMPLE_PLAN_FROM"

export EXPERIMENT_DIR
export N_SHARDS PATCH_SIZE PATCHES_PER_VALID
export NEGATIVE_COUNT NEGATIVE_MIN_DAYS NEGATIVE_MODE CLIMO_WINDOW_DAYS
export WET_THRESHOLD_MM MIN_WET_FRACTION
export FSS_THRESHOLD_MM FSS_WINDOWS MAX_VALID_TIMES SEED
export SAMPLE_PLAN_FROM MAPPING STAGE4_ROOT MANIFEST

sbatch \
  --array="0-$((N_SHARDS - 1))%${MAX_CONCURRENT}" \
  "$(dirname "$0")/../slurm/evaluate_timelag.slurm"

