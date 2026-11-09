#!/usr/bin/env bash
#SBATCH --job-name=eval_timelag
#SBATCH --partition=u1-compute
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
BOOTSTRAP_LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
mkdir -p "${BOOTSTRAP_LOG_DIR}"
LOG_FILE="${BOOTSTRAP_LOG_DIR}/evaluate_timelag_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.log"
exec >"${LOG_FILE}" 2>&1
cd "${PROJECT_DIR}"
source load_env.sh

args=(
  --experiment-dir "$EXPERIMENT_DIR"
  --manifest "$MANIFEST"
  --stage4-root "$STAGE4_ROOT"
  --mapping "$MAPPING"
  --shard-index "$SLURM_ARRAY_TASK_ID"
  --n-shards "$N_SHARDS"
  --patch-size "$PATCH_SIZE"
  --patches-per-valid "$PATCHES_PER_VALID"
  --negative-count "$NEGATIVE_COUNT"
  --negative-min-days "$NEGATIVE_MIN_DAYS"
  --negative-mode "$NEGATIVE_MODE"
  --climo-window-days "$CLIMO_WINDOW_DAYS"
  --wet-threshold-mm "$WET_THRESHOLD_MM"
  --min-wet-fraction "$MIN_WET_FRACTION"
  --fss-threshold-mm "$FSS_THRESHOLD_MM"
  --fss-windows "$FSS_WINDOWS"
  --seed "$SEED"
)
[[ -n "${MAX_VALID_TIMES:-}" ]] && args+=(--max-valid-times "$MAX_VALID_TIMES")
[[ -n "${SAMPLE_PLAN_FROM:-}" ]] && args+=(--sample-plan-from "$SAMPLE_PLAN_FROM")

python scripts/evaluate/timelag.py "${args[@]}"
