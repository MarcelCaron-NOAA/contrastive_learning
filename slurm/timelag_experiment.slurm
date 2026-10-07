#!/usr/bin/env bash
#SBATCH --job-name=timelag_diag
#SBATCH --partition=u1-compute
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/timelag_%A_%a.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/timelag_%A_%a.err

set -euo pipefail

PROJECT_DIR="/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning"
source "$PROJECT_DIR/load_env.sh"

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

if [[ -n "${MAX_VALID_TIMES:-}" ]]; then
  args+=(--max-valid-times "$MAX_VALID_TIMES")
fi

if [[ -n "${SAMPLE_PLAN_FROM:-}" ]]; then
  args+=(--sample-plan-from "$SAMPLE_PLAN_FROM")
fi

python "$PROJECT_DIR/scripts/run_timelag_experiment.py" "${args[@]}"

