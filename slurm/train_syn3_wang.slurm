#!/bin/bash
#SBATCH --job-name=syn3_wang
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

PROJECT_DIR="/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning"
cd "$PROJECT_DIR"
source load_env.sh

RUN_NAME="${RUN_NAME:-synthetic_perturb_syn3_v2}"
OUTPUT_DIR="${DATA_DIR}/derived/experiments/${RUN_NAME}"

mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_${SLURM_JOB_ID}.log"
exec >"$LOG_FILE" 2>&1

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Log: $LOG_FILE"
echo

EXTRA_ARGS=()
if [[ -n "${END_STEP:-}" ]]; then
  EXTRA_ARGS+=(--end-step "$END_STEP")
fi
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

python scripts/train_syn3_wang_adapted.py \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "${BATCH_SIZE:-8}" \
  --patch-size 256 \
  --temperature 0.10 \
  --supcon-weight 1.0 \
  --penalty-weight 1000.0 \
  --wet-threshold-mm 0.1 \
  --min-anchor-wet-fraction 0.001 \
  --learning-rate 3e-4 \
  --warmup-start-lr 3e-5 \
  --warmup-fraction 0.05 \
  --min-learning-rate 3e-6 \
  --weight-decay 1e-4 \
  --validation-samples 512 \
  --validate-every 500 \
  --checkpoint-every 1500 \
  --num-workers 8 \
  --prefetch-factor 2 \
  "${EXTRA_ARGS[@]}"

python scripts/plot_syn3_wang_training.py --experiment-dir "$OUTPUT_DIR"

