#!/bin/bash
#SBATCH --job-name=inv_cont
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
PROJECT_DIR="/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning"
cd "$PROJECT_DIR"
source load_env.sh
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/feature_inversion_continuation_${SLURM_JOB_ID}.log"
exec >"$LOG_FILE" 2>&1

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Log: $LOG_FILE"

CASE_INDEX="${CASE_INDEX:--1}"
N_STEPS="${N_STEPS:-1200}"

BASE_OUT_DIR="${OUT_DIR:-$DATA_DIR/derived/evaluation/feature_inversion_pilot_v1/continuation_v3}"

if [[ "$CASE_INDEX" -ge 0 ]]; then
    CASE_TAG=$(printf "case_%04d" "$CASE_INDEX")
else
    # Prevent auto-selected cases from colliding with each other.
    CASE_TAG="case_auto_${SLURM_JOB_ID}"
fi

OUT_DIR="${BASE_OUT_DIR}/${CASE_TAG}"
mkdir -p "$OUT_DIR"

python scripts/run_feature_inversion_continuation.py \
  --case-index "$CASE_INDEX" \
  --models nfv syn3 syn8 \
  --targets 0.75 0.50 0.25 0.00 -0.25 -0.50 -0.75 -1.00 \
  --n-steps "$N_STEPS" \
  --lambda-mass 0.0 \
  --output-dir "$OUT_DIR"

python scripts/plot_feature_inversion_continuation.py \
  --input-dir "$OUT_DIR" \
  --models nfv syn3 syn8 \
  --targets 0.75 0.50 0.25 0.00 -0.25 -0.50 -0.75 -1.00

echo "Finished: $(date)"
echo "Output: $OUT_DIR"
