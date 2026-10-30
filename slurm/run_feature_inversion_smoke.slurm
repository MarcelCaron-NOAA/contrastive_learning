#!/bin/bash
#SBATCH --job-name=inv_smoke_v2
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
PROJECT_DIR="/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning"
cd "$PROJECT_DIR"
source load_env.sh
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/feature_inversion_smoke_v2_${SLURM_JOB_ID}.log"
exec >"$LOG_FILE" 2>&1

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Log: $LOG_FILE"

CASE_INDEX="${CASE_INDEX:--1}"
N_STEPS="${N_STEPS:-500}"
OUT_DIR="${OUT_DIR:-$DATA_DIR/derived/evaluation/feature_inversion_pilot_v1/smoke_exact_init_v2}"

python scripts/run_feature_inversion.py \
  --case-index "$CASE_INDEX" \
  --models syn3 \
  --targets 0.5 -0.5 \
  --n-steps "$N_STEPS" \
  --output-dir "$OUT_DIR"

python scripts/plot_feature_inversion_pilot.py \
  --input-dir "$OUT_DIR" \
  --models syn3 \
  --targets 0.5 -0.5

echo "Finished: $(date)"
echo "Output: $OUT_DIR"
