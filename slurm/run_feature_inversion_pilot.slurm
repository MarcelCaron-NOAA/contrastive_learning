#!/bin/bash
#SBATCH --job-name=inv_pilot
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
PROJECT_DIR="/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning"
cd "$PROJECT_DIR"
source load_env.sh
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/feature_inversion_pilot_${SLURM_JOB_ID}.log"
exec >"$LOG_FILE" 2>&1

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Log: $LOG_FILE"

CASE_INDEX="${CASE_INDEX:--1}"
N_STEPS="${N_STEPS:-800}"
OUT_DIR="${OUT_DIR:-$DATA_DIR/derived/evaluation/feature_inversion_pilot_v1/pilot}"

python scripts/run_feature_inversion.py \
  --case-index "$CASE_INDEX" \
  --models nfv syn3 syn8 \
  --targets 0.5 0.0 -0.5 -1.0 \
  --n-steps "$N_STEPS" \
  --output-dir "$OUT_DIR"

python scripts/plot_feature_inversion_pilot.py \
  --input-dir "$OUT_DIR" \
  --models nfv syn3 syn8 \
  --targets 0.5 0.0 -0.5 -1.0

echo "Finished: $(date)"
echo "Output: $OUT_DIR"
