#!/bin/bash
#SBATCH --job-name=morph_syn8_v3
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -euo pipefail

PROJECT_DIR="/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning"
cd "$PROJECT_DIR"
source load_env.sh
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/morphology_synthetic_vs_v3_${SLURM_JOB_ID}.log"
exec >"$LOG_FILE" 2>&1

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Log: $LOG_FILE"
echo

python scripts/evaluate_morphology_synthetic_vs_v3.py \
  --case-count 500 \
  --case-seed 20261007 \
  --batch-size 32 \
  --threshold-mm 2.54

python scripts/plot_morphology_synthetic_vs_v3.py \
  --synthetic-label "SYN-8" \
  --natural-label "NFV (V3)"

echo
echo "Finished: $(date)"

