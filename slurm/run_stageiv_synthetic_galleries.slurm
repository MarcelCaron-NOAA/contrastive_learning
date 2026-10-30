#!/bin/bash
#SBATCH --job-name=st4_syn8_gallery
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -euo pipefail

PROJECT_DIR="/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning"
cd "$PROJECT_DIR"
source load_env.sh
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/stageiv_synthetic_galleries_${SLURM_JOB_ID}.log"
exec >"$LOG_FILE" 2>&1

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Log: $LOG_FILE"
echo

python scripts/evaluate_stageiv_synthetic_vs_v3.py
python scripts/plot_stageiv_synthetic_galleries.py --count 5 --synthetic-label "SYN-8" --natural-label "NFV (V3)"

echo
echo "Finished: $(date)"

