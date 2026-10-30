#!/bin/bash
#SBATCH --job-name=final_st4_buckets
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

PROJECT_DIR="/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning"
cd "$PROJECT_DIR"
source load_env.sh

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/final_stageiv_buckets_${SLURM_JOB_ID}.log"
exec >"$LOG_FILE" 2>&1

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Log: $LOG_FILE"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo

python scripts/evaluate_final_stageiv_models.py

python scripts/plot_final_stageiv_buckets.py

echo
echo "Finished: $(date)"
echo "Outputs:"
echo "  $DATA_DIR/derived/evaluation/final_syn_nfv_stageiv_v1"

