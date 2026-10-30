#!/bin/bash
#SBATCH --job-name=st4_fhr_scores
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

PROJECT_DIR="/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning"
cd "${PROJECT_DIR}"
source load_env.sh

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/stageiv_scores_by_forecast_hour_${SLURM_JOB_ID}.log"
exec >"${LOG_FILE}" 2>&1

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Log: ${LOG_FILE}"

python scripts/plot_stageiv_scores_by_forecast_hour.py \
  --archive-dir "${DATA_DIR}/derived/evaluation/stageiv_forecast_hour_v1" \
  --v2-checkpoint "${DATA_DIR}/derived/experiments/hard_ranking_finetune_v1/checkpoint_best.pt" \
  --v3-checkpoint "${DATA_DIR}/derived/experiments/hard_ranking_v3_full_v1/checkpoint_best.pt" \
  --stats-json "${DATA_DIR}/metadata/dataset_v1/precip_normalization_stats_v1.json" \
  --output-dir "${DATA_DIR}/derived/experiments/hard_ranking_v3_full_v1/evaluation/stageiv_by_forecast_hour_v1" \
  --batch-size 32

echo "Finished: $(date)"
