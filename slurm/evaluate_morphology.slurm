#!/bin/bash
#SBATCH --job-name=eval_morph
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
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
BOOTSTRAP_LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
mkdir -p "${BOOTSTRAP_LOG_DIR}"
LOG_FILE="${BOOTSTRAP_LOG_DIR}/evaluate_morphology_${SLURM_JOB_ID}.log"
exec >"${LOG_FILE}" 2>&1
cd "${PROJECT_DIR}"
source load_env.sh

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Log: ${LOG_FILE}"

python scripts/evaluate/morphology.py \
  --case-count "${CASE_COUNT:-500}" \
  --case-seed "${CASE_SEED:-20261007}" \
  --threshold-mm 2.54

python scripts/plot/morphology.py

echo "Finished: $(date)"
