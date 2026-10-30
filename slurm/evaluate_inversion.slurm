#!/bin/bash
#SBATCH --job-name=eval_inversion
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
BOOTSTRAP_LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
mkdir -p "${BOOTSTRAP_LOG_DIR}"
LOG_FILE="${BOOTSTRAP_LOG_DIR}/evaluate_inversion_${SLURM_JOB_ID}.log"
exec >"${LOG_FILE}" 2>&1
cd "${PROJECT_DIR}"
source load_env.sh

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Log: ${LOG_FILE}"

CASE_INDEX="${CASE_INDEX:--1}"
N_STEPS="${N_STEPS:-1200}"
BASE_OUT_DIR="${OUT_DIR:-$DATA_DIR/derived/evaluation/feature_inversion_v1}"
if [[ "${CASE_INDEX}" -ge 0 ]]; then
    CASE_TAG=$(printf "case_%04d" "${CASE_INDEX}")
else
    CASE_TAG="case_auto_${SLURM_JOB_ID}"
fi
CASE_OUT="${BASE_OUT_DIR}/${CASE_TAG}"
mkdir -p "${CASE_OUT}"

python scripts/evaluate/inversion.py \
  --case-index "${CASE_INDEX}" \
  --models nfv syn3 syn8 \
  --targets 0.75 0.50 0.25 0.00 -0.25 -0.50 -0.75 -1.00 \
  --n-steps "${N_STEPS}" \
  --lambda-mass 0.0 \
  --output-dir "${CASE_OUT}"

python scripts/plot/inversion.py \
  --input-dir "${CASE_OUT}" \
  --models nfv syn3 syn8 \
  --targets 0.75 0.50 0.25 0.00 -0.25 -0.50 -0.75 -1.00

echo "Finished: $(date)"
echo "Output: ${CASE_OUT}"
