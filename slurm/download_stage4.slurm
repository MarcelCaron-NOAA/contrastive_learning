#!/usr/bin/env bash
#SBATCH --job-name=stage4_6h
#SBATCH --partition=u1-service
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=06:00:00

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATE_LIST="${1:?Usage: sbatch ... download_stage4.slurm DATE_LIST}"
cd "${PROJECT_DIR}"
source load_env.sh
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/stage4_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.log"
exec >"${LOG_FILE}" 2>&1
DAYS_PER_TASK="${DAYS_PER_TASK:-7}"

START_LINE=$((SLURM_ARRAY_TASK_ID * DAYS_PER_TASK + 1))
END_LINE=$((START_LINE + DAYS_PER_TASK - 1))
mapfile -t DATES < <(sed -n "${START_LINE},${END_LINE}p" "${DATE_LIST}")

if [[ "${#DATES[@]}" -eq 0 ]]; then
    echo "No dates for array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
fi

echo "============================================================"
echo "SLURM_JOB_ID        = ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID  = ${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "DATES               = ${DATES[*]}"
echo "HOST                = $(hostname)"
echo "LOG                 = ${LOG_FILE}"
echo "============================================================"

failures=()
for DATE in "${DATES[@]}"; do
    echo
    echo "############################################################"
    echo "Stage IV: ${DATE}"
    echo "############################################################"
    if ! scripts/stage4_extract_day.sh "${DATE}"; then
        echo "FAILED: ${DATE}" >&2
        failures+=("${DATE}")
    fi
done

if [[ "${#failures[@]}" -gt 0 ]]; then
    echo "Failed dates: ${failures[*]}" >&2
    exit 1
fi

echo "Completed all Stage IV dates for array task ${SLURM_ARRAY_TASK_ID}"
