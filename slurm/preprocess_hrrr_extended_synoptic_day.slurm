#!/usr/bin/env bash
#SBATCH --job-name=hrrr_qpf6_ext
#SBATCH --partition=u1-service
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATE_LIST="${1:?Usage: sbatch ... preprocess_hrrr_extended_synoptic_day.slurm DATE_LIST}"
cd "${PROJECT_DIR}"
source load_env.sh
export BLOSC_NTHREADS="${SLURM_CPUS_PER_TASK:-4}"
DAYS_PER_TASK="${DAYS_PER_TASK:-7}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/hrrr_qpf6_ext_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.log"
exec >"${LOG_FILE}" 2>&1

START_LINE=$((SLURM_ARRAY_TASK_ID * DAYS_PER_TASK + 1))
END_LINE=$((START_LINE + DAYS_PER_TASK - 1))
mapfile -t DATES < <(sed -n "${START_LINE},${END_LINE}p" "${DATE_LIST}")

if [[ "${#DATES[@]}" -eq 0 ]]; then
    echo "No dates found for array task ${SLURM_ARRAY_TASK_ID}" >&2
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

for DATE in "${DATES[@]}"; do
    echo
    echo "############################################################"
    echo "Processing extended synoptic HRRR ${DATE}"
    echo "############################################################"
    python "${PROJECT_DIR}/scripts/preprocess_hrrr_extended_synoptic_day.py" --date "${DATE}"
done

echo
 echo "Completed all dates for array task ${SLURM_ARRAY_TASK_ID}"
