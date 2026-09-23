#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 2 ]] || { echo "Usage: $0 START_YYYYMMDD END_YYYYMMDD" >&2; exit 2; }
START="$1"; END="$2"
PROJECT_DIR="${PROJECT_DIR:-/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning}"
source "${PROJECT_DIR}/load_env.sh"
MAX_CONCURRENT="${MAX_CONCURRENT:-12}"
DATE_LIST="${DATA_DIR}/metadata/date_lists/hrrr_${START}_${END}.txt"
python "${PROJECT_DIR}/scripts/make_hrrr_date_list.py" "${START}" "${END}" --output "${DATE_LIST}"
N="$(wc -l < "${DATE_LIST}")"; [[ "${N}" -gt 0 ]] || exit 2
ARRAY="0-$((N-1))%${MAX_CONCURRENT}"
echo "Date list: ${DATE_LIST}"; echo "Days: ${N}"; echo "Array: ${ARRAY}"; echo "Output: ${DATA_DIR}/forecasts/hrrr/qpf6_native"
CMD=(sbatch "--array=${ARRAY}" "${PROJECT_DIR}/slurm/preprocess_hrrr_day.slurm" "${DATE_LIST}")
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf 'DRY RUN:'; printf ' %q' "${CMD[@]}"; printf '\n'; else "${CMD[@]}"; fi

