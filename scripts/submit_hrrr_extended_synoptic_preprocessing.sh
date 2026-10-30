#!/usr/bin/env bash
set -euo pipefail

START="${1:-20240929}"
END="${2:-20250920}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
source load_env.sh

MAX_CONCURRENT="${MAX_CONCURRENT:-12}"
DAYS_PER_TASK="${DAYS_PER_TASK:-7}"
DATE_LIST="${DATA_DIR}/metadata/date_lists/hrrr_extended_synoptic_${START}_${END}.txt"

python scripts/make_hrrr_date_list.py "${START}" "${END}" --output "${DATE_LIST}"
N="$(wc -l < "${DATE_LIST}")"
N_TASKS=$(( (N + DAYS_PER_TASK - 1) / DAYS_PER_TASK ))
ARRAY="0-$((N_TASKS - 1))%${MAX_CONCURRENT}"

OUT_ROOT="${DATA_DIR}/forecasts/hrrr/qpf6_extended_synoptic"
echo "Date list         : ${DATE_LIST}"
echo "Dates             : ${N}"
echo "Days/task         : ${DAYS_PER_TASK}"
echo "Array tasks       : ${N_TASKS}"
echo "Array             : ${ARRAY}"
echo "Output root       : ${OUT_ROOT}"
echo "Stored cycles     : 00 06 12 18Z"
echo "Stored leads      : F06 F12 F18 F24 F30 F36 F42 F48"

CMD=(
    sbatch
    "--array=${ARRAY}"
    "--export=ALL,DAYS_PER_TASK=${DAYS_PER_TASK}"
    "${PROJECT_DIR}/slurm/preprocess_hrrr_extended_synoptic_day.slurm"
    "${DATE_LIST}"
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'DRY RUN:'; printf ' %q' "${CMD[@]}"; printf '\n'
else
    "${CMD[@]}"
fi
