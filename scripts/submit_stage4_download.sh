#!/usr/bin/env bash
# Submit restartable Stage IV HPSS extraction in multi-day Slurm array tasks.
#
# Usage:
#   MAX_CONCURRENT=2 DAYS_PER_TASK=7 \
#     ./scripts/submit_stage4_download.sh 20201202 20260922
#
# A conservative default is used because HPSS/tape access should not be
# hammered with large numbers of simultaneous requests.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 START_YYYYMMDD END_YYYYMMDD" >&2
    exit 2
fi

START="$1"
END="$2"

PROJECT_DIR="${PROJECT_DIR:-/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning}"
source "${PROJECT_DIR}/load_env.sh"

MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
DAYS_PER_TASK="${DAYS_PER_TASK:-7}"

DATE_LIST="${DATA_DIR}/metadata/date_lists/stage4_${START}_${END}.txt"

python "${PROJECT_DIR}/scripts/make_hrrr_date_list.py" \
    "${START}" "${END}" \
    --output "${DATE_LIST}"

N="$(wc -l < "${DATE_LIST}")"
N_TASKS=$(( (N + DAYS_PER_TASK - 1) / DAYS_PER_TASK ))
ARRAY="0-$((N_TASKS - 1))%${MAX_CONCURRENT}"

echo
echo "Date list       : ${DATE_LIST}"
echo "Number of days  : ${N}"
echo "Days per task   : ${DAYS_PER_TASK}"
echo "Array tasks     : ${N_TASKS}"
echo "Max concurrent  : ${MAX_CONCURRENT}"
echo "Array           : ${ARRAY}"
echo "Output root     : ${DATA_DIR}/observations/stage4/raw"
echo

CMD=(
    sbatch
    "--array=${ARRAY}"
    "--export=ALL,DAYS_PER_TASK=${DAYS_PER_TASK}"
    "${PROJECT_DIR}/slurm/download_stage4.slurm"
    "${DATE_LIST}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY RUN:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
else
    "${CMD[@]}"
fi

