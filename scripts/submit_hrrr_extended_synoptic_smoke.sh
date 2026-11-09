#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
source load_env.sh
DATE="${1:-20250401}"
DATE_LIST="${SCRATCH_DIR}/hrrr_extended_synoptic_smoke_${DATE}.txt"
printf '%s\n' "${DATE}" > "${DATE_LIST}"
echo "Submitting one-day smoke test for ${DATE}"
sbatch --array=0-0 --export=ALL,DAYS_PER_TASK=1 \
  "${PROJECT_DIR}/slurm/preprocess_hrrr_extended_synoptic_day.slurm" "${DATE_LIST}"
echo "After it finishes, inspect:"
echo "  python scripts/inspect_hrrr_extended_store.py ${DATA_DIR}/forecasts/hrrr/qpf6_extended_synoptic/daily/${DATE:0:4}/${DATE:0:6}/hrrr_qpf6_${DATE}.zarr.zip"
