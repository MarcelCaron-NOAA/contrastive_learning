#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning}"
cd "${PROJECT_DIR}"
source load_env.sh

STAGEIV_ROOT="${STAGEIV_ROOT:-${DATA_DIR}/observations/stage4/raw}"
OUT_DIR="${DATA_DIR}/derived/evaluation/stageiv_forecast_hour_v1"
HRRR_MANIFEST="${DATA_DIR}/forecasts/hrrr/qpf6_extended_synoptic/metadata/hrrr_qpf6_manifest.csv.gz"

if [[ ! -f "${HRRR_MANIFEST}" ]]; then
  echo "ERROR: extended HRRR manifest not found:" >&2
  echo "  ${HRRR_MANIFEST}" >&2
  echo >&2
  echo "Build/finalize the extended synoptic HRRR archive first:" >&2
  echo "  ./scripts/submit_hrrr_extended_synoptic_preprocessing.sh" >&2
  echo "  ./scripts/finalize_hrrr_extended_synoptic_archive.sh" >&2
  exit 2
fi

python scripts/make_stageiv_forecast_hour_manifest.py \
  --hrrr-manifest "${HRRR_MANIFEST}" \
  --stageiv-root "${STAGEIV_ROOT}" \
  --output-dir "${OUT_DIR}" \
  --start-date 2024-10-01 \
  --end-date 2025-09-20 \
  --cycles 0 6 12 18 \
  --forecast-hours 6 12 18 24 30 36 42 48 \
  --patch-size 256 \
  --allow-missing-stageiv

echo
echo "Stage-IV/HRRR forecast-hour manifest preflight passed. No GPU was used."
echo "Next step: submit slurm/build_stageiv_forecast_hour_archive.slurm"
