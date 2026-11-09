#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
source load_env.sh

ROOT="${DATA_DIR}/forecasts/hrrr/qpf6_extended_synoptic"
MANIFEST="${ROOT}/metadata/hrrr_qpf6_manifest.csv.gz"

python scripts/build_hrrr_manifest.py \
  --root "${ROOT}" \
  --output "${MANIFEST}"

python scripts/preflight_hrrr_extended_synoptic.py \
  --root "${ROOT}" \
  --manifest "${MANIFEST}" \
  --start 20240929 \
  --end 20250920

echo
echo "Extended HRRR archive finalized and preflight passed:"
echo "  ${MANIFEST}"
