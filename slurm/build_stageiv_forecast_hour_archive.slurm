#!/bin/bash
#SBATCH --job-name=st4_fhr_build
#SBATCH --partition=u1-compute
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
BOOTSTRAP_LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
mkdir -p "${BOOTSTRAP_LOG_DIR}"
LOG_FILE="${BOOTSTRAP_LOG_DIR}/stageiv_forecast_hour_build_${SLURM_JOB_ID}.log"
exec >"${LOG_FILE}" 2>&1
cd "${PROJECT_DIR}"

# ----------------------------------------------------------------------
# 1. Load the project's Conda/Python environment first.
# ----------------------------------------------------------------------
source load_env.sh

# Pin the exact Python interpreter BEFORE loading MET.
PROJECT_PYTHON="${ENV_DIR}/bin/python"

if [[ ! -x "${PROJECT_PYTHON}" ]]; then
    echo "ERROR: project Python not found: ${PROJECT_PYTHON}" >&2
    exit 2
fi

# ----------------------------------------------------------------------
# 2. Logging was initialized before environment bootstrap.
# ----------------------------------------------------------------------
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Log: ${LOG_FILE}"
echo
echo "Project Python:"
echo "  ${PROJECT_PYTHON}"
echo

# ----------------------------------------------------------------------
# 3. Load MET.
#
# Loading MET may modify PATH.  That is okay because all Python calls
# below use PROJECT_PYTHON explicitly.
# ----------------------------------------------------------------------
module load intel-oneapi-compilers/2025.1.1
module use -a /contrib/met/modulefiles/
module load met/12.2.0

echo "Loaded modules:"
module list 2>&1
echo

echo "MET executable:"
command -v regrid_data_plane || {
    echo "ERROR: regrid_data_plane is not on PATH" >&2
    exit 2
}
echo

# ----------------------------------------------------------------------
# 4. Environment preflight.
# ----------------------------------------------------------------------
echo "Checking project Python after MET module load..."

"${PROJECT_PYTHON}" - <<'PY'
import sys
import torch
import numpy
import pandas
import netCDF4

print("Python :", sys.executable)
print("Torch  :", torch.__version__)
print("NumPy  :", numpy.__version__)
print("Pandas :", pandas.__version__)
print("netCDF4:", netCDF4.__version__)
print("Python environment check: PASS")
PY

echo

STAGEIV_ROOT="${STAGEIV_ROOT:-${DATA_DIR}/observations/stage4/raw}"

: "${HRRR_GRID_TEMPLATE:?Set HRRR_GRID_TEMPLATE to a MET-readable native-HRRR gridded file before sbatch}"

PAIR_MANIFEST="${DATA_DIR}/derived/evaluation/stageiv_forecast_hour_v1/pair_manifest.csv.gz"
HRRR_MANIFEST="${DATA_DIR}/forecasts/hrrr/qpf6_extended_synoptic/metadata/hrrr_qpf6_manifest.csv.gz"
ARCHIVE_DIR="${DATA_DIR}/derived/evaluation/stageiv_forecast_hour_v1"

if [[ ! -f "${PAIR_MANIFEST}" ]]; then
    echo "ERROR: ${PAIR_MANIFEST} does not exist." >&2
    echo "Run ./scripts/prepare_stageiv_forecast_hour_manifest.sh first." >&2
    exit 2
fi

if [[ ! -f "${HRRR_MANIFEST}" ]]; then
    echo "ERROR: ${HRRR_MANIFEST} does not exist." >&2
    exit 2
fi

echo "Pair manifest : ${PAIR_MANIFEST}"
echo "HRRR manifest : ${HRRR_MANIFEST}"
echo "Stage IV root : ${STAGEIV_ROOT}"
echo "Output archive: ${ARCHIVE_DIR}"
echo

# ----------------------------------------------------------------------
# 5. Build HRRR/Stage-IV patch archive.
# ----------------------------------------------------------------------
"${PROJECT_PYTHON}" scripts/data/build_stageiv_forecast_hour_archive.py \
    --pair-manifest "${PAIR_MANIFEST}" \
    --hrrr-manifest "${HRRR_MANIFEST}" \
    --output-dir "${ARCHIVE_DIR}" \
    --stageiv-root "${STAGEIV_ROOT}" \
    --hrrr-grid-template "${HRRR_GRID_TEMPLATE}" \
    --patch-attempts 64

# ----------------------------------------------------------------------
# 6. Structural/alignment preflight.
# ----------------------------------------------------------------------
"${PROJECT_PYTHON}" scripts/preflight_stageiv_forecast_hour_archive.py \
    --archive-dir "${ARCHIVE_DIR}" \
    --forecast-hours 6 12 18 24 30 36 42 48 \
    --alignment-count 8 \
    --alignment-fhr 24

# ----------------------------------------------------------------------
# 7. Traditional verification metrics.
# ----------------------------------------------------------------------
"${PROJECT_PYTHON}" scripts/evaluate/forecast_hour_traditional.py \
    --archive-dir "${ARCHIVE_DIR}" \
    --threshold-mm 2.54 \
    --fss-widths 5 10 \
    --batch-size 32

echo
echo "Finished: $(date)"
echo
echo "Before GPU submission, inspect:"
echo "  ${ARCHIVE_DIR}/alignment_check_forecast_hour/"
