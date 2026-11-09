#!/usr/bin/env bash
#SBATCH --job-name=st4_ref_build
#SBATCH --partition=u1-compute
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_DIR}"
source load_env.sh
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/stageiv_ref_build_${SLURM_JOB_ID}.log"
exec >"${LOG_FILE}" 2>&1
PYTHON_EXE="$(which python)"

# MET for Stage-IV regridding.
module load intel-oneapi-compilers/2025.1.1
module use -a /contrib/met/modulefiles/
module load met/12.2.0

echo "Python executable: ${PYTHON_EXE}"
echo "MET executable: $(which regrid_data_plane)"
echo "Log: ${LOG_FILE}"
module list 2>&1

: "${STAGEIV_ROOT:?Set STAGEIV_ROOT to directory containing staged st4_conus.*.06h.grb2 files}"
: "${HRRR_GRID_TEMPLATE:?Set HRRR_GRID_TEMPLATE to a native-HRRR gridded file usable by MET as to_grid}"

command -v regrid_data_plane >/dev/null 2>&1 || {
  echo "ERROR: regrid_data_plane is not on PATH. Load the appropriate MET module/environment." >&2
  exit 2
}

"${PYTHON_EXE}" scripts/make_stageiv_pair_manifest.py \
  --stageiv-root "${STAGEIV_ROOT}" \
  --candidate-count 3000 \
  --patch-size 256

"${PYTHON_EXE}" scripts/build_stageiv_reference_archive.py \
  --stageiv-root "${STAGEIV_ROOT}" \
  --hrrr-grid-template "${HRRR_GRID_TEMPLATE}" \
  --target-count 1000 \
  --regrid-width 2

echo "Stage-IV reference archive build complete."
