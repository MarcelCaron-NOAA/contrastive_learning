#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning}"
source "${PROJECT_DIR}/load_env.sh"

LEAD_EXPERIMENT_DIR="${LEAD_EXPERIMENT_DIR:-${DATA_DIR}/derived/experiments/lead_consistency_v1}"
MAX_CONCURRENT="${MAX_CONCURRENT:-11}"

mkdir -p "${LEAD_EXPERIMENT_DIR}" "${LOG_DIR}"

cat > "${LEAD_EXPERIMENT_DIR}/experiment_config.env" <<EOF
START_DATE=${START_DATE:-2020-12-02}
END_DATE=${END_DATE:-2025-09-23}
SAMPLES_PER_SEASON=${SAMPLES_PER_SEASON:-250}
PATCH_SIZE=${PATCH_SIZE:-256}
PATCH_TRIES=${PATCH_TRIES:-30}
WET_THRESHOLD_MM=${WET_THRESHOLD_MM:-1.0}
MIN_WET_FRACTION=${MIN_WET_FRACTION:-0.01}
FSS_THRESHOLD_MM=${FSS_THRESHOLD_MM:-5.0}
FSS_WINDOW=${FSS_WINDOW:-15}
SEED=${SEED:-20260929}
EOF

echo "Experiment: ${LEAD_EXPERIMENT_DIR}"
echo "Array     : 7-17%${MAX_CONCURRENT}"
cat "${LEAD_EXPERIMENT_DIR}/experiment_config.env"

sbatch \
  --array="7-17%${MAX_CONCURRENT}" \
  --export="ALL,LEAD_EXPERIMENT_DIR=${LEAD_EXPERIMENT_DIR}" \
  "${PROJECT_DIR}/slurm/lead_consistency.slurm"

