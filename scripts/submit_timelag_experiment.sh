#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning}"
source "${PROJECT_DIR}/load_env.sh"

EXPERIMENT_DIR="${EXPERIMENT_DIR:-${DATA_DIR}/derived/experiments/timelag_lag_selection_v1}"
N_SHARDS="${N_SHARDS:-24}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"

mkdir -p "${EXPERIMENT_DIR}"/{shards,figures,summary,grid_mapping}

MAPPING="${EXPERIMENT_DIR}/grid_mapping/stage4_to_hrrr_nearest.npz"
[[ -s "${MAPPING}" ]] || {
    echo "Missing ${MAPPING}; run prepare_timelag_grid_mapping.py first."
    exit 2
}

# Store the full experiment configuration in a shell-readable file.
# This avoids passing comma-containing values such as FSS_WINDOWS through
# sbatch --export, whose own syntax uses commas as separators.
cat > "${EXPERIMENT_DIR}/experiment_config.env" <<EOF
N_SHARDS=${N_SHARDS}
MAX_CONCURRENT=${MAX_CONCURRENT}
PATCH_SIZE=${PATCH_SIZE:-256}
PATCHES_PER_VALID=${PATCHES_PER_VALID:-1}
NEGATIVE_COUNT=${NEGATIVE_COUNT:-3}
NEGATIVE_MIN_DAYS=${NEGATIVE_MIN_DAYS:-90}
WET_THRESHOLD_MM=${WET_THRESHOLD_MM:-1.0}
MIN_WET_FRACTION=${MIN_WET_FRACTION:-0.01}
FSS_THRESHOLD_MM=${FSS_THRESHOLD_MM:-5.0}
FSS_WINDOWS=${FSS_WINDOWS:-1,3,7,15,31,61}
MAX_VALID_TIMES=${MAX_VALID_TIMES:-}
EOF

ARRAY="0-$((N_SHARDS-1))%${MAX_CONCURRENT}"

echo "Experiment: ${EXPERIMENT_DIR}"
echo "Array: ${ARRAY}"
cat "${EXPERIMENT_DIR}/experiment_config.env"

# Only export the experiment path.  The job reads all scientific/runtime
# settings from experiment_config.env.
sbatch \
    --array="${ARRAY}" \
    --export="ALL,EXPERIMENT_DIR=${EXPERIMENT_DIR}" \
    "${PROJECT_DIR}/slurm/timelag_experiment.slurm"

