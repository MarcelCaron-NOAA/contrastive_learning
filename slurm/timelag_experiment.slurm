#!/usr/bin/env bash
#SBATCH --job-name=hrrr_timelag
#SBATCH --partition=u1-compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/timelag_%A_%a.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/timelag_%A_%a.err

set -euo pipefail

PROJECT_DIR="/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning"
source "${PROJECT_DIR}/load_env.sh"

: "${EXPERIMENT_DIR:?EXPERIMENT_DIR was not supplied}"

CONFIG="${EXPERIMENT_DIR}/experiment_config.env"
[[ -s "${CONFIG}" ]] || {
    echo "Missing experiment config: ${CONFIG}"
    exit 2
}

# shellcheck disable=SC1090
source "${CONFIG}"

: "${N_SHARDS:?N_SHARDS missing from config}"

MANIFEST="${DATA_DIR}/forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz"
STAGE4_ROOT="${DATA_DIR}/observations/stage4/raw"
MAPPING="${EXPERIMENT_DIR}/grid_mapping/stage4_to_hrrr_nearest.npz"

ARGS=(
  --manifest "${MANIFEST}"
  --stage4-root "${STAGE4_ROOT}"
  --mapping "${MAPPING}"
  --output-dir "${EXPERIMENT_DIR}"
  --shard-index "${SLURM_ARRAY_TASK_ID}"
  --n-shards "${N_SHARDS}"
  --patch-size "${PATCH_SIZE:-256}"
  --patches-per-valid "${PATCHES_PER_VALID:-1}"
  --negative-count "${NEGATIVE_COUNT:-3}"
  --negative-min-days "${NEGATIVE_MIN_DAYS:-90}"
  --wet-threshold-mm "${WET_THRESHOLD_MM:-1.0}"
  --min-wet-fraction "${MIN_WET_FRACTION:-0.01}"
  --fss-threshold-mm "${FSS_THRESHOLD_MM:-5.0}"
  --fss-windows "${FSS_WINDOWS:-1,3,7,15,31,61}"
)

if [[ -n "${MAX_VALID_TIMES:-}" ]]; then
    ARGS+=(--max-valid-times "${MAX_VALID_TIMES}")
fi

echo "Resolved FSS windows: ${FSS_WINDOWS:-1,3,7,15,31,61}"
echo "Resolved max valid times: ${MAX_VALID_TIMES:-<full run>}"

python "${PROJECT_DIR}/scripts/run_timelag_experiment.py" "${ARGS[@]}"

