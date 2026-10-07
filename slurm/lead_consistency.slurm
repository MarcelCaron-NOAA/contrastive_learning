#!/usr/bin/env bash
#SBATCH --job-name=hrrr_lead_consistency
#SBATCH --partition=u1-compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/lead_consistency_%A_%a.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/lead_consistency_%A_%a.err

set -euo pipefail

PROJECT_DIR="/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning"
source "${PROJECT_DIR}/load_env.sh"

: "${LEAD_EXPERIMENT_DIR:?LEAD_EXPERIMENT_DIR not provided}"

CONFIG="${LEAD_EXPERIMENT_DIR}/experiment_config.env"
[[ -s "${CONFIG}" ]] || {
  echo "Missing config: ${CONFIG}"
  exit 2
}

# shellcheck disable=SC1090
source "${CONFIG}"

python "${PROJECT_DIR}/scripts/run_lead_consistency_experiment.py" \
  --manifest "${DATA_DIR}/forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz" \
  --output-dir "${LEAD_EXPERIMENT_DIR}" \
  --lead-hour "${SLURM_ARRAY_TASK_ID}" \
  --start "${START_DATE}" \
  --end "${END_DATE}" \
  --samples-per-season "${SAMPLES_PER_SEASON}" \
  --patch-size "${PATCH_SIZE}" \
  --patch-tries "${PATCH_TRIES}" \
  --wet-threshold-mm "${WET_THRESHOLD_MM}" \
  --min-wet-fraction "${MIN_WET_FRACTION}" \
  --fss-threshold-mm "${FSS_THRESHOLD_MM}" \
  --fss-window "${FSS_WINDOW}" \
  --seed "${SEED}"

