#!/usr/bin/env bash
# Submit the final Natural Forecast Variability (NFV) training workflow on URSA.
#
# Usage:
#   ./scripts/submit_nfv.sh [full|smoke]
#
# Optional environment overrides:
#   RUN_NAME, WALLTIME, END_STEP, BATCH_SIZE, RESUME_CHECKPOINT, SMOKE_TAG
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"
source load_env.sh

MODE="${1:-full}"
DEFAULT_RUN_NAME="hard_ranking_v3_full_v1"
case "${MODE}" in
    full)
        RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
        WALLTIME="${WALLTIME:-16:00:00}"
        END_STEP="${END_STEP:-}"
        ;;
    smoke)
        SMOKE_TAG="${SMOKE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
        RUN_NAME="${RUN_NAME:-nfv_smoke_${SMOKE_TAG}}"
        WALLTIME="${WALLTIME:-00:45:00}"
        END_STEP="${END_STEP:-10}"
        ;;
    *)
        echo "Usage: $0 [full|smoke]" >&2
        exit 2
        ;;
esac

EXPORTS="ALL,RUN_NAME=${RUN_NAME},BATCH_SIZE=${BATCH_SIZE:-8},END_STEP=${END_STEP},RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}"

jid="$(sbatch --parsable \
    --job-name="nfv_${MODE}" \
    --time="${WALLTIME}" \
    --export="${EXPORTS}" \
    slurm/train_nfv.slurm)"

echo "Submitted NFV ${MODE} run: ${jid}"
echo "Run name: ${RUN_NAME}"
echo "Log: ${LOG_DIR}/${RUN_NAME}_${jid}.log"
