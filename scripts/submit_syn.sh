#!/usr/bin/env bash
# Submit the canonical SYN-3 or SYN-8 training workflow on URSA.
#
# Usage:
#   ./scripts/submit_syn.sh syn3 [full|smoke]
#   ./scripts/submit_syn.sh syn8 [full|smoke]
#
# Optional environment overrides:
#   RUN_NAME, WALLTIME, END_STEP, BATCH_SIZE, RESUME_CHECKPOINT, SMOKE_TAG
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"
source load_env.sh

MODEL="${1:-}"
MODE="${2:-full}"

case "${MODEL}" in
    syn3)
        DEFAULT_RUN_NAME="synthetic_perturb_syn3_v2"
        DEFAULT_WALLTIME="05:00:00"
        ;;
    syn8)
        DEFAULT_RUN_NAME="synthetic_perturb_syn8_v2"
        DEFAULT_WALLTIME="12:00:00"
        ;;
    *)
        echo "Usage: $0 {syn3|syn8} [full|smoke]" >&2
        exit 2
        ;;
esac

case "${MODE}" in
    full)
        RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
        WALLTIME="${WALLTIME:-${DEFAULT_WALLTIME}}"
        END_STEP="${END_STEP:-}"
        ;;
    smoke)
        # Give disposable smoke runs unique output directories by default so a
        # previous smoke test cannot block a later one via overwrite protection.
        SMOKE_TAG="${SMOKE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
        RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}_smoke_${SMOKE_TAG}}"
        WALLTIME="${WALLTIME:-00:45:00}"
        END_STEP="${END_STEP:-10}"
        ;;
    *)
        echo "Mode must be 'full' or 'smoke'; got ${MODE}" >&2
        exit 2
        ;;
esac

EXPORTS="ALL,SYN_MODEL=${MODEL},RUN_NAME=${RUN_NAME},BATCH_SIZE=${BATCH_SIZE:-8},END_STEP=${END_STEP},RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}"

jid="$(sbatch --parsable \
    --job-name="${MODEL}_${MODE}" \
    --time="${WALLTIME}" \
    --export="${EXPORTS}" \
    slurm/train_syn.slurm)"

echo "Submitted ${MODEL^^} ${MODE} run: ${jid}"
echo "Run name: ${RUN_NAME}"
echo "Log: ${LOG_DIR}/${RUN_NAME}_${jid}.log"
