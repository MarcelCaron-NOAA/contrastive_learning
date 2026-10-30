#!/usr/bin/env bash
# Submit one canonical diagnostic workflow on URSA.
#
# Usage:
#   ./scripts/submit_diagnostic.sh morphology
#   ./scripts/submit_diagnostic.sh stageiv
#   ./scripts/submit_diagnostic.sh forecast-hour
#   CASE_INDEX=299 N_STEPS=1200 ./scripts/submit_diagnostic.sh inversion
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"
source load_env.sh

name="${1:-}"
case "${name}" in
    morphology)
        job="$(sbatch --parsable --export=ALL slurm/evaluate_morphology.slurm)"
        log="${LOG_DIR}/evaluate_morphology_${job}.log"
        ;;
    stageiv)
        job="$(sbatch --parsable --export=ALL slurm/evaluate_stageiv.slurm)"
        log="${LOG_DIR}/evaluate_stageiv_${job}.log"
        ;;
    forecast-hour|forecast_hour)
        job="$(sbatch --parsable --export=ALL slurm/evaluate_forecast_hour.slurm)"
        log="${LOG_DIR}/evaluate_forecast_hour_${job}.log"
        ;;
    inversion)
        CASE_INDEX="${CASE_INDEX:--1}"
        N_STEPS="${N_STEPS:-1200}"
        job="$(sbatch --parsable \
            --export=ALL,CASE_INDEX="${CASE_INDEX}",N_STEPS="${N_STEPS}",OUT_DIR="${OUT_DIR:-}" \
            slurm/evaluate_inversion.slurm)"
        log="${LOG_DIR}/evaluate_inversion_${job}.log"
        ;;
    *)
        echo "Usage: $0 {morphology|stageiv|forecast-hour|inversion}" >&2
        exit 2
        ;;
esac

echo "Submitted ${name}: ${job}"
echo "Log: ${log}"
