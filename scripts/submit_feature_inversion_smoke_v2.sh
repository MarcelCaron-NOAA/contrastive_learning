#!/bin/bash
set -euo pipefail
CASE_INDEX="${CASE_INDEX:--1}"
N_STEPS="${N_STEPS:-500}"
OUT_DIR="${OUT_DIR:-}"

echo "Submitting exact-init feature-inversion smoke test: CASE_INDEX=$CASE_INDEX N_STEPS=$N_STEPS"
if [[ -n "$OUT_DIR" ]]; then
  sbatch --export=ALL,CASE_INDEX="$CASE_INDEX",N_STEPS="$N_STEPS",OUT_DIR="$OUT_DIR" slurm/run_feature_inversion_smoke.slurm
else
  sbatch --export=ALL,CASE_INDEX="$CASE_INDEX",N_STEPS="$N_STEPS" slurm/run_feature_inversion_smoke.slurm
fi
