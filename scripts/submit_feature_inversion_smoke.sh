#!/bin/bash
set -euo pipefail
CASE_INDEX="${CASE_INDEX:--1}"
N_STEPS="${N_STEPS:-300}"
echo "Submitting feature-inversion smoke test: CASE_INDEX=$CASE_INDEX N_STEPS=$N_STEPS"
sbatch --export=ALL,CASE_INDEX="$CASE_INDEX",N_STEPS="$N_STEPS" slurm/run_feature_inversion_smoke.slurm
