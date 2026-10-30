#!/bin/bash
set -euo pipefail
CASE_INDEX="${CASE_INDEX:--1}"
N_STEPS="${N_STEPS:-1200}"
echo "Submitting dense feature-inversion continuation: CASE_INDEX=$CASE_INDEX N_STEPS=$N_STEPS"
sbatch --export=ALL,CASE_INDEX="$CASE_INDEX",N_STEPS="$N_STEPS" slurm/run_feature_inversion_continuation.slurm
