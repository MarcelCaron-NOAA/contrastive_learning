#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning}"
cd "$PROJECT_DIR"
source load_env.sh

j1=$(sbatch slurm/run_final_stageiv_buckets.slurm | awk '{print $4}')
j2=$(sbatch slurm/run_final_morphology_models.slurm | awk '{print $4}')

echo "Submitted Stage-IV buckets: $j1"
echo "  log: $LOG_DIR/final_stageiv_buckets_${j1}.log"
echo "Submitted morphology:       $j2"
echo "  log: $LOG_DIR/final_morphology_${j2}.log"

