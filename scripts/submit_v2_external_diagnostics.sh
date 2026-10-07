#!/bin/bash
set -euo pipefail
cd "${PROJECT_DIR:-/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning}"
echo "Submitting V2 external diagnostics..."
J1=$(sbatch --parsable slurm/run_wang_v2.slurm)
J2=$(sbatch --parsable slurm/run_signed_morphology_v2.slurm)
J3=$(sbatch --parsable slurm/run_stageiv_v1_v2.slurm)
J4=$(sbatch --parsable slurm/run_ranking_margin_v1_v2.slurm)
printf 'Wang V1/V2          : %s\nSigned morphology V1/V2: %s\nStage-IV V1/V2       : %s\nRanking margins V1/V2: %s\n' "$J1" "$J2" "$J3" "$J4"

