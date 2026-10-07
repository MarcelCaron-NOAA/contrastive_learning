#!/usr/bin/env bash
set -euo pipefail

cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
mkdir -p /scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs

j1=$(sbatch --parsable slurm/run_training_cue_audit.slurm)
j2=$(sbatch --parsable slurm/run_layerwise_transform_sensitivity.slurm)
j3=$(sbatch --parsable slurm/run_morphology_perturbations.slurm)

echo "Submitted all three representation-characterization experiments:"
echo "  training-cue audit          : ${j1}"
echo "  layerwise/random sensitivity: ${j2}"
echo "  morphology perturbations    : ${j3}"
echo
echo "They are independent jobs. Slurm can run them concurrently when resources are available."
echo "Check with: squeue -j ${j1},${j2},${j3}"

