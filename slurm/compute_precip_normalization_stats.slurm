#!/bin/bash
#SBATCH --job-name=qpf_norm
#SBATCH --partition=u1-compute
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source load_env.sh

NUM_PATCHES="${NUM_PATCHES:-5000}"
PATCH_SIZE="${PATCH_SIZE:-256}"
SEED="${SEED:-20260930}"

python scripts/compute_precip_normalization_stats.py \
  --num-patches "${NUM_PATCHES}" \
  --patch-size "${PATCH_SIZE}" \
  --seed "${SEED}"

