#!/bin/bash
#SBATCH --job-name=dl_io_bench
#SBATCH --partition=u1-compute
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/dataloader_io_%j.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/dataloader_io_%j.err

set -euo pipefail

cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh

# load_env.sh derives OMP/MKL thread counts from SLURM_CPUS_PER_TASK.
# For this benchmark we want one numerical-library thread per DataLoader
# process, not 8 threads inside each worker.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python scripts/dataloader_io_benchmark.py \
  --workers 0,2,4,8 \
  --repeats 2 \
  --warmup-batches 2 \
  --timed-batches 20 \
  --batch-size 4 \
  --negative-count 8 \
  --patch-size 256 \
  --negative-mode climo45 \
  --prefetch-factor 2

