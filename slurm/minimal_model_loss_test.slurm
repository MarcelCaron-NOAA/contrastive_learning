#!/bin/bash
#SBATCH --job-name=contrast_min
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/minimal_model_loss_%j.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/minimal_model_loss_%j.err

set -euo pipefail

cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning

source load_env.sh

python scripts/minimal_model_loss_test.py \
  --negative-mode climo45 \
  --negative-count 8 \
  --batch-size 4 \
  --patch-size 256 \
  --cache-batches 4 \
  --num-workers 0 \
  --steps 100 \
  --temperature 0.10 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4
