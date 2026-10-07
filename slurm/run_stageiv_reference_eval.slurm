#!/bin/bash
#SBATCH --job-name=st4_ref_eval
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/stageiv_ref_eval_%j.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/stageiv_ref_eval_%j.err

set -euo pipefail

cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh

python scripts/evaluate_stageiv_reference.py \
  --batch-size 32 \
  --fss-thresholds 0.1 1 5 10 \
  --fss-widths 1 7 15 31 \
  --representative-fss-threshold 1 \
  --representative-fss-width 15

python scripts/plot_stageiv_galleries.py \
  --count 6 \
  --fss-column budget_fss_t1_w15

