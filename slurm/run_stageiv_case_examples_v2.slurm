#!/bin/bash
#SBATCH -J stageiv_cases_v2
#SBATCH -A ovp
#SBATCH -p u1-compute
#SBATCH -q batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/stageiv_cases_v2_%j.out
#SBATCH -e /scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/stageiv_cases_v2_%j.err

set -euo pipefail
cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh

python scripts/plot_stageiv_case_examples_v2.py \
  --n-per-bucket 6

