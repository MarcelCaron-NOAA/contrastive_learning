#!/usr/bin/env bash
#SBATCH --job-name=test_pytorch_gpu
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/test_gpu_%j.out

source /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning/src/load_env.sh

echo "HOST: $(hostname)"

nvidia-smi

python - <<'PY'
import torch

print()
print("PyTorch version :", torch.__version__)
print("PyTorch CUDA    :", torch.version.cuda)
print("CUDA available  :", torch.cuda.is_available())
print("GPU count       :", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU             :", torch.cuda.get_device_name(0))

    x = torch.rand(1000, 1000, device="cuda")
    y = x @ x
    print("GPU tensor test :", y.device, y.shape)
PY
