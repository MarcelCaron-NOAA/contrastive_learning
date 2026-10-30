#!/usr/bin/env bash
# Fast repository-integrity checks for a clean clone/refactor.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"
source load_env.sh >/dev/null

echo "[1/4] Python tests"
python -m pytest -q tests/test_synthetic_training.py tests/test_contrastive_sample.py

echo "[2/4] Python compilation"
python -m compileall -q src scripts

echo "[3/4] Shell/Slurm syntax"
while IFS= read -r -d '' file; do
    bash -n "${file}"
done < <(find scripts slurm -type f \( -name '*.sh' -o -name '*.slurm' \) -print0)

echo "[4/4] Personal runtime-path scan"
if grep -RInE '/scratch4/[^[:space:]]*Marcel\.Caron|/lfs/[^[:space:]]*Marcel\.Caron' \
    scripts slurm src tests configs --exclude='*.pyc'; then
    echo "ERROR: personal absolute runtime path found in live code." >&2
    exit 1
fi

echo "Repository checks PASS"
