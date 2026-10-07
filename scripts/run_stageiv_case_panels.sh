#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# USER SETTINGS
# -------------------------
EXP_DIR="/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/data/derived/experiments/hard_ranking_finetune_v1"
SCRIPT_DIR="/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning/scripts"

# Your per-case Stage-IV comparison table.
# Adjust this filename to whatever your Stage-IV diagnostic actually wrote.
CASE_TABLE="${EXP_DIR}/evaluation/stageiv_v1_vs_v2/stageiv_case_scores.csv"

OUT_DIR="${EXP_DIR}/evaluation/stageiv_case_triptychs"
MANIFEST="${OUT_DIR}/selected_cases.csv"

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo " Selecting Stage-IV cases"
echo "============================================================"
python "${SCRIPT_DIR}/select_stageiv_case_buckets.py" \
    --input "${CASE_TABLE}" \
    --output "${MANIFEST}" \
    --n-per-bucket 6 \
    --wet-total-thresh-in 0.10 \
    --min-wet-pixels 16

echo "============================================================"
echo " Plotting Stage-IV triptychs"
echo "============================================================"
python "${SCRIPT_DIR}/plot_stageiv_case_triptychs.py" \
    --manifest "${MANIFEST}" \
    --output-dir "${OUT_DIR}/panels" \
    --assume-mm

echo "============================================================"
echo " COMPLETE"
echo "============================================================"
echo "Manifest: ${MANIFEST}"
echo "Panels  : ${OUT_DIR}/panels"
