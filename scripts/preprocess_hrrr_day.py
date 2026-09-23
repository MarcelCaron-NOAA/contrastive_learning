#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_DIR))
from src.data.hrrr_preprocess import process_day

p = argparse.ArgumentParser()
p.add_argument("--date", required=True)
p.add_argument("--output-root", default=str(Path(os.environ.get("DATA_DIR", ".")) / "forecasts" / "hrrr" / "qpf6_native"))
p.add_argument("--scratch-root", default=str(Path(os.environ.get("SCRATCH_DIR", ".")) / "preprocess_hrrr_qpf6"))
p.add_argument("--overwrite", action="store_true")
p.add_argument("--retries", type=int, default=3)
a = p.parse_args()
if len(a.date) != 8 or not a.date.isdigit():
    raise SystemExit("--date must be YYYYMMDD")

year, ym = a.date[:4], a.date[:6]
out = Path(a.output_root) / "daily" / year / ym / f"hrrr_qpf6_{a.date}.zarr.zip"
jobid = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
tmp = Path(a.scratch_root) / f"hrrr_qpf6_{a.date}.{jobid}.partial.zarr.zip"
process_day(a.date, out, tmp, a.overwrite, a.retries)

