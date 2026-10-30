#!/usr/bin/env python3
"""Recompute traditional Stage-IV metrics at inch-based thresholds without GPU inference.

This script reuses the already-computed SYN-3/SYN-8/NFV cosine similarities from
stageiv_per_case_metrics.csv.gz and recalculates only the traditional metrics and
wet-case flags from the archived HRRR/Stage-IV fields.

Definitions:
  0.1 in = 2.54 mm
  0.5 in = 12.70 mm
  FSS-5/FSS-10 use the 0.1-in threshold.
  Non-dry pixel counts use >=0.1 in.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.verification_metrics import contingency, fss

MM_PER_IN = 25.4
THRESH_0P1_IN_MM = 0.1 * MM_PER_IN
THRESH_0P5_IN_MM = 0.5 * MM_PER_IN


def defaults():
    data = Path(os.environ["DATA_DIR"])
    base = data / "derived/evaluation/final_syn_nfv_stageiv_v1"
    return {
        "archive": data / "derived/evaluation/stageiv_reference_v1",
        "input": base / "stageiv_per_case_metrics.csv.gz",
        "output": base / "stageiv_per_case_metrics_inches.csv.gz",
        "metadata": base / "stageiv_metrics_inches_metadata.json",
    }


def counts(f, o, threshold_mm):
    h, m, fa, cn = contingency(f, o, threshold_mm)
    return float(h), float(m), float(fa), float(cn)


def ets(h, m, fa, cn):
    total = h + m + fa + cn
    if total <= 0:
        return np.nan
    href = ((h + fa) * (h + m)) / total
    den = h + m + fa - href
    return np.nan if den == 0 else float((h - href) / den)


def fbias(h, m, fa):
    obs_yes = h + m
    return np.nan if obs_yes == 0 else float((h + fa) / obs_yes)


def pod(h, m):
    den = h + m
    return np.nan if den == 0 else float(h / den)


def sr(h, fa):
    den = h + fa
    return np.nan if den == 0 else float(h / den)


def main():
    d = defaults()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", default=str(d["input"]))
    ap.add_argument("--archive-dir", default=str(d["archive"]))
    ap.add_argument("--output-csv", default=str(d["output"]))
    ap.add_argument("--metadata-json", default=str(d["metadata"]))
    args = ap.parse_args()

    inp = Path(args.input_csv)
    arc = Path(args.archive_dir)
    out = Path(args.output_csv)
    meta_path = Path(args.metadata_json)
    for p in (inp, arc / "hrrr_qpf6_mm.npy", arc / "stageiv_budget_qpf6_mm.npy"):
        if not p.is_file():
            raise SystemExit(f"Required file missing: {p}")

    df = pd.read_csv(inp).copy()
    fcst = np.load(arc / "hrrr_qpf6_mm.npy", mmap_mode="r")
    obs = np.load(arc / "stageiv_budget_qpf6_mm.npy", mmap_mode="r")
    if "archive_row" not in df.columns:
        raise RuntimeError("Input metrics table has no archive_row column")

    columns = {
        "fss5": [], "fss10": [],
        "ets_0p1": [], "ets_0p5": [],
        "fbias_0p1": [], "fbias_0p5": [],
        "pod_0p1": [], "pod_0p5": [],
        "sr_0p1": [], "sr_0p5": [],
        "fcst_wet_pixels": [], "obs_wet_pixels": [],
        "fcst_wet_fraction": [], "obs_wet_fraction": [],
        "fcst_mean_in": [], "obs_mean_in": [],
        "fcst_max_in": [], "obs_max_in": [],
    }

    print(f"Recomputing {len(df):,} Stage-IV cases with inch thresholds")
    print(f"  0.1 in = {THRESH_0P1_IN_MM:.2f} mm")
    print(f"  0.5 in = {THRESH_0P5_IN_MM:.2f} mm")

    for n, row in enumerate(df.itertuples(index=False), start=1):
        j = int(getattr(row, "archive_row"))
        f = np.asarray(fcst[j], dtype=np.float32)
        o = np.asarray(obs[j], dtype=np.float32)

        h01, m01, fa01, cn01 = counts(f, o, THRESH_0P1_IN_MM)
        h05, m05, fa05, cn05 = counts(f, o, THRESH_0P5_IN_MM)
        fwp = int(np.sum(f >= THRESH_0P1_IN_MM))
        owp = int(np.sum(o >= THRESH_0P1_IN_MM))

        columns["fss5"].append(fss(f, o, THRESH_0P1_IN_MM, 5))
        columns["fss10"].append(fss(f, o, THRESH_0P1_IN_MM, 10))
        columns["ets_0p1"].append(ets(h01, m01, fa01, cn01))
        columns["ets_0p5"].append(ets(h05, m05, fa05, cn05))
        columns["fbias_0p1"].append(fbias(h01, m01, fa01))
        columns["fbias_0p5"].append(fbias(h05, m05, fa05))
        columns["pod_0p1"].append(pod(h01, m01))
        columns["pod_0p5"].append(pod(h05, m05))
        columns["sr_0p1"].append(sr(h01, fa01))
        columns["sr_0p5"].append(sr(h05, fa05))
        columns["fcst_wet_pixels"].append(fwp)
        columns["obs_wet_pixels"].append(owp)
        columns["fcst_wet_fraction"].append(float(fwp / f.size))
        columns["obs_wet_fraction"].append(float(owp / o.size))
        columns["fcst_mean_in"].append(float(np.mean(f) / MM_PER_IN))
        columns["obs_mean_in"].append(float(np.mean(o) / MM_PER_IN))
        columns["fcst_max_in"].append(float(np.max(f) / MM_PER_IN))
        columns["obs_max_in"].append(float(np.max(o) / MM_PER_IN))

        if n % 100 == 0 or n == len(df):
            print(f"  {n:5d}/{len(df):5d}", flush=True)

    for k, v in columns.items():
        df[k] = v

    # Avoid leaving stale mm-derived summary columns that could be mistaken for
    # the revised inch-threshold definitions.
    for col in ("fcst_mean_mm", "obs_mean_mm", "fcst_max_mm", "obs_max_mm"):
        if col in df.columns:
            df = df.drop(columns=col)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    meta = {
        "source_metrics_csv": str(inp),
        "archive": str(arc),
        "traditional_thresholds_in": [0.1, 0.5],
        "traditional_thresholds_mm_internal": [THRESH_0P1_IN_MM, THRESH_0P5_IN_MM],
        "fss_threshold_in": 0.1,
        "gallery_wet_threshold_in": 0.1,
        "wet_pixel_definition": "precipitation >= 0.1 in (2.54 mm)",
        "learned_scores_reused": ["syn3", "syn8", "nfv"],
        "fbias_definition": "(hits + false alarms) / (hits + misses)",
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote: {out}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
