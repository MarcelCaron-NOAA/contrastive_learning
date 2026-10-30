#!/usr/bin/env python3
"""Recompute traditional morphology metrics at a 0.1-inch threshold, CPU only.

The expensive learned SYN-3/SYN-8/NFV similarities are reused from the existing
morphology_per_case.csv.gz.  Synthetic fields are deterministically regenerated
from the same selected validation patches so ETS/POD/SR/FSS can be recalculated
at 0.1 in (2.54 mm) without any GPU/model inference.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.verification_metrics import contingency, fss
from src.eval.wang_transforms import transform as wang_transform
from src.eval.morphology_signed_transforms import transform_signed

MM_PER_IN = 25.4
THRESHOLD_IN = 0.1
THRESHOLD_MM = THRESHOLD_IN * MM_PER_IN


def defaults():
    data = Path(os.environ["DATA_DIR"])
    base = data / "derived/evaluation/final_syn_nfv_morphology_v1"
    return {
        "archive": data / "derived/evaluation/wang_transform_v1",
        "per_case": base / "morphology_per_case.csv.gz",
        "selected": base / "selected_cases.csv",
        "out_per_case": base / "morphology_per_case_inches.csv.gz",
        "out_summary": base / "morphology_summary_inches.csv",
        "metadata": base / "morphology_inches_metadata.json",
    }


def transform_field(field, family, bias, case_seed):
    if family in ("displacement", "intensity", "area_size"):
        return wang_transform(field, family, float(bias))
    if family == "smoothing":
        return transform_signed(field, "smoothing_sharpening", float(bias), case_seed=int(case_seed))
    return transform_signed(field, family, float(bias), case_seed=int(case_seed))


def categorical(f, o, threshold_mm):
    h, m, fa, cn = contingency(f, o, threshold_mm)
    total = h + m + fa + cn
    pod = np.nan if h + m == 0 else h / float(h + m)
    sr = np.nan if h + fa == 0 else h / float(h + fa)
    if total <= 0:
        ets = np.nan
    else:
        href = ((h + fa) * (h + m)) / float(total)
        den = h + m + fa - href
        ets = np.nan if den == 0 else (h - href) / den
    return (
        float(ets) if np.isfinite(ets) else np.nan,
        float(pod) if np.isfinite(pod) else np.nan,
        float(sr) if np.isfinite(sr) else np.nan,
    )


def ci95(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(x.mean())
    if len(x) == 1:
        return mean, mean, mean, 1
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    return mean, mean - 1.96 * se, mean + 1.96 * se, int(len(x))


def pearson(a, b):
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    if np.std(aa) == 0 or np.std(bb) == 0:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def main():
    d = defaults()
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", default=str(d["archive"]))
    ap.add_argument("--per-case-csv", default=str(d["per_case"]))
    ap.add_argument("--selected-cases-csv", default=str(d["selected"]))
    ap.add_argument("--output-per-case", default=str(d["out_per_case"]))
    ap.add_argument("--output-summary", default=str(d["out_summary"]))
    ap.add_argument("--metadata-json", default=str(d["metadata"]))
    ap.add_argument("--case-seed", type=int, default=20261007)
    args = ap.parse_args()

    arc = Path(args.archive_dir)
    per_path = Path(args.per_case_csv)
    selected_path = Path(args.selected_cases_csv)
    for p in (arc / "original_patches.npz", per_path, selected_path):
        if not p.is_file():
            raise SystemExit(f"Required file missing: {p}")

    per = pd.read_csv(per_path).copy()
    selected = pd.read_csv(selected_path).copy()
    raw_all = np.load(arc / "original_patches.npz")["qpf6_mm"]

    if "archive_case_index" not in selected.columns:
        raise RuntimeError("selected_cases.csv lacks archive_case_index")
    selected = selected.reset_index(drop=True)
    selected["case_position"] = np.arange(len(selected), dtype=int)
    pos_to_archive = dict(zip(selected.case_position.astype(int), selected.archive_case_index.astype(int)))
    raw = np.asarray(raw_all[selected.archive_case_index.astype(int).to_numpy()], dtype=np.float32)

    # Allocate revised metrics plus raw-space diagnostic quantities.  The latter
    # are useful for diagnosing non-monotonic learned responses such as area-size
    # bumps without rerunning any model inference.
    for col in (
        "ets", "pod", "success_ratio", "fss5", "fss10",
        "pixel_mae_mm", "pixel_rmse_mm", "pixel_corr", "total_precip_ratio",
    ):
        per[col] = np.nan

    groups = list(per.groupby(["family", "bias"], sort=False))
    print(f"Recomputing traditional morphology metrics at {THRESHOLD_IN:.1f} in = {THRESHOLD_MM:.2f} mm")
    print(f"Groups: {len(groups)}; rows: {len(per):,}")

    for gi, ((family, bias), g) in enumerate(groups, start=1):
        # Preserve the exact per-case deterministic seed used by the original evaluator.
        for ridx in g.index:
            pos = int(per.at[ridx, "case_position"])
            archive_idx = int(per.at[ridx, "archive_case_index"])
            if pos not in pos_to_archive or pos_to_archive[pos] != archive_idx:
                raise RuntimeError(f"Case mapping mismatch at row {ridx}: pos={pos} archive={archive_idx}")
            orig = raw[pos]
            pert = np.asarray(
                transform_field(orig, str(family), float(bias), args.case_seed + archive_idx),
                dtype=np.float32,
            )
            ets, pod, sr = categorical(pert, orig, THRESHOLD_MM)
            diff = pert.astype(np.float64) - orig.astype(np.float64)
            orig_sum = float(np.sum(orig, dtype=np.float64))
            pert_sum = float(np.sum(pert, dtype=np.float64))

            per.at[ridx, "ets"] = ets
            per.at[ridx, "pod"] = pod
            per.at[ridx, "success_ratio"] = sr
            per.at[ridx, "fss5"] = fss(pert, orig, THRESHOLD_MM, 5)
            per.at[ridx, "fss10"] = fss(pert, orig, THRESHOLD_MM, 10)
            per.at[ridx, "pixel_mae_mm"] = float(np.mean(np.abs(diff)))
            per.at[ridx, "pixel_rmse_mm"] = float(np.sqrt(np.mean(diff * diff)))
            per.at[ridx, "pixel_corr"] = pearson(pert, orig)
            per.at[ridx, "total_precip_ratio"] = np.nan if orig_sum == 0 else pert_sum / orig_sum

        if gi % 10 == 0 or gi == len(groups):
            print(f"  {gi:3d}/{len(groups):3d} family={family} bias={float(bias):g}", flush=True)

    out_per = Path(args.output_per_case)
    out_sum = Path(args.output_summary)
    out_per.parent.mkdir(parents=True, exist_ok=True)
    per.to_csv(out_per, index=False)

    metrics = (
        "syn3", "syn8", "nfv", "ets", "pod", "success_ratio", "fss5", "fss10",
        "pixel_mae_mm", "pixel_rmse_mm", "pixel_corr", "total_precip_ratio",
    )
    rows = []
    for (family, bias), g in per.groupby(["family", "bias"], sort=False):
        row = {
            "family": family,
            "bias": float(bias),
            "normalized_severity": float(g.normalized_severity.iloc[0]),
            "wang_target_similarity": float(g.wang_target_similarity.iloc[0]),
            "n_rows": int(len(g)),
        }
        for metric in metrics:
            mean, lo, hi, nn = ci95(g[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_lo"] = lo
            row[f"{metric}_ci95_hi"] = hi
            row[f"{metric}_n"] = nn
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_sum, index=False)

    meta = {
        "source_per_case_csv": str(per_path),
        "archive": str(arc),
        "traditional_threshold_in": THRESHOLD_IN,
        "traditional_threshold_mm_internal": THRESHOLD_MM,
        "fss_threshold_in": THRESHOLD_IN,
        "learned_scores_reused": ["syn3", "syn8", "nfv"],
        "raw_space_diagnostics_added": ["pixel_mae_mm", "pixel_rmse_mm", "pixel_corr", "total_precip_ratio"],
        "case_seed": int(args.case_seed),
    }
    Path(args.metadata_json).write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Per-case: {out_per}")
    print(f"Summary : {out_sum}")


if __name__ == "__main__":
    main()
