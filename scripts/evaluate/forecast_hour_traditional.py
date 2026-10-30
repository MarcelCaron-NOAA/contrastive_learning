#!/usr/bin/env python3
"""Compute pooled categorical/FSS verification by HRRR forecast hour.

The canonical threshold is 0.1 inch = 2.54 mm.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter


def fss_components(f: np.ndarray, o: np.ndarray, threshold: float, width: int) -> tuple[float, float]:
    # Batch dimensions are independent; smooth only y/x.
    fb = (f >= threshold).astype(np.float64)
    ob = (o >= threshold).astype(np.float64)
    ff = uniform_filter(fb, size=(1, width, width), mode="constant", cval=0.0)
    oo = uniform_filter(ob, size=(1, width, width), mode="constant", cval=0.0)
    num = float(np.sum((ff - oo) ** 2, dtype=np.float64))
    ref = float(np.sum(ff ** 2 + oo ** 2, dtype=np.float64))
    return num, ref


def pooled_scores(h: int, m: int, fa: int, cn: int) -> dict[str, float]:
    n = h + m + fa + cn
    href = np.nan if n == 0 else ((h + fa) * (h + m)) / float(n)
    ets_denom = np.nan if n == 0 else h + m + fa - href
    return {
        "ets_0p1in": np.nan if not np.isfinite(ets_denom) or ets_denom == 0 else (h - href) / ets_denom,
        "pod_0p1in": np.nan if h + m == 0 else h / float(h + m),
        # 1-FAR == success ratio == hits/(hits+false alarms)
        "success_ratio_0p1in": np.nan if h + fa == 0 else h / float(h + fa),
    }


def main() -> None:
    data = Path(os.environ["DATA_DIR"])
    p = argparse.ArgumentParser()
    p.add_argument("--archive-dir", default=str(data / "derived/evaluation/stageiv_forecast_hour_v1"))
    p.add_argument("--threshold-mm", type=float, default=2.54, help="Internal precipitation threshold in mm (default: 0.1 in = 2.54 mm).")
    p.add_argument("--fss-widths", type=int, nargs="+", default=[5, 10])
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    arc = Path(args.archive_dir)
    cases = pd.read_csv(arc / "cases.csv")
    fcst = np.load(arc / "hrrr_qpf6_mm.npy", mmap_mode="r")
    obs_unique = np.load(arc / "stageiv_budget_qpf6_mm.npy", mmap_mode="r")

    cases["case_index"] = pd.to_numeric(cases["case_index"], errors="raise").astype(int)
    cases["stageiv_index"] = pd.to_numeric(cases["stageiv_index"], errors="raise").astype(int)
    cases["forecast_hour"] = pd.to_numeric(cases["forecast_hour"], errors="raise").astype(int)

    rows = []
    threshold = float(args.threshold_mm)

    for fhr, sub in cases.groupby("forecast_hour", sort=True):
        idx = sub["case_index"].to_numpy(dtype=int)
        oi = sub["stageiv_index"].to_numpy(dtype=int)
        h = m = fa = cn = 0
        comp = {int(w): [0.0, 0.0] for w in args.fss_widths}

        for s in range(0, len(idx), args.batch_size):
            ii = idx[s:s+args.batch_size]
            jj = oi[s:s+args.batch_size]
            f = np.asarray(fcst[ii], dtype=np.float32)
            o = np.asarray(obs_unique[jj], dtype=np.float32)

            fb = f >= threshold
            ob = o >= threshold
            h += int(np.sum(fb & ob))
            m += int(np.sum((~fb) & ob))
            fa += int(np.sum(fb & (~ob)))
            cn += int(np.sum((~fb) & (~ob)))

            for w in args.fss_widths:
                num, ref = fss_components(f, o, threshold, int(w))
                comp[int(w)][0] += num
                comp[int(w)][1] += ref

        r = {
            "forecast_hour": int(fhr),
            "n_cases": int(len(idx)),
            "n_pixels": int((h + m + fa + cn)),
            "hits": h,
            "misses": m,
            "false_alarms": fa,
            "correct_negatives": cn,
        }
        r.update(pooled_scores(h, m, fa, cn))
        for w, (num, ref) in comp.items():
            r[f"fss_0p1in_w{w}"] = np.nan if ref == 0 else 1.0 - num / ref
            r[f"fss_0p1in_w{w}_numerator"] = num
            r[f"fss_0p1in_w{w}_reference"] = ref
        rows.append(r)
        print(
            f"F{int(fhr):02d}: n={len(idx):,} ETS={r['ets_0p1in']:.4f} "
            f"POD={r['pod_0p1in']:.4f} SR={r['success_ratio_0p1in']:.4f} "
            + " ".join(f"FSS-{w}={r[f'fss_0p1in_w{w}']:.4f}" for w in args.fss_widths),
            flush=True,
        )

    out = pd.DataFrame(rows).sort_values("forecast_hour")
    path = arc / "traditional_metrics_by_forecast_hour.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
