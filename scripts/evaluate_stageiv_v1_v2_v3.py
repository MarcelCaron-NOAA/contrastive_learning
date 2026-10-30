#!/usr/bin/env python3
"""Compare V1, V2, and V3 on the HRRR-vs-Stage-IV validation archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data import PrecipTransform
from src.models.checkpoint_loader import load_model_checkpoint

MM_TO_IN = 1.0 / 25.4
THRESH_ETS_MM = 2.54  # 0.10 in


def defaults():
    data = Path(os.environ["DATA_DIR"])
    return {
        "archive": data / "derived/evaluation/stageiv_reference_v1",
        "v1": data / "derived/experiments/groupnorm_pilot_v1/checkpoint_best.pt",
        "v2": data / "derived/experiments/hard_ranking_finetune_v1/checkpoint_best.pt",
        "v3": data / "derived/experiments/hard_ranking_v3_full_v1/checkpoint_best.pt",
        "stats": data / "metadata/dataset_v1/precip_normalization_stats_v1.json",
        "out": data / "derived/experiments/hard_ranking_v3_full_v1/evaluation/stageiv_v1_v2_v3",
    }


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def step(ckpt: dict) -> int:
    return int(ckpt.get("fine_tune_step", ckpt.get("step", -1)))


@torch.no_grad()
def encode(model, raw, pt, dev, bs):
    out = []
    for s in range(0, len(raw), bs):
        x = torch.stack([pt(v) for v in raw[s:s + bs]], 0).to(dev, non_blocking=True)
        _, z = model(x)
        out.append(F.normalize(z.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def fss_score(fcst_mm: np.ndarray, obs_mm: np.ndarray, threshold_mm: float, width: int) -> float:
    from scipy.ndimage import uniform_filter
    f = (np.asarray(fcst_mm) >= threshold_mm).astype(np.float32)
    o = (np.asarray(obs_mm) >= threshold_mm).astype(np.float32)
    ff = uniform_filter(f, size=width, mode="nearest")
    oo = uniform_filter(o, size=width, mode="nearest")
    num = float(np.mean((ff - oo) ** 2))
    den = float(np.mean(ff ** 2 + oo ** 2))
    if den <= 0:
        return np.nan
    return 1.0 - num / den


def ets_score(fcst_mm: np.ndarray, obs_mm: np.ndarray, threshold_mm: float) -> float:
    f = np.asarray(fcst_mm) >= threshold_mm
    o = np.asarray(obs_mm) >= threshold_mm
    h = float(np.logical_and(f, o).sum())
    m = float(np.logical_and(~f, o).sum())
    fa = float(np.logical_and(f, ~o).sum())
    n = float(f.size)
    hr = ((h + m) * (h + fa)) / n if n else np.nan
    den = h + m + fa - hr
    if not np.isfinite(den) or den <= 0:
        return np.nan
    return (h - hr) / den


def pearson_r(f, o):
    f = np.asarray(f, float).ravel()
    o = np.asarray(o, float).ravel()
    if np.std(f) == 0 or np.std(o) == 0:
        return np.nan
    return float(np.corrcoef(f, o)[0, 1])


def _label_objects(mask: np.ndarray):
    from scipy.ndimage import label
    return label(mask, structure=np.ones((3, 3), dtype=np.int8))


def _object_threshold(a: np.ndarray, b: np.ndarray, floor_mm: float) -> float:
    amax = float(np.nanmax(a)) if np.size(a) else 0.0
    bmax = float(np.nanmax(b)) if np.size(b) else 0.0
    return max(float(floor_mm), max(amax, bmax) / 15.0)


def _weighted_centroid(field: np.ndarray):
    w = np.asarray(field, np.float64)
    s = float(w.sum())
    if s <= 0:
        ny, nx = w.shape
        return 0.5 * (ny - 1), 0.5 * (nx - 1)
    yy, xx = np.indices(w.shape, dtype=np.float64)
    return float((w * yy).sum() / s), float((w * xx).sum() / s)


def _scatter_distance(field: np.ndarray, threshold: float) -> float:
    mask = np.asarray(field, np.float32) >= threshold
    labels, nlab = _label_objects(mask)
    total_mass = float(np.sum(field, dtype=np.float64))
    if nlab == 0 or total_mass <= 0:
        return 0.0
    cy, cx = _weighted_centroid(field)
    total = 0.0
    for lab in range(1, nlab + 1):
        obj = labels == lab
        mass = float(np.sum(field[obj], dtype=np.float64))
        if mass <= 0:
            continue
        oy, ox = _weighted_centroid(np.where(obj, field, 0.0))
        total += mass * np.hypot(oy - cy, ox - cx)
    return total / total_mass


def _structure_metric(field: np.ndarray, threshold: float) -> float:
    mask = np.asarray(field, np.float32) >= threshold
    labels, nlab = _label_objects(mask)
    total_mass = float(np.sum(field, dtype=np.float64))
    if nlab == 0 or total_mass <= 0:
        return 0.0
    pieces = []
    weights = []
    for lab in range(1, nlab + 1):
        obj = labels == lab
        mass = float(np.sum(field[obj], dtype=np.float64))
        if mass <= 0:
            continue
        peak = float(np.max(field[obj]))
        if peak <= 0:
            continue
        pieces.append(mass / peak)
        weights.append(mass)
    if not weights:
        return 0.0
    weights = np.asarray(weights, np.float64)
    pieces = np.asarray(pieces, np.float64)
    return float(np.sum(weights * pieces) / np.sum(weights))


def normalized_sal(fcst_mm: np.ndarray, obs_mm: np.ndarray, threshold_floor_mm: float = THRESH_ETS_MM) -> float:
    f = np.maximum(np.asarray(fcst_mm, np.float32), 0.0)
    o = np.maximum(np.asarray(obs_mm, np.float32), 0.0)
    mf = float(np.mean(f))
    mo = float(np.mean(o))
    if mf + mo > 0:
        a = 2.0 * (mf - mo) / (mf + mo)
    else:
        a = 0.0

    thresh = _object_threshold(f, o, threshold_floor_mm)
    cf = _weighted_centroid(f)
    co = _weighted_centroid(o)
    diag = float(np.hypot(*f.shape))
    l1 = np.hypot(cf[0] - co[0], cf[1] - co[1]) / diag if diag > 0 else 0.0
    rf = _scatter_distance(f, thresh)
    ro = _scatter_distance(o, thresh)
    l2 = 2.0 * (rf - ro) / (diag + 1.0e-12) if diag > 0 else 0.0
    l = l1 + l2

    vf = _structure_metric(f, thresh)
    vo = _structure_metric(o, thresh)
    s = 2.0 * (vf - vo) / (vf + vo) if (vf + vo) > 0 else 0.0

    sim = 1.0 - (abs(s) + abs(a) + abs(l)) / 6.0
    return float(np.clip(sim, 0.0, 1.0))


def spearman(x, y) -> float:
    return float(pd.Series(x).rank(method="average").corr(pd.Series(y).rank(method="average")))


def quantiles(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    qs = [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1]
    vals = np.quantile(x, qs)
    return {f"q{int(q * 100):02d}": float(v) for q, v in zip(qs, vals)}


def main():
    d = defaults()
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", default=str(d["archive"]))
    ap.add_argument("--v1-checkpoint", default=str(d["v1"]))
    ap.add_argument("--v2-checkpoint", default=str(d["v2"]))
    ap.add_argument("--v3-checkpoint", default=str(d["v3"]))
    ap.add_argument("--stats-json", default=str(d["stats"]))
    ap.add_argument("--output-dir", default=str(d["out"]))
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    arc = Path(args.archive_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cases = pd.read_csv(arc / "cases.csv")
    fcst = np.load(arc / "hrrr_qpf6_mm.npy", mmap_mode="r")
    obs = np.load(arc / "stageiv_budget_qpf6_mm.npy", mmap_mode="r")
    if not (len(cases) == len(fcst) == len(obs)):
        raise RuntimeError("Stage-IV archive length mismatch")

    dev = torch.device("cuda")
    pt = PrecipTransform.from_stats_json("log1p_zscore", args.stats_json)

    m1, c1 = load_model_checkpoint(args.v1_checkpoint, device=dev)
    m2, c2 = load_model_checkpoint(args.v2_checkpoint, device=dev)
    m3, c3 = load_model_checkpoint(args.v3_checkpoint, device=dev)
    for m in (m1, m2, m3):
        m.eval()

    zf1 = encode(m1, fcst, pt, dev, args.batch_size)
    zo1 = encode(m1, obs, pt, dev, args.batch_size)
    zf2 = encode(m2, fcst, pt, dev, args.batch_size)
    zo2 = encode(m2, obs, pt, dev, args.batch_size)
    zf3 = encode(m3, fcst, pt, dev, args.batch_size)
    zo3 = encode(m3, obs, pt, dev, args.batch_size)
    s1 = np.sum(zf1 * zo1, axis=1)
    s2 = np.sum(zf2 * zo2, axis=1)
    s3 = np.sum(zf3 * zo3, axis=1)

    rec = []
    for i in range(len(cases)):
        f = np.asarray(fcst[i], np.float32)
        o = np.asarray(obs[i], np.float32)
        r = cases.iloc[i].to_dict()
        wet_f = float(np.mean(f >= THRESH_ETS_MM))
        wet_o = float(np.mean(o >= THRESH_ETS_MM))
        rec.append({
            **r,
            "learned_v1": float(s1[i]),
            "learned_v2": float(s2[i]),
            "learned_v3": float(s3[i]),
            "delta_v3_minus_v2": float(s3[i] - s2[i]),
            "delta_v3_minus_v1": float(s3[i] - s1[i]),
            "corr": pearson_r(f, o),
            "fss5_0p1in": fss_score(f, o, THRESH_ETS_MM, 5),
            "fss10_0p1in": fss_score(f, o, THRESH_ETS_MM, 10),
            "ets_0p1in": ets_score(f, o, THRESH_ETS_MM),
            "normalized_sal": normalized_sal(f, o),
            "fcst_wet_fraction_0p1in": wet_f,
            "obs_wet_fraction_0p1in": wet_o,
            "both_wet": bool((wet_f > 0) and (wet_o > 0)),
            "fcst_mean_in": float(np.mean(f) * MM_TO_IN),
            "obs_mean_in": float(np.mean(o) * MM_TO_IN),
            "fcst_max_in": float(np.max(f) * MM_TO_IN),
            "obs_max_in": float(np.max(o) * MM_TO_IN),
        })
    df = pd.DataFrame(rec)
    df.to_csv(out / "per_case_metrics.csv.gz", index=False)

    summary = []
    for name in ["learned_v1", "learned_v2", "learned_v3", "fss5_0p1in", "fss10_0p1in", "ets_0p1in", "normalized_sal"]:
        summary.append({
            "metric": name,
            "mean": float(np.nanmean(df[name])),
            "std": float(np.nanstd(df[name])),
            "spearman_vs_v3": spearman(df[name], df["learned_v3"]),
            **quantiles(df[name]),
        })
    pd.DataFrame(summary).to_csv(out / "score_summary.csv", index=False)

    meta = {
        "archive_dir": str(arc),
        "stats_json": str(args.stats_json),
        "n_cases": len(df),
        "v1_checkpoint": str(args.v1_checkpoint),
        "v1_sha256": sha(args.v1_checkpoint),
        "v1_step": step(c1),
        "v2_checkpoint": str(args.v2_checkpoint),
        "v2_sha256": sha(args.v2_checkpoint),
        "v2_step": step(c2),
        "v3_checkpoint": str(args.v3_checkpoint),
        "v3_sha256": sha(args.v3_checkpoint),
        "v3_step": step(c3),
        "ets_threshold_mm": THRESH_ETS_MM,
        "ets_threshold_in": 0.10,
        "fss_widths": [5, 10],
        "normalized_sal_definition": "1 - (|S| + |A| + |L|)/6 using an object threshold max(0.1in, global max / 15).",
        "sealed_test_used": False,
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    print("=" * 88)
    print("STAGE-IV V1/V2/V3 COMPARISON COMPLETE")
    print("=" * 88)
    print(df[["learned_v1", "learned_v2", "learned_v3", "fss5_0p1in", "fss10_0p1in", "ets_0p1in", "normalized_sal"]].describe().to_string())
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
