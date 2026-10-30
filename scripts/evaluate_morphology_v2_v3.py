#!/usr/bin/env python3
"""Evaluate 8 morphology-transform families for V2 and V3."""
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
from src.eval.wang_transforms import DISPLACEMENT_BIASES, INTENSITY_BIASES, AREA_SIZE_BIASES, transform as wang_transform
from src.eval.morphology_signed_transforms import SIGNED_LEVELS, transform_signed
from src.eval.verification_metrics import continuous_and_wang_metrics

THRESH_MM = 2.54


def defaults():
    data = Path(os.environ["DATA_DIR"])
    return {
        "archive": data / "derived/evaluation/wang_transform_v1",
        "v2": data / "derived/experiments/hard_ranking_finetune_v1/checkpoint_best.pt",
        "v3": data / "derived/experiments/hard_ranking_v3_full_v1/checkpoint_best.pt",
        "stats": data / "metadata/dataset_v1/precip_normalization_stats_v1.json",
        "out": data / "derived/experiments/hard_ranking_v3_full_v1/evaluation/morphology_v2_v3",
    }


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def step(c):
    return int(c.get("fine_tune_step", c.get("step", -1)))


@torch.no_grad()
def encode(model, raw, pt, device, bs):
    out = []
    for s in range(0, len(raw), bs):
        x = torch.stack([pt(v) for v in raw[s:s + bs]], 0).to(device, non_blocking=True)
        _, z = model(x)
        out.append(F.normalize(z.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def ci95(v):
    x = np.asarray(v, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan, np.nan, 0
    m = float(x.mean())
    if len(x) == 1:
        return m, m, m, 1
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    return m, m - 1.96 * se, m + 1.96 * se, len(x)


def total_change_pct(orig, pert):
    a = float(np.sum(orig, dtype=np.float64))
    b = float(np.sum(pert, dtype=np.float64))
    return np.nan if a == 0 else 100.0 * (b - a) / a


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
    pieces = np.asarray(pieces, np.float64)
    weights = np.asarray(weights, np.float64)
    return float(np.sum(weights * pieces) / np.sum(weights))


def normalized_sal(fcst_mm: np.ndarray, obs_mm: np.ndarray, threshold_floor_mm: float = THRESH_MM) -> float:
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


def make_transform(family, bias, field, case_seed):
    if family in ("displacement", "intensity", "area_size"):
        return wang_transform(field, family, float(bias))
    return transform_signed(field, family, float(bias), case_seed=case_seed)


def family_biases():
    return [
        ("displacement", DISPLACEMENT_BIASES),
        ("intensity", INTENSITY_BIASES),
        ("area_size", AREA_SIZE_BIASES),
        ("smoothing_sharpening", np.array(SIGNED_LEVELS["smoothing_sharpening"], dtype=float)),
        ("object_dispersion_cells", np.array(SIGNED_LEVELS["object_dispersion_cells"], dtype=float)),
        ("principal_axis_log2_scale", np.array(SIGNED_LEVELS["principal_axis_log2_scale"], dtype=float)),
        ("rotation_deg", np.array(SIGNED_LEVELS["rotation_deg"], dtype=float)),
        ("tile_fragmentation_cells", np.array(SIGNED_LEVELS["tile_fragmentation_cells"], dtype=float)),
    ]


def main():
    d = defaults()
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", default=str(d["archive"]))
    ap.add_argument("--v2-checkpoint", default=str(d["v2"]))
    ap.add_argument("--v3-checkpoint", default=str(d["v3"]))
    ap.add_argument("--stats-json", default=str(d["stats"]))
    ap.add_argument("--output-dir", default=str(d["out"]))
    ap.add_argument("--case-count", type=int, default=500)
    ap.add_argument("--case-seed", type=int, default=20261007)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threshold-mm", type=float, default=THRESH_MM)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    arc = Path(args.archive_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    allraw = np.load(arc / "original_patches.npz")["qpf6_mm"]
    cases = pd.read_csv(arc / "cases.csv")
    rng = np.random.default_rng(args.case_seed)
    n = min(args.case_count, len(allraw))
    idx = np.sort(rng.choice(len(allraw), size=n, replace=False))
    raw = np.asarray(allraw[idx], np.float32)
    cases.iloc[idx].to_csv(out / "selected_cases.csv", index=False)

    dev = torch.device("cuda")
    v2, c2 = load_model_checkpoint(args.v2_checkpoint, device=dev)
    v3, c3 = load_model_checkpoint(args.v3_checkpoint, device=dev)
    v2.eval(); v3.eval()
    pt = PrecipTransform.from_stats_json("log1p_zscore", args.stats_json)
    z20 = encode(v2, raw, pt, dev, args.batch_size)
    z30 = encode(v3, raw, pt, dev, args.batch_size)

    records = []
    for fam, biases in family_biases():
        print(f"\n{fam}: {len(biases)} biases x {n} cases")
        for j, bias in enumerate(biases):
            trans = np.stack([
                make_transform(fam, float(bias), raw[i], case_seed=args.case_seed + int(idx[i]))
                for i in range(n)
            ]).astype(np.float32)
            z2 = encode(v2, trans, pt, dev, args.batch_size)
            z3 = encode(v3, trans, pt, dev, args.batch_size)
            s2 = np.sum(z20 * z2, axis=1)
            s3 = np.sum(z30 * z3, axis=1)
            for i, (orig, pert) in enumerate(zip(raw, trans)):
                m = continuous_and_wang_metrics(pert, orig, args.threshold_mm)
                records.append({
                    "archive_case_index": int(idx[i]),
                    "case_position": i,
                    "family": fam,
                    "bias": float(bias),
                    "learned_similarity_v2": float(s2[i]),
                    "learned_similarity_v3": float(s3[i]),
                    "learned_similarity_delta_v3_minus_v2": float(s3[i] - s2[i]),
                    "normalized_sal": normalized_sal(pert, orig, args.threshold_mm),
                    "total_precip_change_pct": total_change_pct(orig, pert),
                    **m,
                })
            print(f"  {j + 1:2d}/{len(biases)} bias={float(bias):g}")

    per = pd.DataFrame(records)
    per.to_csv(out / "per_case_metrics.csv.gz", index=False)

    cols = [
        "learned_similarity_v2", "learned_similarity_v3", "ts", "pod",
        "normalized_far", "fss5", "fss10", "normalized_sal", "corr",
        "rmse", "mae", "total_precip_change_pct",
    ]
    rows = []
    for (fam, bias), g in per.groupby(["family", "bias"], sort=False):
        r = {"family": fam, "bias": bias, "n_rows": len(g)}
        for col in cols:
            m, lo, hi, nn = ci95(g[col])
            r[f"{col}_mean"] = m
            r[f"{col}_ci95_lo"] = lo
            r[f"{col}_ci95_hi"] = hi
            r[f"{col}_n"] = nn
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "summary_by_bias.csv", index=False)

    meta = {
        "archive_dir": str(args.archive_dir),
        "v2_checkpoint": str(args.v2_checkpoint),
        "v2_sha256": sha(args.v2_checkpoint),
        "v2_step": step(c2),
        "v3_checkpoint": str(args.v3_checkpoint),
        "v3_sha256": sha(args.v3_checkpoint),
        "v3_step": step(c3),
        "stats_json": str(args.stats_json),
        "case_count": n,
        "case_seed": args.case_seed,
        "threshold_mm": args.threshold_mm,
        "families": {k: list(map(float, v)) for k, v in family_biases()},
        "normalized_sal_definition": "1 - (|S| + |A| + |L|)/6 using object threshold max(threshold_mm, global max / 15).",
        "sealed_test_used": False,
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nOutput: {out}")


if __name__ == "__main__":
    main()
