#!/usr/bin/env python3
"""Evaluate SYN-3 v2, SYN-8 v2, and NFV on eight controlled perturbation families.

The same held-out HRRR patches and exact same transformed fields are used for all
three learned models and all traditional metrics.  The eight families are:
  displacement, intensity, area size, smoothing (one-sided), object
  concentration/dispersion, principal-axis compression/elongation, rotation,
  and tile fragmentation.

The gray reference target used later for plotting is the revised Wang-adapted
training target expressed as similarity:
    s_target = 1 - 0.9 * u^2
where u is normalized perturbation severity in [0,1].
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data import PrecipTransform
from src.eval.verification_metrics import contingency, fss
from src.eval.wang_transforms import (
    DISPLACEMENT_BIASES,
    INTENSITY_BIASES,
    AREA_SIZE_BIASES,
    transform as wang_transform,
)
from src.eval.morphology_signed_transforms import SIGNED_LEVELS, transform_signed
from src.models.checkpoint_loader import load_model_checkpoint


FAMILIES = (
    "displacement",
    "intensity",
    "area_size",
    "smoothing",
    "object_dispersion_cells",
    "principal_axis_log2_scale",
    "rotation_deg",
    "tile_fragmentation_cells",
)

SMOOTHING_LEVELS = np.round(np.arange(0.0, 6.0 + 0.0001, 0.6), 1)


def defaults():
    data = Path(os.environ["DATA_DIR"])
    return {
        "archive": data / "derived/evaluation/wang_transform_v1",
        "syn3": data / "derived/experiments/synthetic_perturb_syn3_v2/checkpoint_best.pt",
        "syn8": data / "derived/experiments/synthetic_perturb_syn8_v2/checkpoint_best.pt",
        "nfv": data / "derived/experiments/hard_ranking_v3_full_v1/checkpoint_best.pt",
        "stats": data / "metadata/dataset_v1/precip_normalization_stats_v1.json",
        "out": data / "derived/evaluation/final_syn_nfv_morphology_v1",
    }


def levels_for(family: str) -> np.ndarray:
    if family == "displacement":
        return np.asarray(DISPLACEMENT_BIASES, dtype=float)
    if family == "intensity":
        return np.asarray(INTENSITY_BIASES, dtype=float)
    if family == "area_size":
        return np.asarray(AREA_SIZE_BIASES, dtype=float)
    if family == "smoothing":
        return np.asarray(SMOOTHING_LEVELS, dtype=float)
    return np.asarray(SIGNED_LEVELS[family], dtype=float)


def transform_field(field, family, bias, case_seed):
    if family in ("displacement", "intensity", "area_size"):
        return wang_transform(field, family, float(bias))
    if family == "smoothing":
        # Positive half only.  At zero this returns the identity field.
        return transform_signed(field, "smoothing_sharpening", float(bias), case_seed=int(case_seed))
    return transform_signed(field, family, float(bias), case_seed=int(case_seed))


def normalized_severity(family: str, bias: float) -> float:
    """Map the physical bias to the revised Wang-adapted severity coordinate u.

    For intensity/area, the two sides deliberately use the discrete Wang ladders:
      0.95..0.50 corresponds to ranks 1..10
      1.10..2.00 corresponds to ranks 1..10
    """
    b = float(bias)
    if family == "displacement":
        u = abs(b) / 10.0
    elif family in ("intensity", "area_size"):
        if np.isclose(b, 1.0):
            u = 0.0
        elif b < 1.0:
            u = (1.0 - b) / 0.5
        else:
            u = (b - 1.0) / 1.0
    elif family == "smoothing":
        u = max(0.0, b) / 6.0
    elif family == "object_dispersion_cells":
        u = abs(b) / 16.0
    elif family == "principal_axis_log2_scale":
        u = abs(b) / 1.0
    elif family == "rotation_deg":
        u = abs(b) / 90.0
    elif family == "tile_fragmentation_cells":
        u = abs(b) / 12.0
    else:
        raise KeyError(family)
    return float(np.clip(u, 0.0, 1.0))


def target_similarity(family: str, bias: float) -> float:
    u = normalized_severity(family, bias)
    return float(1.0 - 0.9 * u * u)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_step(ckpt):
    for key in ("fine_tune_step", "step", "global_step"):
        if key in ckpt:
            try:
                return int(ckpt[key])
            except Exception:
                pass
    return -1


@torch.inference_mode()
def encode(model, raw, ptransform, device, batch_size):
    pieces = []
    for s in range(0, len(raw), batch_size):
        e = min(len(raw), s + batch_size)
        x = torch.stack([ptransform(v) for v in raw[s:e]], dim=0).to(device, non_blocking=True)
        _, z = model(x)
        pieces.append(F.normalize(z.float(), dim=-1).cpu().numpy())
    return np.concatenate(pieces, axis=0)


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


def categorical(f, o, threshold):
    h, m, fa, cn = contingency(f, o, threshold)
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


def main():
    d = defaults()
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", default=str(d["archive"]))
    ap.add_argument("--syn3-checkpoint", default=str(d["syn3"]))
    ap.add_argument("--syn8-checkpoint", default=str(d["syn8"]))
    ap.add_argument("--nfv-checkpoint", default=str(d["nfv"]))
    ap.add_argument("--stats-json", default=str(d["stats"]))
    ap.add_argument("--output-dir", default=str(d["out"]))
    ap.add_argument("--case-count", type=int, default=500)
    ap.add_argument("--case-seed", type=int, default=20261007)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threshold-mm", type=float, default=0.1,
                    help="Threshold for ETS/POD/SR/FSS. Requested default: 0.1 mm.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    arc = Path(args.archive_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    required = [
        arc / "original_patches.npz", arc / "cases.csv", Path(args.stats_json),
        Path(args.syn3_checkpoint), Path(args.syn8_checkpoint), Path(args.nfv_checkpoint),
    ]
    for p in required:
        if not p.is_file():
            raise SystemExit(f"Required file missing: {p}")

    raw_all = np.load(arc / "original_patches.npz")["qpf6_mm"]
    cases = pd.read_csv(arc / "cases.csv")
    if len(raw_all) != len(cases):
        raise RuntimeError("Archive/cases length mismatch")

    rng = np.random.default_rng(args.case_seed)
    n = min(int(args.case_count), len(raw_all))
    idx = np.sort(rng.choice(len(raw_all), size=n, replace=False))
    raw = np.asarray(raw_all[idx], dtype=np.float32)
    cases.iloc[idx].assign(archive_case_index=idx).to_csv(out / "selected_cases.csv", index=False)

    device = torch.device("cuda")
    ptransform = PrecipTransform.from_stats_json("log1p_zscore", args.stats_json)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Cases: {n}; seed: {args.case_seed}; traditional threshold: {args.threshold_mm:g} mm")

    models = {}
    ckpts = {}
    for key, path in (
        ("syn3", args.syn3_checkpoint),
        ("syn8", args.syn8_checkpoint),
        ("nfv", args.nfv_checkpoint),
    ):
        print(f"Loading {key.upper()}: {path}")
        model, ckpt = load_model_checkpoint(path, device=device)
        model.eval()
        models[key] = model
        ckpts[key] = ckpt

    print("Encoding originals...")
    z0 = {key: encode(model, raw, ptransform, device, args.batch_size) for key, model in models.items()}

    records = []
    total_levels = sum(len(levels_for(f)) for f in FAMILIES)
    done = 0
    for family in FAMILIES:
        levels = levels_for(family)
        print(f"\n{family}: {len(levels)} levels x {n} cases", flush=True)
        for bias in levels:
            transformed = np.stack([
                transform_field(raw[i], family, bias, args.case_seed + int(idx[i]))
                for i in range(n)
            ]).astype(np.float32)

            sims = {}
            for key, model in models.items():
                z = encode(model, transformed, ptransform, device, args.batch_size)
                sims[key] = np.sum(z0[key] * z, axis=1)
                del z

            u = normalized_severity(family, float(bias))
            target = target_similarity(family, float(bias))
            for i, (orig, pert) in enumerate(zip(raw, transformed)):
                ets, pod, sr = categorical(pert, orig, args.threshold_mm)
                records.append({
                    "archive_case_index": int(idx[i]),
                    "case_position": int(i),
                    "family": family,
                    "bias": float(bias),
                    "normalized_severity": float(u),
                    "wang_target_similarity": float(target),
                    "syn3": float(sims["syn3"][i]),
                    "syn8": float(sims["syn8"][i]),
                    "nfv": float(sims["nfv"][i]),
                    "ets": ets,
                    "pod": pod,
                    "success_ratio": sr,
                    "fss5": fss(pert, orig, args.threshold_mm, 5),
                    "fss10": fss(pert, orig, args.threshold_mm, 10),
                })

            done += 1
            print(f"  {done:3d}/{total_levels:3d} bias={float(bias):g}", flush=True)
            del transformed, sims

    per = pd.DataFrame(records)
    per_path = out / "morphology_per_case.csv.gz"
    per.to_csv(per_path, index=False)

    metrics = ("syn3", "syn8", "nfv", "ets", "pod", "success_ratio", "fss5", "fss10")
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
    summary_path = out / "morphology_summary.csv"
    summary.to_csv(summary_path, index=False)

    meta = {
        "archive": str(arc),
        "case_count": n,
        "case_seed": int(args.case_seed),
        "traditional_threshold_mm": float(args.threshold_mm),
        "families": {f: levels_for(f).tolist() for f in FAMILIES},
        "smoothing_note": "one-sided Gaussian smoothing only; sharpening is intentionally excluded",
        "target_similarity": "1 - 0.9*u^2, where u is normalized perturbation severity",
        "learned_score_definition": "raw cosine similarity of L2-normalized projection z embeddings",
        "success_ratio_definition": "hits/(hits+false alarms) = 1-FAR",
        "syn3": {"checkpoint": str(args.syn3_checkpoint), "sha256": sha256_file(args.syn3_checkpoint), "step": checkpoint_step(ckpts["syn3"])},
        "syn8": {"checkpoint": str(args.syn8_checkpoint), "sha256": sha256_file(args.syn8_checkpoint), "step": checkpoint_step(ckpts["syn8"])},
        "nfv": {"checkpoint": str(args.nfv_checkpoint), "sha256": sha256_file(args.nfv_checkpoint), "step": checkpoint_step(ckpts["nfv"])},
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    models.clear(); ckpts.clear(); z0.clear()
    gc.collect(); torch.cuda.empty_cache()
    print("\nFINAL MORPHOLOGY EVALUATION COMPLETE")
    print(f"Per-case: {per_path}")
    print(f"Summary : {summary_path}")


if __name__ == "__main__":
    main()

