#!/usr/bin/env python3
"""Eight-family controlled morphology evaluation for V3 and a synthetic model.

The exact same held-out HRRR patches and exact same transformed fields are sent
through both frozen models.  Conventional metrics are computed against the
unperturbed field.  The eight transformation families are the three published
Wang-style families plus the five signed morphology families already used in
this repository.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data import PrecipTransform
from src.eval.verification_metrics import contingency, fss
from src.eval.wang_transforms import DISPLACEMENT_BIASES, INTENSITY_BIASES, AREA_SIZE_BIASES, transform as wang_transform
from src.eval.morphology_signed_transforms import SIGNED_LEVELS, transform_signed
from src.models.checkpoint_loader import load_model_checkpoint


FAMILIES = (
    "displacement",
    "intensity",
    "area_size",
    "smoothing_sharpening",
    "object_dispersion_cells",
    "principal_axis_log2_scale",
    "rotation_deg",
    "tile_fragmentation_cells",
)


def defaults():
    data = Path(os.environ["DATA_DIR"])
    return {
        "archive": data / "derived/evaluation/wang_transform_v1",
        "v3": data / "derived/experiments/hard_ranking_v3_full_v1/checkpoint_best.pt",
        "synthetic": data / "derived/experiments/synthetic_perturb_all8_v1/checkpoint_best.pt",
        "stats": data / "metadata/dataset_v1/precip_normalization_stats_v1.json",
        "out": data / "derived/experiments/synthetic_perturb_all8_v1/evaluation/morphology_vs_v3",
    }


def levels_for(family):
    if family == "displacement":
        return np.asarray(DISPLACEMENT_BIASES, dtype=float)
    if family == "intensity":
        return np.asarray(INTENSITY_BIASES, dtype=float)
    if family == "area_size":
        return np.asarray(AREA_SIZE_BIASES, dtype=float)
    return np.asarray(SIGNED_LEVELS[family], dtype=float)


def transform_field(field, family, bias, case_seed):
    if family in ("displacement", "intensity", "area_size"):
        return wang_transform(field, family, float(bias))
    return transform_signed(field, family, float(bias), case_seed=int(case_seed))


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
    out = []
    for s in range(0, len(raw), batch_size):
        x = torch.stack([ptransform(v) for v in raw[s:s+batch_size]], dim=0).to(device, non_blocking=True)
        _, z = model(x)
        out.append(F.normalize(z.float(), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


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
    return float(ets) if np.isfinite(ets) else np.nan, float(pod) if np.isfinite(pod) else np.nan, float(sr) if np.isfinite(sr) else np.nan


def normalized_severity(family, bias):
    b = float(bias)
    if family == "displacement":
        u = b / 10.0
    elif family in ("intensity", "area_size"):
        u = math.log2(b)
    elif family == "smoothing_sharpening":
        u = b / 6.0
    elif family == "object_dispersion_cells":
        u = b / 16.0
    elif family == "principal_axis_log2_scale":
        u = b
    elif family == "rotation_deg":
        u = b / 90.0
    elif family == "tile_fragmentation_cells":
        u = b / 12.0
    else:
        raise KeyError(family)
    return float(np.clip(u, -1.0, 1.0))


def load_model(path, device):
    model, ckpt = load_model_checkpoint(path, device=device)
    model.eval()
    return model, ckpt


def main():
    d = defaults()
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", default=str(d["archive"]))
    ap.add_argument("--v3-checkpoint", default=str(d["v3"]))
    ap.add_argument("--synthetic-checkpoint", default=str(d["synthetic"]))
    ap.add_argument("--stats-json", default=str(d["stats"]))
    ap.add_argument("--output-dir", default=str(d["out"]))
    ap.add_argument("--case-count", type=int, default=500)
    ap.add_argument("--case-seed", type=int, default=20261007)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threshold-mm", type=float, default=2.54,
                    help="Threshold for ETS/POD/Success Ratio/FSS; default 0.10 inch.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    arc = Path(args.archive_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
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
    print(f"Cases: {n}; seed: {args.case_seed}; threshold: {args.threshold_mm:g} mm")

    v3, ck3 = load_model(args.v3_checkpoint, device)
    syn, cks = load_model(args.synthetic_checkpoint, device)
    print("Encoding originals...")
    z0_v3 = encode(v3, raw, ptransform, device, args.batch_size)
    z0_syn = encode(syn, raw, ptransform, device, args.batch_size)

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

            z_v3 = encode(v3, transformed, ptransform, device, args.batch_size)
            z_syn = encode(syn, transformed, ptransform, device, args.batch_size)
            sim_v3 = np.sum(z0_v3 * z_v3, axis=1)
            sim_syn = np.sum(z0_syn * z_syn, axis=1)
            sev = normalized_severity(family, bias)
            target = 1.0 - sev * sev

            for i, (orig, pert) in enumerate(zip(raw, transformed)):
                ets, pod, sr = categorical(pert, orig, args.threshold_mm)
                records.append({
                    "archive_case_index": int(idx[i]),
                    "case_position": int(i),
                    "family": family,
                    "bias": float(bias),
                    "normalized_severity": float(sev),
                    "synthetic_target": float(target),
                    "learned_v3": float(sim_v3[i]),
                    "learned_synthetic": float(sim_syn[i]),
                    "ets": ets,
                    "pod": pod,
                    "success_ratio": sr,
                    "fss5": fss(pert, orig, args.threshold_mm, 5),
                    "fss10": fss(pert, orig, args.threshold_mm, 10),
                })

            done += 1
            print(f"  {done:3d}/{total_levels:3d} bias={float(bias):g}", flush=True)
            del transformed, z_v3, z_syn

    per = pd.DataFrame(records)
    per.to_csv(out / "morphology_per_case.csv.gz", index=False)

    metrics = ("learned_v3", "learned_synthetic", "ets", "pod", "success_ratio", "fss5", "fss10")
    rows = []
    for (family, bias), g in per.groupby(["family", "bias"], sort=False):
        row = {
            "family": family,
            "bias": float(bias),
            "normalized_severity": float(g.normalized_severity.iloc[0]),
            "synthetic_target": float(g.synthetic_target.iloc[0]),
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
    summary.to_csv(out / "morphology_summary.csv", index=False)

    meta = {
        "archive": str(arc),
        "case_count": n,
        "case_seed": int(args.case_seed),
        "threshold_mm": float(args.threshold_mm),
        "v3_checkpoint": str(args.v3_checkpoint),
        "v3_sha256": sha256_file(args.v3_checkpoint),
        "v3_step": checkpoint_step(ck3),
        "synthetic_checkpoint": str(args.synthetic_checkpoint),
        "synthetic_sha256": sha256_file(args.synthetic_checkpoint),
        "synthetic_step": checkpoint_step(cks),
        "families": {f: levels_for(f).tolist() for f in FAMILIES},
        "synthetic_target": "1 - normalized_severity^2, matching synthetic_perturb_all8_v1 training",
        "learned_score_definition": "raw cosine similarity of L2-normalized projection embeddings",
        "success_ratio_definition": "hits / (hits + false alarms) = 1 - FAR",
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    del v3, syn
    gc.collect(); torch.cuda.empty_cache()
    print(f"\nOutput: {out}")


if __name__ == "__main__":
    main()

