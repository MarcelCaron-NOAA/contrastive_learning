#!/usr/bin/env python3
"""Evaluate Wang-style controlled transforms for V2, with V1 as a baseline."""
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
from src.eval.wang_transforms import DISPLACEMENT_BIASES, INTENSITY_BIASES, AREA_SIZE_BIASES, transform
from src.eval.verification_metrics import continuous_and_wang_metrics


def defaults():
    data = Path(os.environ["DATA_DIR"])
    return {
        "archive": data / "derived/evaluation/wang_transform_v1",
        "v1": data / "derived/experiments/groupnorm_pilot_v1/checkpoint_best.pt",
        "v2": data / "derived/experiments/hard_ranking_finetune_v1/checkpoint_best.pt",
        "stats": data / "metadata/dataset_v1/precip_normalization_stats_v1.json",
        "out": data / "derived/experiments/hard_ranking_finetune_v1/evaluation/wang_v1_vs_v2",
    }


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def encode_z(model, raw_batch, ptransform, device, batch_size):
    out = []
    for s in range(0, len(raw_batch), batch_size):
        x = torch.stack([ptransform(v) for v in raw_batch[s:s+batch_size]], dim=0)
        x = x.to(device, non_blocking=True)
        _, z = model(x)
        out.append(F.normalize(z.float(), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def ci95(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    m = float(x.mean())
    if len(x) == 1:
        return m, m, m, 1
    se = float(x.std(ddof=1)/np.sqrt(len(x)))
    return m, m-1.96*se, m+1.96*se, len(x)


def ckpt_step(ckpt):
    return int(ckpt.get("fine_tune_step", ckpt.get("step", -1)))


def main():
    d = defaults()
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", default=str(d["archive"]))
    ap.add_argument("--baseline-checkpoint", default=str(d["v1"]))
    ap.add_argument("--checkpoint", default=str(d["v2"]))
    ap.add_argument("--stats-json", default=str(d["stats"]))
    ap.add_argument("--output-dir", default=str(d["out"]))
    ap.add_argument("--threshold-mm", type=float, default=2.54)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    archive = Path(args.archive_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    originals = np.load(archive/"original_patches.npz")["qpf6_mm"]
    cases = pd.read_csv(archive/"cases.csv")
    if len(originals) != len(cases):
        raise RuntimeError("Archive/case length mismatch")

    device = torch.device("cuda")
    v1, c1 = load_model_checkpoint(args.baseline_checkpoint, device=device); v1.eval()
    v2, c2 = load_model_checkpoint(args.checkpoint, device=device); v2.eval()
    pt = PrecipTransform.from_stats_json("log1p_zscore", args.stats_json)

    z1_0 = encode_z(v1, originals, pt, device, args.batch_size)
    z2_0 = encode_z(v2, originals, pt, device, args.batch_size)

    families = [
        ("displacement", DISPLACEMENT_BIASES),
        ("intensity", INTENSITY_BIASES),
        ("area_size", AREA_SIZE_BIASES),
    ]
    records = []
    for family, biases in families:
        print(f"\n{family}: {len(biases)} biases x {len(originals)} cases")
        for j, bias in enumerate(biases):
            transformed = np.stack([transform(x, family, float(bias)) for x in originals]).astype(np.float32)
            z1 = encode_z(v1, transformed, pt, device, args.batch_size)
            z2 = encode_z(v2, transformed, pt, device, args.batch_size)
            s1 = np.sum(z1_0*z1, axis=1)
            s2 = np.sum(z2_0*z2, axis=1)
            for i, (orig, pert) in enumerate(zip(originals, transformed)):
                m = continuous_and_wang_metrics(pert, orig, args.threshold_mm)
                records.append({
                    "case_index": i, "family": family, "bias": float(bias),
                    "learned_similarity_z_v1": float(s1[i]),
                    "learned_similarity_z_v2": float(s2[i]),
                    "learned_similarity_z_delta_v2_minus_v1": float(s2[i]-s1[i]),
                    **m,
                })
            print(f"  {j+1:2d}/{len(biases)} bias={float(bias):g}")

    per = pd.DataFrame(records)
    per.to_csv(outdir/"per_case_metrics.csv.gz", index=False)
    metric_cols = ["learned_similarity_z_v1", "learned_similarity_z_v2", "ts", "pod", "normalized_far", "fss5", "fss10", "rmse", "mae", "corr"]
    rows = []
    for (family,bias), g in per.groupby(["family","bias"], sort=False):
        r = {"family":family, "bias":bias}
        for col in metric_cols:
            m,lo,hi,n = ci95(g[col].to_numpy())
            r[f"{col}_mean"] = m; r[f"{col}_ci95_lo"] = lo; r[f"{col}_ci95_hi"] = hi; r[f"{col}_n"] = n
        rows.append(r)
    pd.DataFrame(rows).to_csv(outdir/"summary_by_bias.csv", index=False)

    meta = {
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "baseline_sha256": sha256_file(args.baseline_checkpoint),
        "baseline_step": ckpt_step(c1),
        "v2_checkpoint": str(args.checkpoint),
        "v2_sha256": sha256_file(args.checkpoint),
        "v2_step": ckpt_step(c2),
        "threshold_mm": args.threshold_mm,
        "cases": len(originals),
        "sealed_test_used": False,
    }
    (outdir/"run_metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(f"\nOutput: {outdir}")


if __name__ == "__main__":
    main()

