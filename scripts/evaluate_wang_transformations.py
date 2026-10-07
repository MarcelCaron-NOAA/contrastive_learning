#!/usr/bin/env python3
"""
Evaluate the frozen learned metric and traditional metrics under Wang-style
controlled transformations.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data import PrecipTransform
from src.models.checkpoint_loader import load_model_checkpoint
from src.eval.wang_transforms import (
    DISPLACEMENT_BIASES, INTENSITY_BIASES, AREA_SIZE_BIASES, transform,
)
from src.eval.verification_metrics import continuous_and_wang_metrics


def defaults():
    data = Path(os.environ["DATA_DIR"])
    return {
        "archive": data / "derived/evaluation/wang_transform_v1",
        "checkpoint": data / "derived/experiments/groupnorm_pilot_v1/checkpoint_best.pt",
        "stats": data / "metadata/dataset_v1/precip_normalization_stats_v1.json",
    }


def encode_z(model, raw_batch, precip_transform, device):
    tensors = torch.stack([precip_transform(x) for x in raw_batch], dim=0)
    tensors = tensors.to(device, non_blocking=True)
    with torch.no_grad():
        _, z = model(tensors)
    return F.normalize(z.float(), dim=-1).cpu().numpy()


def ci95(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(x.mean())
    if len(x) == 1:
        return mean, mean, mean, 1
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    return mean, mean - 1.96*se, mean + 1.96*se, len(x)


def main():
    d = defaults()
    p = argparse.ArgumentParser()
    p.add_argument("--archive-dir", default=str(d["archive"]))
    p.add_argument("--checkpoint", default=str(d["checkpoint"]))
    p.add_argument("--stats-json", default=str(d["stats"]))
    p.add_argument(
        "--threshold-mm", type=float, required=True,
        help=(
            "Threshold for TS/POD/FAR/FSS. Wang's exact threshold should be "
            "taken from their released verification code before claiming an "
            "exact metric replication."
        ),
    )
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for learned-metric evaluation.")

    archive_dir = Path(args.archive_dir)
    outdir = archive_dir / "results"
    outdir.mkdir(parents=True, exist_ok=True)

    originals = np.load(archive_dir / "original_patches.npz")["qpf6_mm"]
    cases = pd.read_csv(archive_dir / "cases.csv")
    if len(cases) != len(originals):
        raise RuntimeError("cases.csv and original_patches.npz length mismatch")

    device = torch.device("cuda")
    model, ckpt = load_model_checkpoint(args.checkpoint, device=device)
    model.eval()
    ptransform = PrecipTransform.from_stats_json(
        "log1p_zscore", args.stats_json
    )

    # Encode originals once.
    z0 = []
    for s in range(0, len(originals), args.batch_size):
        z0.append(encode_z(
            model, originals[s:s+args.batch_size], ptransform, device
        ))
    z0 = np.concatenate(z0, axis=0)

    families = [
        ("displacement", DISPLACEMENT_BIASES),
        ("intensity", INTENSITY_BIASES),
        ("area_size", AREA_SIZE_BIASES),
    ]

    records = []
    for family, biases in families:
        print(f"\n{family}: {len(biases)} biases x {len(originals)} cases")
        for ib, bias in enumerate(biases):
            transformed = np.stack(
                [transform(x, family, float(bias)) for x in originals],
                axis=0,
            ).astype(np.float32)

            zt = []
            for s in range(0, len(transformed), args.batch_size):
                zt.append(encode_z(
                    model, transformed[s:s+args.batch_size],
                    ptransform, device
                ))
            zt = np.concatenate(zt, axis=0)
            learned = np.sum(z0 * zt, axis=1)

            for i, (orig, pert) in enumerate(zip(originals, transformed)):
                m = continuous_and_wang_metrics(
                    pert, orig, args.threshold_mm
                )
                records.append({
                    "case_index": i,
                    "family": family,
                    "bias": float(bias),
                    "learned_similarity_z": float(learned[i]),
                    **m,
                })
            print(f"  {ib+1:2d}/{len(biases)} bias={float(bias):g}")

    per_case = pd.DataFrame(records)
    per_case.to_csv(outdir / "per_case_metrics.csv.gz", index=False)

    metric_cols = [
        "learned_similarity_z", "ts", "pod", "normalized_far",
        "fss5", "fss10", "rmse", "mae", "corr",
    ]
    summary_rows = []
    for (family, bias), g in per_case.groupby(["family","bias"], sort=False):
        row = {"family": family, "bias": bias}
        for col in metric_cols:
            mean, lo, hi, n = ci95(g[col].to_numpy())
            row[f"{col}_mean"] = mean
            row[f"{col}_ci95_lo"] = lo
            row[f"{col}_ci95_hi"] = hi
            row[f"{col}_n"] = n
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "summary_by_bias.csv", index=False)

    meta = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(ckpt.get("step", -1)),
        "threshold_mm": args.threshold_mm,
        "cases": len(originals),
        "note": (
            "SAL is intentionally not implemented here until the exact Wang "
            "object-identification settings are confirmed from released code."
        ),
    }
    (outdir / "run_metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(f"\nWrote {outdir}")


if __name__ == "__main__":
    main()

