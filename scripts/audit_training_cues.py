#!/usr/bin/env python3
"""
Audit which physical cues actually distinguish the natural +/-1-h positive
from the climo45 negatives used by the HRRR contrastive training objective.

This script deliberately reproduces the training sampler (split=train,
climo45 negatives, +/-1-h positive, same random crop, K explicit negatives)
without using the neural network.  It asks a simple question for each physical
comparison metric: how often is P more A-like than a sampled N?

Outputs are deterministic for a fixed sampler seed/epoch and include the exact
recipes that were audited.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.data import (
    ContrastiveIndex,
    NegativeRules,
    ContrastiveBatchSampler,
    HRRRArchive,
)

try:
    from scipy.ndimage import uniform_filter
except Exception as e:  # pragma: no cover - runtime environment check
    raise RuntimeError(
        "scipy is required for the vectorized FSS cue audit."
    ) from e


DEFAULT_GRID_SHAPE = (1059, 1799)


def defaults():
    data = Path(os.environ["DATA_DIR"])
    return {
        "anchors": data / "metadata/dataset_v1/eligible_anchors_v1.csv.gz",
        "negative_pool": data / "metadata/dataset_v1/negative_pool_v1.csv.gz",
        "manifest": data / "forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz",
        "outdir": data / "derived/evaluation/representation_characterization_v1/training_cue_audit",
    }


class RawRecipeDataset(Dataset):
    """Materialize sampler recipes as raw mm/6h fields, no ML preprocessing."""
    def __init__(self, manifest_path: str | Path, negative_count: int):
        self.archive = HRRRArchive(manifest_path)
        self.negative_count = int(negative_count)

    def __len__(self):
        return self.archive.n_fields

    def __getitem__(self, recipe):
        a = np.asarray(
            self.archive.read_patch(recipe.anchor_id, recipe.y0, recipe.x0, recipe.patch_size),
            dtype=np.float32,
        )
        p = np.asarray(
            self.archive.read_patch(recipe.positive_id, recipe.y0, recipe.x0, recipe.patch_size),
            dtype=np.float32,
        )
        ns = []
        mask = []
        for nid in recipe.negative_ids:
            n = np.asarray(
                self.archive.read_patch(nid, recipe.y0, recipe.x0, recipe.patch_size),
                dtype=np.float32,
            )
            same = np.array_equal(n, a, equal_nan=True) or np.array_equal(n, p, equal_nan=True)
            ns.append(n)
            mask.append(not same)
        if len(ns) != self.negative_count:
            raise RuntimeError("Recipe negative count mismatch")
        return {
            "anchor": torch.from_numpy(a),
            "positive": torch.from_numpy(p),
            "negatives": torch.from_numpy(np.stack(ns)),
            "negative_mask": torch.tensor(mask, dtype=torch.bool),
            "metadata": asdict(recipe),
        }


def raw_collate(items):
    return {
        "anchor": torch.stack([x["anchor"] for x in items]),
        "positive": torch.stack([x["positive"] for x in items]),
        "negatives": torch.stack([x["negatives"] for x in items]),
        "negative_mask": torch.stack([x["negative_mask"] for x in items]),
        "metadata": [x["metadata"] for x in items],
    }


def finite_nonnegative(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.where(np.isfinite(x), x, 0.0)
    return np.maximum(x, 0.0)


def field_stats_batch(x, wet_thresholds=(0.1, 2.54)):
    """x [B,H,W] -> dict of [B] arrays."""
    x = finite_nonnegative(x)
    out = {
        "mean_mm": x.mean(axis=(-2, -1)),
        "max_mm": x.max(axis=(-2, -1)),
    }
    for t in wet_thresholds:
        key = str(t).replace(".", "p")
        wet = x >= float(t)
        out[f"wetfrac_{key}"] = wet.mean(axis=(-2, -1))

    # P90 among >=0.1-mm wet pixels. Loop over B only; B is small.
    p90 = np.full(x.shape[0], np.nan, dtype=np.float64)
    for i in range(x.shape[0]):
        vals = x[i][x[i] >= 0.1]
        if vals.size:
            p90[i] = np.percentile(vals, 90)
    out["p90wet0p1_mm"] = p90

    # Intensity-weighted centroid using >=0.1-mm pixels.
    H, W = x.shape[-2:]
    yy = np.arange(H, dtype=np.float64)[:, None]
    xx = np.arange(W, dtype=np.float64)[None, :]
    w = np.where(x >= 0.1, x, 0.0).astype(np.float64)
    sw = w.sum(axis=(-2, -1))
    cy = np.full(x.shape[0], np.nan)
    cx = np.full(x.shape[0], np.nan)
    ok = sw > 0
    if np.any(ok):
        cy[ok] = (w[ok] * yy).sum(axis=(-2, -1)) / sw[ok]
        cx[ok] = (w[ok] * xx).sum(axis=(-2, -1)) / sw[ok]
    out["centroid_y"] = cy
    out["centroid_x"] = cx
    return out


def rmse_batch(a, b):
    d = np.asarray(a, np.float32) - np.asarray(b, np.float32)
    return np.sqrt(np.mean(d*d, axis=(-2, -1), dtype=np.float64))


def corr_batch(a, b):
    a = np.asarray(a, np.float64).reshape(len(a), -1)
    b = np.asarray(b, np.float64).reshape(len(b), -1)
    am = a.mean(1, keepdims=True); bm = b.mean(1, keepdims=True)
    ac = a-am; bc = b-bm
    num = np.sum(ac*bc, axis=1)
    den = np.sqrt(np.sum(ac*ac, axis=1) * np.sum(bc*bc, axis=1))
    out = np.full(len(a), np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def ts_batch(a, b, threshold):
    aa = np.asarray(a) >= float(threshold)
    bb = np.asarray(b) >= float(threshold)
    h = np.sum(aa & bb, axis=(-2, -1))
    m = np.sum((~aa) & bb, axis=(-2, -1))
    fa = np.sum(aa & (~bb), axis=(-2, -1))
    den = h + m + fa
    out = np.full(len(a), np.nan)
    ok = den > 0
    out[ok] = h[ok] / den[ok]
    return out


def fss_batch(a, b, threshold, width):
    aa = (np.asarray(a) >= float(threshold)).astype(np.float64)
    bb = (np.asarray(b) >= float(threshold)).astype(np.float64)
    size = (1, int(width), int(width))
    fa = uniform_filter(aa, size=size, mode="constant", cval=0.0)
    fb = uniform_filter(bb, size=size, mode="constant", cval=0.0)
    mse = np.mean((fa-fb)**2, axis=(-2, -1))
    ref = np.mean(fa**2 + fb**2, axis=(-2, -1))
    out = np.full(len(a), np.nan)
    ok = ref > 0
    out[ok] = 1.0 - mse[ok]/ref[ok]
    return out


def pair_metrics(anchor, other):
    """Return raw comparison cues for equally-shaped [B,H,W] arrays."""
    sa = field_stats_batch(anchor)
    so = field_stats_batch(other)
    out = {
        "mean_mm_absdiff": np.abs(sa["mean_mm"] - so["mean_mm"]),
        "max_mm_absdiff": np.abs(sa["max_mm"] - so["max_mm"]),
        "p90wet0p1_absdiff": np.abs(sa["p90wet0p1_mm"] - so["p90wet0p1_mm"]),
        "wetfrac0p1_absdiff": np.abs(sa["wetfrac_0p1"] - so["wetfrac_0p1"]),
        "wetfrac2p54_absdiff": np.abs(sa["wetfrac_2p54"] - so["wetfrac_2p54"]),
        "rmse": rmse_batch(anchor, other),
        "corr": corr_batch(anchor, other),
        "ts2p54": ts_batch(anchor, other, 2.54),
        "fss0p1_w15": fss_batch(anchor, other, 0.1, 15),
        "fss2p54_w15": fss_batch(anchor, other, 2.54, 15),
    }
    dy = sa["centroid_y"] - so["centroid_y"]
    dx = sa["centroid_x"] - so["centroid_x"]
    out["centroid0p1_distance"] = np.sqrt(dy*dy + dx*dx)
    return out


DIRECTIONS = {
    "mean_mm_absdiff": "lower",
    "max_mm_absdiff": "lower",
    "p90wet0p1_absdiff": "lower",
    "wetfrac0p1_absdiff": "lower",
    "wetfrac2p54_absdiff": "lower",
    "centroid0p1_distance": "lower",
    "rmse": "lower",
    "corr": "higher",
    "ts2p54": "higher",
    "fss0p1_w15": "higher",
    "fss2p54_w15": "higher",
}


def oriented(v, direction):
    v = np.asarray(v, dtype=float)
    return v if direction == "higher" else -v


def summarize(pair_df, anchor_df):
    rows = []
    for metric, direction in DIRECTIONS.items():
        sub = pair_df[["pair_type", metric]].dropna()
        pvals = anchor_df[f"p_{metric}"].to_numpy(float)
        top1 = anchor_df[f"p_top1_{metric}"].to_numpy(float)
        margins = anchor_df[f"hard_margin_{metric}"].to_numpy(float)

        # Pairwise P-vs-N rows were precomputed per negative.
        wins = anchor_df[f"pairwise_win_fraction_{metric}"].to_numpy(float)
        valid_w = np.isfinite(wins)
        rows.append({
            "cue": metric,
            "better_direction": direction,
            "positive_median": float(np.nanmedian(pvals)),
            "negative_median": float(np.nanmedian(pair_df.loc[pair_df.pair_type=="N", metric])),
            "p_better_than_sampled_n_pct": float(np.nanmean(wins)*100.0),
            "anchor_top1_pct": float(np.nanmean(top1)*100.0),
            "median_hard_margin_oriented": float(np.nanmedian(margins)),
            "anchors_with_valid_comparison": int(valid_w.sum()),
            "negative_pairs_valid": int(np.isfinite(pair_df.loc[pair_df.pair_type=="N", metric]).sum()),
        })
    return pd.DataFrame(rows)


def plot_summary(summary, outdir):
    import matplotlib.pyplot as plt

    s = summary.sort_values("p_better_than_sampled_n_pct", ascending=True)
    plt.figure(figsize=(10, 6.5))
    plt.barh(s["cue"], s["p_better_than_sampled_n_pct"])
    plt.axvline(50, linestyle="--", linewidth=1)
    plt.xlabel("P is more A-like than sampled N (%)")
    plt.title("Training-cue audit: pairwise discriminating power")
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(outdir/"training_cue_pairwise_win_rate.png", dpi=190)
    plt.close()

    s = summary.sort_values("anchor_top1_pct", ascending=True)
    plt.figure(figsize=(10, 6.5))
    plt.barh(s["cue"], s["anchor_top1_pct"])
    plt.xlabel("P beats all sampled N for this cue (%)")
    plt.title("Training-cue audit: cue-only top-1 rate")
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(outdir/"training_cue_top1_rate.png", dpi=190)
    plt.close()


def main():
    d = defaults()
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default=str(d["anchors"]))
    ap.add_argument("--negative-pool", default=str(d["negative_pool"]))
    ap.add_argument("--manifest", default=str(d["manifest"]))
    ap.add_argument("--output-dir", default=str(d["outdir"]))
    ap.add_argument("--sample-count", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--negative-count", type=int, default=8)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260930,
                    help="Training sampler seed used by the GroupNorm run.")
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--negative-min-days", type=float, default=90.0)
    ap.add_argument("--climo-window-days", type=int, default=45)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    index = ContrastiveIndex(args.anchors, args.negative_pool, args.manifest)
    rules = NegativeRules(
        mode="climo45",
        min_days=args.negative_min_days,
        climo_window_days=args.climo_window_days,
    )
    sampler = ContrastiveBatchSampler(
        index,
        split="train",
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        negative_count=args.negative_count,
        negative_rules=rules,
        seed=args.seed,
        grid_shape=DEFAULT_GRID_SHAPE,
    )
    sampler.set_epoch(args.epoch)
    dataset = RawRecipeDataset(args.manifest, args.negative_count)
    loader_kw = dict(
        dataset=dataset,
        batch_sampler=sampler,
        collate_fn=raw_collate,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    if args.num_workers > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(**loader_kw)

    pair_rows = []
    anchor_rows = []
    recipes = []
    seen = 0
    t0 = time.perf_counter()

    for ib, batch in enumerate(loader):
        a = batch["anchor"].numpy()
        p = batch["positive"].numpy()
        n = batch["negatives"].numpy()  # [B,K,H,W]
        nmask = batch["negative_mask"].numpy().astype(bool)
        meta = batch["metadata"]

        if seen + len(a) > args.sample_count:
            keep = args.sample_count - seen
            a, p, n, nmask = a[:keep], p[:keep], n[:keep], nmask[:keep]
            meta = meta[:keep]
        B, K = n.shape[:2]
        if B == 0:
            break

        pm = pair_metrics(a, p)
        aflat = np.repeat(a, K, axis=0)
        nflat = n.reshape(B*K, *n.shape[2:])
        nm_flat = pair_metrics(aflat, nflat)

        for i in range(B):
            recipes.append(meta[i])
            ar = {
                "audit_index": seen+i,
                "anchor_id": meta[i]["anchor_id"],
                "positive_id": meta[i]["positive_id"],
                "positive_direction": meta[i]["positive_direction"],
                "forecast_hour": meta[i]["anchor_forecast_hour"],
                "valid_time": meta[i]["anchor_valid_time"],
                "y0": meta[i]["y0"],
                "x0": meta[i]["x0"],
            }
            for metric, direction in DIRECTIONS.items():
                pv = float(pm[metric][i]) if np.isfinite(pm[metric][i]) else np.nan
                nv = np.asarray(nm_flat[metric][i*K:(i+1)*K], dtype=float)
                valid = nmask[i] & np.isfinite(nv) & np.isfinite(pv)
                q_p = oriented(np.array([pv]), direction)[0]
                q_n = oriented(nv, direction)
                if np.any(valid):
                    wins = q_p > q_n[valid]
                    ties = np.isclose(q_p, q_n[valid], rtol=0, atol=1e-12)
                    win_fraction = float(np.mean(wins + 0.5*ties))
                    top1 = float(q_p > np.max(q_n[valid]))
                    hard_margin = float(q_p - np.max(q_n[valid]))
                else:
                    win_fraction = top1 = hard_margin = np.nan
                ar[f"p_{metric}"] = pv
                ar[f"pairwise_win_fraction_{metric}"] = win_fraction
                ar[f"p_top1_{metric}"] = top1
                ar[f"hard_margin_{metric}"] = hard_margin
            anchor_rows.append(ar)

            # one P row plus K N rows for inspectable raw metrics
            prow = {
                "audit_index": seen+i,
                "anchor_id": meta[i]["anchor_id"],
                "pair_type": "P",
                "pair_index": 0,
                "other_id": meta[i]["positive_id"],
                "valid_pair": True,
            }
            for metric in DIRECTIONS:
                prow[metric] = float(pm[metric][i]) if np.isfinite(pm[metric][i]) else np.nan
            pair_rows.append(prow)

            for j in range(K):
                nr = {
                    "audit_index": seen+i,
                    "anchor_id": meta[i]["anchor_id"],
                    "pair_type": "N",
                    "pair_index": j,
                    "other_id": meta[i]["negative_ids"][j],
                    "valid_pair": bool(nmask[i,j]),
                }
                for metric in DIRECTIONS:
                    v = nm_flat[metric][i*K+j]
                    nr[metric] = float(v) if np.isfinite(v) else np.nan
                pair_rows.append(nr)

        seen += B
        if ib == 0 or seen % 400 < B or seen >= args.sample_count:
            print(f"audited {seen:,}/{args.sample_count:,} anchors")
        if seen >= args.sample_count:
            break

    pair_df = pd.DataFrame(pair_rows)
    anchor_df = pd.DataFrame(anchor_rows)
    summary = summarize(pair_df[pair_df.valid_pair], anchor_df)

    pair_df.to_csv(outdir/"training_cue_pairs.csv.gz", index=False)
    anchor_df.to_csv(outdir/"training_cue_anchor_summary.csv.gz", index=False)
    summary.to_csv(outdir/"training_cue_summary.csv", index=False)
    with (outdir/"training_cue_recipes.jsonl").open("w") as f:
        for r in recipes:
            f.write(json.dumps(r) + "\n")

    metadata = {
        "sample_count": int(seen),
        "negative_count": args.negative_count,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "sampler_seed": args.seed,
        "sampler_epoch": args.epoch,
        "negative_mode": "climo45",
        "negative_min_days": args.negative_min_days,
        "climo_window_days": args.climo_window_days,
        "split": "train",
        "elapsed_seconds": time.perf_counter()-t0,
        "interpretation": (
            "For each cue, P_better_than_sampled_N asks whether the natural +/-1-h "
            "positive provided a more reliable training cue than the exact negatives "
            "sampled by the training recipe. This diagnoses supervision availability; "
            "it does not prove which cue the network actually used."
        ),
    }
    (outdir/"run_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    plot_summary(summary, outdir)

    print("\n" + "="*92)
    print("TRAINING-CUE AUDIT COMPLETE")
    print("="*92)
    print(summary.to_string(index=False))
    print(f"\nOutput: {outdir}")


if __name__ == "__main__":
    main()

