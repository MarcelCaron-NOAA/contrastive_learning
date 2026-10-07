#!/usr/bin/env python3
"""
FP32 validation-similarity and failure-tail diagnostics.

Purpose
-------
Re-evaluate the fixed 512 validation recipes using float32 inference (no
autocast), then diagnose whether learned retrieval failures are:
  * concentrated in dry/light precipitation,
  * associated with cross-season calendar_any negatives,
  * contradicted by conventional spatial metrics.

The fixed validation set uses calendar_any negatives. Therefore a hardest N
may come from a very different season than A/P; this script records that
explicitly.

Traditional metrics are computed for A-P and A-hardestN:
  * MAE
  * RMSE
  * spatial Pearson correlation
  * FSS at thresholds 0.1, 1, 5, 10 mm/6h
    and windows 1, 7, 15, 31 native grid points

Outputs are exploratory validation diagnostics, not sealed-test estimates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.data import (
    ContrastiveIndex,
    HRRRArchive,
    PrecipTransform,
    ContrastiveDataset,
    ContrastiveRecipe,
)
from src.models.checkpoint_loader import load_model_checkpoint


FSS_THRESHOLDS = (0.1, 1.0, 5.0, 10.0)
FSS_WINDOWS = (1, 7, 15, 31)
INTENSITY_ORDER = ["dry", "0.1-1", "1-5", "5-10", "10-25", "25-50", ">=50"]


def defaults():
    data = Path(os.environ["DATA_DIR"])
    exp = data / "derived/experiments/groupnorm_pilot_v1"
    return {
        "experiment_dir": exp,
        "anchors": data / "metadata/dataset_v1/eligible_anchors_v1.csv.gz",
        "negative_pool": data / "metadata/dataset_v1/negative_pool_v1.csv.gz",
        "manifest": (
            data
            / "forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz"
        ),
        "stats": data / "metadata/dataset_v1/precip_normalization_stats_v1.json",
    }


def load_recipes(path: Path):
    out = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            d["negative_ids"] = tuple(d["negative_ids"])
            out.append(ContrastiveRecipe(**d))
    if not out:
        raise RuntimeError(f"No recipes found in {path}")
    return out


def season_from_month(month):
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def climo_day(ts):
    t = pd.Timestamp(ts)
    return pd.Timestamp(year=2000, month=t.month, day=t.day).dayofyear


def circular_climo_distance_days(a, b):
    da = climo_day(a)
    db = climo_day(b)
    d = abs(da - db)
    return min(d, 366 - d)


def wet_p90(raw, threshold=0.1):
    x = raw.detach().float().cpu().numpy().ravel()
    x = x[np.isfinite(x)]
    x = np.maximum(x, 0.0)
    w = x[x >= threshold]
    if len(w) == 0:
        return 0.0
    return float(np.quantile(w, 0.90))


def intensity_bin(v):
    if not np.isfinite(v) or v < 0.1:
        return "dry"
    if v < 1:
        return "0.1-1"
    if v < 5:
        return "1-5"
    if v < 10:
        return "5-10"
    if v < 25:
        return "10-25"
    if v < 50:
        return "25-50"
    return ">=50"


class RecipeDataset(Dataset):
    def __init__(self, recipes, manifest, transform, negative_count):
        self.recipes = recipes
        self.archive = HRRRArchive(manifest)
        self.materializer = ContrastiveDataset(
            self.archive,
            negative_count=negative_count,
            transform=transform,
        )

    def __len__(self):
        return len(self.recipes)

    def __getitem__(self, i):
        return self.materializer[self.recipes[i]]


def collate(items):
    return {
        "anchor": torch.stack([x["anchor"] for x in items]),
        "positive": torch.stack([x["positive"] for x in items]),
        "negatives": torch.stack([x["negatives"] for x in items]),
        "negative_mask": torch.stack([x["negative_mask"] for x in items]),
        "metadata": [x["metadata"] for x in items],
    }


def load_model(path, device):
    model, ckpt = load_model_checkpoint(path, device=device)
    model.eval()
    return model, ckpt


def encode(model, a, p, n):
    B, K = n.shape[:2]
    x = torch.cat(
        [a, p, n.reshape(B * K, *n.shape[2:])],
        dim=0,
    )
    h, z = model(x)
    h = F.normalize(h, dim=-1)

    return (
        h[:B],
        h[B:2*B],
        h[2*B:].reshape(B, K, -1),
    ), (
        z[:B],
        z[B:2*B],
        z[2*B:].reshape(B, K, -1),
    )


def invert_log1p_zscore(x, mean, std):
    raw = torch.expm1(x.float() * float(std) + float(mean))
    return torch.clamp(raw, min=0.0)


def basic_metrics(x, y):
    # x,y: [B,1,H,W]
    d = x - y
    mae = d.abs().mean(dim=(1,2,3))
    rmse = torch.sqrt((d*d).mean(dim=(1,2,3)))

    xf = x.flatten(1)
    yf = y.flatten(1)
    xc = xf - xf.mean(dim=1, keepdim=True)
    yc = yf - yf.mean(dim=1, keepdim=True)
    num = (xc * yc).sum(dim=1)
    den = torch.sqrt((xc*xc).sum(dim=1) * (yc*yc).sum(dim=1))
    corr = torch.where(
        den > 0,
        num / den,
        torch.full_like(den, float("nan")),
    )
    return mae, rmse, corr


def fss_batch(x, y, threshold, window):
    # x,y [B,1,H,W], physical mm
    bx = (x >= threshold).float()
    by = (y >= threshold).float()

    if window == 1:
        fx, fy = bx, by
    else:
        pad = window // 2
        fx = F.avg_pool2d(
            bx, kernel_size=window, stride=1, padding=pad,
            count_include_pad=False,
        )
        fy = F.avg_pool2d(
            by, kernel_size=window, stride=1, padding=pad,
            count_include_pad=False,
        )

    mse = ((fx - fy) ** 2).mean(dim=(1,2,3))
    ref = (fx**2 + fy**2).mean(dim=(1,2,3))
    return torch.where(
        ref > 0,
        1.0 - mse / ref,
        torch.ones_like(ref),
    )


def numpy(x):
    return x.detach().float().cpu().numpy()


def fixed_scale_gallery(rows, recipes_by_anchor, manifest, outdir, vmax_mm=25.0):
    archive = HRRRArchive(manifest)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        for rank, row in enumerate(rows, start=1):
            aid = str(row["anchor_id"])
            rec = recipes_by_anchor[aid]
            pid = str(row["positive_id"])
            nid = str(row["z_hardest_negative_id_fp32"])

            a = archive.read_patch(aid, rec.y0, rec.x0, rec.patch_size)
            p = archive.read_patch(pid, rec.y0, rec.x0, rec.patch_size)
            n = archive.read_patch(nid, rec.y0, rec.x0, rec.patch_size)

            # Fixed scale across every gallery image. Use log1p so light QPF is
            # visible while retaining physical tick labels.
            vmax_t = np.log1p(vmax_mm)
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
            ims = []
            for ax, arr in zip(axes, (a, p, n)):
                im = ax.imshow(
                    np.log1p(np.maximum(arr, 0.0)),
                    origin="lower",
                    vmin=0.0,
                    vmax=vmax_t,
                    interpolation="nearest",
                )
                ims.append(im)
                ax.set_xticks([])
                ax.set_yticks([])

            axes[0].set_title(
                f"A | F{int(row['anchor_forecast_hour']):02d}\n"
                f"valid {row['anchor_valid_time']}"
            )
            axes[1].set_title(
                f"P\nz={row['z_ap_fp32']:.3f}"
            )
            axes[2].set_title(
                f"hardest N | {row['hardest_n_season']}\n"
                f"z={row['z_an_max_fp32']:.3f} | margin={row['z_margin_fp32']:+.3f}"
            )

            cbar = fig.colorbar(
                ims[-1], ax=axes.ravel().tolist(), shrink=0.88, pad=0.02
            )
            ticks_mm = [0, 0.1, 1, 5, 10, 25]
            ticks_mm = [v for v in ticks_mm if v <= vmax_mm]
            cbar.set_ticks([np.log1p(v) for v in ticks_mm])
            cbar.set_ticklabels([str(v) for v in ticks_mm])
            cbar.set_label("QPF6 (mm), fixed log1p display")

            fig.suptitle(
                f"FP32 wet-case failure #{rank} | "
                f"A season={row['season']} | N season={row['hardest_n_season']} | "
                f"P90wet={row['anchor_wet_p90_mm']:.2f} mm"
            )
            fig.subplots_adjust(
                left=0.02, right=0.91, bottom=0.05, top=0.82, wspace=0.06
            )
            fig.savefig(
                outdir / f"wet_failure_{rank:02d}.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(fig)
    finally:
        archive.close()


def main():
    d = defaults()
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", default=str(d["experiment_dir"]))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--anchors", default=str(d["anchors"]))
    p.add_argument("--negative-pool", default=str(d["negative_pool"]))
    p.add_argument("--manifest", default=str(d["manifest"]))
    p.add_argument("--stats-json", default=str(d["stats"]))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--wet-threshold-mm", type=float, default=0.1)
    p.add_argument("--failure-gallery-count", type=int, default=6)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for FP32 model inference.")

    device = torch.device("cuda")
    exp = Path(args.experiment_dir)
    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else exp / "checkpoint_best.pt"
    )
    recipe_path = exp / "fixed_validation_recipes.jsonl"

    stats_spec = json.loads(Path(args.stats_json).read_text())
    norm_mean = float(stats_spec["log1p"]["mean"])
    norm_std = float(stats_spec["log1p"]["std"])

    model, ckpt = load_model(ckpt_path, device)
    step = int(ckpt.get("step", -1))
    checkpoint_tag = ckpt_path.stem.replace("checkpoint_", "")

    index = ContrastiveIndex(
        args.anchors, args.negative_pool, args.manifest
    )
    transform = PrecipTransform.from_stats_json(
        "log1p_zscore", args.stats_json
    )
    recipes = load_recipes(recipe_path)
    recipes_by_anchor = {r.anchor_id: r for r in recipes}

    negative_count = len(recipes[0].negative_ids)
    dataset = RecipeDataset(
        recipes, args.manifest, transform, negative_count
    )
    kw = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )
    if args.num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(**kw)

    rows = []

    print("=" * 88)
    print("FP32 FIXED-VALIDATION FAILURE-TAIL DIAGNOSTICS")
    print("=" * 88)
    print(f"Checkpoint      : {ckpt_path}")
    print(f"Checkpoint step : {step}")
    print(f"Recipes         : {len(recipes)}")
    print("Inference       : FP32 (autocast disabled)")
    print()

    with torch.no_grad():
        for ib, batch in enumerate(loader, start=1):
            a = batch["anchor"].to(device, non_blocking=True)
            pp = batch["positive"].to(device, non_blocking=True)
            nn = batch["negatives"].to(device, non_blocking=True)
            mask = batch["negative_mask"].to(device, non_blocking=True)

            # Explicitly no autocast here.
            (ha, hp, hn), (za, zp, zn) = encode(model, a, pp, nn)

            z_ap = torch.sum(za * zp, dim=-1)
            z_an = torch.einsum("bd,bkd->bk", za, zn)
            h_ap = torch.sum(ha * hp, dim=-1)
            h_an = torch.einsum("bd,bkd->bk", ha, hn)

            neg_inf = torch.finfo(z_an.dtype).min
            z_an_masked = z_an.masked_fill(~mask, neg_inf)
            hard_idx = z_an_masked.argmax(dim=1)
            B = a.shape[0]
            bidx = torch.arange(B, device=device)

            z_hard = z_an[bidx, hard_idx]
            h_hard_at_z = h_an[bidx, hard_idx]

            # Recover physical mm from the normalized model inputs.
            a_raw = invert_log1p_zscore(a, norm_mean, norm_std)
            p_raw = invert_log1p_zscore(pp, norm_mean, norm_std)
            n_raw = invert_log1p_zscore(nn, norm_mean, norm_std)
            nh_raw = n_raw[bidx, hard_idx]

            ap_mae, ap_rmse, ap_corr = basic_metrics(a_raw, p_raw)
            an_mae, an_rmse, an_corr = basic_metrics(a_raw, nh_raw)

            fss_results = {}
            for thr in FSS_THRESHOLDS:
                for win in FSS_WINDOWS:
                    key = f"fss_t{str(thr).replace('.', 'p')}_w{win}"
                    fss_results[f"ap_{key}"] = fss_batch(
                        a_raw, p_raw, thr, win
                    )
                    fss_results[f"an_{key}"] = fss_batch(
                        a_raw, nh_raw, thr, win
                    )

            for i, meta in enumerate(batch["metadata"]):
                aid = str(meta["anchor_id"])
                pid = str(meta["positive_id"])
                hard_k = int(hard_idx[i].item())
                nid = str(meta["negative_ids"][hard_k])

                am = index.field_metadata(aid)
                nm = index.field_metadata(nid)

                at = pd.Timestamp(am["valid_time"])
                nt = pd.Timestamp(nm["valid_time"])
                aseason = season_from_month(at.month)
                nseason = season_from_month(nt.month)

                p90 = wet_p90(a_raw[i], args.wet_threshold_mm)

                row = {
                    "checkpoint_step": step,
                    "anchor_id": aid,
                    "positive_id": pid,
                    "z_hardest_negative_id_fp32": nid,
                    "anchor_valid_time": str(at),
                    "hardest_n_valid_time": str(nt),
                    "anchor_forecast_hour": int(am["forecast_hour"]),
                    "valid_hour_utc": int(am["valid_hour_utc"]),
                    "season": aseason,
                    "hardest_n_season": nseason,
                    "same_season_hardest_n": int(aseason == nseason),
                    "hardest_n_time_distance_days": abs(
                        (nt - at).total_seconds()
                    ) / 86400.0,
                    "hardest_n_climo_distance_days": (
                        circular_climo_distance_days(at, nt)
                    ),
                    "crop_y0": int(meta["y0"]),
                    "crop_x0": int(meta["x0"]),
                    "patch_size": int(meta["patch_size"]),
                    "anchor_wet_p90_mm": p90,
                    "intensity_bin": intensity_bin(p90),
                    "z_ap_fp32": float(z_ap[i].item()),
                    "z_an_max_fp32": float(z_hard[i].item()),
                    "z_margin_fp32": float((z_ap[i] - z_hard[i]).item()),
                    "z_top1_fp32": int(z_ap[i] > z_hard[i]),
                    "h_ap_fp32": float(h_ap[i].item()),
                    "h_an_at_z_hard_fp32": float(h_hard_at_z[i].item()),
                    "ap_mae_mm": float(ap_mae[i].item()),
                    "an_mae_mm": float(an_mae[i].item()),
                    "ap_rmse_mm": float(ap_rmse[i].item()),
                    "an_rmse_mm": float(an_rmse[i].item()),
                    "ap_corr": float(ap_corr[i].item()),
                    "an_corr": float(an_corr[i].item()),
                }

                for key, values in fss_results.items():
                    row[key] = float(values[i].item())

                # Simple conventional pair-preference flags.
                row["mae_prefers_p"] = int(row["ap_mae_mm"] < row["an_mae_mm"])
                row["rmse_prefers_p"] = int(row["ap_rmse_mm"] < row["an_rmse_mm"])
                row["corr_prefers_p"] = (
                    int(row["ap_corr"] > row["an_corr"])
                    if np.isfinite(row["ap_corr"]) and np.isfinite(row["an_corr"])
                    else np.nan
                )
                for thr in FSS_THRESHOLDS:
                    for win in FSS_WINDOWS:
                        k = f"fss_t{str(thr).replace('.', 'p')}_w{win}"
                        row[f"{k}_prefers_p"] = int(
                            row[f"ap_{k}"] > row[f"an_{k}"]
                        )

                rows.append(row)

            if ib == 1 or ib % 16 == 0 or ib == len(loader):
                print(
                    f"Evaluated batch {ib:3d}/{len(loader)} "
                    f"({min(ib*args.batch_size, len(recipes)):4d} samples)"
                )

    df = pd.DataFrame(rows)
    outdir = exp / "diagnostic_plots" / f"fp32_failure_tail_{checkpoint_tag}_step{step}"
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = exp / f"validation_failure_diagnostics_fp32_{checkpoint_tag}_step{step}.csv"
    df.to_csv(csv_path, index=False)

    # --------------------------------------------------------------
    # Summary by precipitation intensity
    # --------------------------------------------------------------
    summary_rows = []
    for label in INTENSITY_ORDER:
        g = df[df["intensity_bin"] == label]
        if g.empty:
            continue
        fails = g[g["z_margin_fp32"] < 0]
        summary_rows.append({
            "intensity_bin": label,
            "n": len(g),
            "z_ap_mean_fp32": g["z_ap_fp32"].mean(),
            "z_an_max_mean_fp32": g["z_an_max_fp32"].mean(),
            "z_margin_mean_fp32": g["z_margin_fp32"].mean(),
            "retrieval_failure_count": len(fails),
            "retrieval_failure_pct": 100 * len(fails) / len(g),
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = outdir / "failure_rate_by_intensity.csv"
    summary.to_csv(summary_path, index=False)

    # --------------------------------------------------------------
    # Wet failure tail and cross-season context
    # --------------------------------------------------------------
    wet = df[df["intensity_bin"] != "dry"].copy()
    wet_fail = wet[wet["z_margin_fp32"] < 0].sort_values("z_margin_fp32")
    wet_fail_path = outdir / "wet_retrieval_failures.csv"
    wet_fail.to_csv(wet_fail_path, index=False)

    # For a cleaner "meaningful precipitation" cut as well.
    wet1 = df[df["anchor_wet_p90_mm"] >= 1.0].copy()
    wet1_fail = wet1[wet1["z_margin_fp32"] < 0].sort_values("z_margin_fp32")
    wet1_path = outdir / "p90_ge1mm_retrieval_failures.csv"
    wet1_fail.to_csv(wet1_path, index=False)

    # Plot failure rate by intensity.
    plt.figure(figsize=(9, 5.5))
    plt.bar(
        np.arange(len(summary)),
        summary["retrieval_failure_pct"],
    )
    plt.xticks(
        np.arange(len(summary)),
        [
            f"{b}\n(n={n})"
            for b, n in zip(summary["intensity_bin"], summary["n"])
        ],
    )
    plt.ylabel("Hardest-N outranks P (%)")
    plt.xlabel("Anchor wet-pixel P90 intensity bin (mm / 6 h)")
    plt.title("FP32 validation retrieval-failure rate by precipitation intensity")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        outdir / "failure_rate_by_intensity.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    # Fixed-scale gallery of the worst non-dry failures.
    gallery_rows = [
        r for _, r in wet_fail.head(args.failure_gallery_count).iterrows()
    ]
    if gallery_rows:
        fixed_scale_gallery(
            gallery_rows,
            recipes_by_anchor,
            args.manifest,
            outdir / "fixed_scale_wet_failure_gallery",
            vmax_mm=25.0,
        )

    # --------------------------------------------------------------
    # Print compact interpretation aids
    # --------------------------------------------------------------
    print()
    print("=" * 88)
    print("FP32 FAILURE-TAIL SUMMARY")
    print("=" * 88)
    print(summary.to_string(index=False))
    print()

    for label, group in (
        ("all validation", df),
        ("non-dry", wet),
        ("P90wet >= 1 mm", wet1),
    ):
        fails = group[group["z_margin_fp32"] < 0]
        print(
            f"{label:18s}: n={len(group):3d} | failures={len(fails):3d} "
            f"({100*len(fails)/len(group):5.2f}%)"
        )
        if len(fails):
            print(
                f"  failure hardest-N same season: "
                f"{100*fails['same_season_hardest_n'].mean():.1f}%"
            )
            print(
                f"  conventional preference for P among failures: "
                f"MAE={100*fails['mae_prefers_p'].mean():.1f}% | "
                f"RMSE={100*fails['rmse_prefers_p'].mean():.1f}%"
            )
            valid_corr = fails["corr_prefers_p"].dropna()
            if len(valid_corr):
                print(
                    f"  correlation prefers P: "
                    f"{100*valid_corr.mean():.1f}%"
                )

    if len(wet1_fail):
        print()
        cols = [
            "anchor_valid_time",
            "hardest_n_valid_time",
            "season",
            "hardest_n_season",
            "anchor_wet_p90_mm",
            "z_ap_fp32",
            "z_an_max_fp32",
            "z_margin_fp32",
            "ap_rmse_mm",
            "an_rmse_mm",
            "ap_corr",
            "an_corr",
        ]
        print("Worst P90wet >= 1 mm failures:")
        print(wet1_fail[cols].head(10).to_string(index=False))

    print()
    print(f"Per-case CSV     : {csv_path}")
    print(f"Intensity summary: {summary_path}")
    print(f"Wet failures     : {wet_fail_path}")
    print(f"P90>=1 failures  : {wet1_path}")
    print(f"Plots/gallery    : {outdir}")


if __name__ == "__main__":
    main()

