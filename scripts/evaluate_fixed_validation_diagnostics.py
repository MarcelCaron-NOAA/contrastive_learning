#!/usr/bin/env python3
"""
Evaluate the fixed validation recipes per example for stratified diagnostics.

This script re-materializes the already-saved fixed validation recipes and
evaluates one checkpoint (default: checkpoint_best.pt). It writes a CSV with
per-example z/h similarities, loss, margin, top-1 retrieval, and precipitation
regime metadata.

Default wet/dry definition:
    wet if >= 1 grid point has QPF6 >= 0.1 mm
    dry otherwise

Default intensity diagnostic:
    90th percentile among anchor pixels with QPF6 >= 0.1 mm.
This is preferable to the single-grid maximum for regime binning, while max,
mean, p95, p99, and wet fractions are also retained in the output CSV.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.data import HRRRArchive, PrecipTransform, ContrastiveRecipe
from src.models.precip_resnet import PrecipResNet18


def defaults():
    data = Path(os.environ["DATA_DIR"])
    exp = data / "derived/experiments/first_real_train_validation_v1"
    return {
        "experiment_dir": exp,
        "manifest": (
            data
            / "forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz"
        ),
        "stats": data / "metadata/dataset_v1/precip_normalization_stats_v1.json",
    }


def array_equal(a: np.ndarray, b: np.ndarray) -> bool:
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    return a.shape == b.shape and bool(np.all(same))


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def precip_stats(raw: np.ndarray, wet_threshold: float):
    x = np.asarray(raw, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        x = np.array([0.0], dtype=np.float32)
    x = np.maximum(x, 0.0)

    wet = x >= wet_threshold
    wet_values = x[wet]

    return {
        "mean_mm": float(np.mean(x)),
        "max_mm": float(np.max(x)),
        "p95_mm": float(np.quantile(x, 0.95)),
        "p99_mm": float(np.quantile(x, 0.99)),
        "wet_fraction_ge_threshold": float(np.mean(wet)),
        "wet_pixel_p90_mm": (
            float(np.quantile(wet_values, 0.90))
            if wet_values.size
            else 0.0
        ),
        "wet_pixel_count": int(wet_values.size),
    }


class FixedRecipeDataset(Dataset):
    def __init__(
        self,
        recipes,
        manifest,
        transform,
        negative_count,
        wet_threshold,
    ):
        self.recipes = recipes
        self.archive = HRRRArchive(manifest)
        self.transform = transform
        self.negative_count = int(negative_count)
        self.wet_threshold = float(wet_threshold)

    def __len__(self):
        return len(self.recipes)

    def __getitem__(self, idx):
        r = self.recipes[idx]
        a_raw = self.archive.read_patch(
            r.anchor_id, r.y0, r.x0, r.patch_size
        )
        p_raw = self.archive.read_patch(
            r.positive_id, r.y0, r.x0, r.patch_size
        )

        n_tensors = []
        n_mask = []
        for nid in r.negative_ids:
            n_raw = self.archive.read_patch(
                nid, r.y0, r.x0, r.patch_size
            )
            valid = not (
                array_equal(n_raw, a_raw)
                or array_equal(n_raw, p_raw)
            )
            n_tensors.append(self.transform(n_raw))
            n_mask.append(valid)

        ast = precip_stats(a_raw, self.wet_threshold)
        pst = precip_stats(p_raw, self.wet_threshold)

        ts = pd.Timestamp(r.anchor_valid_time)
        meta = {
            "anchor_id": r.anchor_id,
            "positive_id": r.positive_id,
            "positive_direction": r.positive_direction,
            "split": r.split,
            "anchor_valid_time": r.anchor_valid_time,
            "anchor_forecast_hour": r.anchor_forecast_hour,
            "anchor_valid_hour_utc": r.anchor_valid_hour_utc,
            "anchor_climo_day": r.anchor_climo_day,
            "season": season_from_month(ts.month),
            "crop_y0": r.y0,
            "crop_x0": r.x0,
            "patch_size": r.patch_size,
            "negative_candidate_count": r.negative_candidate_count,
            "negative_ids_json": json.dumps(list(r.negative_ids)),
        }
        for key, value in ast.items():
            meta[f"anchor_{key}"] = value
        for key, value in pst.items():
            meta[f"positive_{key}"] = value

        meta["wet_dry"] = (
            "wet"
            if ast["wet_pixel_count"] > 0
            else "dry"
        )

        return {
            "anchor": self.transform(a_raw),
            "positive": self.transform(p_raw),
            "negatives": torch.stack(n_tensors, dim=0),
            "negative_mask": torch.tensor(n_mask, dtype=torch.bool),
            "metadata": meta,
        }


def collate(items):
    return {
        "anchor": torch.stack([x["anchor"] for x in items]),
        "positive": torch.stack([x["positive"] for x in items]),
        "negatives": torch.stack([x["negatives"] for x in items]),
        "negative_mask": torch.stack([x["negative_mask"] for x in items]),
        "metadata": [x["metadata"] for x in items],
    }


def load_recipes(path: Path):
    recipes = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            d["negative_ids"] = tuple(d["negative_ids"])
            recipes.append(ContrastiveRecipe(**d))
    if not recipes:
        raise RuntimeError(f"No recipes found in {path}")
    return recipes


def encode(model, a, p, n):
    B, K = n.shape[:2]
    x = torch.cat(
        [a, p, n.reshape(B * K, *n.shape[2:])],
        dim=0,
    )
    h, z = model(x)
    return (
        h[:B],
        h[B:2 * B],
        h[2 * B:].reshape(B, K, -1),
    ), (
        z[:B],
        z[B:2 * B],
        z[2 * B:].reshape(B, K, -1),
    )


def per_sample_metrics(za, zp, zn, mask, temperature):
    sim_ap = torch.sum(za * zp, dim=-1)
    sim_an = torch.einsum("bd,bkd->bk", za, zn)
    sim_pn = torch.einsum("bd,bkd->bk", zp, zn)

    very_negative = torch.finfo(sim_an.dtype).min
    an_masked = sim_an.masked_fill(~mask, very_negative)
    pn_masked = sim_pn.masked_fill(~mask, very_negative)

    logits_a = torch.cat([sim_ap[:, None], an_masked], dim=1) / temperature
    logits_p = torch.cat([sim_ap[:, None], pn_masked], dim=1) / temperature

    loss_a = -torch.log_softmax(logits_a, dim=1)[:, 0]
    loss_p = -torch.log_softmax(logits_p, dim=1)[:, 0]
    loss = 0.5 * (loss_a + loss_p)

    active = mask.any(dim=1)
    max_an = an_masked.max(dim=1).values
    max_pn = pn_masked.max(dim=1).values

    # Mean similarity over valid explicit negatives for each sample.
    denom = mask.sum(dim=1).clamp_min(1)
    mean_an = (
        sim_an.masked_fill(~mask, 0.0).sum(dim=1) / denom
    )
    mean_pn = (
        sim_pn.masked_fill(~mask, 0.0).sum(dim=1) / denom
    )

    out = {
        "loss": loss,
        "cosine_ap": sim_ap,
        "cosine_an_mean": mean_an,
        "cosine_pn_mean": mean_pn,
        "cosine_an_max": max_an,
        "cosine_pn_max": max_pn,
        "margin_a": sim_ap - max_an,
        "margin_p": sim_ap - max_pn,
        "top1_a": (sim_ap > max_an).float(),
        "top1_p": (sim_ap > max_pn).float(),
        "valid_negative_count": mask.sum(dim=1).float(),
        "active": active.float(),
    }

    # Samples with no valid N should not be interpreted.
    for key in (
        "loss",
        "cosine_an_mean",
        "cosine_pn_mean",
        "cosine_an_max",
        "cosine_pn_max",
        "margin_a",
        "margin_p",
        "top1_a",
        "top1_p",
    ):
        out[key] = torch.where(
            active,
            out[key],
            torch.full_like(out[key], float("nan")),
        )

    return out


def to_numpy_dict(metrics):
    return {
        k: v.detach().float().cpu().numpy()
        for k, v in metrics.items()
    }


def intensity_bin(v):
    # v = anchor wet-pixel 90th percentile in mm/6 h.
    if not np.isfinite(v) or v < 0.1:
        return "dry"
    if v < 1.0:
        return "0.1-1"
    if v < 5.0:
        return "1-5"
    if v < 10.0:
        return "5-10"
    if v < 25.0:
        return "10-25"
    if v < 50.0:
        return "25-50"
    return ">=50"


def main():
    d = defaults()

    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", default=str(d["experiment_dir"]))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--manifest", default=str(d["manifest"]))
    p.add_argument("--stats-json", default=str(d["stats"]))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--wet-threshold-mm", type=float, default=0.1)
    p.add_argument("--allow-cpu", action="store_true")
    args = p.parse_args()

    exp = Path(args.experiment_dir)
    recipe_path = exp / "fixed_validation_recipes.jsonl"
    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else exp / "checkpoint_best.pt"
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        amp = True
    elif args.allow_cpu:
        device = torch.device("cpu")
        amp = False
    else:
        raise SystemExit(
            "CUDA not available. Run on a GPU node or pass --allow-cpu."
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    step = int(ckpt.get("step", -1))

    model_spec = ckpt.get("model", {})
    model = PrecipResNet18(
        projection_hidden_dim=int(
            model_spec.get("projection_hidden_dim", 512)
        ),
        projection_dim=int(model_spec.get("projection_dim", 128)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    temperature = float(
        ckpt.get("criterion", {}).get("temperature", 0.10)
    )
    negative_count = int(
        ckpt.get("validation_sampling", {}).get(
            "negative_count",
            ckpt.get("training_sampling", {}).get("negative_count", 8),
        )
    )

    transform = PrecipTransform.from_stats_json(
        "log1p_zscore",
        args.stats_json,
    )
    recipes = load_recipes(recipe_path)

    dataset = FixedRecipeDataset(
        recipes,
        args.manifest,
        transform,
        negative_count,
        args.wet_threshold_mm,
    )

    kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = args.prefetch_factor

    loader = DataLoader(**kwargs)

    rows = []
    print("=" * 80)
    print("FIXED VALIDATION PER-EXAMPLE DIAGNOSTICS")
    print("=" * 80)
    print(f"Checkpoint       : {ckpt_path}")
    print(f"Checkpoint step  : {step}")
    print(f"Recipes          : {len(recipes)}")
    print(f"Device           : {device}")
    print(f"Temperature      : {temperature}")
    print(f"Wet threshold    : {args.wet_threshold_mm} mm/6h")
    print()

    with torch.no_grad():
        for ib, batch in enumerate(loader, start=1):
            a = batch["anchor"].to(device, non_blocking=True)
            p_ = batch["positive"].to(device, non_blocking=True)
            n = batch["negatives"].to(device, non_blocking=True)
            mask = batch["negative_mask"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp,
            ):
                (ha, hp, hn), (za, zp, zn) = encode(model, a, p_, n)

                zm = per_sample_metrics(
                    za, zp, zn, mask, temperature
                )

                hm = per_sample_metrics(
                    torch.nn.functional.normalize(ha, dim=-1),
                    torch.nn.functional.normalize(hp, dim=-1),
                    torch.nn.functional.normalize(hn, dim=-1),
                    mask,
                    temperature,
                )

            zm = to_numpy_dict(zm)
            hm = to_numpy_dict(hm)

            for i, meta in enumerate(batch["metadata"]):
                row = dict(meta)
                row["checkpoint_step"] = step
                row["temperature"] = temperature
                row["wet_threshold_mm"] = args.wet_threshold_mm

                for key, arr in zm.items():
                    row[f"z_{key}"] = float(arr[i])
                for key, arr in hm.items():
                    row[f"h_{key}"] = float(arr[i])

                row["z_top1_a_pct"] = 100.0 * row.pop("z_top1_a")
                row["z_top1_p_pct"] = 100.0 * row.pop("z_top1_p")
                row["h_top1_a_pct"] = 100.0 * row.pop("h_top1_a")
                row["h_top1_p_pct"] = 100.0 * row.pop("h_top1_p")

                row["intensity_bin"] = intensity_bin(
                    row["anchor_wet_pixel_p90_mm"]
                )
                rows.append(row)

            if ib == 1 or ib % 16 == 0 or ib == len(loader):
                print(
                    f"Evaluated batch {ib:3d}/{len(loader)} "
                    f"({min(ib*args.batch_size, len(recipes)):4d} samples)"
                )

    df = pd.DataFrame(rows)
    out = exp / f"validation_per_example_step{step}.csv"
    df.to_csv(out, index=False)

    print()
    print(f"Wrote: {out}")
    print(
        f"Active samples: "
        f"{int(np.isfinite(df['z_loss']).sum())}/{len(df)}"
    )
    print(
        f"Mean z loss={df['z_loss'].mean():.4f}, "
        f"margin={df['z_margin_a'].mean():+.4f}, "
        f"top1={df['z_top1_a_pct'].mean():.1f}%"
    )


if __name__ == "__main__":
    main()

