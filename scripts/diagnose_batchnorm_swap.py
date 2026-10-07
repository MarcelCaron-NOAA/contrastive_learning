#!/usr/bin/env python3
"""
Diagnose whether BatchNorm running statistics are driving the large
fixed-validation oscillations in the full-epoch run.

The experiment is deliberately surgical. It evaluates four model states:

  1. best_native
       best-checkpoint weights + best-checkpoint BN running stats

  2. last_native
       last-checkpoint weights + last-checkpoint BN running stats

  3. last_weights_best_bn
       LAST learned parameters, but BEST checkpoint BN running_mean/running_var

  4. best_weights_last_bn
       BEST learned parameters, but LAST checkpoint BN running_mean/running_var

If changing only the BN buffers moves validation performance dramatically,
then the late-run oscillations are being driven substantially by BatchNorm
state rather than optimizer-weight drift.

All scoring is FP32. The fixed 512 validation recipes are materialized once
and cached in CPU memory so all four model states see identical tensors.
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
from torch.utils.data import Dataset, DataLoader

from src.data import (
    HRRRArchive,
    PrecipTransform,
    ContrastiveDataset,
    ContrastiveRecipe,
)
from src.models.precip_resnet import PrecipResNet18


def default_exp():
    return (
        Path(os.environ["DATA_DIR"])
        / "derived/experiments/full_epoch_train_validation_v1"
    )


def default_manifest():
    return (
        Path(os.environ["DATA_DIR"])
        / "forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz"
    )


def default_stats():
    return (
        Path(os.environ["DATA_DIR"])
        / "metadata/dataset_v1/precip_normalization_stats_v1.json"
    )


def load_recipes(path):
    out = []
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            d["negative_ids"] = tuple(d["negative_ids"])
            out.append(ContrastiveRecipe(**d))
    return out


class RecipeDataset(Dataset):
    def __init__(self, recipes, manifest, transform):
        self.recipes = recipes
        self.archive = HRRRArchive(manifest)
        self.materializer = ContrastiveDataset(
            self.archive,
            negative_count=len(recipes[0].negative_ids),
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
    }


def build_model(ckpt, device):
    spec = ckpt.get("model", {})
    model = PrecipResNet18(
        projection_hidden_dim=int(spec.get("projection_hidden_dim", 512)),
        projection_dim=int(spec.get("projection_dim", 128)),
    )
    model.to(device)
    return model


def bn_buffer_key(key):
    return (
        key.endswith("running_mean")
        or key.endswith("running_var")
        or key.endswith("num_batches_tracked")
    )


def hybrid_state(parameter_source, bn_source):
    state = {}
    for key, value in parameter_source["model_state_dict"].items():
        if bn_buffer_key(key):
            state[key] = bn_source["model_state_dict"][key].clone()
        else:
            state[key] = value.clone()
    return state


def encode(model, a, p, n):
    B, K = n.shape[:2]
    x = torch.cat(
        [a, p, n.reshape(B * K, *n.shape[2:])],
        dim=0,
    )
    h, z = model(x)

    h = F.normalize(h, dim=-1)

    ha = h[:B]
    hp = h[B:2*B]
    hn = h[2*B:].reshape(B, K, -1)

    za = z[:B]
    zp = z[B:2*B]
    zn = z[2*B:].reshape(B, K, -1)

    return (ha, hp, hn), (za, zp, zn)


def evaluate(model, cached, device, temperature):
    model.eval()

    sums = {
        "n": 0,
        "z_loss": 0.0,
        "z_ap": 0.0,
        "z_an": 0.0,
        "z_margin": 0.0,
        "z_top1": 0.0,
        "h_loss": 0.0,
        "h_ap": 0.0,
        "h_an": 0.0,
        "h_margin": 0.0,
        "h_top1": 0.0,
    }

    with torch.no_grad():
        for batch in cached:
            a = batch["anchor"].to(device, non_blocking=True)
            p = batch["positive"].to(device, non_blocking=True)
            n = batch["negatives"].to(device, non_blocking=True)
            mask = batch["negative_mask"].to(device, non_blocking=True)

            (ha, hp, hn), (za, zp, zn) = encode(model, a, p, n)

            B = a.shape[0]
            target = torch.zeros(B, dtype=torch.long, device=device)

            for prefix, qa, qp, qn in (
                ("z", za, zp, zn),
                ("h", ha, hp, hn),
            ):
                ap = torch.sum(qa * qp, dim=-1)
                an = torch.einsum("bd,bkd->bk", qa, qn)
                pn = torch.einsum("bd,bkd->bk", qp, qn)

                very_neg = torch.finfo(an.dtype).min
                an_masked = an.masked_fill(~mask, very_neg)
                pn_masked = pn.masked_fill(~mask, very_neg)

                logits_a = torch.cat(
                    [ap[:, None], an_masked], dim=1
                ) / temperature
                logits_p = torch.cat(
                    [ap[:, None], pn_masked], dim=1
                ) / temperature

                loss_a = F.cross_entropy(
                    logits_a, target, reduction="none"
                )
                loss_p = F.cross_entropy(
                    logits_p, target, reduction="none"
                )
                loss = 0.5 * (loss_a + loss_p)

                valid_float = mask.float()
                valid_count = valid_float.sum(dim=1)
                an_mean = (
                    (an * valid_float).sum(dim=1)
                    / valid_count.clamp_min(1.0)
                )

                hard = an_masked.max(dim=1).values
                margin = ap - hard
                top1 = (ap > hard).float()

                sums[f"{prefix}_loss"] += float(loss.sum().item())
                sums[f"{prefix}_ap"] += float(ap.sum().item())
                sums[f"{prefix}_an"] += float(an_mean.sum().item())
                sums[f"{prefix}_margin"] += float(margin.sum().item())
                sums[f"{prefix}_top1"] += float(top1.sum().item())

            sums["n"] += B

    n = sums.pop("n")
    out = {k: v / n for k, v in sums.items()}
    out["z_top1_pct"] = 100.0 * out.pop("z_top1")
    out["h_top1_pct"] = 100.0 * out.pop("h_top1")
    return out


def state_diff_stats(best, last):
    best_state = best["model_state_dict"]
    last_state = last["model_state_dict"]

    param_sq = 0.0
    param_ref_sq = 0.0
    bn_sq = 0.0
    bn_ref_sq = 0.0
    bn_keys = 0

    for key in best_state:
        a = best_state[key]
        b = last_state[key]

        if not torch.is_floating_point(a):
            continue

        d2 = float(torch.sum((a.float() - b.float()) ** 2).item())
        r2 = float(torch.sum(a.float() ** 2).item())

        if bn_buffer_key(key):
            bn_sq += d2
            bn_ref_sq += r2
            bn_keys += 1
        else:
            param_sq += d2
            param_ref_sq += r2

    return {
        "parameter_delta_l2": param_sq ** 0.5,
        "parameter_relative_delta": (
            (param_sq / param_ref_sq) ** 0.5
            if param_ref_sq > 0 else np.nan
        ),
        "bn_buffer_delta_l2": bn_sq ** 0.5,
        "bn_buffer_relative_delta": (
            (bn_sq / bn_ref_sq) ** 0.5
            if bn_ref_sq > 0 else np.nan
        ),
        "bn_float_buffer_keys": bn_keys,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", default=str(default_exp()))
    p.add_argument("--manifest", default=str(default_manifest()))
    p.add_argument("--stats-json", default=str(default_stats()))
    p.add_argument("--best-checkpoint", default=None)
    p.add_argument("--last-checkpoint", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.10)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required.")

    device = torch.device("cuda")
    exp = Path(args.experiment_dir)
    best_path = (
        Path(args.best_checkpoint)
        if args.best_checkpoint
        else exp / "checkpoint_best.pt"
    )
    last_path = (
        Path(args.last_checkpoint)
        if args.last_checkpoint
        else exp / "checkpoint_last.pt"
    )

    best = torch.load(best_path, map_location="cpu")
    last = torch.load(last_path, map_location="cpu")

    recipes = load_recipes(exp / "fixed_validation_recipes.jsonl")
    transform = PrecipTransform.from_stats_json(
        "log1p_zscore", args.stats_json
    )
    dataset = RecipeDataset(recipes, args.manifest, transform)

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

    print("=" * 86)
    print("BATCHNORM BUFFER-SWAP DIAGNOSTIC")
    print("=" * 86)
    print(f"Best checkpoint : {best_path} (step {best.get('step', -1)})")
    print(f"Last checkpoint : {last_path} (step {last.get('step', -1)})")
    print(f"Recipes         : {len(recipes)}")
    print("Caching fixed validation tensors once...")

    cached = []
    for ib, batch in enumerate(loader, start=1):
        cached.append({
            "anchor": batch["anchor"].contiguous(),
            "positive": batch["positive"].contiguous(),
            "negatives": batch["negatives"].contiguous(),
            "negative_mask": batch["negative_mask"].contiguous(),
        })
        if ib == 1 or ib % 16 == 0 or ib == len(loader):
            print(f"  cached {ib:3d}/{len(loader)} batches")

    cases = [
        ("best_native", best, best),
        ("last_native", last, last),
        ("last_weights_best_bn", last, best),
        ("best_weights_last_bn", best, last),
    ]

    rows = []

    for label, parameter_source, bn_source in cases:
        model = build_model(parameter_source, device)
        model.load_state_dict(
            hybrid_state(parameter_source, bn_source),
            strict=True,
        )

        metrics = evaluate(
            model,
            cached,
            device,
            args.temperature,
        )

        row = {
            "case": label,
            "parameter_checkpoint_step": int(
                parameter_source.get("step", -1)
            ),
            "bn_checkpoint_step": int(
                bn_source.get("step", -1)
            ),
            **metrics,
        }
        rows.append(row)

        print(
            f"{label:24s} | "
            f"z_loss={metrics['z_loss']:.4f} "
            f"z_AP={metrics['z_ap']:.4f} "
            f"z_AN={metrics['z_an']:.4f} "
            f"z_margin={metrics['z_margin']:+.4f} "
            f"z_top1={metrics['z_top1_pct']:.1f}%"
        )

        del model
        torch.cuda.empty_cache()

    diff = state_diff_stats(best, last)

    outdir = exp / "diagnostic_plots" / "batchnorm_swap"
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    csv_path = outdir / "batchnorm_swap_results.csv"
    df.to_csv(csv_path, index=False)

    diff_path = outdir / "best_to_last_state_differences.json"
    diff_path.write_text(json.dumps(diff, indent=2))

    print()
    print("State differences (best -> last):")
    for k, v in diff.items():
        print(f"  {k}: {v}")
    print()
    print(f"Results: {csv_path}")
    print(f"State diff: {diff_path}")


if __name__ == "__main__":
    main()

