#!/usr/bin/env python3
"""
Evaluate existing BatchNorm checkpoints using instantaneous batch statistics.

Purpose
-------
The trained BN ResNet showed large fixed-validation oscillations because eval()
uses running BatchNorm statistics, while optimization used current-minibatch
statistics. This diagnostic evaluates the SAME fixed validation recipes in:

  1) ordinary eval mode (running BN statistics)
  2) batch-stat mode with logical batch size 4  -> 40 physical images
  3) batch-stat mode with logical batch size 8  -> 80 physical images
  4) batch-stat mode with logical batch size 16 -> 160 physical images

No gradients are used. BN running buffers are not updated: track_running_stats
is temporarily disabled while BN layers are put into training mode.

This is diagnostic only. A verification metric whose answer depends on the
other samples in the inference minibatch would be undesirable. The goal is to
identify whether train/eval BN mismatch explains the bad final checkpoint and
whether increasing BN batch size materially stabilizes the geometry.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.data import (
    HRRRArchive,
    PrecipTransform,
    ContrastiveDataset,
    ContrastiveRecipe,
)
from src.models.precip_resnet import PrecipResNet18


def defaults():
    data = Path(os.environ["DATA_DIR"])
    exp = data / "derived/experiments/full_epoch_train_validation_v1"
    return {
        "experiment": exp,
        "manifest": (
            data
            / "forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz"
        ),
        "stats": data / "metadata/dataset_v1/precip_normalization_stats_v1.json",
    }


def load_recipes(path):
    recipes = []
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            d["negative_ids"] = tuple(d["negative_ids"])
            recipes.append(ContrastiveRecipe(**d))
    return recipes


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
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model


def bn_modules(model):
    return [
        m for m in model.modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm)
    ]


class BatchStatMode:
    """
    Use current-batch BN statistics without modifying running buffers.
    """
    def __init__(self, model):
        self.model = model
        self.saved = []

    def __enter__(self):
        self.model.eval()
        for bn in bn_modules(self.model):
            self.saved.append(
                (
                    bn,
                    bn.training,
                    bn.track_running_stats,
                )
            )
            bn.train()
            bn.track_running_stats = False
        return self.model

    def __exit__(self, exc_type, exc, tb):
        for bn, training, track in self.saved:
            bn.track_running_stats = track
            bn.train(training)
        self.model.eval()


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
        h[B:2 * B],
        h[2 * B:].reshape(B, K, -1),
    ), (
        z[:B],
        z[B:2 * B],
        z[2 * B:].reshape(B, K, -1),
    )


def eval_batches(model, tensors, batch_size, device, temperature, batch_stats):
    n_total = tensors["anchor"].shape[0]
    accum = {
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

    def run():
        with torch.no_grad():
            for i in range(0, n_total, batch_size):
                a = tensors["anchor"][i:i+batch_size].to(
                    device, non_blocking=True
                )
                p = tensors["positive"][i:i+batch_size].to(
                    device, non_blocking=True
                )
                n = tensors["negatives"][i:i+batch_size].to(
                    device, non_blocking=True
                )
                mask = tensors["negative_mask"][i:i+batch_size].to(
                    device, non_blocking=True
                )

                B = a.shape[0]
                target = torch.zeros(B, dtype=torch.long, device=device)

                (ha, hp, hn), (za, zp, zn) = encode(model, a, p, n)

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

                    loss = 0.5 * (
                        F.cross_entropy(
                            logits_a, target, reduction="none"
                        )
                        + F.cross_entropy(
                            logits_p, target, reduction="none"
                        )
                    )

                    valid_float = mask.float()
                    valid_count = valid_float.sum(dim=1).clamp_min(1.0)
                    an_mean = (
                        (an * valid_float).sum(dim=1)
                        / valid_count
                    )
                    hardest = an_masked.max(dim=1).values
                    margin = ap - hardest
                    top1 = (ap > hardest).float()

                    accum[f"{prefix}_loss"] += float(loss.sum())
                    accum[f"{prefix}_ap"] += float(ap.sum())
                    accum[f"{prefix}_an"] += float(an_mean.sum())
                    accum[f"{prefix}_margin"] += float(margin.sum())
                    accum[f"{prefix}_top1"] += float(top1.sum())

                accum["n"] += B

    if batch_stats:
        with BatchStatMode(model):
            run()
    else:
        model.eval()
        run()

    count = accum.pop("n")
    out = {k: v / count for k, v in accum.items()}
    out["z_top1_pct"] = 100.0 * out.pop("z_top1")
    out["h_top1_pct"] = 100.0 * out.pop("h_top1")
    return out


def main():
    d = defaults()
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", default=str(d["experiment"]))
    p.add_argument("--manifest", default=str(d["manifest"]))
    p.add_argument("--stats-json", default=str(d["stats"]))
    p.add_argument(
        "--batch-stat-sizes",
        type=int,
        nargs="+",
        default=[4, 8, 16],
    )
    p.add_argument("--materialize-batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.10)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required.")

    device = torch.device("cuda")
    exp = Path(args.experiment_dir)

    recipes = load_recipes(exp / "fixed_validation_recipes.jsonl")
    if any(len(recipes) % b != 0 for b in args.batch_stat_sizes):
        raise SystemExit(
            "For this diagnostic, each --batch-stat-sizes value must divide "
            f"the fixed validation count ({len(recipes)})."
        )

    transform = PrecipTransform.from_stats_json(
        "log1p_zscore", args.stats_json
    )
    dataset = RecipeDataset(recipes, args.manifest, transform)

    loader_kw = dict(
        dataset=dataset,
        batch_size=args.materialize_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )
    if args.num_workers > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = args.prefetch_factor

    print("=" * 90)
    print("BATCHNORM TRAIN-STAT / EVAL-STAT DIAGNOSTIC")
    print("=" * 90)
    print(f"GPU                  : {torch.cuda.get_device_name(0)}")
    print(f"Fixed recipes        : {len(recipes)}")
    print(f"Batch-stat sizes     : {args.batch_stat_sizes}")
    print(
        "Physical BN sizes    : "
        + ", ".join(
            str(b * 10) for b in args.batch_stat_sizes
        )
        + " images"
    )
    print()
    print("Materializing fixed validation tensors once...")

    chunks = {
        "anchor": [],
        "positive": [],
        "negatives": [],
        "negative_mask": [],
    }
    for ib, batch in enumerate(DataLoader(**loader_kw), start=1):
        for k in chunks:
            chunks[k].append(batch[k].contiguous())
        if ib == 1 or ib % 16 == 0:
            print(f"  materialized batch {ib}")

    tensors = {
        k: torch.cat(v, dim=0)
        for k, v in chunks.items()
    }
    print(
        f"Materialized {tensors['anchor'].shape[0]} logical samples."
    )
    print()

    cases = [
        ("best", exp / "checkpoint_best.pt"),
        ("last", exp / "checkpoint_last.pt"),
    ]

    rows = []
    outdir = exp / "diagnostic_plots" / "bn_batch_statistics"
    outdir.mkdir(parents=True, exist_ok=True)

    for label, path in cases:
        ckpt = torch.load(path, map_location="cpu")
        model = build_model(ckpt, device)

        print("-" * 90)
        print(
            f"{label.upper()} checkpoint | step={ckpt.get('step', -1)}"
        )
        print("-" * 90)

        native = eval_batches(
            model,
            tensors,
            batch_size=8,
            device=device,
            temperature=args.temperature,
            batch_stats=False,
        )
        rows.append({
            "checkpoint": label,
            "source_step": int(ckpt.get("step", -1)),
            "mode": "eval_running_stats",
            "logical_batch_size": 8,
            "physical_bn_batch_size": np.nan,
            **native,
        })
        print(
            f"running stats        | "
            f"loss={native['z_loss']:.4f} "
            f"AP={native['z_ap']:.4f} "
            f"AN={native['z_an']:.4f} "
            f"margin={native['z_margin']:+.4f} "
            f"top1={native['z_top1_pct']:.1f}%"
        )

        for b in args.batch_stat_sizes:
            metrics = eval_batches(
                model,
                tensors,
                batch_size=b,
                device=device,
                temperature=args.temperature,
                batch_stats=True,
            )
            rows.append({
                "checkpoint": label,
                "source_step": int(ckpt.get("step", -1)),
                "mode": "instantaneous_batch_stats",
                "logical_batch_size": b,
                "physical_bn_batch_size": b * 10,
                **metrics,
            })
            print(
                f"batch stats B={b:2d} "
                f"(physical={b*10:3d}) | "
                f"loss={metrics['z_loss']:.4f} "
                f"AP={metrics['z_ap']:.4f} "
                f"AN={metrics['z_an']:.4f} "
                f"margin={metrics['z_margin']:+.4f} "
                f"top1={metrics['z_top1_pct']:.1f}%"
            )

        del model
        torch.cuda.empty_cache()
        print()

    df = pd.DataFrame(rows)
    csv_path = outdir / "bn_batch_stat_diagnostic.csv"
    df.to_csv(csv_path, index=False)

    print(f"Results: {csv_path}")


if __name__ == "__main__":
    main()

