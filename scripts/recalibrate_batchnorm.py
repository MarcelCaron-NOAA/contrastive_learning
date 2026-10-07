#!/usr/bin/env python3
"""
Post-hoc BatchNorm recalibration experiment for the full-epoch HRRR
contrastive model.

Scientific question
-------------------
The full-epoch run showed large fixed-validation oscillations even after the
optimizer learning rate became very small. A prior BN-buffer swap experiment
showed that replacing only BatchNorm running_mean/running_var strongly changed
validation performance.

This script asks a cleaner question:

    If model weights are frozen and BatchNorm running statistics are recomputed
    from a large, representative sample of TRAINING data only, how much of the
    evaluation degradation disappears?

No gradients and no optimizer steps are used. Trainable parameters are asserted
to remain bitwise unchanged.

Calibration data
----------------
A deterministic set of training recipes is generated once using the same v1
relationship definition as training:
  * split=train
  * climo45 negatives
  * +/-1 h positive
  * same crop within A/P/N
  * K=8 explicit negatives
  * log1p + training-derived z-score transform

By default, 10,000 logical training anchors are used. With batch size 8 and
K=8, each BN update sees 80 physical precipitation images
(A + P + 8 N for each of 8 logical samples).

Recalibration algorithm
-----------------------
1. Load a checkpoint.
2. Put the overall model in eval mode.
3. Put only BatchNorm layers in train mode.
4. Reset running_mean, running_var, num_batches_tracked.
5. Temporarily set BN momentum=None, giving PyTorch's cumulative moving average.
6. Forward the fixed training-calibration recipes in FP32 with no gradients.
7. Restore BN momentum and switch to eval mode.
8. Save a new checkpoint containing the identical learned weights and newly
   estimated BN buffers.
9. Evaluate the exact fixed 512-case validation set in FP32.

Outputs
-------
For both checkpoint_best.pt and checkpoint_last.pt:
  checkpoint_*_bnrecal.pt

Plus:
  bn_recalibration_comparison.csv
  bn_recalibration_metadata.json
  bn_recalibration_recipes_*.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.data import (
    ContrastiveIndex,
    NegativeRules,
    ContrastiveBatchSampler,
    HRRRArchive,
    ContrastiveDataset,
    ContrastiveRecipe,
    PrecipTransform,
)
from src.models.precip_resnet import PrecipResNet18


def default_paths():
    data = Path(os.environ["DATA_DIR"])
    exp = data / "derived/experiments/full_epoch_train_validation_v1"
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


def load_recipes(path: Path) -> list[ContrastiveRecipe]:
    recipes = []
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            d["negative_ids"] = tuple(d["negative_ids"])
            recipes.append(ContrastiveRecipe(**d))
    if not recipes:
        raise RuntimeError(f"No recipes found in {path}")
    return recipes


def write_recipes(path: Path, recipes: Iterable[ContrastiveRecipe]) -> None:
    with Path(path).open("w") as f:
        for r in recipes:
            f.write(json.dumps(asdict(r)) + "\n")


class RecipeDataset(Dataset):
    def __init__(
        self,
        recipes: list[ContrastiveRecipe],
        manifest: str | Path,
        transform,
    ):
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


def loader_kwargs(dataset, batch_size, workers, prefetch_factor):
    kw = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate,
        pin_memory=True,
    )
    if workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = prefetch_factor
    return kw


def build_model(ckpt, device):
    spec = ckpt.get("model", {})
    model = PrecipResNet18(
        projection_hidden_dim=int(spec.get("projection_hidden_dim", 512)),
        projection_dim=int(spec.get("projection_dim", 128)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model


def named_parameter_digest(model: nn.Module) -> str:
    """Bitwise digest over all trainable/model parameters, excluding buffers."""
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        h.update(name.encode("utf-8"))
        arr = param.detach().cpu().contiguous().numpy()
        h.update(arr.tobytes())
    return h.hexdigest()


def batchnorm_modules(model: nn.Module):
    return [
        m for m in model.modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm)
    ]


def reset_and_prepare_bn(model: nn.Module):
    bns = batchnorm_modules(model)
    if not bns:
        raise RuntimeError("No BatchNorm modules found.")

    model.eval()

    old_momenta = {}
    for i, bn in enumerate(bns):
        old_momenta[i] = bn.momentum

        if bn.running_mean is None or bn.running_var is None:
            raise RuntimeError(
                "Encountered BatchNorm with track_running_stats=False; "
                "this recalibration script expects running buffers."
            )

        bn.running_mean.zero_()
        bn.running_var.fill_(1.0)
        bn.num_batches_tracked.zero_()

        # Cumulative moving average across calibration minibatches.
        bn.momentum = None
        bn.train()

    return bns, old_momenta


def finish_bn(model, bns, old_momenta):
    for i, bn in enumerate(bns):
        bn.momentum = old_momenta[i]
        bn.eval()
    model.eval()


def forward_physical_batch(model, batch, device):
    a = batch["anchor"].to(device, non_blocking=True)
    p = batch["positive"].to(device, non_blocking=True)
    n = batch["negatives"].to(device, non_blocking=True)
    B, K = n.shape[:2]

    x = torch.cat(
        [a, p, n.reshape(B * K, *n.shape[2:])],
        dim=0,
    )
    model(x)


@torch.no_grad()
def recalibrate_bn(model, loader, device):
    bns, old_momenta = reset_and_prepare_bn(model)

    logical_seen = 0
    physical_seen = 0

    for ib, batch in enumerate(loader, start=1):
        B = int(batch["anchor"].shape[0])
        K = int(batch["negatives"].shape[1])

        # Deliberately FP32: no autocast during population-stat estimation.
        forward_physical_batch(model, batch, device)

        logical_seen += B
        physical_seen += B * (K + 2)

        if ib == 1 or ib % 250 == 0 or ib == len(loader):
            print(
                f"  BN calibration batch {ib:4d}/{len(loader)} | "
                f"logical={logical_seen:,} | physical={physical_seen:,}"
            )

    finish_bn(model, bns, old_momenta)

    return {
        "bn_module_count": len(bns),
        "logical_samples_seen": logical_seen,
        "physical_images_seen": physical_seen,
        "calibration_batches": len(loader),
    }


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


@torch.no_grad()
def evaluate_fixed_validation(model, cached, device, temperature):
    model.eval()

    totals = {
        "n": 0,
        "z_loss": 0.0,
        "z_cosine_ap": 0.0,
        "z_cosine_an": 0.0,
        "z_margin_a": 0.0,
        "z_top1_a": 0.0,
        "h_loss": 0.0,
        "h_cosine_ap": 0.0,
        "h_cosine_an": 0.0,
        "h_margin_a": 0.0,
        "h_top1_a": 0.0,
    }

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

            loss = 0.5 * (
                F.cross_entropy(logits_a, target, reduction="none")
                + F.cross_entropy(logits_p, target, reduction="none")
            )

            valid_float = mask.float()
            valid_count = valid_float.sum(dim=1).clamp_min(1.0)
            an_mean = (an * valid_float).sum(dim=1) / valid_count

            hard = an_masked.max(dim=1).values
            margin = ap - hard
            top1 = (ap > hard).float()

            totals[f"{prefix}_loss"] += float(loss.sum().item())
            totals[f"{prefix}_cosine_ap"] += float(ap.sum().item())
            totals[f"{prefix}_cosine_an"] += float(an_mean.sum().item())
            totals[f"{prefix}_margin_a"] += float(margin.sum().item())
            totals[f"{prefix}_top1_a"] += float(top1.sum().item())

        totals["n"] += B

    count = totals.pop("n")
    out = {}
    for key, value in totals.items():
        out[key] = value / count

    out["z_top1_a_pct"] = 100.0 * out.pop("z_top1_a")
    out["h_top1_a_pct"] = 100.0 * out.pop("h_top1_a")
    return out


def build_calibration_recipes(
    *,
    index,
    count,
    batch_size,
    patch_size,
    negative_count,
    seed,
    negative_min_days,
    climo_window_days,
):
    rules = NegativeRules(
        mode="climo45",
        min_days=negative_min_days,
        climo_window_days=climo_window_days,
    )
    sampler = ContrastiveBatchSampler(
        index,
        split="train",
        batch_size=batch_size,
        patch_size=patch_size,
        negative_count=negative_count,
        negative_rules=rules,
        seed=seed,
    )
    # A distinct deterministic epoch from the training run.
    sampler.set_epoch(91)

    recipes = []
    for batch in sampler:
        need = count - len(recipes)
        if need <= 0:
            break
        recipes.extend(batch[:need])
        if len(recipes) >= count:
            break

    if len(recipes) != count:
        raise RuntimeError(
            f"Requested {count} calibration anchors; built {len(recipes)}."
        )
    return recipes


def save_recalibrated_checkpoint(
    original_ckpt,
    model,
    output_path,
    metadata,
):
    new_ckpt = dict(original_ckpt)
    new_ckpt["model_state_dict"] = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }
    new_ckpt["bn_recalibration"] = metadata
    torch.save(new_ckpt, output_path)


def main():
    d = default_paths()
    p = argparse.ArgumentParser()

    p.add_argument("--experiment-dir", default=str(d["experiment_dir"]))
    p.add_argument("--anchors", default=str(d["anchors"]))
    p.add_argument("--negative-pool", default=str(d["negative_pool"]))
    p.add_argument("--manifest", default=str(d["manifest"]))
    p.add_argument("--stats-json", default=str(d["stats"]))

    p.add_argument("--best-checkpoint", default=None)
    p.add_argument("--last-checkpoint", default=None)

    p.add_argument("--calibration-anchors", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--negative-count", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--negative-min-days", type=float, default=90.0)
    p.add_argument("--climo-window-days", type=int, default=45)
    p.add_argument("--seed", type=int, default=20261002)

    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.10)

    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required.")

    if args.calibration_anchors < args.batch_size:
        raise SystemExit("--calibration-anchors must be >= batch size.")

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

    outdir = exp / "bn_recalibration"
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print("POST-HOC BATCHNORM RECALIBRATION EXPERIMENT")
    print("=" * 92)
    print(f"GPU                  : {torch.cuda.get_device_name(0)}")
    print(f"Best checkpoint      : {best_path}")
    print(f"Last checkpoint      : {last_path}")
    print(f"Calibration anchors  : {args.calibration_anchors:,}")
    print(f"Logical batch size   : {args.batch_size}")
    print(f"K negatives          : {args.negative_count}")
    print(
        f"Physical images/batch: "
        f"{args.batch_size * (args.negative_count + 2)}"
    )
    print("Calibration split    : train")
    print("Calibration negatives: climo45")
    print("Calibration precision: FP32")
    print()

    print("Loading metadata index...")
    index = ContrastiveIndex(
        args.anchors,
        args.negative_pool,
        args.manifest,
    )
    transform = PrecipTransform.from_stats_json(
        "log1p_zscore",
        args.stats_json,
    )

    recipe_path = outdir / (
        f"bn_recalibration_recipes_"
        f"{args.calibration_anchors}_seed{args.seed}.jsonl"
    )

    if recipe_path.exists():
        print(f"Using existing calibration recipes: {recipe_path}")
        calibration_recipes = load_recipes(recipe_path)
        if len(calibration_recipes) != args.calibration_anchors:
            raise RuntimeError(
                f"{recipe_path} contains {len(calibration_recipes)} recipes; "
                f"expected {args.calibration_anchors}."
            )
    else:
        print("Building deterministic training-only calibration recipes...")
        calibration_recipes = build_calibration_recipes(
            index=index,
            count=args.calibration_anchors,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            negative_count=args.negative_count,
            seed=args.seed,
            negative_min_days=args.negative_min_days,
            climo_window_days=args.climo_window_days,
        )
        write_recipes(recipe_path, calibration_recipes)
        print(f"Wrote calibration recipes: {recipe_path}")

    print("Loading fixed validation recipes...")
    validation_recipes = load_recipes(
        exp / "fixed_validation_recipes.jsonl"
    )

    calibration_dataset = RecipeDataset(
        calibration_recipes,
        args.manifest,
        transform,
    )
    calibration_loader = DataLoader(
        **loader_kwargs(
            calibration_dataset,
            args.batch_size,
            args.num_workers,
            args.prefetch_factor,
        )
    )

    validation_dataset = RecipeDataset(
        validation_recipes,
        args.manifest,
        transform,
    )
    validation_loader = DataLoader(
        **loader_kwargs(
            validation_dataset,
            args.batch_size,
            args.num_workers,
            args.prefetch_factor,
        )
    )

    print("Caching the fixed 512-case validation set once...")
    cached_validation = []
    for ib, batch in enumerate(validation_loader, start=1):
        cached_validation.append({
            "anchor": batch["anchor"].contiguous(),
            "positive": batch["positive"].contiguous(),
            "negatives": batch["negatives"].contiguous(),
            "negative_mask": batch["negative_mask"].contiguous(),
        })
        if ib == 1 or ib % 16 == 0 or ib == len(validation_loader):
            print(f"  cached {ib:3d}/{len(validation_loader)} batches")
    print()

    cases = [
        ("best", best_path),
        ("last", last_path),
    ]

    rows = []
    metadata_all = {
        "version": 1,
        "purpose": (
            "post-hoc training-only BatchNorm population-stat recalibration"
        ),
        "calibration_recipe_path": str(recipe_path),
        "calibration_anchor_count": args.calibration_anchors,
        "batch_size": args.batch_size,
        "negative_count": args.negative_count,
        "patch_size": args.patch_size,
        "negative_mode": "climo45",
        "negative_min_days": args.negative_min_days,
        "climo_window_days": args.climo_window_days,
        "seed": args.seed,
        "precision": "FP32",
        "checkpoint_results": {},
    }

    for label, ckpt_path in cases:
        print("-" * 92)
        print(f"CHECKPOINT: {label.upper()} | {ckpt_path}")
        print("-" * 92)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = build_model(ckpt, device)

        digest_before = named_parameter_digest(model)
        native = evaluate_fixed_validation(
            model,
            cached_validation,
            device,
            args.temperature,
        )

        print(
            f"Native     | z_loss={native['z_loss']:.4f} | "
            f"AP={native['z_cosine_ap']:.4f} | "
            f"AN={native['z_cosine_an']:.4f} | "
            f"margin={native['z_margin_a']:+.4f} | "
            f"top1={native['z_top1_a_pct']:.1f}%"
        )

        print("Recomputing BatchNorm running statistics from training data...")
        recal_meta = recalibrate_bn(
            model,
            calibration_loader,
            device,
        )

        digest_after = named_parameter_digest(model)
        if digest_before != digest_after:
            raise RuntimeError(
                "Model parameters changed during BN recalibration. "
                "This experiment must modify buffers only."
            )

        recal = evaluate_fixed_validation(
            model,
            cached_validation,
            device,
            args.temperature,
        )

        print(
            f"Recalibrated| z_loss={recal['z_loss']:.4f} | "
            f"AP={recal['z_cosine_ap']:.4f} | "
            f"AN={recal['z_cosine_an']:.4f} | "
            f"margin={recal['z_margin_a']:+.4f} | "
            f"top1={recal['z_top1_a_pct']:.1f}%"
        )

        out_ckpt = outdir / f"checkpoint_{label}_bnrecal.pt"
        checkpoint_meta = {
            **recal_meta,
            "source_checkpoint": str(ckpt_path),
            "source_step": int(ckpt.get("step", -1)),
            "parameter_digest_before": digest_before,
            "parameter_digest_after": digest_after,
            "validation_before": native,
            "validation_after": recal,
        }
        save_recalibrated_checkpoint(
            ckpt,
            model,
            out_ckpt,
            checkpoint_meta,
        )

        rows.append({
            "checkpoint": label,
            "source_step": int(ckpt.get("step", -1)),
            "state": "native",
            **native,
        })
        rows.append({
            "checkpoint": label,
            "source_step": int(ckpt.get("step", -1)),
            "state": "bn_recalibrated",
            **recal,
        })

        metadata_all["checkpoint_results"][label] = {
            **checkpoint_meta,
            "output_checkpoint": str(out_ckpt),
        }

        print(f"Saved: {out_ckpt}")
        print()

        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    csv_path = outdir / "bn_recalibration_comparison.csv"
    df.to_csv(csv_path, index=False)

    metadata_path = outdir / "bn_recalibration_metadata.json"
    metadata_path.write_text(json.dumps(metadata_all, indent=2) + "\n")

    print("=" * 92)
    print("BATCHNORM RECALIBRATION COMPARISON")
    print("=" * 92)
    print(
        df[
            [
                "checkpoint",
                "source_step",
                "state",
                "z_loss",
                "z_cosine_ap",
                "z_cosine_an",
                "z_margin_a",
                "z_top1_a_pct",
                "h_loss",
                "h_margin_a",
                "h_top1_a_pct",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Comparison CSV : {csv_path}")
    print(f"Metadata JSON  : {metadata_path}")
    print(f"Output dir     : {outdir}")
    print("=" * 92)


if __name__ == "__main__":
    main()

