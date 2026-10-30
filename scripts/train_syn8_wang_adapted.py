#!/usr/bin/env python3
"""
Train revised SYN-8: Wang-adapted synthetic precipitation similarity with eight perturbation families.

Scientific design
-----------------
This experiment deliberately keeps this project's GroupNorm ResNet-18,
projection head, HRRR 256x256 patches, log1p-zscore preprocessing, optimizer,
and one-full-anchor-epoch schedule, while adapting the synthetic supervision
much more closely to the released Wang et al. CLPFV training code.

Each underlying HRRR case contributes nine views per visit:
    O : original
    plus one independently sampled perturbation from each of eight families:
    displacement, intensity, area size, smoothing, object
    concentration/dispersion, principal-axis deformation, rotation, and
    tile fragmentation.

For each of the eight perturbations independently:
    severity rank r ~ Uniform{1,...,10}
    direction      ~ {-1,+1}, except smoothing which is one-sided

The physical bias mapping follows Wang's published/released levels:
    displacement: +/- r grid cells
    intensity decrease: 1 - 0.05*r  (0.95 ... 0.50)
    intensity increase: 1 + 0.10*r  (1.10 ... 2.00)
    area size: same factor ladder as intensity
    smoothing: +0.6*r sigma (0.6 ... 6.0; sharpening is diagnostic-only)
    object concentration/dispersion: +/-round(1.6*r) cells
    principal-axis deformation: +/-0.1*r log2 scale
    rotation: +/-9*r degrees
    tile fragmentation: +/-round(1.2*r) cells

The loss is
    L = L_SupCon + penalty_weight * L_penalty

where SupCon uses all nine views of the same underlying HRRR field as one
class, and all views from other active fields in the minibatch as negatives.
The Wang-style custom penalty is computed only original-to-perturbation:

    d_cos = 1 - cosine(z_original, z_perturbed)
    d_target(r) = 0.009 * r^2
    L_penalty = mean |d_cos - d_target|

Thus the equivalent target cosine similarity is
    s_target(r) = 1 - 0.009*r^2,
which runs from 0.991 at rank 1 to 0.100 at rank 10.

The default penalty weight is 1000, matching the released Wang training
notebook's explicit multiplication of the custom loss by 1000.  Unlike the
released notebook, we use a proper mean over valid original/perturbation pairs
before applying that explicit weight.

No separately sampled negative HRRR fields are needed: the other underlying
cases in the SupCon minibatch are the negatives.  With batch_size=8 this is
8 cases x 9 views = 72 physical model inputs per optimizer step.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.data import ContrastiveIndex, HRRRArchive, PrecipTransform
from src.models.precip_resnet_gn import PrecipResNet18GN
from src.eval.wang_transforms import displacement, intensity, area_size
from src.eval.morphology_signed_transforms import (
    smooth_or_sharpen,
    signed_object_dispersion,
    principal_axis_deform,
    rotate,
    signed_tile_fragment,
)


DEFAULT_GRID_SHAPE = (1059, 1799)
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
VIEWS_PER_CASE = 1 + len(FAMILIES)


def defaults():
    data = Path(os.environ["DATA_DIR"])
    return {
        "anchors": data / "metadata/dataset_v1/eligible_anchors_v1.csv.gz",
        "negative_pool": data / "metadata/dataset_v1/negative_pool_v1.csv.gz",
        "manifest": data / "forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz",
        "stats": data / "metadata/dataset_v1/precip_normalization_stats_v1.json",
        "output_dir": data / "derived/experiments/synthetic_perturb_syn8_v2",
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stable_u32(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little", signed=False)


def _rng_for(base_seed: int, epoch: int, stream: int, key: str = ""):
    ss = np.random.SeedSequence(
        [int(base_seed), int(epoch), int(stream), int(_stable_u32(key))]
    )
    return np.random.default_rng(ss)


def _array_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    return bool(np.all(same))


def shutdown_iterator(iterator) -> None:
    fn = getattr(iterator, "_shutdown_workers", None)
    if callable(fn):
        fn()


def loader_kwargs(dataset, sampler, workers, pin_memory, prefetch_factor):
    kw = dict(
        dataset=dataset,
        batch_sampler=sampler,
        collate_fn=syn8_collate,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    if workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = prefetch_factor
    return kw


def rank_to_target_distance(rank: torch.Tensor | np.ndarray | int | float):
    """Wang released-code target: [0, .009, .036, ..., .9] for ranks 0..10."""
    return 0.009 * (rank ** 2)


def syn8_bias(family: str, rank: int, direction: int) -> float:
    """Map rank 1..10 and signed direction to each family's physical bias.

    The three Wang families use the released CLPFV ladders exactly.  The five
    added morphology families use rank/10 of the same signed maximum magnitude
    used in this project's morphology diagnostics.  Integer-cell transforms
    are rounded to the nearest nonzero cell.
    """
    r = int(rank)
    sgn = int(direction)
    if not (1 <= r <= 10):
        raise ValueError(f"rank must be in 1..10; got {rank}")
    if sgn not in (-1, 1):
        raise ValueError(f"direction must be +/-1; got {direction}")

    if family == "displacement":
        return float(sgn * r)
    if family in ("intensity", "area_size"):
        return float(1.0 - 0.05 * r) if sgn < 0 else float(1.0 + 0.10 * r)
    if family == "smoothing":
        # Smoothing is intentionally one-sided. Sharpening is retained only
        # as an out-of-sample morphology diagnostic because the two operators
        # are not mathematical inverses and empirically show asymmetric skill.
        return float(0.6 * r)              # +6 sigma at rank 10
    if family == "object_dispersion_cells":
        mag = max(1, int(np.rint(1.6 * r)))  # +/-16 cells at rank 10
        return float(sgn * mag)
    if family == "principal_axis_log2_scale":
        return float(sgn * 0.1 * r)        # +/-1 log2 scale at rank 10
    if family == "rotation_deg":
        return float(sgn * 9.0 * r)        # +/-90 deg at rank 10
    if family == "tile_fragmentation_cells":
        mag = max(1, int(np.rint(1.2 * r)))  # +/-12 cells at rank 10
        return float(sgn * mag)
    raise KeyError(family)


def apply_transform(field: np.ndarray, family: str, bias: float, case_seed: int) -> np.ndarray:
    if family == "displacement":
        return displacement(field, int(round(bias)))
    if family == "intensity":
        return intensity(field, float(bias))
    if family == "area_size":
        return area_size(field, float(bias))
    if family == "smoothing":
        if bias <= 0:
            raise ValueError(f"smoothing bias must be positive; got {bias}")
        return smooth_or_sharpen(field, float(bias))
    if family == "object_dispersion_cells":
        return signed_object_dispersion(field, float(bias), seed=int(case_seed))
    if family == "principal_axis_log2_scale":
        return principal_axis_deform(field, float(bias))
    if family == "rotation_deg":
        return rotate(field, float(bias))
    if family == "tile_fragmentation_cells":
        return signed_tile_fragment(field, float(bias), seed=int(case_seed))
    raise KeyError(family)


@dataclass(frozen=True)
class Syn8Recipe:
    anchor_id: str
    y0: int
    x0: int
    patch_size: int
    ranks: tuple[int, ...]
    directions: tuple[int, ...]
    biases: tuple[float, ...]
    transform_seeds: tuple[int, ...]
    split: str
    anchor_valid_time: str
    anchor_forecast_hour: int
    epoch: int


class Syn8BatchSampler:
    """Deterministic one-epoch sampler of anchors, crops, ranks, and directions."""

    def __init__(
        self,
        index: ContrastiveIndex,
        *,
        split: str,
        batch_size: int = 8,
        patch_size: int = 256,
        seed: int = 20261014,
        grid_shape=DEFAULT_GRID_SHAPE,
        drop_last: bool = False,
    ):
        self.index = index
        self.split = str(split)
        self.batch_size = int(batch_size)
        self.patch_size = int(patch_size)
        self.seed = int(seed)
        self.grid_shape = tuple(int(x) for x in grid_shape)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        if self.batch_size < 2:
            raise ValueError("batch_size must be >=2 for SupCon negatives")
        ny, nx = self.grid_shape
        if self.patch_size < 1 or self.patch_size > ny or self.patch_size > nx:
            raise ValueError(f"Invalid patch_size={self.patch_size} for grid {self.grid_shape}")
        self._anchors = self.index.anchors_for_split(self.split)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def n_anchors(self) -> int:
        return len(self._anchors)

    def __len__(self) -> int:
        n = len(self._anchors)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def _recipe(self, row) -> Syn8Recipe:
        aid = str(row["anchor_id"])
        rng = _rng_for(self.seed, self.epoch, 101, aid)
        ny, nx = self.grid_shape
        y0 = int(rng.integers(0, ny - self.patch_size + 1))
        x0 = int(rng.integers(0, nx - self.patch_size + 1))

        ranks = []
        directions = []
        biases = []
        transform_seeds = []
        for family in FAMILIES:
            frng = _rng_for(self.seed, self.epoch, 103, f"{aid}|{family}")
            rank = int(frng.integers(1, 11))
            # Smoothing is one-sided by design. All other signed transforms
            # retain independently sampled negative/positive directions.
            direction = 1 if family == "smoothing" else (-1 if int(frng.integers(0, 2)) == 0 else 1)
            ranks.append(rank)
            directions.append(direction)
            biases.append(syn8_bias(family, rank, direction))
            transform_seeds.append(_stable_u32(f"{self.seed}|{self.epoch}|{aid}|{family}|transform"))

        return Syn8Recipe(
            anchor_id=aid,
            y0=y0,
            x0=x0,
            patch_size=self.patch_size,
            ranks=tuple(ranks),
            directions=tuple(directions),
            biases=tuple(biases),
            transform_seeds=tuple(transform_seeds),
            split=str(row["split"]),
            anchor_valid_time=str(row["valid_time"]),
            anchor_forecast_hour=int(row["forecast_hour"]),
            epoch=self.epoch,
        )

    def __iter__(self) -> Iterator[list[Syn8Recipe]]:
        rng = _rng_for(self.seed, self.epoch, 97, "anchor_order")
        order = rng.permutation(len(self._anchors))
        for start in range(0, len(order), self.batch_size):
            idxs = order[start:start + self.batch_size]
            if len(idxs) < self.batch_size and self.drop_last:
                continue
            yield [self._recipe(self._anchors.iloc[int(i)]) for i in idxs]


class Syn8Dataset(Dataset):
    def __init__(
        self,
        archive: HRRRArchive,
        *,
        transform: PrecipTransform,
        wet_threshold_mm: float = 0.1,
        min_anchor_wet_fraction: float = 0.001,
    ):
        self.archive = archive
        self.transform = transform
        self.wet_threshold_mm = float(wet_threshold_mm)
        self.min_anchor_wet_fraction = float(min_anchor_wet_fraction)

    def __len__(self):
        return self.archive.n_fields

    def __getitem__(self, recipe: Syn8Recipe):
        if not isinstance(recipe, Syn8Recipe):
            raise TypeError("Syn8Dataset expects Syn8Recipe indices")

        raw = np.asarray(
            self.archive.read_patch(
                recipe.anchor_id, recipe.y0, recipe.x0, recipe.patch_size
            ),
            dtype=np.float32,
        )
        wet_fraction = float(np.mean(raw >= self.wet_threshold_mm))
        active_case = wet_fraction >= self.min_anchor_wet_fraction

        transformed = []
        pair_mask = []
        for family, bias, case_seed in zip(FAMILIES, recipe.biases, recipe.transform_seeds):
            p_raw = np.asarray(apply_transform(raw, family, bias, case_seed), dtype=np.float32)
            changed = not _array_equal(raw, p_raw)
            transformed.append(self.transform(p_raw))
            pair_mask.append(bool(active_case and changed))

        views = torch.stack([self.transform(raw), *transformed], dim=0)
        ranks = torch.tensor(recipe.ranks, dtype=torch.float32)
        target_distances = 0.009 * ranks.pow(2)

        return {
            "views": views,  # [9,1,H,W] original first
            "active_case": torch.tensor(active_case, dtype=torch.bool),
            "pair_mask": torch.tensor(pair_mask, dtype=torch.bool),
            "ranks": ranks,
            "directions": torch.tensor(recipe.directions, dtype=torch.int8),
            "biases": torch.tensor(recipe.biases, dtype=torch.float32),
            "target_distances": target_distances,
            "anchor_wet_fraction": torch.tensor(wet_fraction, dtype=torch.float32),
            "metadata": asdict(recipe),
        }


def syn8_collate(items):
    if not items:
        raise ValueError("Cannot collate an empty batch")
    tensor_keys = (
        "views", "active_case", "pair_mask", "ranks", "directions",
        "biases", "target_distances", "anchor_wet_fraction",
    )
    out = {k: torch.stack([x[k] for x in items], dim=0) for k in tensor_keys}
    out["metadata"] = [x["metadata"] for x in items]
    return out


def move_batch(batch, device):
    return {
        k: batch[k].to(device, non_blocking=True)
        for k in (
            "views", "active_case", "pair_mask", "ranks", "directions",
            "biases", "target_distances", "anchor_wet_fraction",
        )
    }


def encode_views(model, batch, use_amp: bool):
    views = batch["views"]  # [B,V,1,H,W]
    B, V = views.shape[:2]
    x = views.reshape(B * V, *views.shape[2:])
    with torch.autocast(
        device_type=views.device.type,
        dtype=torch.bfloat16,
        enabled=use_amp,
    ):
        _, z = model(x)
    # Ensure all similarity and loss math is FP32 and normalized.
    z = F.normalize(z.float(), dim=-1)
    return z.reshape(B, V, -1)


def supervised_contrastive_loss(z: torch.Tensor, active_case: torch.Tensor, temperature: float):
    """Standard multi-positive SupCon over active cases and all 9 views/case."""
    active_idx = torch.where(active_case.bool())[0]
    if len(active_idx) < 2:
        zero = z.sum() * 0.0
        nan = torch.tensor(float("nan"), device=z.device)
        return zero, {"active_views": int(len(active_idx) * z.shape[1]), "retrieval_top1_pct": nan}

    zz = z[active_idx]  # [A,V,D]
    A, V, D = zz.shape
    flat = zz.reshape(A * V, D)
    labels = torch.arange(A, device=z.device).repeat_interleave(V)

    logits = torch.matmul(flat, flat.T) / float(temperature)
    self_mask = torch.eye(A * V, device=z.device, dtype=torch.bool)
    logits = logits.masked_fill(self_mask, float("-inf"))

    same = labels[:, None].eq(labels[None, :]) & ~self_mask
    log_denom = torch.logsumexp(logits, dim=1, keepdim=True)
    log_prob = logits - log_denom
    pos_count = same.sum(dim=1)
    loss_per_anchor = -(log_prob.masked_fill(~same, 0.0).sum(dim=1) / pos_count.clamp_min(1))
    loss = loss_per_anchor.mean()

    sim = torch.matmul(flat, flat.T).masked_fill(self_mask, float("-inf"))
    nn_idx = sim.argmax(dim=1)
    top1 = labels[nn_idx].eq(labels).float().mean() * 100.0
    return loss, {"active_views": int(A * V), "retrieval_top1_pct": top1.detach()}


def syn8_objective(z, batch, *, temperature: float, penalty_weight: float, supcon_weight: float):
    B, V, D = z.shape
    if V != VIEWS_PER_CASE:
        raise RuntimeError(f"Expected {VIEWS_PER_CASE} views/case; got {V}")

    supcon, smeta = supervised_contrastive_loss(z, batch["active_case"], temperature)

    zo = z[:, 0, :]
    zp = z[:, 1:, :]
    similarity = torch.einsum("bd,btd->bt", zo, zp)
    cosine_distance = 1.0 - similarity
    target_distance = batch["target_distances"].float()
    target_similarity = 1.0 - target_distance
    pair_mask = batch["pair_mask"].bool()
    abs_error = torch.abs(cosine_distance - target_distance)

    n_pairs = int(pair_mask.sum().item())
    if n_pairs <= 0:
        penalty = z.sum() * 0.0
        sim_mean = dist_mean = target_dist_mean = target_sim_mean = mae = torch.tensor(float("nan"), device=z.device)
    else:
        penalty = abs_error[pair_mask].mean()
        sim_mean = similarity[pair_mask].mean().detach()
        dist_mean = cosine_distance[pair_mask].mean().detach()
        target_dist_mean = target_distance[pair_mask].mean().detach()
        target_sim_mean = target_similarity[pair_mask].mean().detach()
        mae = abs_error[pair_mask].mean().detach()

    supcon_contribution = float(supcon_weight) * supcon
    penalty_contribution = float(penalty_weight) * penalty
    total = supcon_contribution + penalty_contribution

    per_transform = []
    for t, family in enumerate(FAMILIES):
        m = pair_mask[:, t]
        count = int(m.sum().item())
        if count:
            per_transform.append({
                "family": family,
                "count": count,
                "rank": batch["ranks"][m, t].float().mean().detach(),
                "sim": similarity[m, t].mean().detach(),
                "distance": cosine_distance[m, t].mean().detach(),
                "target_distance": target_distance[m, t].mean().detach(),
                "target_similarity": target_similarity[m, t].mean().detach(),
                "mae": abs_error[m, t].mean().detach(),
            })
        else:
            nan = torch.tensor(float("nan"), device=z.device)
            per_transform.append({
                "family": family, "count": 0, "rank": nan, "sim": nan,
                "distance": nan, "target_distance": nan,
                "target_similarity": nan, "mae": nan,
            })

    return total, {
        "supcon_raw": supcon.detach(),
        "penalty_mae": penalty.detach(),
        "supcon_contribution": supcon_contribution.detach(),
        "penalty_contribution": penalty_contribution.detach(),
        "similarity_mean": sim_mean,
        "distance_mean": dist_mean,
        "target_distance_mean": target_dist_mean,
        "target_similarity_mean": target_sim_mean,
        "target_mae": mae,
        "retrieval_top1_pct": smeta["retrieval_top1_pct"],
        "active_view_count": int(smeta["active_views"]),
        "active_case_count": int(batch["active_case"].sum().item()),
        "active_case_fraction": batch["active_case"].float().mean().detach(),
        "active_pair_count": n_pairs,
        "per_transform": per_transform,
    }


def fval(x):
    if torch.is_tensor(x):
        return float(x.detach().float().cpu())
    return float(x)


class ValidationAccumulator:
    def __init__(self):
        self.view_count = 0
        self.pair_count = 0
        self.case_count = 0
        self.supcon_sum = 0.0
        self.top1_sum = 0.0
        self.penalty_sum = 0.0
        self.sim_sum = 0.0
        self.dist_sum = 0.0
        self.target_dist_sum = 0.0
        self.target_sim_sum = 0.0
        self.pt = {
            f: {"count": 0, "rank": 0.0, "sim": 0.0, "distance": 0.0,
                "target_distance": 0.0, "target_similarity": 0.0, "mae": 0.0}
            for f in FAMILIES
        }

    def update(self, m):
        nv = int(m["active_view_count"])
        npair = int(m["active_pair_count"])
        ncase = int(m["active_case_count"])
        if nv > 0 and np.isfinite(fval(m["supcon_raw"])):
            self.view_count += nv
            self.supcon_sum += nv * fval(m["supcon_raw"])
            if np.isfinite(fval(m["retrieval_top1_pct"])):
                self.top1_sum += nv * fval(m["retrieval_top1_pct"])
        self.case_count += ncase
        if npair > 0:
            self.pair_count += npair
            self.penalty_sum += npair * fval(m["penalty_mae"])
            self.sim_sum += npair * fval(m["similarity_mean"])
            self.dist_sum += npair * fval(m["distance_mean"])
            self.target_dist_sum += npair * fval(m["target_distance_mean"])
            self.target_sim_sum += npair * fval(m["target_similarity_mean"])
        for stat in m["per_transform"]:
            f = stat["family"]
            c = int(stat["count"])
            if c <= 0:
                continue
            self.pt[f]["count"] += c
            for k in ("rank", "sim", "distance", "target_distance", "target_similarity", "mae"):
                v = fval(stat[k])
                if np.isfinite(v):
                    self.pt[f][k] += c * v

    def result(self, supcon_weight: float, penalty_weight: float):
        if self.pair_count <= 0:
            raise RuntimeError("Validation had no active original/perturbation pairs")
        supcon = self.supcon_sum / self.view_count if self.view_count else float("nan")
        top1 = self.top1_sum / self.view_count if self.view_count else float("nan")
        penalty = self.penalty_sum / self.pair_count
        out = {
            "supcon_raw": supcon,
            "penalty_mae": penalty,
            "supcon_contribution": float(supcon_weight) * supcon,
            "penalty_contribution": float(penalty_weight) * penalty,
            "total_objective": float(supcon_weight) * supcon + float(penalty_weight) * penalty,
            "similarity_mean": self.sim_sum / self.pair_count,
            "distance_mean": self.dist_sum / self.pair_count,
            "target_distance_mean": self.target_dist_sum / self.pair_count,
            "target_similarity_mean": self.target_sim_sum / self.pair_count,
            "target_mae": penalty,
            "retrieval_top1_pct": top1,
            "active_view_count": self.view_count,
            "active_pair_count": self.pair_count,
            "active_case_count": self.case_count,
        }
        for f in FAMILIES:
            c = self.pt[f]["count"]
            out[f + "_active_count"] = c
            for k in ("rank", "sim", "distance", "target_distance", "target_similarity", "mae"):
                out[f + "_" + k] = self.pt[f][k] / c if c else float("nan")
        return out


@torch.no_grad()
def evaluate_cached(model, cached_validation, device, use_amp, args):
    model.eval()
    acc = ValidationAccumulator()
    for cpu_batch in cached_validation:
        batch = move_batch(cpu_batch, device)
        z = encode_views(model, batch, use_amp)
        _, metrics = syn8_objective(
            z, batch,
            temperature=args.temperature,
            penalty_weight=args.penalty_weight,
            supcon_weight=args.supcon_weight,
        )
        acc.update(metrics)
    return acc.result(args.supcon_weight, args.penalty_weight)


def learning_rate_for_step(step, total_steps, warmup_steps, start_lr, peak_lr, min_lr):
    if step <= warmup_steps:
        if warmup_steps <= 1:
            return float(peak_lr)
        frac = (step - 1) / float(warmup_steps - 1)
        return float(start_lr + frac * (peak_lr - start_lr))
    remain = max(1, total_steps - warmup_steps)
    frac = min(1.0, max(0.0, (step - warmup_steps) / float(remain)))
    c = 0.5 * (1.0 + math.cos(math.pi * frac))
    return float(min_lr + c * (peak_lr - min_lr))


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def save_checkpoint(path, *, model, optimizer, step, args, best_val, full_epoch_steps, warmup_steps, n_train_anchors):
    torch.save(
        {
            "checkpoint_schema_version": 4,
            "step": int(step),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_validation": dict(best_val),
            "model": {
                "name": "PrecipResNet18GN",
                "normalization": "GroupNorm32",
                "projection_hidden_dim": args.projection_hidden_dim,
                "projection_dim": args.projection_dim,
                "verification_embedding": "normalized projection head",
            },
            "preprocessing": {
                "transform": "log1p_zscore",
                "stats_json": str(args.stats_json),
            },
            "experiment": {
                "name": "SYN-8 Wang-adapted v2",
                "transforms": list(FAMILIES),
                "severity": "discrete ranks 1..10",
                "target_cosine_distance": "0.009 * rank^2",
                "equivalent_target_similarity": "1 - 0.009 * rank^2",
                "global_objective": "multi-positive supervised contrastive",
                "temperature": args.temperature,
                "supcon_weight": args.supcon_weight,
                "penalty_weight": args.penalty_weight,
                "views_per_case": VIEWS_PER_CASE,
                "wet_threshold_mm": args.wet_threshold_mm,
                "min_anchor_wet_fraction": args.min_anchor_wet_fraction,
            },
            "training_sampling": {
                "split": "train",
                "batch_size_underlying_cases": args.batch_size,
                "patch_size": args.patch_size,
                "n_train_anchors": int(n_train_anchors),
                "full_epoch_steps": int(full_epoch_steps),
            },
            "validation_sampling": {
                "split": "validation",
                "validation_samples": args.validation_samples,
                "validation_seed": args.validation_seed,
            },
            "learning_rate_schedule": {
                "type": "linear_warmup_then_cosine_decay_no_restarts",
                "warmup_start_lr": args.warmup_start_lr,
                "peak_lr": args.learning_rate,
                "min_lr": args.min_learning_rate,
                "warmup_fraction": args.warmup_fraction,
                "warmup_steps": int(warmup_steps),
                "schedule_reference_steps": int(full_epoch_steps),
            },
            "seed": args.seed,
        },
        path,
    )


def row_from_metrics(step, lr, total, metrics, data_wait=None, compute_sec=None):
    row = {
        "step": int(step),
        "learning_rate": float(lr),
        "total_loss": fval(total),
        "supcon_raw_loss": fval(metrics["supcon_raw"]),
        "penalty_mae": fval(metrics["penalty_mae"]),
        "supcon_contribution": fval(metrics["supcon_contribution"]),
        "penalty_contribution": fval(metrics["penalty_contribution"]),
        "similarity_mean": fval(metrics["similarity_mean"]),
        "distance_mean": fval(metrics["distance_mean"]),
        "target_distance_mean": fval(metrics["target_distance_mean"]),
        "target_similarity_mean": fval(metrics["target_similarity_mean"]),
        "target_mae": fval(metrics["target_mae"]),
        "retrieval_top1_pct": fval(metrics["retrieval_top1_pct"]),
        "active_case_count": int(metrics["active_case_count"]),
        "active_case_fraction": fval(metrics["active_case_fraction"]),
        "active_pair_count": int(metrics["active_pair_count"]),
    }
    if data_wait is not None:
        row["data_wait_seconds"] = float(data_wait)
    if compute_sec is not None:
        row["gpu_compute_seconds"] = float(compute_sec)
    for stat in metrics["per_transform"]:
        f = stat["family"]
        row[f + "_active_count"] = int(stat["count"])
        for k in ("rank", "sim", "distance", "target_distance", "target_similarity", "mae"):
            row[f + "_" + k] = fval(stat[k])
    return row


def main():
    d = defaults()
    p = argparse.ArgumentParser()
    p.add_argument("--anchors", default=str(d["anchors"]))
    p.add_argument("--negative-pool", default=str(d["negative_pool"]), help="Loaded only because ContrastiveIndex uses the existing project metadata API; no separately sampled negatives are used in SYN-8.")
    p.add_argument("--manifest", default=str(d["manifest"]))
    p.add_argument("--stats-json", default=str(d["stats"]))
    p.add_argument("--output-dir", default=str(d["output_dir"]))
    p.add_argument("--resume-checkpoint", default=None)
    p.add_argument("--end-step", type=int, default=None)

    p.add_argument("--batch-size", type=int, default=8, help="Underlying HRRR cases per optimizer step. Physical inputs = 9x this value.")
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=2)

    p.add_argument("--wet-threshold-mm", type=float, default=0.1)
    p.add_argument("--min-anchor-wet-fraction", type=float, default=0.001)

    p.add_argument("--validation-samples", type=int, default=512)
    p.add_argument("--validate-every", type=int, default=500)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--checkpoint-every", type=int, default=1500)

    p.add_argument("--projection-hidden-dim", type=int, default=512)
    p.add_argument("--projection-dim", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.10)
    p.add_argument("--supcon-weight", type=float, default=1.0)
    p.add_argument("--penalty-weight", type=float, default=1000.0)

    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--warmup-start-lr", type=float, default=3e-5)
    p.add_argument("--warmup-fraction", type=float, default=0.05)
    p.add_argument("--min-learning-rate", type=float, default=3e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--validation-seed", type=int, default=20261015)
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    if args.batch_size < 2:
        raise SystemExit("--batch-size must be >=2")
    if args.validation_samples % args.batch_size != 0:
        raise SystemExit("--validation-samples must be divisible by --batch-size")
    if not (0 <= args.min_anchor_wet_fraction <= 1):
        raise SystemExit("--min-anchor-wet-fraction must be in [0,1]")
    if not (0 <= args.warmup_fraction < 1):
        raise SystemExit("--warmup-fraction must be in [0,1)")
    if args.temperature <= 0:
        raise SystemExit("--temperature must be >0")
    if args.supcon_weight < 0 or args.penalty_weight < 0:
        raise SystemExit("loss weights must be nonnegative")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    use_amp = not args.no_amp

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    config_path = outdir / "config.json"
    if config_path.exists() and args.resume_checkpoint is None:
        raise RuntimeError(
            f"{config_path} exists; refusing to overwrite. Use a new --output-dir or --resume-checkpoint."
        )

    index = ContrastiveIndex(args.anchors, args.negative_pool, args.manifest)
    precip_transform = PrecipTransform.from_stats_json("log1p_zscore", args.stats_json)

    train_sampler = Syn8BatchSampler(
        index, split="train", batch_size=args.batch_size, patch_size=args.patch_size,
        seed=args.seed,
    )
    train_sampler.set_epoch(0)
    val_sampler = Syn8BatchSampler(
        index, split="validation", batch_size=args.batch_size, patch_size=args.patch_size,
        seed=args.validation_seed,
    )
    val_sampler.set_epoch(0)

    full_epoch_steps = len(train_sampler)
    n_train_anchors = train_sampler.n_anchors
    target_end_step = full_epoch_steps if args.end_step is None else int(args.end_step)
    if target_end_step < 1 or target_end_step > full_epoch_steps:
        raise RuntimeError(f"target end step {target_end_step} must be in [1,{full_epoch_steps}]")
    warmup_steps = max(1, int(round(full_epoch_steps * args.warmup_fraction)))

    print("=" * 100)
    print("REVISED SYN-8 / WANG-ADAPTED GROUPNORM TRAINING")
    print("=" * 100)
    print(f"GPU                       : {torch.cuda.get_device_name(0)}")
    print(f"Output                    : {outdir}")
    print(f"Transforms                : {', '.join(FAMILIES)}")
    print("Severity                  : discrete ranks 1..10; signed direction except smoothing")
    print("Target cosine distance    : 0.009 * rank^2")
    print("Equivalent similarity     : 1 - 0.009 * rank^2 (0.991 -> 0.100)")
    print(f"Underlying cases / step   : {args.batch_size}")
    print(f"Views / case              : {VIEWS_PER_CASE}")
    print(f"Physical fields / step    : {args.batch_size * VIEWS_PER_CASE}")
    print(f"Global objective          : SupCon, temperature={args.temperature:g}")
    print(f"Loss weights              : SupCon={args.supcon_weight:g}, penalty={args.penalty_weight:g}")
    print(f"Anchor wet mask           : >= {args.min_anchor_wet_fraction:.4f} at {args.wet_threshold_mm:g} mm")
    print(f"AMP forward               : {'bf16' if use_amp else 'off'}; similarity/loss FP32")
    print(f"Train anchors             : {n_train_anchors:,}")
    print(f"Full deterministic epoch  : {full_epoch_steps:,} steps")
    print(f"Target end step           : {target_end_step:,}")
    print(f"Warmup steps              : {warmup_steps:,}")
    print()

    config = {
        **vars(args),
        "experiment_name": "SYN-8 Wang-adapted v2 (smoothing-only training)",
        "families": list(FAMILIES),
        "views_per_case": VIEWS_PER_CASE,
        "physical_fields_per_step": args.batch_size * VIEWS_PER_CASE,
        "target_distance_table": {str(r): 0.009 * r * r for r in range(1, 11)},
        "rank_to_bias": {
            "displacement": "+/- rank cells",
            "intensity": "lower=1-0.05*rank; upper=1+0.10*rank",
            "area_size": "lower=1-0.05*rank; upper=1+0.10*rank",
            "smoothing": "+0.6*rank sigma (one-sided; sharpening diagnostic-only)",
            "object_dispersion_cells": "+/-round(1.6*rank) cells",
            "principal_axis_log2_scale": "+/-0.1*rank",
            "rotation_deg": "+/-9*rank degrees",
            "tile_fragmentation_cells": "+/-round(1.2*rank) cells",
        },
        "target_similarity_table": {str(r): 1.0 - 0.009 * r * r for r in range(1, 11)},
        "full_epoch_steps": full_epoch_steps,
        "target_end_step": target_end_step,
        "n_train_anchors": n_train_anchors,
        "warmup_steps": warmup_steps,
    }
    if config_path.exists():
        old = json.loads(config_path.read_text())
        # Resume is intentionally strict about the scientific definition.
        for k in ("experiment_name", "families", "views_per_case", "target_distance_table"):
            if old.get(k) != config.get(k):
                raise RuntimeError(f"Resume config mismatch for {k}: {old.get(k)} != {config.get(k)}")
    else:
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    # Fixed validation cache.
    val_archive = HRRRArchive(args.manifest)
    val_dataset = Syn8Dataset(
        val_archive, transform=precip_transform,
        wet_threshold_mm=args.wet_threshold_mm,
        min_anchor_wet_fraction=args.min_anchor_wet_fraction,
    )
    n_val_batches = args.validation_samples // args.batch_size

    class _ValSlice:
        def __iter__(self_nonlocal):
            return itertools.islice(iter(val_sampler), n_val_batches)
        def __len__(self_nonlocal):
            return n_val_batches

    val_loader = DataLoader(
        **loader_kwargs(val_dataset, _ValSlice(), args.num_workers, True, args.prefetch_factor)
    )
    val_iter = iter(val_loader)
    cached_validation = []
    validation_recipes = []
    t0 = time.perf_counter()
    print("Building fixed synthetic validation cache...")
    try:
        for i in range(n_val_batches):
            b = next(val_iter)
            cached_validation.append({
                k: b[k].clone()
                for k in (
                    "views", "active_case", "pair_mask", "ranks", "directions",
                    "biases", "target_distances", "anchor_wet_fraction",
                )
            })
            validation_recipes.extend(b["metadata"])
            if i == 0 or (i + 1) % 16 == 0 or i + 1 == n_val_batches:
                print(f"  cached {i+1:3d}/{n_val_batches} batches")
    finally:
        shutdown_iterator(val_iter)
        val_archive.close()
        del val_iter, val_loader, val_dataset, val_archive

    recipe_path = outdir / "fixed_validation_recipes.jsonl"
    recipe_text = "".join(json.dumps(x, sort_keys=True) + "\n" for x in validation_recipes)
    if recipe_path.exists():
        if recipe_path.read_text() != recipe_text:
            raise RuntimeError("Fixed validation recipes changed on resume")
    else:
        recipe_path.write_text(recipe_text)
    print(f"Validation cache built in {time.perf_counter()-t0:.1f}s")
    print()

    seed_everything(args.seed)
    model = PrecipResNet18GN(
        projection_hidden_dim=args.projection_hidden_dim,
        projection_dim=args.projection_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    start_step = 0
    best_val = {"loss": float("inf"), "step": -1}
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt["step"])
        best_val = dict(ckpt.get("best_validation", best_val))
        print(f"Resumed from {args.resume_checkpoint} at step {start_step}")
        if start_step >= target_end_step:
            raise RuntimeError("Resume checkpoint is already at/after target end step")

    train_hist_path = outdir / "training_history.csv"
    val_hist_path = outdir / "validation_history.csv"
    train_rows = pd.read_csv(train_hist_path).to_dict("records") if train_hist_path.exists() else []
    val_rows = pd.read_csv(val_hist_path).to_dict("records") if val_hist_path.exists() else []

    if start_step == 0:
        vm0 = evaluate_cached(model, cached_validation, device, use_amp, args)
        val_rows.append({"step": 0, "learning_rate": 0.0, **vm0})
        best_val = {"loss": float(vm0["total_objective"]), "step": 0}
        print(
            f"VAL step=0 | total={vm0['total_objective']:.4f} "
            f"SupCon={vm0['supcon_raw']:.4f} "
            f"P={vm0['penalty_mae']:.4f} x{args.penalty_weight:g}={vm0['penalty_contribution']:.2f} | "
            f"d={vm0['distance_mean']:.3f} target_d={vm0['target_distance_mean']:.3f} "
            f"MAE={vm0['target_mae']:.3f} retrieval={vm0['retrieval_top1_pct']:.1f}%"
        )
        save_checkpoint(
            outdir / "checkpoint_best.pt", model=model, optimizer=optimizer, step=0,
            args=args, best_val=best_val, full_epoch_steps=full_epoch_steps,
            warmup_steps=warmup_steps, n_train_anchors=n_train_anchors,
        )
        write_csv(val_hist_path, val_rows)

    train_archive = HRRRArchive(args.manifest)
    train_dataset = Syn8Dataset(
        train_archive, transform=precip_transform,
        wet_threshold_mm=args.wet_threshold_mm,
        min_anchor_wet_fraction=args.min_anchor_wet_fraction,
    )

    base_iter = iter(train_sampler)
    recipe_iter = itertools.islice(base_iter, start_step, target_end_step)

    class _RecipeSlice:
        def __iter__(self_nonlocal):
            return recipe_iter
        def __len__(self_nonlocal):
            return target_end_step - start_step

    train_loader = DataLoader(
        **loader_kwargs(train_dataset, _RecipeSlice(), args.num_workers, True, args.prefetch_factor)
    )
    train_iter = iter(train_loader)

    run_start = time.perf_counter()
    data_wait_sum = 0.0
    compute_sum = 0.0
    completed_steps = 0

    try:
        for step in range(start_step + 1, target_end_step + 1):
            wait0 = time.perf_counter()
            cpu_batch = next(train_iter)
            data_wait = time.perf_counter() - wait0
            data_wait_sum += data_wait

            lr = learning_rate_for_step(
                step, full_epoch_steps, warmup_steps,
                args.warmup_start_lr, args.learning_rate, args.min_learning_rate,
            )
            for g in optimizer.param_groups:
                g["lr"] = lr

            compute0 = time.perf_counter()
            batch = move_batch(cpu_batch, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            z = encode_views(model, batch, use_amp)
            total, metrics = syn8_objective(
                z, batch,
                temperature=args.temperature,
                penalty_weight=args.penalty_weight,
                supcon_weight=args.supcon_weight,
            )
            total.backward()
            optimizer.step()
            compute_sec = time.perf_counter() - compute0
            compute_sum += compute_sec
            completed_steps += 1

            row = row_from_metrics(step, lr, total, metrics, data_wait, compute_sec)
            train_rows.append(row)

            if step == 1 or step % args.print_every == 0:
                print(
                    f"TRAIN step={step:5d}/{target_end_step} lr={lr:.3e} "
                    f"total={row['total_loss']:.3f} | "
                    f"SupCon={row['supcon_raw_loss']:.3f} "
                    f"P={row['penalty_mae']:.4f} x{args.penalty_weight:g}={row['penalty_contribution']:.2f} | "
                    f"sim={row['similarity_mean']:.3f} d={row['distance_mean']:.3f} "
                    f"target_d={row['target_distance_mean']:.3f} MAE={row['target_mae']:.3f} "
                    f"retrieval={row['retrieval_top1_pct']:.1f}% active_cases={row['active_case_count']}/{args.batch_size} | "
                    f"wait={data_wait:.3f}s compute={compute_sec:.3f}s"
                )

            if step % args.validate_every == 0 or step == target_end_step:
                vm = evaluate_cached(model, cached_validation, device, use_amp, args)
                val_rows.append({"step": step, "learning_rate": lr, **vm})
                print(
                    f"VAL step={step:5d} | total={vm['total_objective']:.3f} "
                    f"SupCon={vm['supcon_raw']:.3f} "
                    f"P={vm['penalty_mae']:.4f} x{args.penalty_weight:g}={vm['penalty_contribution']:.2f} | "
                    f"sim={vm['similarity_mean']:.3f} d={vm['distance_mean']:.3f} "
                    f"target_d={vm['target_distance_mean']:.3f} MAE={vm['target_mae']:.3f} "
                    f"retrieval={vm['retrieval_top1_pct']:.1f}%"
                )
                parts = []
                for f in FAMILIES:
                    parts.append(
                        f"{f}:r={vm[f+'_rank']:.2f}/S={vm[f+'_sim']:.3f}/"
                        f"d={vm[f+'_distance']:.3f}/Td={vm[f+'_target_distance']:.3f}/"
                        f"MAE={vm[f+'_mae']:.3f}"
                    )
                print("  " + " | ".join(parts))

                if vm["total_objective"] < best_val["loss"]:
                    best_val = {"loss": float(vm["total_objective"]), "step": step}
                    save_checkpoint(
                        outdir / "checkpoint_best.pt", model=model, optimizer=optimizer,
                        step=step, args=args, best_val=best_val,
                        full_epoch_steps=full_epoch_steps, warmup_steps=warmup_steps,
                        n_train_anchors=n_train_anchors,
                    )
                    print(f"  NEW BEST validation objective at step {step}")
                write_csv(val_hist_path, val_rows)

            if step % args.checkpoint_every == 0 and step < target_end_step:
                save_checkpoint(
                    outdir / f"checkpoint_step{step:05d}.pt", model=model, optimizer=optimizer,
                    step=step, args=args, best_val=best_val,
                    full_epoch_steps=full_epoch_steps, warmup_steps=warmup_steps,
                    n_train_anchors=n_train_anchors,
                )
                write_csv(train_hist_path, train_rows)

        write_csv(train_hist_path, train_rows)
        write_csv(val_hist_path, val_rows)
        save_checkpoint(
            outdir / "checkpoint_last.pt", model=model, optimizer=optimizer,
            step=target_end_step, args=args, best_val=best_val,
            full_epoch_steps=full_epoch_steps, warmup_steps=warmup_steps,
            n_train_anchors=n_train_anchors,
        )
    finally:
        shutdown_iterator(train_iter)
        train_archive.close()

    elapsed = time.perf_counter() - run_start
    summary = {
        "experiment": "SYN-8 Wang-adapted v2",
        "best_validation": best_val,
        "full_epoch_steps": full_epoch_steps,
        "target_end_step": target_end_step,
        "elapsed_seconds": elapsed,
        "mean_data_wait_seconds_per_step": data_wait_sum / max(1, completed_steps),
        "mean_gpu_compute_seconds_per_step": compute_sum / max(1, completed_steps),
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print()
    print("=" * 100)
    print("RUN COMPLETE")
    print("=" * 100)
    print(f"Best validation objective : {best_val['loss']:.6f} at step {best_val['step']}")
    print(f"Mean data wait / step     : {summary['mean_data_wait_seconds_per_step']:.3f}s")
    print(f"Mean GPU compute / step   : {summary['mean_gpu_compute_seconds_per_step']:.3f}s")
    print(f"Output                    : {outdir}")


if __name__ == "__main__":
    main()

