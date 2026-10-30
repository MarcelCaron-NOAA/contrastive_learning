#!/usr/bin/env python3
"""
Train a GroupNorm ResNet-18 precipitation similarity model with synthetic
single-axis perturbations and a Wang-style contrastive + magnitude penalty.

Each logical training sample contains:
    A                 : one original HRRR QPF6 patch
    P_t, t=1..T       : one independently perturbed copy of A for each enabled
                        transformation family
    N_k, k=1..K       : unrelated original HRRR patches sampled using the same
                        metadata-aware negative rules used by prior experiments

The optimized projection-space objective is

    L = lambda_global * L_InfoNCE / log(K+1)
        + lambda_penalty * mean_t | cos(A, P_t) - q_t |

with

    q_t = 1 - (1 - target_floor) * u_t^2,

where u_t is a dimensionless signed severity in [-1, 1].  The square makes
positive/negative directions equivalent in target severity while retaining the
sign in the actual synthetic transform.

This is deliberately Wang-inspired rather than a literal transcription of
Eq. (9) from Wang et al. (2026).  The published equation is ambiguous as a
cosine-similarity target: used literally, zero perturbation would target zero
similarity.  The parabolic target here directly encodes the behavior described
in the paper and used as the reference parabola in its perturbation figures:
identity -> 1, progressively larger errors -> lower similarity.

Profiles
--------
all8:
    displacement, intensity, area_size, smoothing_sharpening,
    object_dispersion_cells, principal_axis_log2_scale, rotation_deg,
    tile_fragmentation_cells

wang3:
    displacement, intensity, area_size

Magnitude modes
---------------
continuous:
    Samples one normalized severity u ~ Uniform(-1,1) independently for each
    enabled transformation and converts u to the physical bias range.

discrete:
    Samples one value from the canonical diagnostic/published bias levels.
    For the Wang three, these are the Wang-style levels already defined in
    src/eval/wang_transforms.py.

Only ONE magnitude per enabled transformation is materialized for an anchor on
one visit.  Thus all8 with K=4 uses B*(1+8+4) model inputs per step, not all
magnitudes at once.
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
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.data import ContrastiveIndex, NegativeRules, HRRRArchive, PrecipTransform
from src.models.precip_resnet_gn import PrecipResNet18GN
from src.eval.wang_transforms import (
    DISPLACEMENT_BIASES,
    INTENSITY_BIASES,
    AREA_SIZE_BIASES,
    displacement,
    intensity,
    area_size,
)
from src.eval.morphology_signed_transforms import (
    SIGNED_LEVELS,
    smooth_or_sharpen,
    signed_object_dispersion,
    principal_axis_deform,
    rotate,
    signed_tile_fragment,
)


DEFAULT_GRID_SHAPE = (1059, 1799)
ALL8 = (
    "displacement",
    "intensity",
    "area_size",
    "smoothing_sharpening",
    "object_dispersion_cells",
    "principal_axis_log2_scale",
    "rotation_deg",
    "tile_fragmentation_cells",
)
WANG3 = ("displacement", "intensity", "area_size")
VALID_TRANSFORMS = set(ALL8)


def defaults():
    data = Path(os.environ["DATA_DIR"])
    return {
        "anchors": data / "metadata/dataset_v1/eligible_anchors_v1.csv.gz",
        "negative_pool": data / "metadata/dataset_v1/negative_pool_v1.csv.gz",
        "manifest": data / "forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz",
        "stats": data / "metadata/dataset_v1/precip_normalization_stats_v1.json",
        "output_dir": data / "derived/experiments/synthetic_perturb_all8_v1",
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
        collate_fn=synthetic_collate,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    if workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = prefetch_factor
    return kw


def canonical_levels(family: str) -> np.ndarray:
    if family == "displacement":
        return np.asarray(DISPLACEMENT_BIASES, dtype=float)
    if family == "intensity":
        return np.asarray(INTENSITY_BIASES, dtype=float)
    if family == "area_size":
        return np.asarray(AREA_SIZE_BIASES, dtype=float)
    if family in SIGNED_LEVELS:
        return np.asarray(SIGNED_LEVELS[family], dtype=float)
    raise KeyError(family)


def identity_bias(family: str) -> float:
    if family in ("intensity", "area_size"):
        return 1.0
    return 0.0


def bias_to_severity(family: str, bias: float) -> float:
    """Map each physical bias to a comparable signed severity in [-1,1]."""
    b = float(bias)
    if family == "displacement":
        u = b / 10.0
    elif family in ("intensity", "area_size"):
        if b <= 0:
            raise ValueError(f"{family} factor must be > 0")
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


def severity_to_bias(family: str, severity: float) -> float:
    u = float(np.clip(severity, -1.0, 1.0))
    if family == "displacement":
        b = int(np.rint(10.0 * u))
        if b == 0 and not np.isclose(u, 0.0):
            b = 1 if u > 0 else -1
        return float(b)
    if family in ("intensity", "area_size"):
        return float(2.0 ** u)
    if family == "smoothing_sharpening":
        return float(6.0 * u)
    if family == "object_dispersion_cells":
        b = int(np.rint(16.0 * u))
        if b == 0 and not np.isclose(u, 0.0):
            b = 1 if u > 0 else -1
        return float(b)
    if family == "principal_axis_log2_scale":
        return u
    if family == "rotation_deg":
        return float(90.0 * u)
    if family == "tile_fragmentation_cells":
        b = int(np.rint(12.0 * u))
        if b == 0 and not np.isclose(u, 0.0):
            b = 1 if u > 0 else -1
        return float(b)
    raise KeyError(family)


def sample_bias(
    family: str,
    rng: np.random.Generator,
    magnitude_mode: str,
    include_identity_levels: bool,
) -> tuple[float, float]:
    if magnitude_mode == "continuous":
        # Probability of drawing exact identity from a continuous distribution is zero.
        u = float(rng.uniform(-1.0, 1.0))
        b = severity_to_bias(family, u)
        return float(b), bias_to_severity(family, b)

    if magnitude_mode == "discrete":
        levels = canonical_levels(family)
        if not include_identity_levels:
            ident = identity_bias(family)
            levels = levels[~np.isclose(levels, ident)]
        if len(levels) == 0:
            raise RuntimeError(f"No non-identity levels remain for {family}")
        b = float(levels[int(rng.integers(0, len(levels)))])
        return b, bias_to_severity(family, b)

    raise ValueError(f"Unknown magnitude_mode={magnitude_mode!r}")


def apply_transform(field: np.ndarray, family: str, bias: float, case_seed: int) -> np.ndarray:
    if family == "displacement":
        return displacement(field, int(round(bias)))
    if family == "intensity":
        return intensity(field, float(bias))
    if family == "area_size":
        return area_size(field, float(bias))
    if family == "smoothing_sharpening":
        return smooth_or_sharpen(field, float(bias))
    if family == "object_dispersion_cells":
        return signed_object_dispersion(field, float(bias), seed=int(case_seed))
    if family == "principal_axis_log2_scale":
        return principal_axis_deform(field, float(bias))
    if family == "rotation_deg":
        return rotate(field, float(bias))
    if family == "tile_fragmentation_cells":
        return signed_tile_fragment(field, float(bias), seed=int(case_seed))
    raise ValueError(f"Unknown transform family {family!r}")


@dataclass(frozen=True)
class SyntheticRecipe:
    anchor_id: str
    negative_ids: Tuple[str, ...]
    y0: int
    x0: int
    patch_size: int
    transform_families: Tuple[str, ...]
    transform_biases: Tuple[float, ...]
    transform_severities: Tuple[float, ...]
    transform_seeds: Tuple[int, ...]
    split: str
    anchor_valid_unix_s: int
    anchor_valid_time: str
    anchor_forecast_hour: int
    anchor_valid_hour_utc: int
    anchor_climo_day: int
    negative_candidate_count: int
    epoch: int


class SyntheticPerturbationBatchSampler:
    """Deterministic one-epoch sampler of anchors, synthetic biases, and N fields."""

    def __init__(
        self,
        index: ContrastiveIndex,
        *,
        split: str,
        transform_families: tuple[str, ...],
        magnitude_mode: str,
        include_identity_levels: bool,
        batch_size: int = 8,
        patch_size: int = 256,
        negative_count: int = 4,
        negative_rules: NegativeRules | None = None,
        seed: int = 20261007,
        grid_shape=DEFAULT_GRID_SHAPE,
        drop_last: bool = False,
    ):
        self.index = index
        self.split = str(split)
        self.transform_families = tuple(transform_families)
        self.magnitude_mode = str(magnitude_mode)
        self.include_identity_levels = bool(include_identity_levels)
        self.batch_size = int(batch_size)
        self.patch_size = int(patch_size)
        self.negative_count = int(negative_count)
        self.rules = negative_rules or NegativeRules()
        self.seed = int(seed)
        self.grid_shape = tuple(int(x) for x in grid_shape)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        unknown = set(self.transform_families) - VALID_TRANSFORMS
        if unknown:
            raise ValueError(f"Unknown transforms: {sorted(unknown)}")
        if not self.transform_families:
            raise ValueError("At least one transformation must be enabled")
        if self.batch_size < 1 or self.negative_count < 1:
            raise ValueError("batch_size and negative_count must be >=1")

        ny, nx = self.grid_shape
        if self.patch_size < 1 or self.patch_size > ny or self.patch_size > nx:
            raise ValueError(f"Invalid patch_size={self.patch_size} for grid {self.grid_shape}")

        # Use the same eligible-anchor population as prior experiments to isolate
        # objective changes rather than silently changing the training cases.
        self._anchors = self.index.anchors_for_split(self.split)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def n_anchors(self) -> int:
        return len(self._anchors)

    def __len__(self) -> int:
        n = len(self._anchors)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def _crop_and_transforms(self, row):
        aid = str(row["anchor_id"])
        crng = _rng_for(self.seed, self.epoch, 101, aid)
        ny, nx = self.grid_shape
        y0 = int(crng.integers(0, ny - self.patch_size + 1))
        x0 = int(crng.integers(0, nx - self.patch_size + 1))

        biases = []
        severities = []
        seeds = []
        for family in self.transform_families:
            frng = _rng_for(self.seed, self.epoch, 103, f"{aid}|{family}")
            bias, sev = sample_bias(
                family,
                frng,
                self.magnitude_mode,
                self.include_identity_levels,
            )
            biases.append(float(bias))
            severities.append(float(sev))
            seeds.append(int(frng.integers(0, 2**31 - 1)))
        return y0, x0, tuple(biases), tuple(severities), tuple(seeds)

    def __iter__(self) -> Iterator[list[SyntheticRecipe]]:
        order_rng = _rng_for(self.seed, self.epoch, 97, "anchor_order")
        order = order_rng.permutation(len(self._anchors))

        for start in range(0, len(order), self.batch_size):
            idxs = order[start:start + self.batch_size]
            if len(idxs) < self.batch_size and self.drop_last:
                continue
            rows = [self._anchors.iloc[int(i)] for i in idxs]

            prepared = []
            forbidden = set()
            for row in rows:
                aid = str(row["anchor_id"])
                y0, x0, biases, severities, seeds = self._crop_and_transforms(row)
                prepared.append((row, aid, y0, x0, biases, severities, seeds))
                forbidden.add(aid)

            used_negative_ids = set()
            recipes = []
            for row, aid, y0, x0, biases, severities, seeds in prepared:
                candidates = self.index.negative_candidates(row, self.rules)
                if len(candidates) < self.negative_count:
                    raise RuntimeError(
                        f"Anchor {aid} has only {len(candidates)} negative candidates; "
                        f"need {self.negative_count}."
                    )

                nrng = _rng_for(self.seed, self.epoch, 107, aid)
                perm = nrng.permutation(len(candidates))
                chosen = []
                for ii in perm:
                    nid = str(candidates.iloc[int(ii)]["negative_id"])
                    if nid in forbidden or nid in used_negative_ids:
                        continue
                    chosen.append(nid)
                    used_negative_ids.add(nid)
                    if len(chosen) == self.negative_count:
                        break
                if len(chosen) < self.negative_count:
                    raise RuntimeError(
                        f"Could not draw {self.negative_count} unique negatives for {aid}; "
                        f"got {len(chosen)}."
                    )

                recipes.append(
                    SyntheticRecipe(
                        anchor_id=aid,
                        negative_ids=tuple(chosen),
                        y0=int(y0),
                        x0=int(x0),
                        patch_size=self.patch_size,
                        transform_families=self.transform_families,
                        transform_biases=biases,
                        transform_severities=severities,
                        transform_seeds=seeds,
                        split=str(row["split"]),
                        anchor_valid_unix_s=int(row["valid_unix_s"]),
                        anchor_valid_time=str(row["valid_time"]),
                        anchor_forecast_hour=int(row["forecast_hour"]),
                        anchor_valid_hour_utc=int(row["valid_hour_utc"]),
                        anchor_climo_day=int(row["climo_day"]),
                        negative_candidate_count=int(len(candidates)),
                        epoch=self.epoch,
                    )
                )
            if recipes:
                yield recipes


class SyntheticPerturbationDataset(Dataset):
    def __init__(
        self,
        archive: HRRRArchive,
        *,
        negative_count: int,
        transform: PrecipTransform,
        wet_threshold_mm: float = 0.1,
        min_anchor_wet_fraction: float = 0.001,
    ):
        self.archive = archive
        self.negative_count = int(negative_count)
        self.transform = transform
        self.wet_threshold_mm = float(wet_threshold_mm)
        self.min_anchor_wet_fraction = float(min_anchor_wet_fraction)

    def __len__(self):
        return self.archive.n_fields

    def __getitem__(self, recipe: SyntheticRecipe):
        if not isinstance(recipe, SyntheticRecipe):
            raise TypeError("SyntheticPerturbationDataset expects SyntheticRecipe indices")
        if len(recipe.negative_ids) != self.negative_count:
            raise RuntimeError("Recipe negative count mismatch")

        a_raw = self.archive.read_patch(
            recipe.anchor_id, recipe.y0, recipe.x0, recipe.patch_size
        )
        a_raw = np.asarray(a_raw, dtype=np.float32)
        wet_fraction = float(np.mean(a_raw >= self.wet_threshold_mm))
        anchor_active = wet_fraction >= self.min_anchor_wet_fraction

        positives = []
        positive_mask = []
        for family, bias, tseed in zip(
            recipe.transform_families,
            recipe.transform_biases,
            recipe.transform_seeds,
        ):
            p_raw = np.asarray(
                apply_transform(a_raw, family, bias, int(tseed)),
                dtype=np.float32,
            )
            changed = not _array_equal(a_raw, p_raw)
            positives.append(self.transform(p_raw))
            positive_mask.append(bool(anchor_active and changed))

        negatives = []
        negative_mask = []
        for nid in recipe.negative_ids:
            n_raw = np.asarray(
                self.archive.read_patch(nid, recipe.y0, recipe.x0, recipe.patch_size),
                dtype=np.float32,
            )
            valid = not _array_equal(n_raw, a_raw)
            negatives.append(self.transform(n_raw))
            negative_mask.append(bool(valid))

        return {
            "anchor": self.transform(a_raw),
            "positives": torch.stack(positives, dim=0),
            "negatives": torch.stack(negatives, dim=0),
            "positive_mask": torch.tensor(positive_mask, dtype=torch.bool),
            "negative_mask": torch.tensor(negative_mask, dtype=torch.bool),
            "severities": torch.tensor(recipe.transform_severities, dtype=torch.float32),
            "biases": torch.tensor(recipe.transform_biases, dtype=torch.float32),
            "anchor_wet_fraction": torch.tensor(wet_fraction, dtype=torch.float32),
            "metadata": asdict(recipe),
        }


def synthetic_collate(items):
    if not items:
        raise ValueError("Cannot collate an empty batch")
    tensor_keys = (
        "anchor", "positives", "negatives", "positive_mask", "negative_mask",
        "severities", "biases", "anchor_wet_fraction",
    )
    out = {k: torch.stack([x[k] for x in items], dim=0) for k in tensor_keys}
    out["metadata"] = [x["metadata"] for x in items]
    return out


def move_batch(batch, device):
    return {
        "anchor": batch["anchor"].to(device, non_blocking=True),
        "positives": batch["positives"].to(device, non_blocking=True),
        "negatives": batch["negatives"].to(device, non_blocking=True),
        "positive_mask": batch["positive_mask"].to(device, non_blocking=True),
        "negative_mask": batch["negative_mask"].to(device, non_blocking=True),
        "severities": batch["severities"].to(device, non_blocking=True),
        "biases": batch["biases"].to(device, non_blocking=True),
        "anchor_wet_fraction": batch["anchor_wet_fraction"].to(device, non_blocking=True),
    }


def encode_batch(model, batch, use_amp: bool):
    a = batch["anchor"]              # [B,1,H,W]
    p = batch["positives"]           # [B,T,1,H,W]
    n = batch["negatives"]           # [B,K,1,H,W]
    B, T = p.shape[:2]
    K = n.shape[1]

    x = torch.cat(
        [a, p.reshape(B*T, *p.shape[2:]), n.reshape(B*K, *n.shape[2:])],
        dim=0,
    )
    with torch.autocast(
        device_type=a.device.type,
        dtype=torch.bfloat16,
        enabled=use_amp,
    ):
        _, z = model(x)

    # Similarity/loss math is intentionally FP32.
    z = F.normalize(z.float(), dim=-1)
    za = z[:B]
    zp = z[B:B+B*T].reshape(B, T, -1)
    zn = z[B+B*T:].reshape(B, K, -1)
    return za, zp, zn


def parse_transform_weights(specs: list[str] | None, families: tuple[str, ...]) -> torch.Tensor:
    weights = {f: 1.0 for f in families}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"Expected --transform-weight FAMILY=VALUE, got {spec!r}")
        name, value = spec.split("=", 1)
        name = name.strip()
        if name not in weights:
            raise ValueError(f"Weight specified for inactive/unknown transform {name!r}")
        value = float(value)
        if value <= 0:
            raise ValueError("Transform weights must be >0")
        weights[name] = value
    return torch.tensor([weights[f] for f in families], dtype=torch.float32)


def synthetic_objective(
    za,
    zp,
    zn,
    positive_mask,
    negative_mask,
    severities,
    transform_weights,
    *,
    temperature: float,
    target_floor: float,
    lambda_global: float,
    lambda_penalty: float,
    symmetric_global: bool,
):
    """Return total loss plus scalar and per-transform diagnostics."""
    B, T, D = zp.shape
    K = zn.shape[1]
    posmask = positive_mask.bool()
    negmask = negative_mask.bool()
    active = posmask & negmask.any(dim=1, keepdim=True)

    # Graph-connected zero if a pathological minibatch has no active synthetic pair.
    if not torch.any(active):
        zero = (za.sum() + zp.sum() + zn.sum()) * 0.0
        return zero, {
            "global_raw": zero.detach(),
            "global_norm": zero.detach(),
            "penalty": zero.detach(),
            "global_contribution": zero.detach(),
            "penalty_contribution": zero.detach(),
            "sim_mean": torch.tensor(float("nan"), device=za.device),
            "target_mean": torch.tensor(float("nan"), device=za.device),
            "target_mae": torch.tensor(float("nan"), device=za.device),
            "top1_pct": torch.tensor(float("nan"), device=za.device),
            "active_pair_count": active.sum().detach(),
            "active_pair_fraction": active.float().mean().detach(),
            "per_transform": [],
        }

    sim_ap = torch.einsum("bd,btd->bt", za, zp)
    sim_an = torch.einsum("bd,bkd->bk", za, zn)
    sim_pn = torch.einsum("btd,bkd->btk", zp, zn)

    very_negative = torch.finfo(sim_ap.dtype).min
    an_expand = sim_an[:, None, :].expand(B, T, K)
    nmask_expand = negmask[:, None, :].expand(B, T, K)
    an_masked = an_expand.masked_fill(~nmask_expand, very_negative)
    pn_masked = sim_pn.masked_fill(~nmask_expand, very_negative)

    logits_a = torch.cat([sim_ap[..., None], an_masked], dim=-1) / float(temperature)
    target_cls = torch.zeros(B*T, dtype=torch.long, device=za.device)
    ce_a = F.cross_entropy(logits_a.reshape(B*T, K+1), target_cls, reduction="none").reshape(B, T)

    if symmetric_global:
        logits_p = torch.cat([sim_ap[..., None], pn_masked], dim=-1) / float(temperature)
        ce_p = F.cross_entropy(logits_p.reshape(B*T, K+1), target_cls, reduction="none").reshape(B, T)
        ce = 0.5 * (ce_a + ce_p)
    else:
        ce = ce_a

    w = transform_weights.to(device=za.device, dtype=sim_ap.dtype)[None, :]
    active_w = active.float() * w
    denom = torch.clamp(active_w.sum(), min=1.0)
    global_raw = (ce * active_w).sum() / denom
    global_norm = global_raw / math.log(K + 1.0)

    target_sim = 1.0 - (1.0 - float(target_floor)) * severities.float().pow(2)
    target_sim = target_sim.clamp(-1.0, 1.0)
    abs_err = torch.abs(sim_ap - target_sim)
    penalty = (abs_err * active_w).sum() / denom

    global_contribution = float(lambda_global) * global_norm
    penalty_contribution = float(lambda_penalty) * penalty
    total = global_contribution + penalty_contribution

    max_an = an_masked.max(dim=-1).values
    top1 = sim_ap > max_an

    per_transform = []
    for t in range(T):
        m = active[:, t]
        count = int(m.sum().item())
        if count:
            per_transform.append({
                "count": count,
                "sim": sim_ap[m, t].mean().detach(),
                "target": target_sim[m, t].mean().detach(),
                "mae": abs_err[m, t].mean().detach(),
                "global_raw": ce[m, t].mean().detach(),
                "top1_pct": (top1[m, t].float().mean() * 100.0).detach(),
                "abs_severity": severities[m, t].abs().mean().detach(),
            })
        else:
            nan = torch.tensor(float("nan"), device=za.device)
            per_transform.append({
                "count": 0, "sim": nan, "target": nan, "mae": nan,
                "global_raw": nan, "top1_pct": nan, "abs_severity": nan,
            })

    all_active = active
    return total, {
        "global_raw": global_raw.detach(),
        "global_norm": global_norm.detach(),
        "penalty": penalty.detach(),
        "global_contribution": global_contribution.detach(),
        "penalty_contribution": penalty_contribution.detach(),
        "sim_mean": sim_ap[all_active].mean().detach(),
        "target_mean": target_sim[all_active].mean().detach(),
        "target_mae": abs_err[all_active].mean().detach(),
        "top1_pct": (top1[all_active].float().mean() * 100.0).detach(),
        "active_pair_count": active.sum().detach(),
        "active_pair_fraction": active.float().mean().detach(),
        "per_transform": per_transform,
    }


def fval(x):
    if torch.is_tensor(x):
        return float(x.detach().float().cpu())
    return float(x)


class ValidationAccumulator:
    def __init__(self, families):
        self.families = tuple(families)
        self.total_pairs = 0
        self.scalar_sums = {
            "global_raw": 0.0,
            "global_norm": 0.0,
            "penalty": 0.0,
            "global_contribution": 0.0,
            "penalty_contribution": 0.0,
            "sim_mean": 0.0,
            "target_mean": 0.0,
            "target_mae": 0.0,
            "top1_pct": 0.0,
        }
        self.pt = {
            f: {"count": 0, "sim": 0.0, "target": 0.0, "mae": 0.0,
                "global_raw": 0.0, "top1_pct": 0.0, "abs_severity": 0.0}
            for f in self.families
        }

    def update(self, metrics):
        n = int(metrics["active_pair_count"].item())
        if n <= 0:
            return
        self.total_pairs += n
        for k in self.scalar_sums:
            v = fval(metrics[k])
            if np.isfinite(v):
                self.scalar_sums[k] += n * v
        for fam, stat in zip(self.families, metrics["per_transform"]):
            c = int(stat["count"])
            if c <= 0:
                continue
            self.pt[fam]["count"] += c
            for k in ("sim", "target", "mae", "global_raw", "top1_pct", "abs_severity"):
                v = fval(stat[k])
                if np.isfinite(v):
                    self.pt[fam][k] += c * v

    def result(self):
        if self.total_pairs <= 0:
            raise RuntimeError("Validation had no active synthetic pairs")
        out = {k: v / self.total_pairs for k, v in self.scalar_sums.items()}
        out["active_pair_count"] = self.total_pairs
        for fam in self.families:
            c = self.pt[fam]["count"]
            prefix = fam + "_"
            out[prefix + "active_count"] = c
            for k in ("sim", "target", "mae", "global_raw", "top1_pct", "abs_severity"):
                out[prefix + k] = self.pt[fam][k] / c if c else float("nan")
        return out


@torch.no_grad()
def evaluate_cached(
    model,
    cached_validation,
    device,
    use_amp,
    transform_weights,
    args,
    families,
):
    model.eval()
    acc = ValidationAccumulator(families)
    for cpu_batch in cached_validation:
        batch = move_batch(cpu_batch, device)
        za, zp, zn = encode_batch(model, batch, use_amp)
        _, metrics = synthetic_objective(
            za, zp, zn,
            batch["positive_mask"], batch["negative_mask"], batch["severities"],
            transform_weights,
            temperature=args.temperature,
            target_floor=args.target_floor,
            lambda_global=args.lambda_global,
            lambda_penalty=args.lambda_penalty,
            symmetric_global=args.symmetric_global,
        )
        acc.update(metrics)
    out = acc.result()
    out["total_objective"] = (
        args.lambda_global * out["global_norm"]
        + args.lambda_penalty * out["penalty"]
    )
    return out


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


def save_checkpoint(
    path,
    *,
    model,
    optimizer,
    step,
    args,
    families,
    transform_weights,
    best_val,
    full_epoch_steps,
    warmup_steps,
    n_train_anchors,
):
    torch.save(
        {
            "checkpoint_schema_version": 3,
            "step": int(step),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_validation": dict(best_val),
            "model": {
                "name": "PrecipResNet18GN",
                "normalization": "GroupNorm32",
                "projection_hidden_dim": args.projection_hidden_dim,
                "projection_dim": args.projection_dim,
            },
            "preprocessing": {
                "transform": "log1p_zscore",
                "stats_json": str(args.stats_json),
            },
            "experiment": {
                "name": "synthetic_perturbation_wang_style",
                "profile": args.profile,
                "transforms": list(families),
                "magnitude_mode": args.magnitude_mode,
                "include_identity_levels": bool(args.include_identity_levels),
                "target_formula": "1 - (1-target_floor)*normalized_severity^2",
                "target_floor": float(args.target_floor),
                "transform_weights": {
                    f: float(w) for f, w in zip(families, transform_weights.tolist())
                },
                "negative_count": args.negative_count,
                "symmetric_global": bool(args.symmetric_global),
                "temperature": args.temperature,
                "lambda_global": args.lambda_global,
                "lambda_penalty": args.lambda_penalty,
                "wet_threshold_mm": args.wet_threshold_mm,
                "min_anchor_wet_fraction": args.min_anchor_wet_fraction,
            },
            "training_sampling": {
                "split": "train",
                "negative_mode": "climo45",
                "negative_min_days": args.negative_min_days,
                "climo_window_days": args.climo_window_days,
                "patch_size": args.patch_size,
                "n_train_anchors": int(n_train_anchors),
                "full_epoch_steps": int(full_epoch_steps),
            },
            "validation_sampling": {
                "split": "validation",
                "negative_mode": "calendar_any",
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


def resolve_families(args) -> tuple[str, ...]:
    if args.transforms:
        families = tuple(args.transforms)
    elif args.profile == "all8":
        families = ALL8
    elif args.profile == "wang3":
        families = WANG3
    else:
        raise ValueError(args.profile)
    if len(set(families)) != len(families):
        raise SystemExit("Duplicate transformation names are not allowed")
    unknown = set(families) - VALID_TRANSFORMS
    if unknown:
        raise SystemExit(f"Unknown transforms: {sorted(unknown)}")
    return families


def row_from_metrics(step, lr, total, metrics, families, data_wait=None, compute_sec=None):
    row = {
        "step": int(step),
        "learning_rate": float(lr),
        "total_loss": fval(total),
        "global_raw_loss": fval(metrics["global_raw"]),
        "global_norm_loss": fval(metrics["global_norm"]),
        "penalty_loss": fval(metrics["penalty"]),
        "global_contribution": fval(metrics["global_contribution"]),
        "penalty_contribution": fval(metrics["penalty_contribution"]),
        "similarity_mean": fval(metrics["sim_mean"]),
        "target_mean": fval(metrics["target_mean"]),
        "target_mae": fval(metrics["target_mae"]),
        "global_top1_pct": fval(metrics["top1_pct"]),
        "active_pair_count": int(metrics["active_pair_count"].item()),
        "active_pair_fraction": fval(metrics["active_pair_fraction"]),
    }
    if data_wait is not None:
        row["data_wait_seconds"] = float(data_wait)
    if compute_sec is not None:
        row["gpu_compute_seconds"] = float(compute_sec)
    for fam, stat in zip(families, metrics["per_transform"]):
        row[f"{fam}_active_count"] = int(stat["count"])
        for k in ("sim", "target", "mae", "global_raw", "top1_pct", "abs_severity"):
            row[f"{fam}_{k}"] = fval(stat[k])
    return row


def main():
    d = defaults()
    p = argparse.ArgumentParser()
    p.add_argument("--anchors", default=str(d["anchors"]))
    p.add_argument("--negative-pool", default=str(d["negative_pool"]))
    p.add_argument("--manifest", default=str(d["manifest"]))
    p.add_argument("--stats-json", default=str(d["stats"]))
    p.add_argument("--output-dir", default=str(d["output_dir"]))
    p.add_argument("--resume-checkpoint", default=None)
    p.add_argument("--end-step", type=int, default=None)

    p.add_argument("--profile", choices=("all8", "wang3"), default="all8")
    p.add_argument(
        "--transforms", nargs="+", default=None,
        help="Optional explicit transform list; overrides --profile.",
    )
    p.add_argument("--magnitude-mode", choices=("continuous", "discrete"), default="continuous")
    p.add_argument("--include-identity-levels", action="store_true")
    p.add_argument(
        "--transform-weight", action="append", default=None,
        help="Optional repeatable FAMILY=VALUE weighting for both global and penalty averages.",
    )

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--negative-count", type=int, default=4)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--negative-min-days", type=float, default=90.0)
    p.add_argument("--climo-window-days", type=int, default=45)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=2)

    p.add_argument("--wet-threshold-mm", type=float, default=0.1)
    p.add_argument("--min-anchor-wet-fraction", type=float, default=0.001)

    p.add_argument("--validation-samples", type=int, default=512)
    p.add_argument("--validate-every", type=int, default=500)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--checkpoint-every", type=int, default=2500)

    p.add_argument("--projection-hidden-dim", type=int, default=512)
    p.add_argument("--projection-dim", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.10)
    p.add_argument("--target-floor", type=float, default=0.0)
    p.add_argument("--lambda-global", type=float, default=1.0)
    p.add_argument("--lambda-penalty", type=float, default=1.0)
    p.add_argument("--symmetric-global", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--warmup-start-lr", type=float, default=3e-5)
    p.add_argument("--warmup-fraction", type=float, default=0.05)
    p.add_argument("--min-learning-rate", type=float, default=3e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--seed", type=int, default=20261007)
    p.add_argument("--validation-seed", type=int, default=20261008)
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    families = resolve_families(args)
    transform_weights = parse_transform_weights(args.transform_weight, families)

    if args.negative_count < 1:
        raise SystemExit("--negative-count must be >=1")
    if args.validation_samples % args.batch_size != 0:
        raise SystemExit("--validation-samples must be divisible by --batch-size")
    if not (0.0 <= args.target_floor <= 1.0):
        raise SystemExit("--target-floor must be in [0,1]")
    if not (0.0 <= args.min_anchor_wet_fraction <= 1.0):
        raise SystemExit("--min-anchor-wet-fraction must be in [0,1]")
    if not (0.0 <= args.warmup_fraction < 1.0):
        raise SystemExit("--warmup-fraction must be in [0,1)")
    if args.temperature <= 0:
        raise SystemExit("--temperature must be >0")
    if args.lambda_global < 0 or args.lambda_penalty < 0:
        raise SystemExit("loss lambdas must be nonnegative")
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
            f"{config_path} exists; refusing to overwrite. Use a new --output-dir "
            "or provide --resume-checkpoint."
        )

    print("=" * 100)
    print("SYNTHETIC PERTURBATION / WANG-STYLE GROUPNORM TRAINING")
    print("=" * 100)
    print(f"GPU                      : {torch.cuda.get_device_name(0)}")
    print(f"Output                   : {outdir}")
    print(f"Profile                  : {args.profile}")
    print(f"Transforms ({len(families):d})           : {', '.join(families)}")
    print(f"Magnitude mode           : {args.magnitude_mode}")
    print(f"Target curve             : 1 - (1-{args.target_floor:g})*u^2")
    print(f"Batch size               : {args.batch_size}")
    print(f"Negatives K              : {args.negative_count}")
    print(f"Physical fields / step   : {args.batch_size * (1 + len(families) + args.negative_count)}")
    print(f"Global InfoNCE symmetric : {args.symmetric_global}")
    print(f"Lambdas                  : global={args.lambda_global} penalty={args.lambda_penalty}")
    print(f"Anchor wet mask          : >= {args.min_anchor_wet_fraction:.4f} at {args.wet_threshold_mm:g} mm")
    print(f"AMP forward              : {'bf16' if use_amp else 'off'}; similarity/loss FP32")
    print()

    index = ContrastiveIndex(args.anchors, args.negative_pool, args.manifest)
    precip_transform = PrecipTransform.from_stats_json("log1p_zscore", args.stats_json)

    train_sampler = SyntheticPerturbationBatchSampler(
        index,
        split="train",
        transform_families=families,
        magnitude_mode=args.magnitude_mode,
        include_identity_levels=args.include_identity_levels,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        negative_count=args.negative_count,
        negative_rules=NegativeRules(
            mode="climo45",
            min_days=args.negative_min_days,
            climo_window_days=args.climo_window_days,
        ),
        seed=args.seed,
    )
    train_sampler.set_epoch(0)

    val_sampler = SyntheticPerturbationBatchSampler(
        index,
        split="validation",
        transform_families=families,
        magnitude_mode=args.magnitude_mode,
        include_identity_levels=args.include_identity_levels,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        negative_count=args.negative_count,
        negative_rules=NegativeRules(
            mode="calendar_any",
            min_days=args.negative_min_days,
            climo_window_days=args.climo_window_days,
        ),
        seed=args.validation_seed,
    )
    val_sampler.set_epoch(0)

    full_epoch_steps = len(train_sampler)
    n_train_anchors = train_sampler.n_anchors
    target_end_step = full_epoch_steps if args.end_step is None else int(args.end_step)
    if target_end_step < 1 or target_end_step > full_epoch_steps:
        raise RuntimeError(
            f"target end step {target_end_step} must be in [1,{full_epoch_steps}]"
        )
    warmup_steps = max(1, int(round(full_epoch_steps * args.warmup_fraction)))
    if warmup_steps >= full_epoch_steps:
        raise RuntimeError("Warmup consumes complete epoch")

    print(f"Train anchors             : {n_train_anchors:,}")
    print(f"Full deterministic epoch : {full_epoch_steps:,} steps")
    print(f"Target end step          : {target_end_step:,}")
    print(f"Warmup steps             : {warmup_steps:,}")
    print()

    config = {
        **vars(args),
        "transforms_resolved": list(families),
        "transform_weights_resolved": {
            f: float(w) for f, w in zip(families, transform_weights.tolist())
        },
        "full_epoch_steps": full_epoch_steps,
        "physical_fields_per_step": args.batch_size * (1 + len(families) + args.negative_count),
        "severity_definitions": {
            "displacement": "bias/10",
            "intensity": "log2(factor)",
            "area_size": "log2(scale)",
            "smoothing_sharpening": "sigma_bias/6",
            "object_dispersion_cells": "cells/16",
            "principal_axis_log2_scale": "bias",
            "rotation_deg": "degrees/90",
            "tile_fragmentation_cells": "cells/12",
        },
    }
    if config_path.exists():
        old = json.loads(config_path.read_text())
        # Do not require exact path equality for resume, but protect the core scientific contract.
        for key in (
            "profile", "transforms_resolved", "magnitude_mode", "target_floor",
            "negative_count", "batch_size", "patch_size", "lambda_global",
            "lambda_penalty", "symmetric_global",
        ):
            if old.get(key) != config.get(key):
                raise RuntimeError(f"Resume config mismatch for {key}: {old.get(key)!r} vs {config.get(key)!r}")
    else:
        config_path.write_text(json.dumps(config, indent=2, default=str) + "\n")

    # --------------------------------------------------------------
    # Fixed validation cache
    # --------------------------------------------------------------
    print("Building fixed synthetic validation cache...")
    val_archive = HRRRArchive(args.manifest)
    val_dataset = SyntheticPerturbationDataset(
        val_archive,
        negative_count=args.negative_count,
        transform=precip_transform,
        wet_threshold_mm=args.wet_threshold_mm,
        min_anchor_wet_fraction=args.min_anchor_wet_fraction,
    )
    val_loader = DataLoader(
        **loader_kwargs(val_dataset, val_sampler, args.num_workers, True, args.prefetch_factor)
    )
    val_iter = iter(val_loader)
    n_val_batches = args.validation_samples // args.batch_size
    cached_validation = []
    validation_recipes = []
    t0 = time.perf_counter()
    try:
        for i in range(n_val_batches):
            b = next(val_iter)
            cached_validation.append({
                k: b[k].clone()
                for k in (
                    "anchor", "positives", "negatives", "positive_mask", "negative_mask",
                    "severities", "biases", "anchor_wet_fraction",
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

    # --------------------------------------------------------------
    # Model / optimizer / optional resume
    # --------------------------------------------------------------
    seed_everything(args.seed)
    model = PrecipResNet18GN(
        projection_hidden_dim=args.projection_hidden_dim,
        projection_dim=args.projection_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

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

    transform_weights_device = transform_weights.to(device)

    if start_step == 0:
        vm0 = evaluate_cached(
            model, cached_validation, device, use_amp,
            transform_weights_device, args, families,
        )
        vrow0 = {"step": 0, "learning_rate": 0.0, **vm0}
        val_rows.append(vrow0)
        best_val = {"loss": float(vm0["total_objective"]), "step": 0}
        print(
            f"VAL step=0 | total={vm0['total_objective']:.4f} "
            f"G={vm0['global_norm']:.4f} P={vm0['penalty']:.4f} "
            f"sim={vm0['sim_mean']:.3f} target={vm0['target_mean']:.3f} "
            f"MAE={vm0['target_mae']:.3f}"
        )
        save_checkpoint(
            outdir / "checkpoint_best.pt",
            model=model, optimizer=optimizer, step=0, args=args,
            families=families, transform_weights=transform_weights,
            best_val=best_val, full_epoch_steps=full_epoch_steps,
            warmup_steps=warmup_steps, n_train_anchors=n_train_anchors,
        )
        write_csv(val_hist_path, val_rows)

    train_archive = HRRRArchive(args.manifest)
    train_dataset = SyntheticPerturbationDataset(
        train_archive,
        negative_count=args.negative_count,
        transform=precip_transform,
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
            za, zp, zn = encode_batch(model, batch, use_amp)
            total, metrics = synthetic_objective(
                za, zp, zn,
                batch["positive_mask"], batch["negative_mask"], batch["severities"],
                transform_weights_device,
                temperature=args.temperature,
                target_floor=args.target_floor,
                lambda_global=args.lambda_global,
                lambda_penalty=args.lambda_penalty,
                symmetric_global=args.symmetric_global,
            )
            total.backward()
            optimizer.step()
            compute_sec = time.perf_counter() - compute0
            compute_sum += compute_sec

            row = row_from_metrics(
                step, lr, total, metrics, families,
                data_wait=data_wait, compute_sec=compute_sec,
            )
            train_rows.append(row)

            if step == 1 or step % args.print_every == 0:
                print(
                    f"TRAIN step={step:5d}/{target_end_step} "
                    f"lr={lr:.3e} total={row['total_loss']:.4f} | "
                    f"G={row['global_norm_loss']:.3f} P={row['penalty_loss']:.3f} | "
                    f"sim={row['similarity_mean']:.3f} target={row['target_mean']:.3f} "
                    f"MAE={row['target_mae']:.3f} top1={row['global_top1_pct']:.1f}% | "
                    f"active={100*row['active_pair_fraction']:.1f}% "
                    f"wait={data_wait:.3f}s compute={compute_sec:.3f}s"
                )

            if step % args.validate_every == 0 or step == target_end_step:
                vm = evaluate_cached(
                    model, cached_validation, device, use_amp,
                    transform_weights_device, args, families,
                )
                vrow = {"step": step, "learning_rate": lr, **vm}
                val_rows.append(vrow)
                print(
                    f"VAL step={step:5d} | total={vm['total_objective']:.4f} "
                    f"G={vm['global_norm']:.4f} P={vm['penalty']:.4f} | "
                    f"sim={vm['sim_mean']:.3f} target={vm['target_mean']:.3f} "
                    f"MAE={vm['target_mae']:.3f} top1={vm['top1_pct']:.1f}%"
                )
                parts = []
                for fam in families:
                    parts.append(
                        f"{fam}:S={vm[fam+'_sim']:.3f}/T={vm[fam+'_target']:.3f}/"
                        f"MAE={vm[fam+'_mae']:.3f}"
                    )
                print("  " + " | ".join(parts))

                if vm["total_objective"] < best_val["loss"]:
                    best_val = {"loss": float(vm["total_objective"]), "step": step}
                    save_checkpoint(
                        outdir / "checkpoint_best.pt",
                        model=model, optimizer=optimizer, step=step, args=args,
                        families=families, transform_weights=transform_weights,
                        best_val=best_val, full_epoch_steps=full_epoch_steps,
                        warmup_steps=warmup_steps, n_train_anchors=n_train_anchors,
                    )
                    print(f"  NEW BEST validation objective at step {step}")
                write_csv(val_hist_path, val_rows)

            if step % args.checkpoint_every == 0 and step < target_end_step:
                save_checkpoint(
                    outdir / f"checkpoint_step{step:05d}.pt",
                    model=model, optimizer=optimizer, step=step, args=args,
                    families=families, transform_weights=transform_weights,
                    best_val=best_val, full_epoch_steps=full_epoch_steps,
                    warmup_steps=warmup_steps, n_train_anchors=n_train_anchors,
                )
                write_csv(train_hist_path, train_rows)

        write_csv(train_hist_path, train_rows)
        write_csv(val_hist_path, val_rows)
        save_checkpoint(
            outdir / "checkpoint_last.pt",
            model=model, optimizer=optimizer, step=target_end_step, args=args,
            families=families, transform_weights=transform_weights,
            best_val=best_val, full_epoch_steps=full_epoch_steps,
            warmup_steps=warmup_steps, n_train_anchors=n_train_anchors,
        )
    finally:
        shutdown_iterator(train_iter)
        train_archive.close()

    elapsed = time.perf_counter() - run_start
    summary = {
        "experiment": "synthetic_perturbation_wang_style",
        "profile": args.profile,
        "transforms": list(families),
        "best_validation": best_val,
        "full_epoch_steps": full_epoch_steps,
        "target_end_step": target_end_step,
        "elapsed_seconds": elapsed,
        "mean_data_wait_seconds_per_step": data_wait_sum / max(1, target_end_step - start_step),
        "mean_gpu_compute_seconds_per_step": compute_sum / max(1, target_end_step - start_step),
        "checkpoint_best": str(outdir / "checkpoint_best.pt"),
        "checkpoint_last": str(outdir / "checkpoint_last.pt"),
        "training_history": str(train_hist_path),
        "validation_history": str(val_hist_path),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

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

