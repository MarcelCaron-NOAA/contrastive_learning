#!/usr/bin/env python3
"""
Explicit-negative symmetric InfoNCE for natural HRRR A/P/N relationships.

Only explicitly sampled negatives are used. Other minibatch examples are not
silently turned into negatives.

Dry-field edge case:
If every N for one logical sample is masked because it is exactly identical to
A or P, that sample contributes no contrastive loss on that visit. Other
samples in the minibatch continue normally.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SymmetricExplicitInfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.10):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)

    def forward(self, z_a, z_p, z_n, negative_mask):
        """
        z_a: [B,D] L2-normalized
        z_p: [B,D] L2-normalized
        z_n: [B,K,D] L2-normalized
        negative_mask: [B,K] bool
        """
        if z_a.ndim != 2 or z_p.ndim != 2 or z_n.ndim != 3:
            raise ValueError("Unexpected embedding dimensions")
        if z_a.shape != z_p.shape:
            raise ValueError("A and P embeddings must have the same shape")
        if z_n.shape[0] != z_a.shape[0] or z_n.shape[2] != z_a.shape[1]:
            raise ValueError("N embedding shape is incompatible with A/P")
        if negative_mask.shape != z_n.shape[:2]:
            raise ValueError("negative_mask shape does not match N")

        mask = negative_mask.bool()
        active = mask.any(dim=1)

        # Preserve a graph-connected zero loss for the pathological case where
        # the whole minibatch has no usable negatives.
        if not torch.any(active):
            zero = (z_a.sum() + z_p.sum() + z_n.sum()) * 0.0
            nan = torch.tensor(float("nan"), device=z_a.device)
            metrics = {
                "loss": zero.detach(),
                "cosine_ap": nan,
                "cosine_an": nan,
                "cosine_pn": nan,
                "margin_a": nan,
                "margin_p": nan,
                "top1_a_pct": nan,
                "top1_p_pct": nan,
                "valid_negative_fraction": mask.float().mean().detach(),
                "active_query_fraction": active.float().mean().detach(),
                "active_query_count": active.sum().detach(),
            }
            return zero, metrics

        # Samples with zero valid negatives contribute no loss this visit.
        za = z_a[active]
        zp = z_p[active]
        zn = z_n[active]
        m = mask[active]

        sim_ap = torch.sum(za * zp, dim=-1)
        sim_an = torch.einsum("bd,bkd->bk", za, zn)
        sim_pn = torch.einsum("bd,bkd->bk", zp, zn)

        very_negative = torch.finfo(sim_an.dtype).min
        sim_an_masked = sim_an.masked_fill(~m, very_negative)
        sim_pn_masked = sim_pn.masked_fill(~m, very_negative)

        logits_a = torch.cat(
            [sim_ap[:, None], sim_an_masked], dim=1
        ) / self.temperature
        logits_p = torch.cat(
            [sim_ap[:, None], sim_pn_masked], dim=1
        ) / self.temperature

        target = torch.zeros(
            za.shape[0], dtype=torch.long, device=za.device
        )

        loss_a = F.cross_entropy(logits_a, target)
        loss_p = F.cross_entropy(logits_p, target)
        loss = 0.5 * (loss_a + loss_p)

        valid_an = sim_an[m]
        valid_pn = sim_pn[m]
        max_an = sim_an_masked.max(dim=1).values
        max_pn = sim_pn_masked.max(dim=1).values

        metrics = {
            "loss": loss.detach(),
            "cosine_ap": sim_ap.mean().detach(),
            "cosine_an": valid_an.mean().detach(),
            "cosine_pn": valid_pn.mean().detach(),
            "margin_a": (sim_ap - max_an).mean().detach(),
            "margin_p": (sim_ap - max_pn).mean().detach(),
            "top1_a_pct": (
                (sim_ap > max_an).float().mean() * 100.0
            ).detach(),
            "top1_p_pct": (
                (sim_ap > max_pn).float().mean() * 100.0
            ).detach(),
            "valid_negative_fraction": mask.float().mean().detach(),
            "active_query_fraction": active.float().mean().detach(),
            "active_query_count": active.sum().detach(),
        }
        return loss, metrics

