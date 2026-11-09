#!/usr/bin/env python3
"""Smooth partial-order ranking loss for cosine similarities."""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


class OrderedSimilarityRankingLoss(torch.nn.Module):
    """
    Enforce similarity ordering to one anchor using all ordered view pairs.

    For views [v0, v1, ..., vm] whose desired order is
        sim(a,v0) > sim(a,v1) > ... > sim(a,vm),
    every i<j contributes
        softplus((sim_j - sim_i) / temperature).

    Equality therefore has raw loss log(2), which is also returned as the
    normalization reference. Exact-identical raw field pairs can be masked by
    the dataset so impossible strict-order labels do not contribute.
    """

    def __init__(self, temperature: float = 0.10):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)

    def forward(self, z_anchor, z_views, pair_mask=None):
        """
        z_anchor: [B,D], L2-normalized
        z_views : [B,M,D], L2-normalized, ordered best -> worst
        pair_mask: optional [B, M*(M-1)/2] bool in lexicographic (i,j) order
        """
        if z_anchor.ndim != 2 or z_views.ndim != 3:
            raise ValueError("Unexpected embedding dimensions")
        if z_anchor.shape[0] != z_views.shape[0] or z_anchor.shape[1] != z_views.shape[2]:
            raise ValueError("Anchor/view embedding shapes are incompatible")

        sims = torch.einsum("bd,bmd->bm", z_anchor, z_views)
        m = sims.shape[1]
        pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]

        deltas = torch.stack(
            [(sims[:, j] - sims[:, i]) / self.temperature for i, j in pairs],
            dim=1,
        )
        losses = F.softplus(deltas)

        if pair_mask is None:
            mask = torch.ones_like(losses, dtype=torch.bool)
        else:
            if pair_mask.shape != losses.shape:
                raise ValueError(
                    f"pair_mask shape {pair_mask.shape} does not match {losses.shape}"
                )
            mask = pair_mask.bool()

        if torch.any(mask):
            loss = losses[mask].mean()
            pairwise_pct = ((deltas < 0) & mask).sum().float() / mask.sum().float() * 100.0
        else:
            loss = (z_anchor.sum() + z_views.sum()) * 0.0
            pairwise_pct = torch.tensor(float("nan"), device=z_anchor.device)

        # Adjacent ordering is the cleanest full-chain diagnostic.
        adjacent_indices = []
        for q, (i, j) in enumerate(pairs):
            if j == i + 1:
                adjacent_indices.append(q)

        adj_mask = mask[:, adjacent_indices]
        adj_good = deltas[:, adjacent_indices] < 0
        eligible_full = adj_mask.all(dim=1)
        if torch.any(eligible_full):
            full_pct = adj_good[eligible_full].all(dim=1).float().mean() * 100.0
        else:
            full_pct = torch.tensor(float("nan"), device=z_anchor.device)

        metrics = {
            "loss": loss.detach(),
            "normalized_loss": (loss / math.log(2.0)).detach(),
            "pairwise_order_pct": pairwise_pct.detach(),
            "full_order_pct": full_pct.detach(),
            "valid_pair_fraction": mask.float().mean().detach(),
            "valid_pair_count": mask.sum().detach(),
            "similarities": sims.detach(),
        }
        return loss, metrics

