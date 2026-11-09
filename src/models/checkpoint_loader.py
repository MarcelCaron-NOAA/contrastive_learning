#!/usr/bin/env python3
"""
Checkpoint-aware model loader for precipitation contrastive experiments.

Keeping model construction behind one helper lets validation scripts evaluate
BatchNorm and GroupNorm checkpoints without duplicating architecture-selection
logic.
"""

from __future__ import annotations

import torch

from src.models.precip_resnet import PrecipResNet18
from src.models.precip_resnet_gn import PrecipResNet18GN


def build_model_from_checkpoint(ckpt, device=None):
    spec = ckpt.get("model", {})
    name = spec.get("name", "PrecipResNet18")

    kwargs = {
        "projection_hidden_dim": int(
            spec.get("projection_hidden_dim", 512)
        ),
        "projection_dim": int(spec.get("projection_dim", 128)),
    }

    if name == "PrecipResNet18":
        model = PrecipResNet18(**kwargs)
    elif name == "PrecipResNet18GN":
        model = PrecipResNet18GN(**kwargs)
    else:
        raise ValueError(f"Unsupported checkpoint model name: {name!r}")

    model.load_state_dict(ckpt["model_state_dict"])

    if device is not None:
        model.to(device)

    return model


def load_model_checkpoint(path, device=None):
    ckpt = torch.load(path, map_location="cpu")
    model = build_model_from_checkpoint(ckpt, device=device)
    return model, ckpt

