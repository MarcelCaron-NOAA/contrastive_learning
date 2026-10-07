#!/usr/bin/env python3
"""
Layerwise controlled-transformation sensitivity for a trained GroupNorm model
versus an otherwise identical randomly initialized GroupNorm ResNet-18.

Purpose
-------
Separate invariances introduced by the architecture from invariances learned by
contrastive training.  The same Wang-style displacement, intensity, and area
transformations are measured after:
  stem -> layer1 -> layer2 -> layer3 -> layer4/h -> projection z

Spatial feature maps are global-average-pooled before cosine similarity so each
stage has one vector per field.  The layer4 pooled vector is exactly the
backbone h representation in this architecture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data import PrecipTransform
from src.models.checkpoint_loader import load_model_checkpoint
from src.models.precip_resnet_gn import PrecipResNet18GN
from src.eval.wang_transforms import (
    DISPLACEMENT_BIASES, INTENSITY_BIASES, AREA_SIZE_BIASES, transform,
)


LAYERS = ["stem", "layer1", "layer2", "layer3", "h", "z"]


def defaults():
    data = Path(os.environ["DATA_DIR"])
    return {
        "archive": data / "derived/evaluation/wang_transform_v1",
        "checkpoint": data / "derived/experiments/groupnorm_pilot_v1/checkpoint_best.pt",
        "stats": data / "metadata/dataset_v1/precip_normalization_stats_v1.json",
        "outdir": data / "derived/evaluation/representation_characterization_v1/layerwise_transform_sensitivity",
    }


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def poolnorm(x):
    if x.ndim == 4:
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
    return F.normalize(x.float(), dim=-1)


@torch.no_grad()
def extract_features(model, x):
    b = model.backbone
    x = b.conv1(x)
    x = b.bn1(x)  # GroupNorm in the GN model
    x = b.relu(x)
    x = b.maxpool(x)
    out = {"stem": poolnorm(x)}

    x = b.layer1(x); out["layer1"] = poolnorm(x)
    x = b.layer2(x); out["layer2"] = poolnorm(x)
    x = b.layer3(x); out["layer3"] = poolnorm(x)
    x = b.layer4(x)
    h = torch.flatten(b.avgpool(x), 1)
    out["h"] = F.normalize(h.float(), dim=-1)
    z = model.projection_head(h)
    out["z"] = F.normalize(z.float(), dim=-1)
    return out


def tensorize(raw, ptransform, device):
    return torch.stack([ptransform(x) for x in raw], dim=0).to(device, non_blocking=True)


@torch.no_grad()
def encode_all(model, raw, ptransform, device, batch_size):
    parts = {k: [] for k in LAYERS}
    for s in range(0, len(raw), batch_size):
        x = tensorize(raw[s:s+batch_size], ptransform, device)
        feat = extract_features(model, x)
        for k in LAYERS:
            parts[k].append(feat[k].cpu().numpy())
    return {k: np.concatenate(parts[k], axis=0) for k in LAYERS}


def ci95(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan, np.nan, 0
    m = float(np.mean(x))
    if len(x) == 1:
        return m, m, m, 1
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    return m, m-1.96*se, m+1.96*se, len(x)


def make_random_model(ckpt, device, seed):
    spec = ckpt.get("model", {})
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = PrecipResNet18GN(
        projection_hidden_dim=int(spec.get("projection_hidden_dim", 512)),
        projection_dim=int(spec.get("projection_dim", 128)),
    ).to(device)
    model.eval()
    return model


def plot_family(summary, family, model_kind, outdir):
    import matplotlib.pyplot as plt
    g = summary[(summary.family == family) & (summary.model_kind == model_kind)]
    plt.figure(figsize=(8.5, 5.8))
    for layer in LAYERS:
        q = g[g.layer == layer].sort_values("bias")
        plt.plot(q.bias, q.cosine_mean, marker="o", label=layer)
    plt.xlabel("bias")
    plt.ylabel("mean cosine similarity to untransformed field")
    plt.title(f"{model_kind}: layerwise response to {family}")
    plt.grid(alpha=0.2)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(outdir/f"{model_kind}_{family}_layerwise_similarity.png", dpi=190)
    plt.close()

    plt.figure(figsize=(8.5, 5.8))
    for layer in LAYERS:
        q = g[g.layer == layer].sort_values("bias")
        plt.plot(q.bias, 1.0-q.cosine_mean, marker="o", label=layer)
    plt.xlabel("bias")
    plt.ylabel("1 - mean cosine similarity")
    plt.title(f"{model_kind}: representation sensitivity to {family}")
    plt.grid(alpha=0.2)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(outdir/f"{model_kind}_{family}_layerwise_distance.png", dpi=190)
    plt.close()


def main():
    d = defaults()
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", default=str(d["archive"]))
    ap.add_argument("--checkpoint", default=str(d["checkpoint"]))
    ap.add_argument("--stats-json", default=str(d["stats"]))
    ap.add_argument("--output-dir", default=str(d["outdir"]))
    ap.add_argument("--case-count", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--case-seed", type=int, default=20261005)
    ap.add_argument("--random-model-seed", type=int, default=20261006)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required.")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    archive = Path(args.archive_dir)
    all_raw = np.load(archive/"original_patches.npz")["qpf6_mm"]
    cases = pd.read_csv(archive/"cases.csv")
    if len(all_raw) != len(cases):
        raise RuntimeError("Archive/cases length mismatch")

    rng = np.random.default_rng(args.case_seed)
    n = min(args.case_count, len(all_raw))
    idx = np.sort(rng.choice(len(all_raw), size=n, replace=False))
    raw = np.asarray(all_raw[idx], dtype=np.float32)
    cases.iloc[idx].to_csv(outdir/"selected_cases.csv", index=False)

    device = torch.device("cuda")
    trained, ckpt = load_model_checkpoint(args.checkpoint, device=device)
    trained.eval()
    if ckpt.get("model", {}).get("name") != "PrecipResNet18GN":
        raise RuntimeError("This experiment expects a GroupNorm checkpoint.")
    random_model = make_random_model(ckpt, device, args.random_model_seed)
    ptransform = PrecipTransform.from_stats_json("log1p_zscore", args.stats_json)

    print(f"cases={n} | checkpoint step={ckpt.get('step',-1)}")
    print("Encoding untransformed fields...")
    base = {
        "trained": encode_all(trained, raw, ptransform, device, args.batch_size),
        "random": encode_all(random_model, raw, ptransform, device, args.batch_size),
    }

    families = [
        ("displacement", DISPLACEMENT_BIASES),
        ("intensity", INTENSITY_BIASES),
        ("area_size", AREA_SIZE_BIASES),
    ]
    models = {"trained": trained, "random": random_model}
    records = []

    for family, biases in families:
        print(f"\n{family}: {len(biases)} biases")
        for ib, bias in enumerate(biases):
            # Generate/infer in batches so the 2000-case archive is never duplicated in RAM.
            sim_parts = {
                mk: {layer: [] for layer in LAYERS}
                for mk in models
            }
            for s in range(0, n, args.batch_size):
                e = min(n, s+args.batch_size)
                transformed = np.stack([
                    transform(x, family, float(bias)) for x in raw[s:e]
                ]).astype(np.float32)
                tx = tensorize(transformed, ptransform, device)
                for mk, model in models.items():
                    feat = extract_features(model, tx)
                    for layer in LAYERS:
                        b0 = base[mk][layer][s:e]
                        ft = feat[layer].cpu().numpy()
                        sim_parts[mk][layer].append(np.sum(b0*ft, axis=1))

            for mk in models:
                for layer in LAYERS:
                    sims = np.concatenate(sim_parts[mk][layer])
                    for case_pos, sim in enumerate(sims):
                        records.append({
                            "model_kind": mk,
                            "family": family,
                            "bias": float(bias),
                            "layer": layer,
                            "case_position": int(case_pos),
                            "archive_case_index": int(idx[case_pos]),
                            "cosine_similarity": float(sim),
                        })
            print(f"  {ib+1:2d}/{len(biases)} bias={float(bias):g}")

    per_case = pd.DataFrame(records)
    per_case.to_csv(outdir/"layerwise_transform_per_case.csv.gz", index=False)

    summary_rows = []
    for (mk, family, bias, layer), g in per_case.groupby(
        ["model_kind","family","bias","layer"], sort=False
    ):
        mean, lo, hi, nn = ci95(g.cosine_similarity.to_numpy())
        summary_rows.append({
            "model_kind": mk,
            "family": family,
            "bias": bias,
            "layer": layer,
            "cosine_mean": mean,
            "cosine_ci95_lo": lo,
            "cosine_ci95_hi": hi,
            "cosine_median": float(np.nanmedian(g.cosine_similarity)),
            "n": nn,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir/"layerwise_transform_summary.csv", index=False)

    for family, _ in families:
        for mk in models:
            plot_family(summary, family, mk, outdir)

    meta = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": int(ckpt.get("step", -1)),
        "case_count": n,
        "case_seed": args.case_seed,
        "random_model_seed": args.random_model_seed,
        "layers": LAYERS,
        "spatial_layer_reduction": "global average pooling then L2 normalization",
        "note": (
            "If an invariance appears similarly in the random and trained networks, "
            "the architecture/preprocessing is implicated. If it emerges primarily in "
            "later trained layers, the contrastive objective is implicated."
        ),
    }
    (outdir/"run_metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(f"\nOutput: {outdir}")


if __name__ == "__main__":
    main()

