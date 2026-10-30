#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.models.checkpoint_loader import load_model_checkpoint
from feature_inversion_common import (
    MM_PER_IN, default_archive_dir, default_stats_path, default_checkpoints,
    read_log1p_stats, resolve_case_index,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def log1p_param_init(field_mm: np.ndarray) -> torch.Tensor:
    field = np.maximum(np.asarray(field_mm, np.float32), 0.0)
    q = np.log1p(field).astype(np.float32)
    return torch.from_numpy(q[None, None, ...].copy())


def field_from_param(q: torch.Tensor, cap_mm: float) -> torch.Tensor:
    qmax = float(np.log1p(float(cap_mm)))
    return torch.expm1(torch.clamp(q, min=0.0, max=qmax))


def preprocess_mm(x_mm: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (torch.log1p(torch.clamp_min(x_mm, 0.0)) - float(mean)) / float(std)


def get_embedding(model, x_model: torch.Tensor, space: str) -> torch.Tensor:
    h, z = model(x_model)
    if space == "z":
        return F.normalize(z, dim=-1)
    if space == "h":
        return F.normalize(h, dim=-1)
    raise ValueError(space)


def total_variation01(x01: torch.Tensor) -> torch.Tensor:
    dy = (x01[..., 1:, :] - x01[..., :-1, :]).abs().mean()
    dx = (x01[..., :, 1:] - x01[..., :, :-1]).abs().mean()
    return dx + dy


def smooth_restart(q_start: torch.Tensor, scale: float, seed: int, cap_mm: float) -> torch.Tensor:
    """Add deterministic low-frequency noise in log1p-precipitation space.

    scale=0 returns the continuation solution exactly. Positive scales create
    alternate nearby basins without injecting pixel-scale checkerboard noise.
    The perturbed state may both add and remove precipitation.
    """
    q = q_start.detach().clone()
    if float(scale) == 0.0:
        return q

    # Generate on CPU for deterministic behavior independent of GPU details.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    coarse_h = max(4, q.shape[-2] // 16)
    coarse_w = max(4, q.shape[-1] // 16)
    coarse = torch.randn((1, 1, coarse_h, coarse_w), generator=gen, dtype=torch.float32)
    noise = F.interpolate(coarse, size=q.shape[-2:], mode="bicubic", align_corners=False)
    noise = (noise - noise.mean()) / torch.clamp(noise.std(), min=1.0e-6)
    noise = noise.to(device=q.device, dtype=q.dtype)

    qmax = float(np.log1p(float(cap_mm)))
    return torch.clamp(q + float(scale) * noise, min=0.0, max=qmax)


def candidate_key(summary: dict, target_tol: float):
    """Prefer a regularized solution inside tolerance; otherwise closest target."""
    err = float(summary["similarity_error"])
    if err <= float(target_tol):
        return (0, float(summary["loss_total"]), err)
    return (1, err, float(summary["loss_total"]))


def optimize_target_once(
    *, model, z_ref, q_start, original_mm, cap_mm, norm_mean, norm_std,
    target, n_steps, lr, lambda_tv, lambda_mass, lambda_proximity,
    space, log_every, output_dir, target_tol, early_stop_patience,
):
    q = torch.nn.Parameter(q_start.detach().clone())
    optimizer = torch.optim.Adam([q], lr=float(lr))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(n_steps)), eta_min=float(lr) * 0.10
    )

    orig_mean = original_mm.mean().detach()
    denom_mean = torch.clamp(orig_mean, min=0.10)
    orig01 = original_mm / float(cap_mm)

    best_in_tol = None
    best_closest = None
    history = []
    no_improve_in_tol = 0
    best_in_tol_total = np.inf
    t0 = time.time()

    for step in range(1, int(n_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        x_mm = field_from_param(q, cap_mm)
        z = get_embedding(model, preprocess_mm(x_mm, norm_mean, norm_std), space)
        cosine = F.cosine_similarity(z, z_ref, dim=-1).mean()

        sim_loss = (cosine - float(target)).square()
        x01 = x_mm / float(cap_mm)
        tv = total_variation01(x01)
        mass = ((x_mm.mean() - orig_mean) / denom_mean).square()
        proximity = (x01 - orig01).square().mean()
        total = (
            sim_loss
            + float(lambda_tv) * tv
            + float(lambda_mass) * mass
            + float(lambda_proximity) * proximity
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_([q], max_norm=20.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            rec = {
                "step": step,
                "target_similarity": float(target),
                "achieved_similarity": float(cosine.item()),
                "similarity_error": float(abs(cosine.item() - float(target))),
                "loss_total": float(total.item()),
                "loss_similarity": float(sim_loss.item()),
                "loss_tv": float(tv.item()),
                "loss_mass": float(mass.item()),
                "loss_proximity": float(proximity.item()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "mean_mm": float(x_mm.mean().item()),
                "max_mm": float(x_mm.max().item()),
            }
            history.append(rec)
            q_here = torch.log1p(torch.clamp_min(x_mm.detach(), 0.0))
            cand = {"record": rec.copy(), "q": q_here, "field_mm": x_mm.detach().clone()}

            if best_closest is None or (
                rec["similarity_error"], rec["loss_total"]
            ) < (
                best_closest["record"]["similarity_error"], best_closest["record"]["loss_total"]
            ):
                best_closest = cand

            if rec["similarity_error"] <= float(target_tol):
                if rec["loss_total"] < best_in_tol_total - 1.0e-9:
                    best_in_tol = cand
                    best_in_tol_total = rec["loss_total"]
                    no_improve_in_tol = 0
                else:
                    no_improve_in_tol += 1
            else:
                no_improve_in_tol = 0

        if step == 1 or step % int(log_every) == 0 or step == int(n_steps):
            print(
                f"      step {step:4d}/{n_steps} target={target:+.2f} "
                f"S={rec['achieved_similarity']:+.4f} err={rec['similarity_error']:.4f} "
                f"L={rec['loss_total']:.5f} TV={rec['loss_tv']:.5f} "
                f"mass={rec['loss_mass']:.5f}"
            )

        if (
            best_in_tol is not None
            and int(early_stop_patience) > 0
            and no_improve_in_tol >= int(early_stop_patience)
        ):
            print(f"      early stop at step {step}: target reached and regularized loss plateaued")
            break

    selected = best_in_tol if best_in_tol is not None else best_closest
    assert selected is not None

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_dir / "optimization_history.csv.gz", index=False)
    field_np = selected["field_mm"].detach().cpu().numpy()[0, 0].astype(np.float32)
    np.save(output_dir / "optimized_precip_mm.npy", field_np)

    original_np = original_mm.detach().cpu().numpy()[0, 0].astype(np.float32)
    diff_mm = field_np - original_np
    abs_diff_mm = np.abs(diff_mm)
    pos_diff = np.maximum(diff_mm, 0.0)
    neg_diff = np.maximum(-diff_mm, 0.0)
    threshold01 = 0.01 * MM_PER_IN
    orig_wet01 = original_np >= threshold01
    final_wet01 = field_np >= threshold01

    summary = {
        **selected["record"],
        "reached_target_tolerance": bool(selected["record"]["similarity_error"] <= float(target_tol)),
        "target_tolerance": float(target_tol),
        "elapsed_s": time.time() - t0,
        "steps_completed": len(history),
        "cap_mm": float(cap_mm),
        "cap_in": float(cap_mm / MM_PER_IN),
        "lambda_tv": float(lambda_tv),
        "lambda_mass": float(lambda_mass),
        "lambda_proximity": float(lambda_proximity),
        "precip_mae_mm": float(abs_diff_mm.mean()),
        "precip_rmse_mm": float(np.sqrt(np.mean(diff_mm.astype(np.float64) ** 2))),
        "precip_max_abs_change_mm": float(abs_diff_mm.max()),
        "total_added_mm_pixel_sum": float(pos_diff.sum(dtype=np.float64)),
        "total_removed_mm_pixel_sum": float(neg_diff.sum(dtype=np.float64)),
        "mean_added_mm_per_pixel": float(pos_diff.mean()),
        "mean_removed_mm_per_pixel": float(neg_diff.mean()),
        "fraction_original_wet_0p01in_eroded_below_0p01in": (
            float(np.mean(~final_wet01[orig_wet01])) if np.any(orig_wet01) else float("nan")
        ),
        "fraction_original_dry_0p01in_grown_to_0p01in": (
            float(np.mean(final_wet01[~orig_wet01])) if np.any(~orig_wet01) else float("nan")
        ),
    }
    for thr in (0.001, 0.01, 0.05, 0.10):
        summary[f"fraction_abs_change_ge_{thr:g}in"] = float(np.mean(abs_diff_mm >= thr * MM_PER_IN))

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return selected["q"].detach(), summary


def parse_targets(values):
    return [float(v) for v in values]


def main():
    dcp = default_checkpoints()
    ap = argparse.ArgumentParser(description="Dense continuation feature-space inversion with adaptive restarts.")
    ap.add_argument("--archive-dir", default=str(default_archive_dir()))
    ap.add_argument("--stats", default=str(default_stats_path()))
    ap.add_argument("--case-index", type=int, default=-1)
    ap.add_argument("--min-wet-pixels", type=int, default=64)
    ap.add_argument("--models", nargs="+", choices=["nfv", "syn3", "syn8"], default=["nfv", "syn3", "syn8"])
    ap.add_argument("--nfv-checkpoint", default=str(dcp["nfv"]))
    ap.add_argument("--syn3-checkpoint", default=str(dcp["syn3"]))
    ap.add_argument("--syn8-checkpoint", default=str(dcp["syn8"]))
    ap.add_argument(
        "--targets", nargs="+",
        default=["0.75", "0.50", "0.25", "0.00", "-0.25", "-0.50", "-0.75", "-1.00"],
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--space", choices=["z", "h"], default="z")
    ap.add_argument("--n-steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--cap-in", type=float, default=10.0)
    ap.add_argument("--lambda-tv", type=float, default=0.02)
    # Intentionally zero by default for this exploratory continuation run:
    # net precipitation may be added OR removed if that helps reach the target.
    ap.add_argument("--lambda-mass", type=float, default=0.0)
    ap.add_argument("--lambda-proximity", type=float, default=0.002)
    ap.add_argument("--target-tol", type=float, default=0.01)
    ap.add_argument("--restart-trigger", type=float, default=0.03)
    ap.add_argument("--restart-noise-scales", nargs="+", default=["0.0", "0.03", "0.08"])
    ap.add_argument("--restart-seed", type=int, default=20261016)
    ap.add_argument("--early-stop-patience", type=int, default=150)
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    targets = parse_targets(args.targets)
    noise_scales = [float(v) for v in args.restart_noise_scales]
    if not noise_scales or noise_scales[0] != 0.0:
        noise_scales = [0.0] + noise_scales

    archive = Path(args.archive_dir)
    stats_path = Path(args.stats)
    outroot = Path(args.output_dir)
    outroot.mkdir(parents=True, exist_ok=True)

    case_index, original_np, case_meta, _ = resolve_case_index(
        archive, args.case_index, min_wet_pixels=args.min_wet_pixels
    )
    original_np = np.maximum(np.asarray(original_np, dtype=np.float32), 0.0)
    if not np.isfinite(original_np).all():
        raise RuntimeError("Selected Stage-IV case contains non-finite values")

    cap_mm = max(float(args.cap_in) * MM_PER_IN, float(original_np.max()) * 1.10 + 1.0)
    norm_mean, norm_std = read_log1p_stats(stats_path)
    original_mm = torch.from_numpy(original_np[None, None, ...].copy()).to(device=device, dtype=torch.float32)
    np.save(outroot / "original_stageiv_precip_mm.npy", original_np)

    meta = {
        "case": case_meta,
        "case_index": int(case_index),
        "space": args.space,
        "targets": targets,
        "continuation": True,
        "adaptive_restarts": True,
        "restart_noise_scales_log1p": noise_scales,
        "restart_trigger_similarity_error": args.restart_trigger,
        "target_tolerance": args.target_tol,
        "n_steps_max_per_restart": args.n_steps,
        "lr": args.lr,
        "cap_in_effective": cap_mm / MM_PER_IN,
        "constraints": {
            "nonnegative": True,
            "hard_upper_cap": True,
            "net_mass_constraint": False if args.lambda_mass == 0 else True,
            "lambda_tv": args.lambda_tv,
            "lambda_mass": args.lambda_mass,
            "lambda_proximity": args.lambda_proximity,
            "note": "Precipitation may be both added and removed; positive pixels can erode all the way to zero.",
        },
        "interpretation": "Each target is a constrained local inversion, not a unique inverse image.",
    }
    (outroot / "experiment_metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")

    print("=" * 100)
    print("DENSE CONTINUATION FEATURE-SPACE INVERSION")
    print("=" * 100)
    print(f"Stage-IV case : {case_index} {case_meta.get('label', '')}")
    print(f"Original max  : {original_np.max()/MM_PER_IN:.3f} in")
    print(f"Models        : {args.models}")
    print(f"Targets       : {targets}")
    print(f"Max steps     : {args.n_steps} / restart")
    print(f"Restarts      : primary + adaptive smooth-noise scales {noise_scales[1:]}")
    print(f"Mass penalty  : {args.lambda_mass} (0 means net precipitation may freely change)")

    ckpts = {
        "nfv": Path(args.nfv_checkpoint),
        "syn3": Path(args.syn3_checkpoint),
        "syn8": Path(args.syn8_checkpoint),
    }
    all_selected_rows = []

    for model_i, model_name in enumerate(args.models):
        ckpt_path = ckpts[model_name]
        if not ckpt_path.exists():
            raise FileNotFoundError(f"{model_name} checkpoint not found: {ckpt_path}")

        print("\n" + "-" * 100)
        print(f"MODEL {model_name.upper()}: {ckpt_path}")
        model, _ = load_model_checkpoint(ckpt_path, device=device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            z_ref = get_embedding(model, preprocess_mm(original_mm, norm_mean, norm_std), args.space).detach()
            self_cos = float(F.cosine_similarity(z_ref, z_ref, dim=-1).item())
        print(f"Original self-similarity: {self_cos:.9f}")

        model_out = outroot / model_name
        model_out.mkdir(parents=True, exist_ok=True)
        (model_out / "checkpoint.json").write_text(json.dumps({
            "path": str(ckpt_path), "sha256": sha256_file(ckpt_path), "self_similarity": self_cos,
        }, indent=2) + "\n")

        q_current = log1p_param_init(original_np).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            x_init = field_from_param(q_current, cap_mm)
            init_err_mm = float((x_init - original_mm).abs().max().item())
            init_z = get_embedding(model, preprocess_mm(x_init, norm_mean, norm_std), args.space)
            init_cos = float(F.cosine_similarity(init_z, z_ref, dim=-1).item())
        print(f"Exact-init check: max |Δprecip|={init_err_mm:.9g} mm, S={init_cos:.9f}")
        if init_err_mm > 1.0e-4 or init_cos < 0.999999:
            raise RuntimeError("Exact initialization sanity check failed")

        for target_i, target in enumerate(targets):
            tag = ("p" if target >= 0 else "m") + f"{abs(target):.2f}".replace(".", "p")
            target_dir = model_out / f"target_{tag}"
            target_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n  TARGET {target:+.2f}")

            restart_results = []
            scales_to_run = [noise_scales[0]]
            # Primary continuation run first.
            for restart_i, scale in enumerate(scales_to_run):
                restart_dir = target_dir / f"restart_{restart_i:02d}"
                q_start = smooth_restart(
                    q_current, scale,
                    args.restart_seed + 10000 * model_i + 100 * target_i + restart_i,
                    cap_mm,
                )
                print(f"    Restart {restart_i}: log1p smooth-noise scale={scale:.3f}")
                q_best, summary = optimize_target_once(
                    model=model, z_ref=z_ref, q_start=q_start,
                    original_mm=original_mm, cap_mm=cap_mm,
                    norm_mean=norm_mean, norm_std=norm_std,
                    target=target, n_steps=args.n_steps, lr=args.lr,
                    lambda_tv=args.lambda_tv, lambda_mass=args.lambda_mass,
                    lambda_proximity=args.lambda_proximity,
                    space=args.space, log_every=args.log_every,
                    output_dir=restart_dir, target_tol=args.target_tol,
                    early_stop_patience=args.early_stop_patience,
                )
                summary["restart_index"] = restart_i
                summary["restart_noise_scale_log1p"] = float(scale)
                restart_results.append({"q": q_best, "summary": summary, "dir": restart_dir})

            # If the primary continuation path missed appreciably, try two or
            # more nearby low-frequency starts. This avoids spending restart
            # compute on easy targets that are already reached cleanly.
            primary_err = restart_results[0]["summary"]["similarity_error"]
            if primary_err > float(args.restart_trigger):
                for restart_i, scale in enumerate(noise_scales[1:], start=1):
                    restart_dir = target_dir / f"restart_{restart_i:02d}"
                    q_start = smooth_restart(
                        q_current, scale,
                        args.restart_seed + 10000 * model_i + 100 * target_i + restart_i,
                        cap_mm,
                    )
                    print(f"    Restart {restart_i}: log1p smooth-noise scale={scale:.3f}")
                    q_best, summary = optimize_target_once(
                        model=model, z_ref=z_ref, q_start=q_start,
                        original_mm=original_mm, cap_mm=cap_mm,
                        norm_mean=norm_mean, norm_std=norm_std,
                        target=target, n_steps=args.n_steps, lr=args.lr,
                        lambda_tv=args.lambda_tv, lambda_mass=args.lambda_mass,
                        lambda_proximity=args.lambda_proximity,
                        space=args.space, log_every=args.log_every,
                        output_dir=restart_dir, target_tol=args.target_tol,
                        early_stop_patience=args.early_stop_patience,
                    )
                    summary["restart_index"] = restart_i
                    summary["restart_noise_scale_log1p"] = float(scale)
                    restart_results.append({"q": q_best, "summary": summary, "dir": restart_dir})

            # Select the best restart. If any restart reaches tolerance, prefer
            # the cleanest regularized solution among them. Otherwise choose
            # the closest achievable similarity.
            selected = min(
                restart_results,
                key=lambda r: candidate_key(r["summary"], args.target_tol),
            )
            q_current = selected["q"].detach().clone()
            s = selected["summary"].copy()
            s["selected_restart_index"] = int(s["restart_index"])
            s["selected_restart_noise_scale_log1p"] = float(s["restart_noise_scale_log1p"])
            s["n_restarts_run"] = len(restart_results)

            shutil.copy2(selected["dir"] / "optimized_precip_mm.npy", target_dir / "optimized_precip_mm.npy")
            (target_dir / "summary.json").write_text(json.dumps(s, indent=2) + "\n")
            pd.DataFrame([r["summary"] for r in restart_results]).to_csv(
                target_dir / "restart_summary.csv", index=False
            )

            all_selected_rows.append({"model": model_name, **s})
            print(
                f"  SELECTED target={target:+.2f} achieved={s['achieved_similarity']:+.4f} "
                f"err={s['similarity_error']:.4f} restart={s['selected_restart_index']} "
                f"reached_tol={s['reached_target_tolerance']}"
            )
            print(
                f"           added={s['mean_added_mm_per_pixel']/MM_PER_IN:.5f} in/pixel, "
                f"removed={s['mean_removed_mm_per_pixel']/MM_PER_IN:.5f} in/pixel, "
                f"wet->dry@0.01in={100*s['fraction_original_wet_0p01in_eroded_below_0p01in']:.2f}%"
            )

        del model
        torch.cuda.empty_cache()

    summary_df = pd.DataFrame(all_selected_rows)
    summary_df.to_csv(outroot / "continuation_summary.csv", index=False)
    print("\n" + "=" * 100)
    print("CONTINUATION INVERSION COMPLETE")
    print("=" * 100)
    print(f"Output: {outroot}")
    print(summary_df[[
        "model", "target_similarity", "achieved_similarity", "similarity_error",
        "selected_restart_index", "n_restarts_run"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
