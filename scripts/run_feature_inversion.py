#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
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
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def log1p_param_init(field_mm: np.ndarray) -> torch.Tensor:
    """Exact initialization in log1p-precipitation space.

    Unlike the previous inverse-sigmoid parameterization, this preserves exact
    zeros exactly: 0 mm -> log1p(0) = 0 -> expm1(0) = 0.
    """
    field = np.maximum(np.asarray(field_mm, np.float32), 0.0)
    q = np.log1p(field).astype(np.float32)
    return torch.from_numpy(q[None, None, ...].copy())


def field_from_param(q: torch.Tensor, cap_mm: float) -> torch.Tensor:
    """Map an optimizable log1p field to bounded nonnegative precipitation.

    The hard bounds are applied in log1p space.  At q=log1p(original), the
    reconstructed field is exactly the original field, including exact zeros.
    """
    qmax = float(np.log1p(float(cap_mm)))
    q_bounded = torch.clamp(q, min=0.0, max=qmax)
    return torch.expm1(q_bounded)


def preprocess_mm(x_mm: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (torch.log1p(torch.clamp_min(x_mm, 0.0)) - float(mean)) / float(std)


def get_embedding(model, x_model: torch.Tensor, space: str) -> torch.Tensor:
    h, z = model(x_model)
    if space == 'z':
        return F.normalize(z, dim=-1)
    if space == 'h':
        return F.normalize(h, dim=-1)
    raise ValueError(space)


def total_variation01(x01: torch.Tensor) -> torch.Tensor:
    dy = (x01[..., 1:, :] - x01[..., :-1, :]).abs().mean()
    dx = (x01[..., :, 1:] - x01[..., :, :-1]).abs().mean()
    return dx + dy


def optimize_target(
    *, model, z_ref, q_start, original_mm, cap_mm, norm_mean, norm_std,
    target, n_steps, lr, lambda_tv, lambda_mass, lambda_proximity,
    space, log_every, output_dir,
):
    device = q_start.device
    q = torch.nn.Parameter(q_start.detach().clone())
    optimizer = torch.optim.Adam([q], lr=float(lr))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(n_steps)), eta_min=float(lr) * 0.10)

    orig_mean = original_mm.mean().detach()
    denom_mean = torch.clamp(orig_mean, min=0.10)
    orig01 = original_mm / float(cap_mm)

    best = None
    history = []
    t0 = time.time()

    for step in range(1, int(n_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        x_mm = field_from_param(q, cap_mm)
        x_model = preprocess_mm(x_mm, norm_mean, norm_std)
        z = get_embedding(model, x_model, space)
        cosine = F.cosine_similarity(z, z_ref, dim=-1).mean()

        sim_loss = (cosine - float(target)).square()
        x01 = x_mm / float(cap_mm)
        tv = total_variation01(x01)
        mass = ((x_mm.mean() - orig_mean) / denom_mean).square()
        proximity = (x01 - orig01).square().mean()
        total = sim_loss + float(lambda_tv) * tv + float(lambda_mass) * mass + float(lambda_proximity) * proximity
        total.backward()
        torch.nn.utils.clip_grad_norm_([q], max_norm=20.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            rec = {
                'step': step,
                'target_similarity': float(target),
                'achieved_similarity': float(cosine.item()),
                'similarity_error': float(abs(cosine.item() - float(target))),
                'loss_total': float(total.item()),
                'loss_similarity': float(sim_loss.item()),
                'loss_tv': float(tv.item()),
                'loss_mass': float(mass.item()),
                'loss_proximity': float(proximity.item()),
                'lr': float(optimizer.param_groups[0]['lr']),
                'mean_mm': float(x_mm.mean().item()),
                'max_mm': float(x_mm.max().item()),
            }
            history.append(rec)
            if best is None or rec['loss_total'] < best['record']['loss_total']:
                # rec/x_mm describe the parameter state BEFORE optimizer.step().
                # Reconstruct that state by using the current bounded field itself as
                # the next log1p starting point; this keeps continuation consistent
                # with the saved best image.
                q_best = torch.log1p(torch.clamp_min(x_mm.detach(), 0.0))
                best = {'record': rec.copy(), 'q': q_best, 'field_mm': x_mm.detach().clone()}

        if step == 1 or step % int(log_every) == 0 or step == int(n_steps):
            print(
                f"    step {step:4d}/{n_steps} target={target:+.2f} "
                f"S={rec['achieved_similarity']:+.4f} err={rec['similarity_error']:.4f} "
                f"L={rec['loss_total']:.5f} sim={rec['loss_similarity']:.5f} "
                f"TV={rec['loss_tv']:.5f} mass={rec['loss_mass']:.5f}"
            )

    assert best is not None
    pd.DataFrame(history).to_csv(output_dir / 'optimization_history.csv.gz', index=False)
    best_mm = best['field_mm'].detach().cpu().numpy()[0, 0].astype(np.float32)
    np.save(output_dir / 'optimized_precip_mm.npy', best_mm)

    original_np = original_mm.detach().cpu().numpy()[0, 0].astype(np.float32)
    diff_mm = best_mm - original_np
    abs_diff_mm = np.abs(diff_mm)
    change_thresholds_in = (0.001, 0.01, 0.05, 0.10)
    change_fracs = {
        f'fraction_abs_change_ge_{thr:g}in': float(np.mean(abs_diff_mm >= thr * MM_PER_IN))
        for thr in change_thresholds_in
    }
    original_dry = original_np <= 0.0
    newly_wet = {}
    for thr in (0.001, 0.01, 0.10):
        newly_wet[f'fraction_originally_dry_now_ge_{thr:g}in'] = (
            float(np.mean(best_mm[original_dry] >= thr * MM_PER_IN))
            if np.any(original_dry) else float('nan')
        )

    summary = {
        **best['record'],
        'elapsed_s': time.time() - t0,
        'cap_mm': float(cap_mm),
        'cap_in': float(cap_mm / MM_PER_IN),
        'lambda_tv': float(lambda_tv),
        'lambda_mass': float(lambda_mass),
        'lambda_proximity': float(lambda_proximity),
        'n_steps': int(n_steps),
        'learning_rate_initial': float(lr),
        'precip_mae_mm': float(abs_diff_mm.mean()),
        'precip_rmse_mm': float(np.sqrt(np.mean(diff_mm.astype(np.float64) ** 2))),
        'precip_max_abs_change_mm': float(abs_diff_mm.max()),
        **change_fracs,
        **newly_wet,
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return best['q'].detach(), summary


def parse_targets(values):
    return [float(v) for v in values]


def main():
    dcp = default_checkpoints()
    ap = argparse.ArgumentParser(description='Constrained feature-space inversion of Stage-IV precipitation.')
    ap.add_argument('--archive-dir', default=str(default_archive_dir()))
    ap.add_argument('--stats', default=str(default_stats_path()))
    ap.add_argument('--case-index', type=int, default=-1, help='Stage-IV archive case index; -1 auto-selects a high-impact centered case.')
    ap.add_argument('--min-wet-pixels', type=int, default=64)
    ap.add_argument('--models', nargs='+', choices=['nfv','syn3','syn8'], default=['syn3'])
    ap.add_argument('--nfv-checkpoint', default=str(dcp['nfv']))
    ap.add_argument('--syn3-checkpoint', default=str(dcp['syn3']))
    ap.add_argument('--syn8-checkpoint', default=str(dcp['syn8']))
    ap.add_argument('--targets', nargs='+', default=['0.5','0.0','-0.5','-1.0'])
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--space', choices=['z','h'], default='z')
    ap.add_argument('--n-steps', type=int, default=800)
    ap.add_argument('--lr', type=float, default=0.02)
    ap.add_argument('--cap-in', type=float, default=10.0)
    ap.add_argument('--lambda-tv', type=float, default=0.02)
    ap.add_argument('--lambda-mass', type=float, default=0.01)
    ap.add_argument('--lambda-proximity', type=float, default=0.002)
    ap.add_argument('--log-every', type=int, default=100)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU required')
    device = torch.device('cuda')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

    archive = Path(args.archive_dir)
    stats_path = Path(args.stats)
    outroot = Path(args.output_dir)
    outroot.mkdir(parents=True, exist_ok=True)

    case_index, original_np, case_meta, _ = resolve_case_index(
        archive, args.case_index, min_wet_pixels=args.min_wet_pixels
    )
    original_np = np.asarray(original_np, dtype=np.float32)
    if not np.isfinite(original_np).all():
        raise RuntimeError('Selected Stage-IV case contains non-finite values')
    original_np = np.maximum(original_np, 0.0)

    cap_mm = max(float(args.cap_in) * MM_PER_IN, float(original_np.max()) * 1.10 + 1.0)
    norm_mean, norm_std = read_log1p_stats(stats_path)
    original_mm = torch.from_numpy(original_np[None, None, ...].copy()).to(device=device, dtype=torch.float32)
    np.save(outroot / 'original_stageiv_precip_mm.npy', original_np)

    experiment_meta = {
        'case': case_meta,
        'archive_dir': str(archive),
        'stats': str(stats_path),
        'space': args.space,
        'targets': parse_targets(args.targets),
        'cap_in_requested': args.cap_in,
        'cap_in_effective': cap_mm / MM_PER_IN,
        'constraints': {
            'nonnegative': True,
            'hard_upper_bound_in_log1p_space': True,
            'exact_zero_preserving_initialization': True,
            'lambda_tv': args.lambda_tv,
            'lambda_mass': args.lambda_mass,
            'lambda_proximity': args.lambda_proximity,
        },
        'optimizer': {'name':'Adam','lr':args.lr,'n_steps':args.n_steps,'scheduler':'CosineAnnealingLR'},
        'note': 'Feature inversion is non-unique. Each output is one constrained local solution, not THE physical inverse of the embedding. Initialization exactly reproduces the original precipitation field, including zero-valued pixels.',
    }
    (outroot / 'experiment_metadata.json').write_text(json.dumps(experiment_meta, indent=2, default=str) + '\n')

    print('=' * 96)
    print('FEATURE-SPACE INVERSION')
    print('=' * 96)
    print(f"Stage-IV case : {case_index} {case_meta.get('label','')}")
    print(f"Original max  : {original_np.max()/MM_PER_IN:.3f} in")
    print(f"Effective cap : {cap_mm/MM_PER_IN:.3f} in")
    print(f"Target space  : {args.space}")
    print(f"Models        : {args.models}")
    print(f"Targets       : {parse_targets(args.targets)}")
    print(f"Iterations    : {args.n_steps} / target")
    print()

    ckpts = {
        'nfv': Path(args.nfv_checkpoint),
        'syn3': Path(args.syn3_checkpoint),
        'syn8': Path(args.syn8_checkpoint),
    }
    all_rows = []

    for model_name in args.models:
        ckpt_path = ckpts[model_name]
        if not ckpt_path.exists():
            raise FileNotFoundError(f'{model_name} checkpoint not found: {ckpt_path}')
        print('-' * 96)
        print(f'MODEL {model_name.upper()}: {ckpt_path}')
        model, _ = load_model_checkpoint(ckpt_path, device=device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            x0_model = preprocess_mm(original_mm, norm_mean, norm_std)
            z_ref = get_embedding(model, x0_model, args.space).detach()
            self_cos = float(F.cosine_similarity(z_ref, z_ref, dim=-1).item())
        print(f'Original self-similarity: {self_cos:.6f}')

        model_out = outroot / model_name
        model_out.mkdir(parents=True, exist_ok=True)
        (model_out / 'checkpoint.json').write_text(json.dumps({
            'path': str(ckpt_path), 'sha256': sha256_file(ckpt_path), 'self_similarity': self_cos
        }, indent=2) + '\n')

        # Continuation path: each lower target starts from the previous solution.
        # Exact log1p initialization preserves zero-valued pixels as exactly zero.
        q_current = log1p_param_init(original_np).to(device=device, dtype=torch.float32)

        # Mandatory sanity check: before any optimization, reconstruction and
        # similarity must reproduce the original essentially exactly.
        with torch.no_grad():
            x_init = field_from_param(q_current, cap_mm)
            max_init_abs_mm = float((x_init - original_mm).abs().max().item())
            init_z = get_embedding(model, preprocess_mm(x_init, norm_mean, norm_std), args.space)
            init_cos = float(F.cosine_similarity(init_z, z_ref, dim=-1).item())
        print(f'Exact-init check: max |Δprecip|={max_init_abs_mm:.9g} mm, S={init_cos:.9f}')
        if max_init_abs_mm > 1.0e-4 or init_cos < 0.999999:
            raise RuntimeError(
                f'Feature-inversion initialization does not reproduce the original: '
                f'max_abs_mm={max_init_abs_mm}, similarity={init_cos}'
            )
        for target in parse_targets(args.targets):
            tag = ('p' if target >= 0 else 'm') + f'{abs(target):.2f}'.replace('.', 'p')
            target_dir = model_out / f'target_{tag}'
            target_dir.mkdir(parents=True, exist_ok=True)
            print(f'  Target {target:+.2f}')
            q_current, summary = optimize_target(
                model=model, z_ref=z_ref, q_start=q_current,
                original_mm=original_mm, cap_mm=cap_mm,
                norm_mean=norm_mean, norm_std=norm_std,
                target=target, n_steps=args.n_steps, lr=args.lr,
                lambda_tv=args.lambda_tv, lambda_mass=args.lambda_mass,
                lambda_proximity=args.lambda_proximity,
                space=args.space, log_every=args.log_every,
                output_dir=target_dir,
            )
            all_rows.append({'model':model_name, **summary})
            print(
                f"  BEST target={target:+.2f} achieved={summary['achieved_similarity']:+.4f} "
                f"err={summary['similarity_error']:.4f} max={summary['max_mm']/MM_PER_IN:.2f} in"
            )
            print(
                f"       precip MAE={summary['precip_mae_mm']/MM_PER_IN:.5f} in, "
                f"RMSE={summary['precip_rmse_mm']/MM_PER_IN:.5f} in; "
                f"|Δ|>=0.01 in on {100*summary['fraction_abs_change_ge_0.01in']:.2f}% of pixels; "
                f"|Δ|>=0.10 in on {100*summary['fraction_abs_change_ge_0.1in']:.2f}%"
            )

        del model
        torch.cuda.empty_cache()

    pd.DataFrame(all_rows).to_csv(outroot / 'inversion_summary.csv', index=False)
    print()
    print('=' * 96)
    print('INVERSION COMPLETE')
    print('=' * 96)
    print(f'Output: {outroot}')
    print((pd.DataFrame(all_rows)[['model','target_similarity','achieved_similarity','similarity_error','mean_mm','max_mm']]).to_string(index=False))


if __name__ == '__main__':
    main()
