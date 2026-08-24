"""
@file train_classical.py
@date 7-3-2026
@version 0.2.0

Utility script to train the classical PINN with stabilized training defaults.

Usage:
> python -m scripts.train_classical --epochs 8000 --lbfgs
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch
from torch import nn
from tqdm import trange
import numpy as np

from src.classical.model import ClassicalPINN
from src.common.losses import total_loss
from src.common.metrics import evaluate
from src.common.training_utils import (
    CollocationPoints,
    LearningRateAnnealing,
    LossWeights,
    build_collocation,
    resample_collocation,
)
from src.common.utils import ensure_dir, get_device, set_seed
from src.common.visualize import plot_error_map, plot_loss_curve, plot_prediction_heatmap, plot_snapshots, plot_exact_heatmap

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train classical PINN with stabilized defaults.")
    p.add_argument("--epochs", type=int, default=8000)
    p.add_argument("--n_pde", type=int, default=2000,
                   help="Interior collocation points (lower = faster epochs).")
    p.add_argument("--n_ic", type=int, default=400)
    p.add_argument("--n_bc", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-4, help="Lower default LR reduces second-derivative blow-ups.")
    p.add_argument("--hidden_features", type=int, default=128)
    p.add_argument("--hidden_layers", type=int, default=1)
    p.add_argument("--activation", type=str, default="tanh", choices=["siren", "tanh", "relu"])
    p.add_argument("--lbfgs", action="store_true", help="Run LBFGS after Adam if relative L2 is below threshold.")
    p.add_argument("--lbfgs_steps", type=int, default=500)
    p.add_argument("--lbfgs_max_rel_l2", type=float, default=0.5,
                   help="Skip LBFGS if Adam ends with relative L2 above this (Fix 7).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--use_paper_literal_velocity", action="store_true")

    # Manual Loss Weights
    p.add_argument("--w_pde", type=float, default=1.0)
    p.add_argument("--w_mse", type=float, default=1000.0)
    p.add_argument("--w_velocity", type=float, default=1000.0)
    p.add_argument("--w_b1", type=float, default=0.1)
    p.add_argument("--w_b2", type=float, default=0.1)

    # Adaptive gradient balancing (Wang et al. 2021)
    p.add_argument("--lra", action=argparse.BooleanOptionalAction, default=False,
                   help="Adaptive loss weights; off by default with hard IC ansatz.")
    p.add_argument("--lra_alpha", type=float, default=0.9)
    p.add_argument("--lra_every", type=int, default=100)

    # Fixed Collocation
    p.add_argument("--fixed_collocation", action=argparse.BooleanOptionalAction, default=True,
                   help="Reuse the same collocation points each epoch.")

    # Hard IC Ansatz
    p.add_argument(
        "--hard_ic_ansatz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce ICs via u = E_exact(x,t) + t^2 * (1 - t)^2 * NN(x,t).",
    )

    # Causal PDE Weighting
    p.add_argument("--causal_pde", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--causal_epsilon", type=float, default=1.0)
    p.add_argument("--causal_bins", type=int, default=32)

    # Stability Fix
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Max gradient norm; 0 disables clipping.")

    return p.parse_args()

def _loss_kwargs(args: argparse.Namespace, weights: LossWeights, model: ClassicalPINN) -> dict:
    return {
        "weights": weights,
        "use_paper_literal_velocity": args.use_paper_literal_velocity,
        "use_causal_pde": args.causal_pde,
        "skip_ic_losses": model.hard_ic_ansatz,
        "causal_epsilon": args.causal_epsilon,
        "causal_bins": args.causal_bins,
    }

def _effective_weights(weights: LossWeights, lra: LearningRateAnnealing | None) -> LossWeights:
    if lra is not None:
        return lra.weights
    
    return weights

def _run_training_step(
    model: ClassicalPINN,
    optimizer: torch.optim.Optimizer,
    colloc: CollocationPoints,
    args: argparse.Namespace,
    weights: LossWeights,
    lra: LearningRateAnnealing | None,
) -> dict[str, torch.Tensor]:
    optimizer.zero_grad()
    active_weights = _effective_weights(weights, lra)

    if lra is not None and lra.needs_probe():
        probe = total_loss(
            model,
            colloc.x_pde,
            colloc.t_pde,
            colloc.x_ic,
            colloc.t_bc,
            **_loss_kwargs(args, active_weights, model),
        )
        lra.maybe_update(model, probe)
        active_weights = lra.weights

    losses = total_loss(
        model,
        colloc.x_pde,
        colloc.t_pde,
        colloc.x_ic,
        colloc.t_bc,
        **_loss_kwargs(args, active_weights, model),
    )
    losses["total"].backward()

    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

    optimizer.step()
    return losses

def _has_nan_params(model: nn.Module) -> bool:
    return any(not torch.isfinite(p).all() for p in model.parameters())

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    ensure_dir(args.out_dir)

    weights = LossWeights(
        pde=args.w_pde,
        mse=args.w_mse,
        velocity=args.w_velocity,
        b1=args.w_b1,
        b2=args.w_b2,
    )
    lra = (
        LearningRateAnnealing(weights, alpha=args.lra_alpha, update_every=args.lra_every)
        if args.lra
        else None
    )

    model = ClassicalPINN(
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        activation=args.activation,
        hard_ic_ansatz=args.hard_ic_ansatz,
    ).to(device)
    model.train()

    colloc = build_collocation(
        args.n_pde, args.n_ic, args.n_bc, device, seed=args.seed, fixed=args.fixed_collocation,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history: dict[str, list[float]] = defaultdict(list)

    pbar = trange(args.epochs, desc="Adam")
    for epoch in pbar:
        if not args.fixed_collocation:
            colloc = resample_collocation(
                colloc, args.n_pde, args.n_ic, args.n_bc, device, seed=args.seed + epoch,
            )

        losses = _run_training_step(model, optimizer, colloc, args, weights, lra)

        loss_total = losses["total"].item()
        if not torch.isfinite(losses["total"]) or _has_nan_params(model):
            print(f"\n[ERROR] NaN/Inf detected at epoch {epoch}. Stopping early.")
            print("[HINT] Try: --no-lra --no-causal_pde --grad_clip 0.5 --lr 5e-5")
            return

        for k, v in losses.items():
            history[k].append(v.item())

        if epoch % args.log_every == 0:
            metrics = evaluate(model, device=device)
            rel_l2 = metrics["relative_l2"]
            history["relative_l2"].append(rel_l2)
            history.setdefault("relative_l2_epoch", []).append(epoch)
            rel_str = f"{rel_l2:.2e}" if torch.isfinite(torch.tensor(rel_l2)) else "nan"
            pbar.set_postfix(total=f"{loss_total:.2e}", rel_l2=rel_str)

    metrics = evaluate(model, device=device)
    print(f"[INFO] After Adam: relative_l2={metrics['relative_l2']:.4e}, max_norm={metrics['max_norm']:.4e}")

    if args.lbfgs:
        if metrics["relative_l2"] > args.lbfgs_max_rel_l2:
            print(
                f"[WARN] Skipping LBFGS: relative_l2={metrics['relative_l2']:.4e} "
                f"> threshold {args.lbfgs_max_rel_l2} (Fix 7)."
            )
        else:
            print("[INFO] Starting LBFGS fine-tune stage on fixed collocation...")
            lbfgs_opt = torch.optim.LBFGS(
                model.parameters(),
                lr=1.0,
                max_iter=args.lbfgs_steps,
                history_size=50,
                line_search_fn="strong_wolfe",
            )

            def closure():
                lbfgs_opt.zero_grad()
                step_losses = total_loss(
                    model,
                    colloc.x_pde,
                    colloc.t_pde,
                    colloc.x_ic,
                    colloc.t_bc,
                    **_loss_kwargs(args, _effective_weights(weights, lra), model),
                )
                step_losses["total"].backward()

                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                for k, v in step_losses.items():
                    history[k].append(v.item())

                return step_losses["total"]

            lbfgs_opt.step(closure)
            metrics = evaluate(model, device=device)

            print(f"[INFO] After LBFGS: relative_l2={metrics['relative_l2']:.4e}, max_norm={metrics['max_norm']:.4e}")

    model.eval()
    torch.save(model.state_dict(), f"{args.out_dir}/classical_pinn.pt")

    plot_loss_curve(dict(history), f"{args.out_dir}/classical_loss_curve.png")
    plot_prediction_heatmap(
        model,
        f"{args.out_dir}/classical_prediction_heatmap.png",
        title="Classical PINN: Predicted u(x,t)",
        device=device,
    )
    plot_error_map(model, f"{args.out_dir}/classical_error_map.png", device=device)
    plot_snapshots(model, f"{args.out_dir}/classical_snapshots.png", device=device)
    plot_exact_heatmap(f"{args.out_dir}/classical_exact_solution.png")

    print(f"[INFO] Final metrics: relative_l2={metrics['relative_l2']:.4e}, max_norm={metrics['max_norm']:.4e}")
    print(f"[INFO] Saved checkpoint + plots to {args.out_dir}/")

if __name__ == "__main__":
    main()
