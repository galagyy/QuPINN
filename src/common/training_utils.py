"""
@file training_utils.py
@date 7-5-2026
@version 0.1.0

Training helpers: fixed collocation grids, loss weights, and LRA (Wang et al. 2021).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from src.common.exact_solution import sample_boundary_time, sample_initial, sample_interior


@dataclass
class LossWeights:
    """
    Manual loss weights (Fix 1 / Fix 5).

    IC terms are up-weighted so they are not drowned out by the PDE residual.
    BC terms are down-weighted because u=0 satisfies them without learning the wave.
    """

    pde: float = 1.0
    mse: float = 1000.0
    velocity: float = 1000.0
    b1: float = 0.1
    b2: float = 0.1

    def as_dict(self) -> dict[str, float]:
        return {
            "pde": self.pde,
            "mse": self.mse,
            "velocity": self.velocity,
            "b1": self.b1,
            "b2": self.b2,
        }


@dataclass
class CollocationPoints:
    """Fixed collocation tensors reused every epoch (Fix 3)."""

    x_pde: torch.Tensor
    t_pde: torch.Tensor
    x_ic: torch.Tensor
    t_bc: torch.Tensor


def build_collocation(
    n_pde: int,
    n_ic: int,
    n_bc: int,
    device: torch.device,
    seed: int = 0,
    fixed: bool = True,
) -> CollocationPoints:
    """
    Build collocation points for PINN training.

    When fixed=True, draws once via quasi-random Sobol (interior) and uniform
    linspace (IC/BC) so PDE gaps do not move between epochs.
    """
    if fixed:
        sobol = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed)
        samples = sobol.draw(n_pde).to(device)
        x_pde = samples[:, 0].contiguous().reshape(-1, 1)
        t_pde = samples[:, 1].contiguous().reshape(-1, 1)

        x_ic = torch.linspace(0.0, 1.0, n_ic, device=device).reshape(-1, 1)
        t_bc = torch.linspace(0.0, 1.0, n_bc, device=device).reshape(-1, 1)
    else:
        gen = torch.Generator(device=device).manual_seed(seed)
        x_pde, t_pde = sample_interior(n_pde, device=device, generator=gen)
        x_ic = sample_initial(n_ic, device=device, generator=gen)
        t_bc = sample_boundary_time(n_bc, device=device, generator=gen)

    return CollocationPoints(x_pde=x_pde, t_pde=t_pde, x_ic=x_ic, t_bc=t_bc)


def resample_collocation(
    points: CollocationPoints,
    n_pde: int,
    n_ic: int,
    n_bc: int,
    device: torch.device,
    seed: int,
) -> CollocationPoints:
    """Resample collocation (legacy random mode)."""
    return build_collocation(n_pde, n_ic, n_bc, device, seed=seed, fixed=False)


class LearningRateAnnealing:
    """
    Adaptive loss weights from gradient statistics (Fix 2).

    Wang, Teng & Perdikaris (2021), SIAM J. Sci. Comput. — balances back-propagated
    gradient magnitudes across loss components.
    """

    def __init__(
        self,
        base_weights: LossWeights,
        alpha: float = 0.9,
        update_every: int = 100,
        weight_min: float = 1e-2,
        weight_max: float = 1e3,
    ) -> None:
        self.weights = base_weights
        self.alpha = alpha
        self.update_every = update_every
        self.weight_min = weight_min
        self.weight_max = weight_max
        self._step = 0

    def needs_probe(self) -> bool:
        """True when the next step will recompute gradient-based weights."""
        return (self._step + 1) % self.update_every == 0

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def maybe_update(
        self,
        model: nn.Module,
        raw_losses: dict[str, torch.Tensor],
    ) -> LossWeights:
        self._step += 1
        if self._step % self.update_every != 0:
            return self.weights

        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return self.weights

        pde_grads = torch.autograd.grad(
            raw_losses["pde"],
            params,
            retain_graph=True,
            allow_unused=True,
        )
        max_grad_pde = max(
            (g.abs().max().item() for g in pde_grads if g is not None),
            default=1.0,
        )
        if not (max_grad_pde > 0 and max_grad_pde < float("inf")):
            return self.weights

        updated = LossWeights(**self.weights.as_dict())
        for key in ("mse", "velocity", "b1", "b2"):
            if key not in raw_losses:
                continue
            if not raw_losses[key].requires_grad:
                continue
            grads = torch.autograd.grad(
                raw_losses[key],
                params,
                retain_graph=True,
                allow_unused=True,
            )
            mean_grad = sum(
                g.abs().mean().item() for g in grads if g is not None
            ) / max(len(params), 1)
            if mean_grad < 1e-12:
                continue
            lambda_hat = self._clamp(
                max_grad_pde / mean_grad,
                self.weight_min,
                self.weight_max,
            )
            current = getattr(updated, key)
            new_weight = (1.0 - self.alpha) * current + self.alpha * lambda_hat
            setattr(updated, key, self._clamp(new_weight, self.weight_min, self.weight_max))

        self.weights = updated
        return self.weights
