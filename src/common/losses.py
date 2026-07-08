"""
@file losses.py
@date 7-5-2026
@version 0.1.0
@author Saumil Sharma and Om Kasar

Responsible for showing the loss functions for any model; both models expose
the same forward signature, so we are able to use either.

All second derivatives use torch.autograd.grad with create_graph = True so that
they remain differentiable during optimizations.
"""

from __future__ import annotations
import torch
from src.common.exact_solution import exact_solution_torch
from src.common.training_utils import LossWeights

def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """
    Compute first-order gradients of outputs with respect to inputs.

    @param outputs: Tensor whose derivatives are required.
    @type outputs: torch.Tensor
    @param inputs: Tensor with respect to which differentiation is performed.
    @type inputs: torch.Tensor

    @return: Gradient tensor matching the shape of inputs.
    @rtype: torch.Tensor
    """

    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]

def _pde_residual(model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Pointwise wave-equation residual u_tt - u_xx."""
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)

    u = model(x, t)
    u_x = _grad(u, x)
    u_xx = _grad(u_x, x)
    u_t = _grad(u, t)
    u_tt = _grad(u_t, t)

    return u_tt - u_xx

def loss_pde(model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean-squared PDE residual for the 1D wave equation u_tt - u_xx = 0.

    @param model: PINN model exposing forward(x, t) -> u.
    @type model: torch.nn.Module
    @param x: Interior spatial collocation points.
    @type x: torch.Tensor
    @param t: Interior temporal collocation points.
    @type t: torch.Tensor

    @return: Scalar mean-squared PDE residual loss.
    @rtype: torch.Tensor
    """
    residual = _pde_residual(model, x, t)

    return torch.mean(residual ** 2)

def loss_pde_causal(
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    num_time_bins: int = 32,
    epsilon: float = 5.0,
) -> torch.Tensor:
    """
    Causal PDE loss for hyperbolic problems (Fix 6).

    Weights earlier time bins fully and later bins only after earlier residuals
    are small (Wang et al. 2024, CMAME — respecting causality in PINN training).

    @param model: PINN model exposing forward(x, t) -> u.
    @type model: torch.nn.Module
    @param x: Interior spatial collocation points.
    @type x: torch.Tensor
    @param t: Interior temporal collocation points.
    @type t: torch.Tensor
    @param num_time_bins: Number of temporal bins for causal weighting.
    @type num_time_bins: int
    @param epsilon: Sharpness of the causal weight decay.
    @type epsilon: float

    @return: Scalar causally weighted PDE loss.
    @rtype: torch.Tensor
    """
    residual = _pde_residual(model, x, t)
    residual_sq = residual.contiguous().view(-1)

    # Column slices like samples[:, 1:2] are non-contiguous; flatten() does not copy.
    t_flat = t.detach().contiguous().view(-1)

    step = 1.0 / num_time_bins
    boundaries = torch.linspace(
        step,
        1.0 - step,
        num_time_bins - 1,
        device=t.device,
        dtype=t.dtype,
    )
    bin_idx = torch.bucketize(t_flat, boundaries)
    bin_losses: list[torch.Tensor] = []

    for i in range(num_time_bins):
        mask = bin_idx == i

        if mask.any():
            bin_losses.append(residual_sq[mask].mean())
        else:
            bin_losses.append(torch.zeros((), device=t.device, dtype=residual.dtype))

    cumulative = torch.zeros((), device=t.device, dtype=residual.dtype)
    weighted_sum = torch.zeros((), device=t.device, dtype=residual.dtype)

    for i, bin_loss in enumerate(bin_losses):
        if i == 0:
            weight = torch.ones((), device=t.device, dtype=residual.dtype)
        else:
            weight = torch.exp(-epsilon * cumulative)

        weighted_sum = weighted_sum + weight * bin_loss
        cumulative = cumulative + bin_loss.detach()

    return weighted_sum / num_time_bins

def loss_ic_mse(model, x0: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean-squared error between the model and exact solution at t = 0.

    @param model: PINN model exposing forward(x, t) -> u.
    @type model: torch.nn.Module
    @param x0: Spatial collocation points on the initial time slice.
    @type x0: torch.Tensor

    @return: Scalar initial-condition MSE loss.
    @rtype: torch.Tensor
    """
    t0 = torch.zeros_like(x0)
    u0 = model(x0, t0)
    e0 = exact_solution_torch(x0, t0)

    return torch.mean((u0 - e0) ** 2)

def loss_velocity_ic(model, x0: torch.Tensor) -> torch.Tensor:
    """
    Penalize mismatch between du/dt at t = 0 and the analytical initial velocity.

    @param model: PINN model exposing forward(x, t) -> u.
    @type model: torch.nn.Module
    @param x0: Spatial collocation points on the initial time slice.
    @type x0: torch.Tensor
    @param use_paper_literal: If True, compare against dE/dx instead of dE/dt.
    @type use_paper_literal: bool

    @return: Scalar initial-velocity MSE loss.
    @rtype: torch.Tensor
    """
    x0 = x0.clone().requires_grad_(True)
    t0 = torch.zeros_like(x0).requires_grad_(True)

    u0 = model(x0, t0)
    u_t0 = _grad(u0, t0)

    return torch.mean((u_t0) ** 2)

def loss_lower_boundary(model, t: torch.Tensor) -> torch.Tensor:
    """
    Enforce the first periodic boundary condition by verifying u(0, t) = 0 for all collocation points.

    @param model: PINN model exposing forward(x, t) -> u.
    @type model: torch.nn.Module
    @param t: Temporal collocation points on the domain boundary.
    @type t: torch.Tensor

    @return: Scalar boundary-value MSE loss.
    @rtype: torch.Tensor
    """
    x0 = torch.zeros_like(t)

    u0 = model(x0, t)

    return torch.mean((u0) ** 2)

def loss_upper_boundary(model, t: torch.Tensor) -> torch.Tensor:
    """
    Enforce the second periodic boundary condition by verifying u(1, t) = 0 for all collocation points.

    @param model: PINN model exposing forward(x, t) -> u.
    @type model: torch.nn.Module
    @param t: Temporal collocation points on the domain boundary.
    @type t: torch.Tensor

    @return: Scalar boundary-derivative MSE loss.
    @rtype: torch.Tensor
    """
    x1 = torch.ones_like(t)

    u1 = model(x1, t)

    return torch.mean((u1) ** 2)

def total_loss(
    model,
    x_pde: torch.Tensor,
    t_pde: torch.Tensor,
    x_ic: torch.Tensor,
    t_bc: torch.Tensor,
    weights: LossWeights | None = None,
    use_paper_literal_velocity: bool = False,
    use_causal_pde: bool = True,
    skip_ic_losses: bool = False,
    causal_epsilon: float = 5.0,
    causal_bins: int = 32,
) -> dict[str, torch.Tensor]:
    """
    Aggregate all physics-informed loss components into a single training objective.

    @param model: PINN model exposing forward(x, t) -> u.
    @type model: torch.nn.Module
    @param x_pde: Spatial collocation points for the PDE residual.
    @type x_pde: torch.Tensor
    @param t_pde: Temporal collocation points for the PDE residual.
    @type t_pde: torch.Tensor
    @param x_ic: Spatial collocation points for initial-condition losses.
    @type x_ic: torch.Tensor
    @param t_bc: Temporal collocation points for boundary-condition losses.
    @type t_bc: torch.Tensor
    @param weights: Per-component loss weights (Fix 1).
    @type weights: LossWeights or None
    @param use_paper_literal_velocity: If True, use dE/dx for the velocity IC target.
    @type use_paper_literal_velocity: bool
    @param use_causal_pde: If True, apply causal time weighting to PDE loss (Fix 6).
    @type use_causal_pde: bool
    @param skip_ic_losses: Skip IC penalties when hard ansatz satisfies them (Fix 4).
    @type skip_ic_losses: bool
    @param causal_epsilon: Causal decay sharpness.
    @type causal_epsilon: float
    @param causal_bins: Number of causal time bins.
    @type causal_bins: int

    @return: Dictionary of total and per-component scalar loss tensors.
    @rtype: dict[str, torch.Tensor]
    """

    if weights is None:
        weights = LossWeights()

    if use_causal_pde:
        l_pde = loss_pde_causal(
            model, x_pde, t_pde,
            num_time_bins=causal_bins,
            epsilon=causal_epsilon,
        )
    else:
        l_pde = loss_pde(model, x_pde, t_pde)

    if skip_ic_losses:
        l_mse = torch.zeros((), device=x_pde.device, dtype=l_pde.dtype)
        l_v = torch.zeros((), device=x_pde.device, dtype=l_pde.dtype)
    else:
        l_mse = loss_ic_mse(model, x_ic)
        l_v = loss_velocity_ic(model, x_ic, use_paper_literal=use_paper_literal_velocity)

    l_b1 = loss_lower_boundary(model, t_bc)
    l_b2 = loss_upper_boundary(model, t_bc)

    total = (
        weights.pde * l_pde
        + weights.mse * l_mse
        + weights.velocity * l_v
        + weights.b1 * l_b1
        + weights.b2 * l_b2
    )

    return {
        "total": total,
        "pde": l_pde,
        "mse": l_mse,
        "velocity": l_v,
        "b1": l_b1,
        "b2": l_b2,
    }