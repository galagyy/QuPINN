"""
@file metrics.py
@date 7-5-2026
@version 0.1.0
@author Saumil Sharma and Om Kasar

This file shows the metrics for both the QuPINN and PINN.
"""

from __future__ import annotations

import torch

from src.common.exact_solution import exact_solution_torch

@torch.no_grad()
def relative_l2_error(model, x: torch.Tensor, t: torch.Tensor) -> float:
    """
    Compute the relative L2 error between model predictions and the exact solution.

    @param model: PINN model exposing forward(x, t) -> u.
    @type model: torch.nn.Module
    @param x: Spatial evaluation coordinates.
    @type x: torch.Tensor
    @param t: Temporal evaluation coordinates.
    @type t: torch.Tensor

    @return: Relative L2 error as a Python float.
    @rtype: float
    """
    u = model(x, t)
    e = exact_solution_torch(x, t)

    num = torch.sum((u - e) ** 2)
    den = torch.sum(e ** 2)

    if not torch.isfinite(num) or not torch.isfinite(den) or den <= 0:
        return float("nan")

    return torch.sqrt(num / den).item()


@torch.no_grad()
def max_norm_error(model, x: torch.Tensor, t: torch.Tensor) -> float:
    """
    Compute the maximum absolute pointwise error against the exact solution.

    @param model: PINN model exposing forward(x, t) -> u.
    @type model: torch.nn.Module
    @param x: Spatial evaluation coordinates.
    @type x: torch.Tensor
    @param t: Temporal evaluation coordinates.
    @type t: torch.Tensor

    @return: Maximum absolute error as a Python float.
    @rtype: float
    """
    u = model(x, t)
    e = exact_solution_torch(x, t)

    return torch.max(torch.abs(u - e)).item()


@torch.no_grad()
def evaluate(model, n_x: int = 200, n_t: int = 200, device = None) -> dict[str, float]:
    """
    Evaluate standard error metrics on a uniform grid over the spatiotemporal domain.

    @param model: PINN model exposing forward(x, t) -> u.
    @type model: torch.nn.Module
    @param n_x: Number of spatial grid points on [0, 1].
    @type n_x: int
    @param n_t: Number of temporal grid points on [0, 1].
    @type n_t: int
    @param device: Torch device on which to run evaluation.
    @type device: torch.device or None

    @return: Dictionary containing relative L2 and max-norm error metrics.
    @rtype: dict[str, float]
    """
    xs = torch.linspace(0.0, 1.0, n_x, device = device)
    ts = torch.linspace(0.0, 1.0, n_t, device = device)

    X, T = torch.meshgrid(xs, ts, indexing = "xy")
    x = X.reshape(-1, 1)
    t = T.reshape(-1, 1)

    return {
        "relative_l2": relative_l2_error(model, x, t),
        "max_norm": max_norm_error(model, x, t),
    }
