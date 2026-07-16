"""
@file exact_solution.py
@date 7-5-2026
@version 0.1.1
@author Saumil Sharma and Om Kasar

Responsible for providing the analytical benchmark for the 1D wave equation:

`E(x, t) = sin(5*pi*x)*cos(5*pi*t) + 2*sin(7*pi*x)*cos(7*pi*t) where x ∈ [0, 1], t ∈ [0, 1]`

which is derived from the general solution `SUM(E0sin(k_n * x)cos(w_n * t))` with E0 = 1, wavelength
lamba = 1 (=> k = 2*pi) and the dispersion relation w = c*k = k in normalized units of
of `c = 1`.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

# Set constants for used multiples of pi
FIVE_PI = 5.0 * np.pi
SEVEN_PI = 7.0 * np.pi

def exact_solution_torch(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the exact wave-equation solution E(x, t) = sin(5*pi*x)*cos(5*pi*t) + 2*sin(7*pi*x)*cos(7*pi*t).

    @param x: Spatial coordinates on the domain [0, 1].
    @type x: torch.Tensor
    @param t: Temporal coordinates on the domain [0, 1].
    @type t: torch.Tensor

    @return: Exact solution values at each (x, t) pair.
    @rtype: torch.Tensor
    """
    return torch.sin(FIVE_PI * x) * torch.cos(FIVE_PI * t) + 2 * torch.sin(SEVEN_PI * x) * torch.cos(SEVEN_PI * t)

def exact_solution_numpy(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Evaluate the exact wave-equation solution E(x, t) = sin(5*pi*x)*cos(5*pi*t) + 2*sin(7*pi*x)*cos(7*pi*t) in NumPy.

    @param x: Spatial coordinates on the domain [0, 1].
    @type x: np.ndarray
    @param t: Temporal coordinates on the domain [0, 1].
    @type t: np.ndarray

    @return: Exact solution values at each (x, t) pair.
    @rtype: np.ndarray
    """
    return np.sin(FIVE_PI * x) * np.cos(FIVE_PI * t) + 2 * np.sin(SEVEN_PI * x) * np.cos(SEVEN_PI * t)

def dExact_dx_torch(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Compute the analytical spatial derivative dE/dx of the exact solution.

    @param x: Spatial coordinates on the domain [0, 1].
    @type x: torch.Tensor
    @param t: Temporal coordinates on the domain [0, 1].
    @type t: torch.Tensor

    @return: Spatial derivative values at each (x, t) pair.
    @rtype: torch.Tensor
    """
    return (FIVE_PI * torch.cos(FIVE_PI * x) * torch.cos(FIVE_PI * t) + 2.0 * SEVEN_PI * torch.cos(SEVEN_PI * x) * torch.cos(SEVEN_PI * t))

def dExact_dt_torch(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Compute the analytical temporal derivative dE/dt of the exact solution.

    @param x: Spatial coordinates on the domain [0, 1].
    @type x: torch.Tensor
    @param t: Temporal coordinates on the domain [0, 1].
    @type t: torch.Tensor

    @return: Temporal derivative values at each (x, t) pair.
    @rtype: torch.Tensor
    """
    return (
        -FIVE_PI * torch.sin(FIVE_PI * x) * torch.sin(FIVE_PI * t)
        - 2.0 * SEVEN_PI * torch.sin(SEVEN_PI * x) * torch.sin(SEVEN_PI * t)
    )

def sample_interior(n: int, device = None, generator = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Draw random interior collocation points uniformly from the spatiotemporal domain.

    @param n: Number of (x, t) pairs to sample.
    @type n: int
    @param device: Torch device on which to allocate the sampled tensors.
    @type device: torch.device or None
    @param generator: Optional random number generator for reproducible sampling.
    @type generator: torch.Generator or None

    @return: Tuple of spatial and temporal sample tensors, each of shape (n, 1).
    @rtype: tuple[torch.Tensor, torch.Tensor]
    """
    x = torch.rand(n, 1, device = device, generator = generator)
    t = torch.rand(n, 1, device = device, generator = generator)

    return x, t

def sample_initial(n: int, device = None, generator = None) -> torch.Tensor:
    """
    Draw random spatial coordinates for initial-condition collocation at t = 0.

    @param n: Number of spatial points to sample.
    @type n: int
    @param device: Torch device on which to allocate the sampled tensor.
    @type device: torch.device or None
    @param generator: Optional random number generator for reproducible sampling.
    @type generator: torch.Generator or None

    @return: Spatial sample tensor of shape (n, 1) on [0, 1].
    @rtype: torch.Tensor
    """
    return torch.rand(n, 1, device = device, generator = generator)

def sample_boundary_time(n: int, device = None, generator = None) -> torch.Tensor:
    """
    Draw random temporal coordinates for boundary-condition collocation.

    @param n: Number of time points to sample.
    @type n: int
    @param device: Torch device on which to allocate the sampled tensor.
    @type device: torch.device or None
    @param generator: Optional random number generator for reproducible sampling.
    @type generator: torch.Generator or None

    @return: Temporal sample tensor of shape (n, 1) on [0, 1].
    @rtype: torch.Tensor
    """
    return torch.rand(n, 1, device=device, generator=generator)

def grid(n_x: int = 200, n_t: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a uniform meshgrid over the normalized spatiotemporal domain.

    @param n_x: Number of spatial grid points on [0, 1].
    @type n_x: int
    @param n_t: Number of temporal grid points on [0, 1].
    @type n_t: int

    @return: Meshgrid arrays (X, T) suitable for evaluating the exact solution.
    @rtype: tuple[np.ndarray, np.ndarray]
    """
    xs = np.linspace(0.0, 1.0, n_x)
    ts = np.linspace(0.0, 1.0, n_t)

    X, T = np.meshgrid(xs, ts)

    return X, T

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity-check / plot the exact benchmark solution.")
    parser.add_argument("--plot", action="store_true", help="Save a heatmap of E(x,t) to outputs/.")
    args = parser.parse_args()

    if args.plot:
        from src.common.utils import ensure_dir
        from src.common.visualize import plot_exact_heatmap

        X, T = grid()
        E = exact_solution_numpy(X, T)

        ensure_dir("outputs")
        plot_exact_heatmap(X, T, E, save_path="outputs/exact_heatmap.png")
        print("[INFO] Saved outputs/exact_heatmap.png")
