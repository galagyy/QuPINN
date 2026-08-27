"""
@file visualize.py
@date 7-5-2026
@version 0.1.0
@author Saumil Sharma and Om Kasar

This file is responsible for plotting PINNs. As of this version, it is only responsible for
classical PINNs.
"""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_exact_heatmap(save_path: str) -> None:
    """
    Save a heatmap of the exact benchmark solution E(x, t).

    @param save_path: Output file path for the saved figure.
    @type save_path: str

    @return: None
    @rtype: None
    """
    x = np.linspace(0, 1, 512)
    t = np.linspace(0, 1, 512)
    X, T = np.meshgrid(x, t)

    # Plot exact solution for comparison
    E = np.sin(5 * np.pi * X) * np.cos(5 * np.pi * T) + 2 * np.sin(7 * np.pi * X) * np.cos(7 * np.pi * T)

    fig, ax = plt.subplots(figsize=(6, 5))
    v = float(np.max(np.abs(E)))
    im = ax.pcolormesh(X, T, E, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")

    ax.set_title(
        r"Exact Solution: $E(x,t)=\sin(5\pi x)\cos(5\pi t)+2\sin(7\pi x)\cos(7\pi t)$"
    )
    ax.set_xlabel("x (normalized)")
    ax.set_ylabel("t (normalized)")

    fig.colorbar(im, ax=ax, label="E(x,t)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)

    plt.close(fig)

@torch.no_grad()
def _model_grid(model, n_x: int = 200, n_t: int = 200, device = None):
    """
    Evaluate model predictions on a uniform spatiotemporal grid.

    @param model: PINN model exposing forward(x, t) -> u.
    @type model: torch.nn.Module
    @param n_x: Number of spatial grid points on [0, 1].
    @type n_x: int
    @param n_t: Number of temporal grid points on [0, 1].
    @type n_t: int
    @param device: Torch device on which to run inference.
    @type device: torch.device or None

    @return: Tuple of NumPy meshgrid arrays (X, T) and predicted values U.
    @rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    xs = torch.linspace(0.0, 1.0, n_x, device = device)
    ts = torch.linspace(0.0, 1.0, n_t, device = device)

    Xg, Tg = torch.meshgrid(xs, ts, indexing = "xy")
    x = Xg.reshape(-1, 1)
    t = Tg.reshape(-1, 1)
    u = model(x, t).reshape(Xg.shape)

    return Xg.cpu().numpy(), Tg.cpu().numpy(), u.cpu().numpy()

def plot_prediction_heatmap(model, save_path: str, title: str = "Predicted u(x,t)", device = None) -> None:
    """
    Save a heatmap of model predictions over the spatiotemporal domain.

    @param model: PINN model exposing forward(x, t) -> u.
    @type model: torch.nn.Module
    @param save_path: Output file path for the saved figure.
    @type save_path: str
    @param title: Plot title displayed above the heatmap.
    @type title: str
    @param device: Torch device on which to run inference.
    @type device: torch.device or None

    @return: None.
    @rtype: None
    """
    X, T, U = _model_grid(model, device=device)

    fig, ax = plt.subplots(figsize=(6, 5))
    v = float(np.max(np.abs(U)))
    im = ax.pcolormesh(X, T, U, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")

    ax.set_xlabel("x (normalized)")
    ax.set_ylabel("t (normalized)")
    ax.set_title(title)

    fig.colorbar(im, ax=ax, label="u(x,t)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)

    plt.close(fig)


def plot_error_map(model, save_path: str, device=None) -> None:
    """
    Save a heatmap of absolute error between model predictions and the exact solution.

    @param model: PINN model exposing forward(x, t) -> u.
    @type model: torch.nn.Module
    @param save_path: Output file path for the saved figure.
    @type save_path: str
    @param device: Torch device on which to run inference.
    @type device: torch.device or None

    @return: None.
    @rtype: None
    """
    from src.common.exact_solution import exact_solution_numpy

    X, T, U = _model_grid(model, device=device)
    E = exact_solution_numpy(X, T)
    err = np.abs(U - E)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(X, T, err, cmap="viridis", shading="auto")

    ax.set_title("|Classical PINN - Exact|")
    ax.set_xlabel("x (normalized)")
    ax.set_ylabel("t (normalized)")

    fig.colorbar(im, ax=ax, label = "Absolute error")
    fig.tight_layout()
    fig.savefig(save_path, dpi = 150)

    plt.close(fig)


def plot_snapshots(model, save_path: str, times = (0.25, 0.5, 0.75), device = None) -> None:
    """
    Save line plots comparing exact and predicted solutions at selected time slices.

    @param model: PINN model exposing forward(x, t) -> u.
    @type model: torch.nn.Module
    @param save_path: Output file path for the saved figure.
    @type save_path: str
    @param times: Sequence of normalized time values at which to plot snapshots.
    @type times: tuple[float, ...]
    @param device: Torch device on which to run inference.
    @type device: torch.device or None

    @return: None.
    @rtype: None
    """
    from src.common.exact_solution import exact_solution_numpy

    xs = np.linspace(0.0, 1.0, 400)
    x_t = torch.tensor(xs, dtype=torch.float32, device=device).reshape(-1, 1)

    fig, axes = plt.subplots(1, len(times), figsize=(4 * len(times), 3.5), sharey=True)
    if len(times) == 1:
        axes = [axes]
    for col, tv in enumerate(times):
        t_t = torch.full_like(x_t, tv)
        exact = exact_solution_numpy(xs, np.full_like(xs, tv))

        with torch.no_grad():
            u = model(x_t, t_t).cpu().numpy().reshape(-1)

        axes[col].plot(xs, exact, "b-", label="Exact")
        axes[col].plot(xs, u, "r--", label="Prediction")
        axes[col].set_title(f"t = {tv:.2f}")
        axes[col].set_xlabel("x")
        axes[col].set_ylim(-3.2, 3.2)

    axes[0].set_ylabel("u(x,t)")

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc = "lower center", ncol = 2)
    fig.suptitle("Exact vs. Predicted Solution: Classical PINN")
    fig.tight_layout(rect = (0, 0.08, 1, 1))
    fig.savefig(save_path, dpi = 150)

    plt.close(fig)


def plot_loss_curve(history: dict[str, list[float]], save_path: str) -> None:
    """
    Save a log-scale plot of training loss components over epochs.

    @param history: Mapping from loss component names to per-epoch loss values.
    @type history: dict[str, list[float]]
    @param save_path: Output file path for the saved figure.
    @type save_path: str

    @return: None.
    @rtype: None
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    relative_l2 = history.get("relative_l2")
    rel_epochs = history.get("relative_l2_epoch")

    for key, vals in history.items():
        if key in ("relative_l2", "relative_l2_epoch"):
            continue
        ax.plot(vals, label=key)

    ax.set_title("Training loss components")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend(loc="upper left")

    if relative_l2:
        ax2 = ax.twinx()
        x_rel = rel_epochs if rel_epochs else list(range(len(relative_l2)))
        ax2.plot(x_rel, relative_l2, color="black", linestyle=":", linewidth=2, label="relative_l2")
        ax2.set_ylabel("relative L2 error")
        ax2.set_yscale("log")
        ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(save_path, dpi = 150)

    plt.close(fig)
