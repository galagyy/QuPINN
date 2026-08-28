# QuPINN

A research codebase for comparing classical, hybrid, and quantum physics-informed neural networks.

## Overview

The QuPINN project studies if quantum/hybrid physics-informed neural networks (PINNs) solve differential equations more efficiently than classical PINNs. The benchmark used is a simplified solution to the 1D Wave Equation for comparison.

The study aims to make the study as reproducible as possible, keeping the domain, reference solution, collocation points, optimizer budget, random seeds, and evaluation grid fixed as a result. Quantum results will additionally report circuit depth, qubit count, shots, encoding, and simulator or hardware backend.

## Status

Currently, the repository only contains the classical PINN algorithm and the common modules. The quantum/hybrid model and the overall comparison are still in progress.

## Research Question

Under matched training configurations and an identical benchmark, how do classical, hybrid, and quantum models compare in:

- solution accuracy 
- PDE-residual & loss reduction
- parameter count & computational cost
- training stability & sensitivity to collocation points

Our code follows conventions in previous PINN studies, where neural networks are trained to minimize violations of the benchmark equations and initial/boundary conditions rather than only on solutions [[1](#reference-1), [2](#reference-2)].

## Benchmark problem

The baseline solves the normalized wave equation, considering $c = 1$:

$$
\frac{\partial^2 u}{\partial t^2}
- \frac{\partial^2 u}{\partial x^2} = 0,
\qquad (x,t) \in [0,1] \times [0,1],
$$

with homogeneous Dirichlet boundary conditions and the benchmark solution adapted from Dashtbayaz [[3](#reference-3)] for result comparisons:

$$
E(x,t) = \sin(5\pi x)\cos(5\pi t)
	+ 2\sin(7\pi x)\cos(7\pi t).
$$

The classical model receives a point $(x,t)$ and outputs $u_{\theta}(x,t)$. Training combines the wave-equation residual with the initial condition and boundary condition corrections. With the initial condition ansatz, the expanded expression is:

$$
u_\theta(x,t) = E(x,t) + t^2(1-t)^2 \cdot N_\theta(x,t)
$$

The prescribed displacement and initial velocity are tracked by robust loss functions located in the ```common/``` directory. Spatial and temporal partial derivatives are obtained with PyTorch's automatic differentiation.

## PINN Configurations

The comparison will use a 3 layer, feed forward network as outlined in Dashtbayaz's paper [[3](#reference-3)] with varying hidden layer widths for ease of comparison with the other two architectures.

For the classical activation function, we chose a sin/SIREN option, motivated by its ability to represent signal-like functions with periodic activations. This allows for daptive loss balancing and causal weighting to be included in PINN stabilization techniques, as illustrated in Dashtbayaz's & Sitzmann's paper [[3](#reference-3), [4](#reference-4)].

## Evaluation Metrics

After training, models are evaluated on the following:

- **relative $L^2$ error**: the normalized difference between the prediction function and $E(x,t)$
- **maximum pointwise error**: $\|u_\theta-E\|_\infty$
- **PDE residual loss**: the mean squared value of $\frac{\partial^2 u}{\partial t^2} - \frac{\partial^2 u}{\partial x^2}$ at collocation points.
- **Total loss**: Aggregate loss from each loss component.

## Installation

Use Python 3.10 or newer, create new environment, and use the following commands:

```bash
python -m venv .venv

# Windows PowerShell activation
.venv\Scripts\Activate.ps1

# macOS/Linux activation
source .venv/bin/activate

python -m pip install -r requirements.txt
```

## License

QuPINN is licensed under the **GNU Affero General Public License v3.0**. See [LICENSE](LICENSE).

## References

1. M. Raissi, P. Perdikaris, and G. E. Karniadakis, “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,” *Journal of Computational Physics*, 378, 686-707, 2019. [doi:10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)
2. G. E. Karniadakis et al., “Physics-informed machine learning,” *Nature Reviews Physics*, 3, 422-440, 2021. [doi:10.1038/s42254-021-00314-5](https://doi.org/10.1038/s42254-021-00314-5)
3. N. Hosseini Dashtbayaz, G. Farhani, B. Wang, and C. X. Ling, “Physics-Informed Neural Networks: Minimizing Residual Loss with Wide Networks and Effective Activations,” *Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence Main Track*, 5853-5861, 2024. [doi:10.24963/ijcai.2024/647](https://doi.org/10.24963/ijcai.2024/647)
4. V. Sitzmann et al., “Implicit neural representations with periodic activation functions,” *Advances in Neural Information Processing Systems*, 33, 2020. [arXiv:2006.09661](https://arxiv.org/abs/2006.09661)