# QuPINN

A research codebase for comparing classical, hybrid, and quantum physics-informed neural networks.

## Overview

The QuPINN project studies if quantum/hybrid neural-network ansatzes solve differential equations more efficiently than classical physics-informed neural networks (PINNs). The benchmark used is a simplified solution to the 1D Wave Equation for comparison.

## Status

Currently, the repository only contains the classical PINN algorithm and the common modules. The quantum/hybrid model and the overall comparison are still in progress.

## Research Question

Under matched training configurations and an identical benchmark, how do classical, hybrid, and quantum models compare in:

- solution accuracy 
- PDE-residual & loss reduction
- parameter count & computational cost
- training stability & sensitivity to collocation points

Our code follows the broader PINN literature, where neural networks are trained to minimize violations of the benchmark equations and initial/boundary conditions rather than relying only on labeled solution data [[1](doi:10.1016/j.jcp.2018.10.045), [2](doi:10.1016/j.jcp.2018.10.045)].

## License

QuPINN is licensed under the **GNU Affero General Public License v3.0**. See [LICENSE](LICENSE).

## References

1. M. Raissi, P. Perdikaris, and G. E. Karniadakis, “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,” *Journal of Computational Physics*, 378, 686-707, 2019. [doi:10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)
2. G. E. Karniadakis et al., “Physics-informed machine learning,” *Nature Reviews Physics*, 3, 422-440, 2021. [doi:10.1038/s42254-021-00314-5](https://doi.org/10.1038/s42254-021-00314-5)