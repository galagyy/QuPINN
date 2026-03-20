#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module defines the protocol for PINN models (`PINNModel`) and
implements the autograd-based residual calculation for the 1D
wave function derived from Maxwell's equations.

Authors: Saumil Sharma, Om Kasar
Created: 2026-03-19
Version: 1.0.0
"""

import torch
from typing import Protocol


class PINNModel(Protocol):
    """Protocol defining the expected interface for a Wave-Equation PINN."""

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor: ...


def compute_dimensionless_wave_residual(
    model: PINNModel, 
    x_tilde: torch.Tensor, 
    t_tilde: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the dimensionless PDE residual.
    Note: Assumes inputs are already normalized to [0, 1].
    """
    x_tilde.requires_grad_(True)
    t_tilde.requires_grad_(True)
    
    E: torch.Tensor = model(x_tilde, t_tilde)
    
    # First Derivative
    E_t: torch.Tensor = torch.autograd.grad(E, t_tilde, torch.ones_like(E), create_graph=True)[0]
    E_x: torch.Tensor = torch.autograd.grad(E, x_tilde, torch.ones_like(E), create_graph=True)[0]
    
    # Second Derivative
    E_tt: torch.Tensor = torch.autograd.grad(E_t, t_tilde, torch.ones_like(E_t), create_graph=True)[0]
    E_xx: torch.Tensor = torch.autograd.grad(E_x, x_tilde, torch.ones_like(E_x), create_graph=True)[0]
    
    return E_tt - E_xx
