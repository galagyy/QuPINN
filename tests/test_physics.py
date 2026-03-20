#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module tests the shape of the residual.

Authors: Saumil Sharma, Om Kasar
Created: 2026-03-19
Version: 1.0.0
"""

import torch
from physics.wave_equation import compute_dimensionless_wave_residual


def test_wave_residual_shape() -> None:
    """Ensure the residual has the proper tensor shape."""

    class MockModel:
        def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return x**2 + t**2

    x = torch.randn(10, 1)
    t = torch.randn(10, 1)

    res = compute_dimensionless_wave_residual(MockModel(), x, t)

    assert res.shape == (10, 1)
    print("[PASS] Residual shape is correct.")


if __name__ == "__main__":
    test_wave_residual_shape()
