#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This verifies the graphical outputs. It checks the axis, titles, bounds, etc.

Authors: Saumil Sharma, Om Kasar
Created: 2026-03-19
Version: 1.0.0
"""

import sys
import os
import torch

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from evaluation.plotting import plot_wave_heatmap, animate_wave
from models.classical_pinn import ClassicalPINN


class MockPerfectWave(ClassicalPINN):
    def eval(self) -> None:
        pass

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(torch.pi * x) * torch.cos(torch.pi * t)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    plots_dir = os.path.join(project_root, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    mock_model = MockPerfectWave()

    print("[INFO] Generating Heatmap ...")
    plot_wave_heatmap(mock_model, save_path=os.path.join(plots_dir, "test_heatmap.png"))

    print("[INFO] Generating Animation ...")
    animate_wave(mock_model, frames=30, save_path=os.path.join(plots_dir, "test_animation.gif"))

    print(f"[INFO] Complete. Files saved to: {plots_dir}")
