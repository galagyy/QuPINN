"""
@file model.py
@date 7-3-2026
@version 0.1.0
@author Saumil Sharma and Om Kasar

This is a model for a classical PINN with the following specifications:
- input of (x,t), output of predicted u(x,t)
- sin/SIREN activation
- Adam optimizer in `train_classical.py`, with an optional LBFGS second stage
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.common.exact_solution import FIVE_PI, SEVEN_PI

def _init_siren_linear(
    linear: nn.Linear,
    in_features: int,
    omega_0: float,
    is_first: bool,
) -> None:
    """SIREN weight init (Sitzmann et al. 2020, NeurIPS)."""
    with torch.no_grad():
        if is_first:
            bound = 1.0 / in_features
        else:
            bound = np.sqrt(6.0 / in_features) / omega_0
        linear.weight.uniform_(-bound, bound)
        if linear.bias is not None:
            linear.bias.uniform_(-bound, bound)


def _init_xavier_linear(linear: nn.Linear) -> None:
    """Initialize a standard activation layer with Xavier uniform weights."""
    nn.init.xavier_uniform_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class SineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, is_first: bool = False, omega_0: float = 30.0):
        """
        Initialize a SIREN sine activation layer with frequency-scaled linear weights.

        @param in_features: Number of input features.
        @type in_features: int
        @param out_features: Number of output features.
        @type out_features: int
        @param is_first: Whether this is the first layer in the network.
        @type is_first: bool
        @param omega_0: Frequency scaling factor for the sine activation.
        @type omega_0: float

        @return: None.
        @rtype: None
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        _init_siren_linear(self.linear, in_features, omega_0, is_first)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the SIREN sine activation to a linear projection of the input.

        @param x: Input feature tensor.
        @type x: torch.Tensor

        @return: Activated output tensor.
        @rtype: torch.Tensor
        """
        return torch.sin(self.omega_0 * self.linear(x))
    
class ClassicalPINN(nn.Module):
    def __init__(
            self,
            hidden_features: int = 32,
            hidden_layers: int = 4,
            activation: str = "siren",
            first_omega_0: float = 30.0,
            hidden_omega_0: float = 30.0,
            hard_ic_ansatz: bool = True,
    ):
        """
        Build a classical PINN for the 1D wave equation with selectable activations.

        @param hidden_features: Width of each hidden layer.
        @type hidden_features: int
        @param hidden_layers: Total number of hidden layers in the network.
        @type hidden_layers: int
        @param activation: Activation family; one of 'siren', 'tanh', or 'relu'.
        @type activation: str
        @param first_omega_0: Omega scaling for the first SIREN layer.
        @type first_omega_0: float
        @param hidden_omega_0: Omega scaling for subsequent SIREN layers.
        @type hidden_omega_0: float
        @param hard_ic_ansatz: If True, enforce ICs via u = E(x,t) + t^2 * (1 - t)^2 * NN(x, t) (Fix 4).
        @type hard_ic_ansatz: bool

        @return: None.
        @rtype: None
        """
        super().__init__()
        self.activation = activation
        self.hard_ic_ansatz = hard_ic_ansatz
        self.hidden_omega_0 = hidden_omega_0

        if activation == "siren":
            layers: list[nn.Module] = [SineLayer(2, hidden_features, is_first=True, omega_0=first_omega_0)]

            for _ in range(hidden_layers - 1):
                layers.append(SineLayer(hidden_features, hidden_features, omega_0=hidden_omega_0))

            final = nn.Linear(hidden_features, 1)
            _init_siren_linear(final, hidden_features, hidden_omega_0, is_first=False)
            layers.append(final)
            self.net = nn.Sequential(*layers)

        elif activation in ("tanh", "relu"):
            act_cls = nn.Tanh if activation == "tanh" else nn.ReLU
            input_layer = nn.Linear(2, hidden_features)
            _init_xavier_linear(input_layer)
            layers: list[nn.Module] = [input_layer, act_cls()]

            for _ in range(hidden_layers - 1):
                hidden_layer = nn.Linear(hidden_features, hidden_features)
                _init_xavier_linear(hidden_layer)
                layers += [hidden_layer, act_cls()]

            output_layer = nn.Linear(hidden_features, 1)
            _init_xavier_linear(output_layer)
            layers.append(output_layer)
            self.net = nn.Sequential(*layers)

        else:
            raise ValueError(f"Unknown activation method '{activation}'. Use 'siren', 'tanh', or 'relu'.")
        

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict u(x, t) by concatenating spatial and temporal inputs and passing them through the network.

        With hard_ic_ansatz=True:
            u(x,t) = E_exact(x,t) + t^2 * (1 - t)^2 * NN(x,t)
        so u(x,0) and u_t(x,0) match the exact wave ICs by construction.

        @param x: Spatial input coordinates.
        @type x: torch.Tensor
        @param t: Temporal input coordinates.
        @type t: torch.Tensor

        @return: Predicted solution values u(x, t).
        @rtype: torch.Tensor
        """
        xt = torch.cat([x, t], dim=-1)
        nn_out = self.net(xt)

        if self.hard_ic_ansatz:
            particular = (
                torch.sin(FIVE_PI * x) * torch.cos(FIVE_PI * t)
                + 2 * torch.sin(SEVEN_PI * x) * torch.cos(SEVEN_PI * t)
            )
            return particular + t**2 * (1 - t)**2 * nn_out

        return nn_out