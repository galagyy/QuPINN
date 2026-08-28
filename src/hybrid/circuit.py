# Small differentiable quantum model for the wave-equation PINN.

from __future__ import annotations

import pennylane as qml
import torch
from torch import nn

N_QUBITS = 6
N_LAYERS = 3

_device = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(_device, interface="torch", diff_method="best")
def quantum_circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
	"""Encode ``(x, t)`` and return one trainable expectation value."""
	qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
	qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
	return qml.expval(qml.PauliZ(0))