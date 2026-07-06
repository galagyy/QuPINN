"""
@file utils.py
@date 7-5-2026
@version 0.1.0
@author Saumil Sharma and Om Kasar

This is the utility class for the entire project containing generic functions.
"""

import os
import random

import numpy as np
import torch

def set_seed(seed: int = 0) -> None:
    """
    Seed Python, NumPy, and PyTorch random number generators for reproducibility.

    @param seed: Integer seed applied to all supported RNG backends.
    @type seed: int

    @return: None.
    @rtype: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if (torch.cuda.is_available()):
        torch.cuda.manual_seed_all(seed)

def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Select the best available Torch compute device.

    @param prefer_cuda: If True and CUDA is available, return a CUDA device.
    @type prefer_cuda: bool

    @return: Selected Torch device instance.
    @rtype: torch.device
    """
    if (prefer_cuda and torch.cuda.is_available()):
        return torch.device("cuda")
    
    return torch.device("cpu")

def ensure_dir(directory: str) -> str:
    """
    Create a directory and all missing parent directories if needed.

    @param directory: Path to the directory that must exist.
    @type directory: str

    @return: The same directory path after creation.
    @rtype: str
    """
    os.makedirs(directory, exist_ok=True)
    return directory
