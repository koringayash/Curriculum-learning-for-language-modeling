"""
config_global.py
Global configuration shared across all modules.
Handles project paths, device auto-detection, and reproducibility.
"""
import os
import torch
import random
import numpy as np

# 📍 Project root (absolute)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def get_device() -> str:
    """Auto-detect best available device: CUDA -> MPS -> CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def set_seed(seed: int = 42) -> None:
    """Set random seeds for python, numpy, and torch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN for reproducible training on same hardware
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = get_device()
RANDOM_SEED = 42