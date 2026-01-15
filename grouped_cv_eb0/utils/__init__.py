"""
Utility functions for grouped CV pipeline.
"""

import random
import numpy as np
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RANDOM_SEED


def set_seed(seed=None):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed (default from config)
    """
    if seed is None:
        seed = RANDOM_SEED
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"  ✓ Random seed set to {seed}")


def get_device_info():
    """
    Get information about available devices.
    
    Returns:
        dict: Device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        info['memory_allocated'] = torch.cuda.memory_allocated(0) / 1e9  # GB
        info['memory_reserved'] = torch.cuda.memory_reserved(0) / 1e9  # GB
    
    return info


def print_device_info():
    """Print device information."""
    info = get_device_info()
    
    print(f"\n  Device Information:")
    print(f"    CUDA available: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"    Device: {info['device_name']}")
        print(f"    CUDA version: {info['cuda_version']}")
        print(f"    cuDNN version: {info['cudnn_version']}")
    else:
        print(f"    Using CPU")


def count_parameters(model, trainable_only=False):
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: Only count trainable parameters
        
    Returns:
        int: Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_number(n):
    """Format large numbers with K/M/B suffixes."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
