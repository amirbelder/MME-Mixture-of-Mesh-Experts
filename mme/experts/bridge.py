"""Numpy bridge between torch and tf tensors.

Explicit, small, and side-effect-free. All conversions go through numpy;
this means gradients do NOT cross the framework boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def to_numpy(x: Any) -> np.ndarray:
    """Convert a torch / tf / numpy array-like to numpy."""
    if isinstance(x, np.ndarray):
        return x
    # torch tensor?
    try:
        import torch  # noqa: F401

        if hasattr(x, "detach") and hasattr(x, "cpu"):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    # tf tensor?
    try:
        import tensorflow as tf  # noqa: F401

        if hasattr(x, "numpy"):
            return x.numpy()
    except ImportError:
        pass
    return np.asarray(x)


def to_torch(x: Any) -> "object":
    """Convert to torch.Tensor (importing torch lazily)."""
    import torch

    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(to_numpy(x))


def to_tf(x: Any) -> "object":
    """Convert to tf.Tensor (importing tensorflow lazily)."""
    import tensorflow as tf

    return tf.convert_to_tensor(to_numpy(x))
