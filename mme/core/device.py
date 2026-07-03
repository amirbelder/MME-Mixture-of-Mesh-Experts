"""Device helpers for torch and (optional) tensorflow."""

from __future__ import annotations

from typing import Optional


def torch_device(prefer: str = "auto") -> "object":
    """Return a torch.device, importing torch lazily.

    ``prefer`` is one of: "auto", "cpu", "cuda", or "cuda:N". Auto picks CUDA
    if available.
    """
    import torch

    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(prefer)


def tf_visible_devices(prefer: str = "auto") -> Optional[list]:
    """Return the list of visible logical devices for TF; None if TF not installed."""
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError:
        return None
    return tf.config.list_logical_devices()
