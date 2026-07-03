"""Pytest bootstrap: make tf tests skip cleanly and register a helper."""

import pytest


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _has_tf() -> bool:
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        return False
    return True


needs_torch = pytest.mark.skipif(not _has_torch(), reason="requires torch")
needs_tf = pytest.mark.skipif(not _has_tf(), reason="requires tensorflow")
