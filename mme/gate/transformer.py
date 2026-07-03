"""Small transformer encoder used by the MME gate.

Kept intentionally tiny — depth 1-2 works well for the paper's gate. Batch-first.
"""

from __future__ import annotations


def build_small_transformer(
    d_model: int,
    n_heads: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.0,
):
    """Return a ``torch.nn.TransformerEncoder``. Imports torch lazily."""
    import torch.nn as nn

    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=True,
        activation="gelu",
    )
    return nn.TransformerEncoder(layer, num_layers=num_layers)
