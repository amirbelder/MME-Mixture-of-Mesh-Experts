"""Simple metrics."""

from __future__ import annotations


def accuracy(logits, targets) -> float:
    import torch

    preds = torch.as_tensor(logits).argmax(dim=-1)
    return float((preds == torch.as_tensor(targets)).float().mean())
