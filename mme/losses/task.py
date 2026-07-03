"""Task loss — cross-entropy for classification / segmentation."""

from __future__ import annotations


def task_ce_loss(logits, targets):
    """Cross-entropy over the last dim of ``logits`` against integer ``targets``.

    Shapes:
        classification: logits (B, C), targets (B,)
        segmentation:   logits (B, N, C) or (N, C), targets (B, N) or (N,)
    """
    import torch
    import torch.nn.functional as F

    logits = torch.as_tensor(logits)
    targets = torch.as_tensor(targets).long()
    if logits.dim() == 2:
        return F.cross_entropy(logits, targets)
    if logits.dim() == 3:
        b, n, c = logits.shape
        return F.cross_entropy(logits.reshape(b * n, c), targets.reshape(b * n))
    raise ValueError(f"unsupported logits shape {tuple(logits.shape)}")
