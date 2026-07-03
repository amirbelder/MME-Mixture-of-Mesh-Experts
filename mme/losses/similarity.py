"""Similarity term — encourages experts to agree on the *right* answer.

TODO(amir): the paper defines similarity as KL between each expert's softmax
prediction and the gated MoE prediction. Verify direction (which is the
reference distribution) matches your implementation.
"""

from __future__ import annotations

from typing import Sequence


def similarity_loss(expert_logits: Sequence, moe_logits) -> "object":
    """Mean KL(softmax(moe) || softmax(expert)) over experts."""
    import torch
    import torch.nn.functional as F

    moe_logp = F.log_softmax(torch.as_tensor(moe_logits).reshape(-1).float(), dim=-1)
    moe_p = moe_logp.exp()
    losses = []
    for e in expert_logits:
        e_logp = F.log_softmax(torch.as_tensor(e).reshape(-1).float(), dim=-1)
        losses.append(F.kl_div(e_logp, moe_p, reduction="sum"))
    return torch.stack(losses).mean()
