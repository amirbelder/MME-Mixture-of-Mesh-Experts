"""Diversity term — encourages experts' features (or predictions) to disagree.

TODO(amir): the paper's exact form uses cosine-similarity between L2-
normalized expert feature vectors, averaged over the upper triangle. Verify
the sign / normalization matches your reference implementation.
"""

from __future__ import annotations

from typing import Sequence


def diversity_loss(expert_features: Sequence) -> "object":
    """Mean pairwise cosine similarity of L2-normalized expert features.

    Lower is more diverse — we *return* this quantity so it can be added
    directly to the total loss with a positive coefficient. Users may sign-
    flip to *encourage* diversity depending on their formulation.
    """
    import torch
    import torch.nn.functional as F

    feats = torch.stack(
        [torch.as_tensor(f).reshape(-1).float() for f in expert_features], dim=0
    )
    feats = F.normalize(feats, dim=-1)
    sim = feats @ feats.t()  # (E, E)
    e = sim.shape[0]
    if e < 2:
        return torch.zeros((), device=sim.device)
    mask = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
    return sim[mask].mean()
