"""Retrieval eval — mean average precision using expert/moe features."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from mme.core.mesh import Mesh
from mme.core.moe import MMEModel


def evaluate_retrieval(model: MMEModel, meshes: Sequence[Mesh]) -> dict:
    """Compute mAP@all using the MoE final logits as feature vectors.

    Each mesh is a query; positives are same-``label`` meshes. Cosine
    similarity. Simple and honest for a toy setting.
    """
    import torch

    model.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for m in meshes:
            logits = model.forward([m])
            feats.append(logits.reshape(-1).cpu().numpy())
            labels.append(m.label if m.label is not None else -1)
    f = np.stack(feats).astype(np.float32)
    f = f / np.clip(np.linalg.norm(f, axis=1, keepdims=True), 1e-8, None)
    sims = f @ f.T
    labels_arr = np.asarray(labels)
    aps = []
    for i in range(len(meshes)):
        order = np.argsort(-sims[i])
        order = order[order != i]
        rel = (labels_arr[order] == labels_arr[i]).astype(np.float32)
        if rel.sum() == 0:
            continue
        precision = np.cumsum(rel) / (np.arange(len(rel)) + 1)
        ap = float((precision * rel).sum() / rel.sum())
        aps.append(ap)
    return {"mAP": float(np.mean(aps)) if aps else 0.0, "count": len(meshes)}
