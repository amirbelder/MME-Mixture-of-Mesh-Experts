"""Segmentation eval — per-face mIoU. Skeleton implementation.

Assumes ``MMEModel.forward`` returns per-face logits of shape
``(1, num_faces, num_classes)`` and each mesh carries a per-face label array
in ``mesh.face_features`` where the first column is the target class.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from mme.core.mesh import Mesh
from mme.core.moe import MMEModel


def evaluate_segmentation(model: MMEModel, meshes: Sequence[Mesh]) -> dict:
    import torch

    model.eval()
    ious_per_class: dict = {}
    with torch.no_grad():
        for m in meshes:
            if m.face_features is None:
                raise ValueError(
                    "segmentation eval requires per-face labels in mesh.face_features"
                )
            targets = m.face_features[:, 0].astype(np.int64)
            logits = model.forward([m])
            # Accept either (1, F, C) or (F, C).
            if logits.dim() == 3:
                logits = logits.squeeze(0)
            preds = logits.argmax(dim=-1).cpu().numpy()
            classes = np.unique(np.concatenate([preds, targets]))
            for c in classes:
                p = preds == c
                t = targets == c
                inter = np.logical_and(p, t).sum()
                union = np.logical_or(p, t).sum()
                if union == 0:
                    continue
                ious_per_class.setdefault(int(c), []).append(inter / union)
    miou = (
        float(np.mean([np.mean(v) for v in ious_per_class.values()]))
        if ious_per_class
        else 0.0
    )
    return {"mIoU": miou, "count": len(meshes)}
