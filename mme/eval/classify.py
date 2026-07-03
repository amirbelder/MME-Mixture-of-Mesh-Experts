"""Classification evaluation."""

from __future__ import annotations

from typing import Sequence

from mme.core.mesh import Mesh
from mme.core.moe import MMEModel
from mme.training.metrics import accuracy


def evaluate_classification(
    model: MMEModel, meshes: Sequence[Mesh], batch_size: int = 8
) -> dict:
    import torch

    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for start in range(0, len(meshes), batch_size):
            batch = list(meshes[start : start + batch_size])
            targets = torch.tensor(
                [m.label if m.label is not None else 0 for m in batch], dtype=torch.long
            )
            logits = model.forward(batch)
            preds = logits.argmax(dim=-1)
            correct += int((preds.cpu() == targets).sum())
            total += len(batch)
            all_preds.append(preds.cpu())
            all_targets.append(targets)
    return {
        "accuracy": correct / max(1, total),
        "count": total,
    }
