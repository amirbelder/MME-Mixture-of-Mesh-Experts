"""Three tiny torch experts operating on a per-mesh feature vector.

Importing this module registers ``toy_mlp_small``, ``toy_mlp_wide``, and
``toy_mlp_deep`` with the MME registry.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from mme.core.mesh import Mesh
from mme.experts.base import ExpertOutput
from mme.experts.registry import register_expert
from mme.experts.torch_expert import TorchExpert


INPUT_DIM = 32


def _preprocess(mesh: Mesh) -> torch.Tensor:
    x = mesh.sampled_vertex_features(dim=INPUT_DIM)
    return torch.from_numpy(x).float()


class _ToyMLP(TorchExpert, nn.Module):
    """Shared implementation. Subclasses just choose hidden width/depth."""

    def __init__(
        self, num_classes: int, hidden: int, depth: int, activation: str = "relu"
    ) -> None:
        nn.Module.__init__(self)
        TorchExpert.__init__(self, num_classes=num_classes)
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]
        layers: list = [nn.Linear(INPUT_DIM, hidden), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), act()]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, num_classes)
        self.feature_dim = hidden

    def preprocess(self, mesh: Mesh) -> torch.Tensor:
        return _preprocess(mesh)

    def forward(self, x: torch.Tensor) -> ExpertOutput:
        feats = self.backbone(x)
        logits = self.head(feats)
        return ExpertOutput(logits=logits, features=feats)


@register_expert("toy_mlp_small")
class ToyMLPSmall(_ToyMLP):
    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes=num_classes, hidden=32, depth=1, activation="relu")


@register_expert("toy_mlp_wide")
class ToyMLPWide(_ToyMLP):
    def __init__(self, num_classes: int) -> None:
        super().__init__(
            num_classes=num_classes, hidden=128, depth=1, activation="gelu"
        )


@register_expert("toy_mlp_deep")
class ToyMLPDeep(_ToyMLP):
    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes=num_classes, hidden=64, depth=3, activation="tanh")
