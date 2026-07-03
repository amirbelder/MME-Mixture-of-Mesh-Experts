"""Minimal script: register experts, build the model, run one forward pass."""

from __future__ import annotations

import examples.toy_expert  # noqa: F401 — registers 3 experts
from mme.core.moe import MMEModel
from mme.data.synthetic import make_synthetic_mesh
from mme.experts.registry import get_expert, list_experts
from mme.gate.mme_gate_torch import MMEGateTorch


def main() -> None:
    print("registered experts:", list_experts())

    num_classes = 4
    experts = [
        get_expert("toy_mlp_small", num_classes=num_classes),
        get_expert("toy_mlp_wide", num_classes=num_classes),
        get_expert("toy_mlp_deep", num_classes=num_classes),
    ]
    gate = MMEGateTorch(
        num_experts=len(experts), feature_dim=32, walk_len=16, num_walks=4
    )
    model = MMEModel(experts=experts, gate=gate)

    mesh = make_synthetic_mesh(shape="sphere", seed=0)
    logits = model.forward([mesh])
    print("logits:", logits.shape, logits.detach().cpu().numpy())
    print("gate  :", model.last_gate_weights.detach().cpu().numpy())


if __name__ == "__main__":
    main()
