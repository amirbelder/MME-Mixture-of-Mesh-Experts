"""Mixed-framework MoE: two torch experts + one TF Keras expert.

Requires ``pip install mme[all]``. The TF expert is forward-only inside the
MoE — see ``docs/mixing_frameworks.md``.
"""

from __future__ import annotations

import examples.toy_expert  # noqa: F401  — registers torch experts
import numpy as np
from mme.core.mesh import Mesh
from mme.core.moe import MMEModel
from mme.data.synthetic import make_synthetic_mesh
from mme.experts.base import ExpertOutput
from mme.experts.registry import get_expert, register_expert
from mme.experts.tf_expert import TFExpert
from mme.gate.mme_gate_torch import MMEGateTorch


@register_expert("tf_toy_expert")
class TFToyExpert(TFExpert):
    framework = "tf"

    def __init__(self, num_classes: int, hidden: int = 64) -> None:
        super().__init__(num_classes=num_classes)
        import tensorflow as tf

        self._tf = tf
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(32,)),
                tf.keras.layers.Dense(hidden, activation="relu"),
                tf.keras.layers.Dense(num_classes),
            ]
        )
        self.feature_dim = hidden

    def preprocess(self, mesh: Mesh):
        tf = self._tf
        x = mesh.sampled_vertex_features(dim=32)
        return tf.convert_to_tensor(x[None])  # (1, 32)

    def forward(self, x) -> ExpertOutput:
        tf = self._tf
        # Extract features from the penultimate layer.
        feats = tf.keras.Model(
            inputs=self.model.inputs, outputs=self.model.layers[0].output
        )(x)
        logits = self.model(x)
        return ExpertOutput(
            logits=tf.squeeze(logits, axis=0), features=tf.squeeze(feats, axis=0)
        )

    def trainable_variables(self):
        return self.model.trainable_variables


def main() -> None:
    num_classes = 4
    experts = [
        get_expert("toy_mlp_small", num_classes=num_classes),
        get_expert("toy_mlp_wide", num_classes=num_classes),
        get_expert("tf_toy_expert", num_classes=num_classes),
    ]
    gate = MMEGateTorch(
        num_experts=len(experts), feature_dim=32, walk_len=16, num_walks=4
    )
    model = MMEModel(experts=experts, gate=gate)

    mesh = make_synthetic_mesh(shape="cube", seed=0)
    logits = model.forward([mesh])
    print("logits:", logits.detach().cpu().numpy())
    print("gate  :", model.last_gate_weights.detach().cpu().numpy())
    print(
        "(TF expert contributed frozen features; gradients did not cross the boundary.)"
    )


if __name__ == "__main__":
    main()
