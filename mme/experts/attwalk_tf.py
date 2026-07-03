"""AttWalkExpert — MME expert wrapping the paper's TF WalkTransformer.

Uses the same walk-sampling + feature-composition pipeline as
:class:`mme.gate.walk_hier_gate.WalkHierGate` — samplers from
:mod:`mme.gate.walk_algorithms`, feature builders from
:mod:`mme.gate.walk_features`. This guarantees the ``(num_walks, walk_len,
D)`` tensor fed to the vendored TF model matches ``train_val.py`` bit-for-bit.

Same numpy-bridge story as :class:`WalkHierGate` — gradients don't cross
between torch and TF. Train AttWalk with its own tf.GradientTape (either in
its own env, or inside the ``_maybe_tf_expert_step`` hook of
``examples/train_shrec11_joint.py``).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
from mme.core.mesh import Mesh
from mme.experts.base import ExpertOutput
from mme.experts.tf_expert import TFExpert
from mme.gate.walk_algorithms import sample_paper_walks
from mme.gate.walk_features import (
    batch_compose_walk_features,
    net_input_dim,
    norm_model,
)


class AttWalkExpert(TFExpert):
    """MME expert backed by the vendored TF WalkTransformer (AttWalk).

    Args (walk-side — MUST match how the paper trains AttWalk):
        walk_len: walk length (paper: 200 for classification).
        num_walks: walks per mesh (paper: 32).
        walk_alg: sampler name (default ``"random_global_jumps"``).
        net_input: feature spec (default ``("xyz",)`` — 3-dim positions).
        normalize: apply :func:`norm_model` to vertices first.

    Args (model-side):
        num_classes, d_model, num_layers, num_heads, dff, rate,
        out_features, last_layer_activation.

    Args (misc):
        walk_features_fn: optional override — signature
            ``(mesh, walks) -> np.ndarray[num_walks, walk_len, D]``.
        checkpoint_path: optional ``.keras`` file to restore on first forward.
        seed: RNG seed for walk sampling (rotates per forward).
        name: registered expert name.
    """

    framework = "tf"

    def __init__(
        self,
        num_classes: int,
        walk_len: int = 200,
        num_walks: int = 32,
        walk_alg: str = "random_global_jumps",
        net_input: Sequence[str] = ("xyz",),
        normalize: bool = True,
        walk_features_fn: Optional[Callable[[Mesh, np.ndarray], np.ndarray]] = None,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dff: int = 256,
        rate: float = 0.25,
        out_features: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        seed: int = 0,
        last_layer_activation: Optional[str] = None,
        name: str = "attwalk",
        constant_jump_k: int = 10,
    ) -> None:
        super().__init__(num_classes=num_classes)
        self.name = name
        self.walk_len = walk_len
        self.num_walks = num_walks
        self.walk_alg = walk_alg
        self.net_input = tuple(net_input)
        self.normalize = normalize
        self.walk_features_fn = walk_features_fn
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        self.out_features = out_features if out_features is not None else d_model
        self.checkpoint_path = checkpoint_path
        self.seed = int(seed)
        self.last_layer_activation = last_layer_activation
        self.constant_jump_k = constant_jump_k

        self._tf_model = None
        self._net_input_dim = None

    # ------------------------------------------------------------------
    def _build(self, net_input_dim_: int) -> None:
        from mme.gate.walk_hier_transformer_tf import WalkTransformer

        self._tf_model = WalkTransformer(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.dff,
            input_vocab_size=net_input_dim_,
            out_features=self.out_features,
            pe_input=self.walk_len,
            pe_target=self.walk_len,
            num_classes=self.num_classes,
            net_input_dim=net_input_dim_,
            last_layer_activation=self.last_layer_activation,
            one_label_per_model=True,
            rate=self.rate,
        )
        self._net_input_dim = net_input_dim_
        if self.checkpoint_path is not None:
            self._tf_model.load_weights(self.checkpoint_path)

    # ------------------------------------------------------------------
    def trainable_variables(self):
        if self._tf_model is None:
            return []
        return self._tf_model.trainable_variables

    # ------------------------------------------------------------------
    def _build_walk_tensor(self, mesh: Mesh) -> np.ndarray:
        if self.walk_features_fn is not None:
            seqs, _ = sample_paper_walks(
                mesh,
                num_walks=self.num_walks,
                walk_len=self.walk_len,
                walk_alg=self.walk_alg,
                seed=self.seed,
                constant_jump_k=self.constant_jump_k,
            )
            feats = np.asarray(self.walk_features_fn(mesh, seqs), dtype=np.float32)
            if feats.ndim != 3:
                raise ValueError(
                    f"walk_features_fn must return (num_walks, walk_len, D); got {feats.shape}"
                )
            return feats

        vertices = (
            norm_model(mesh.vertices)
            if self.normalize
            else mesh.vertices.astype(np.float32)
        )
        seqs, jumps = sample_paper_walks(
            mesh,
            num_walks=self.num_walks,
            walk_len=self.walk_len,
            walk_alg=self.walk_alg,
            seed=self.seed,
            constant_jump_k=self.constant_jump_k,
        )
        return batch_compose_walk_features(
            vertices, seqs, jumps, self.walk_len, spec=self.net_input
        )

    # ------------------------------------------------------------------
    def preprocess(self, mesh: Mesh) -> np.ndarray:
        feats = self._build_walk_tensor(mesh)
        self.seed += 1
        return feats

    def forward(self, inputs: np.ndarray) -> ExpertOutput:
        import tensorflow as tf

        if self._tf_model is None:
            self._build(net_input_dim_=int(inputs.shape[-1]))
        elif inputs.shape[-1] != self._net_input_dim:
            raise ValueError(
                f"walk feature dim changed: model was built for "
                f"{self._net_input_dim}, got {inputs.shape[-1]}"
            )

        tf_inp = tf.constant(inputs)
        # Per-walk logits → mean across walks → per-mesh (num_classes,).
        tf_logits = self._tf_model(tf_inp, training=False, classify=True)
        logits = tf_logits.numpy().mean(axis=0)

        # Penultimate features, pooled the same way.
        tf_feats = self._tf_model(tf_inp, training=False, classify=False)
        features = np.asarray(tf_feats).mean(axis=(0, 1))

        return ExpertOutput(logits=logits, features=features)

    # ------------------------------------------------------------------
    @property
    def expected_input_dim(self) -> int:
        return net_input_dim(self.net_input)
