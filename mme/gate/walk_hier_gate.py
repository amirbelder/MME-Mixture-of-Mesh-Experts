"""WalkHierGate — MMEModel adapter around the paper's TF WalkHierTransformer.

Walks are sampled and encoded EXACTLY like the reference project's
``train_val.py``:

    1. Optionally normalize vertices via :func:`mme.gate.walk_features.norm_model`
       (center + scale — matches ``dataset.norm_model``).
    2. Sample ``num_walks`` walks of ``walk_len + 1`` vertices using
       :mod:`mme.gate.walk_algorithms` (default: ``random_global_jumps`` —
       the paper's default).
    3. Compose per-vertex features via
       :func:`mme.gate.walk_features.compose_walk_features` per the
       ``net_input`` list (default: ``('xyz',)`` — 3-dim positions).
    4. Feed ``(num_walks, walk_len, D)`` to the vendored TF
       :class:`WalkHierTransformer` (identical to reference ``train_step``'s
       ``dnn_model(model_ftrs)`` after its reshape).
    5. Mean-pool per-walk logits into a single ``(num_experts,)`` torch tensor.

Gradients do NOT flow across the numpy bridge. Train the gate with its own
``tf.GradientTape`` alongside the MoE step, or freeze it.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
from mme.core.mesh import Mesh
from mme.experts.base import ExpertOutput
from mme.gate.walk_algorithms import sample_paper_walks
from mme.gate.walk_features import (
    batch_compose_walk_features,
    net_input_dim,
    norm_model,
)


class WalkHierGate:
    """Bridge from the TF WalkHierTransformer to the MMEModel gate interface.

    Args:
        num_experts: number of experts to score (== ``num_classes`` in the TF model).
        walk_len: vertices per random walk (paper: 100–800).
        num_walks: walks per mesh (paper: 8–32).
        walk_alg: sampler name from :data:`mme.gate.walk_algorithms.WALK_ALGORITHMS`.
            Default ``"random_global_jumps"`` matches the paper.
        net_input: feature composition, matching ``params.net_input`` in the
            reference. Default ``("xyz",)`` = 3-dim positions.
        normalize: apply :func:`norm_model` to vertices before feature build.
        walk_features_fn: optional override callable
            ``(mesh, walks) -> np.ndarray[num_walks, walk_len, D]`` — takes
            precedence over ``net_input`` / ``normalize`` if provided.
        d_model, num_layers, num_heads, dff, jump_every_k, pooling,
        concat_xyz, recurrent, rate, global_dim_mult, num_scales,
        last_layer_activation: passed to :class:`WalkHierTransformer` unchanged.
        seed: RNG seed for walk sampling (rotates per forward for stability).

    Notes:
        - The underlying TF model is built lazily on the first ``forward``
          call, using the observed feature dim.
        - Call ``load_weights(path)`` after the first forward to restore a
          ``.keras`` checkpoint from the reference project.
    """

    num_experts: int

    def __init__(
        self,
        num_experts: int,
        walk_len: int = 100,
        num_walks: int = 32,
        walk_alg: str = "random_global_jumps",
        net_input: Sequence[str] = ("xyz",),
        normalize: bool = True,
        walk_features_fn: Optional[Callable[[Mesh, np.ndarray], np.ndarray]] = None,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dff: int = 256,
        jump_every_k: int = 20,
        pooling: bool = True,
        concat_xyz: bool = False,
        recurrent: bool = False,
        rate: float = 0.0,
        global_dim_mult: int = 1,
        num_scales: Optional[int] = None,
        seed: int = 0,
        last_layer_activation: Optional[str] = None,
        path: str = None,
        constant_jump_k: int = 10,

    ) -> None:
        self.num_experts = num_experts
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
        self.jump_every_k = jump_every_k
        self.pooling = pooling
        self.concat_xyz = concat_xyz
        self.recurrent = recurrent
        self.rate = rate
        self.global_dim_mult = global_dim_mult
        self.num_scales = num_scales
        self.last_layer_activation = last_layer_activation
        self.seed = int(seed)
        self.constant_jump_k = constant_jump_k

        self._tf_model = None
        self._net_input_dim = None

    # ------------------------------------------------------------------
    def _build(self, net_input_dim_: int) -> None:
        from mme.gate.walk_hier_transformer_tf import WalkHierTransformer
        if self.path is not None:
            self.load_weights(self.path)
        else:
            self._tf_model = WalkHierTransformer(
                num_layers=self.num_layers,
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.dff,
                input_vocab_size=net_input_dim_,
                out_features=self.num_experts,
                pe_input=self.walk_len,
                pe_target=self.walk_len,
                num_classes=self.num_experts,
                net_input_dim=net_input_dim_,
                seq_len=self.walk_len,
                last_layer_activation=self.last_layer_activation,
                one_label_per_model=True,
                rate=self.rate,
                jump_every_k=self.jump_every_k,
                pooling=self.pooling,
                concat_xyz=self.concat_xyz,
                num_scales=self.num_scales,
                global_dim_mult=self.global_dim_mult,
                recurrent=self.recurrent,
            )
            self._net_input_dim = net_input_dim_

    # ------------------------------------------------------------------
    def parameters(self):
        return []  # TF vars aren't torch params — gradients don't cross the bridge.

    def to(self, device):
        return self

    def train(self, mode: bool = True):
        return self

    # ------------------------------------------------------------------
    def _build_walk_tensor(self, mesh: Mesh) -> np.ndarray:
        """Return ``(num_walks, walk_len, D)`` — matches ``train_val.py``.

        User override via ``walk_features_fn`` wins; otherwise composes per
        ``net_input`` after (optional) ``norm_model`` normalization.
        """
        if self.walk_features_fn is not None:
            # Legacy path: user supplies both walks and features in one shot.
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
    def forward(
        self,
        mesh: Mesh,
        expert_outputs: Sequence[ExpertOutput],
        **_ignored,
    ):
        import torch

        if len(expert_outputs) != self.num_experts:
            raise ValueError(
                f"WalkHierGate: expected {self.num_experts} experts, got {len(expert_outputs)}"
            )

        feats = self._build_walk_tensor(mesh)  # (num_walks, walk_len, D)
        self.seed += 1

        if self._tf_model is None:
            self._build(net_input_dim_=int(feats.shape[-1]))
        elif feats.shape[-1] != self._net_input_dim:
            raise ValueError(
                f"walk feature dim changed: model was built for "
                f"{self._net_input_dim}, got {feats.shape[-1]}"
            )

        import tensorflow as tf

        tf_inp = tf.constant(feats)
        with tf.device(
            "/CPU:0" if not tf.config.list_physical_devices("GPU") else "/GPU:0"
        ):
            tf_logits = self._tf_model(tf_inp, training=False, classify=True)
        # Reference test_step treats each walk as its own prediction with the
        # same label; we mean-pool per-walk logits to get one (num_experts,)
        # vector per mesh.
        scores = tf_logits.numpy().mean(axis=0).astype(np.float32)
        return torch.from_numpy(scores)

    # ------------------------------------------------------------------
    def load_weights(self, path: str) -> None:
        if self._tf_model is None:
            raise RuntimeError(
                "call forward() at least once (or _build()) before load_weights"
            )
        self._tf_model.load_weights(path)

    def save_weights(self, path: str) -> None:
        if self._tf_model is None:
            raise RuntimeError("model not built yet — call forward() first")
        self._tf_model.save_weights(path)

    # ------------------------------------------------------------------
    @property
    def expected_input_dim(self) -> int:
        """The D that this gate will feed to the TF model, without running it."""
        return net_input_dim(self.net_input)
