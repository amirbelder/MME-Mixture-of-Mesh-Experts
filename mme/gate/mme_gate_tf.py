"""MME gate — TensorFlow mirror.

Provided for parity with the paper's TF reference. Not used by the default
``MMEModel`` (which is torch-hosted). Keep it functionally aligned with
``mme_gate_torch.py`` so users can train a TF-only MoE if they prefer.

TODO(amir): replace with your paper's TF gate; this is a straightforward
port from the torch version for reference.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from mme.core.mesh import Mesh
from mme.experts.base import ExpertOutput
from mme.gate.random_walk import sample_walks


class MMEGateTF:
    def __init__(
        self,
        num_experts: int,
        feature_dim: int = 32,
        walk_len: int = 32,
        num_walks: int = 8,
        num_heads: int = 4,
        num_layers: int = 2,
        seed: int = 0,
    ) -> None:
        try:
            import tensorflow as tf
        except ImportError as e:  # pragma: no cover — TF is an optional extra
            raise ImportError(
                "MMEGateTF requires tensorflow. Install with mme[tf]."
            ) from e

        self._tf = tf
        self.num_experts = num_experts
        self.feature_dim = feature_dim
        self.walk_len = walk_len
        self.num_walks = num_walks
        self.seed = seed

        self.walk_input_proj = tf.keras.layers.Dense(feature_dim)
        self.walk_pe = self._sinusoidal_pe(walk_len, feature_dim)
        self.walk_transformer_layers = [
            tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=feature_dim)
            for _ in range(num_layers)
        ]
        self.walk_ffn = [
            tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(feature_dim * 4, activation="gelu"),
                    tf.keras.layers.Dense(feature_dim),
                ]
            )
            for _ in range(num_layers)
        ]
        self.expert_proj = tf.keras.layers.Dense(feature_dim)
        self.cross_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=feature_dim
        )
        self.score_head = tf.keras.layers.Dense(1)

    def _sinusoidal_pe(self, length: int, dim: int):
        tf = self._tf
        pe = np.zeros((length, dim), dtype=np.float32)
        position = np.arange(length)[:, None]
        div_term = np.exp(np.arange(0, dim, 2) * (-np.log(10000.0) / dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return tf.constant(pe)

    def forward(self, mesh: Mesh, expert_outputs: Sequence[ExpertOutput]):
        tf = self._tf
        if len(expert_outputs) != self.num_experts:
            raise ValueError(
                f"gate expected {self.num_experts} experts, got {len(expert_outputs)}"
            )
        walks = sample_walks(
            mesh, num_walks=self.num_walks, walk_len=self.walk_len, seed=self.seed
        )
        self.seed += 1
        pos = mesh.vertices[walks]
        vert_normals = np.zeros_like(mesh.vertices)
        counts = np.zeros(mesh.num_vertices, dtype=np.float32)
        if mesh.num_faces > 0:
            fn = mesh.face_normals
            for k in range(3):
                np.add.at(vert_normals, mesh.faces[:, k], fn)
                np.add.at(counts, mesh.faces[:, k], 1.0)
            counts = np.clip(counts, 1.0, None)
            vert_normals = vert_normals / counts[:, None]
        vn = vert_normals[walks]
        is_repeat = np.zeros(walks.shape, dtype=np.float32)
        is_repeat[:, 1:] = (walks[:, 1:] == walks[:, :-1]).astype(np.float32)
        feats = np.concatenate([pos, vn, is_repeat[..., None]], axis=-1).astype(
            np.float32
        )
        x = tf.constant(feats)
        x = self.walk_input_proj(x) + self.walk_pe[None]
        for attn, ffn in zip(self.walk_transformer_layers, self.walk_ffn):
            x = attn(x, x) + x
            x = ffn(x) + x
        walk_tokens = tf.reduce_mean(x, axis=1)
        query = tf.reduce_mean(walk_tokens, axis=0, keepdims=True)[None]
        expert_feats = []
        for o in expert_outputs:
            f = o.features if o.features is not None else o.logits
            f = tf.reshape(tf.cast(tf.convert_to_tensor(f), tf.float32), [-1])
            if int(f.shape[0]) != self.feature_dim:
                # Simple lazy dense projection.
                proj = tf.keras.layers.Dense(self.feature_dim)
                f = proj(f[None])[0]
            expert_feats.append(f)
        keys = tf.stack(expert_feats, axis=0)[None]
        keys = self.expert_proj(keys)
        attn_out = self.cross_attn(query, keys)
        scores = tf.squeeze(self.score_head(attn_out + keys), axis=-1)
        return tf.squeeze(scores, axis=0)
