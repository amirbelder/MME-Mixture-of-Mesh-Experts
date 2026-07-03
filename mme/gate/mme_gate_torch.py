"""MME gate — PyTorch (canonical implementation).

TODO(amir): This is a re-implementation from the paper description.
Replace with your reference torch gate when convenient and verify shapes /
init / walk encoding match. Kept small on purpose:

  1. Sample ``num_walks`` random walks of ``walk_len`` vertices on the mesh.
  2. Embed vertex features (positions + normals summary) at every walked
     vertex, add positional encoding along the walk, run a small
     ``TransformerEncoder`` to get one summary token per walk.
  3. Cross-attend the pooled walk summary against the stacked expert
     features to produce per-expert scores.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from mme.core.mesh import Mesh
from mme.experts.base import ExpertOutput
from mme.gate.random_walk import sample_walks
from mme.gate.transformer import build_small_transformer


def _sinusoidal_pe(length: int, dim: int):
    """Standard sinusoidal positional encoding as a torch tensor of shape (length, dim)."""
    import torch

    pe = torch.zeros(length, dim)
    position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-np.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


class MMEGateTorch:
    """Random-walk transformer gate over experts (paper-based, torch).

    Args:
        num_experts: number of experts to score.
        feature_dim: hidden width for the walk transformer AND the expected
            per-expert feature width. Expert ``features`` will be projected
            to this width if they differ.
        walk_len: vertices per random walk.
        num_walks: number of random walks per mesh.
        num_heads: transformer heads.
        num_layers: transformer encoder depth.
        seed: RNG seed for walk sampling (walks are also refreshed per call).
    """

    num_experts: int

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
        import torch.nn as nn

        self.num_experts = num_experts
        self.feature_dim = feature_dim
        self.walk_len = walk_len
        self.num_walks = num_walks
        self.seed = seed

        # Input to the walk transformer: at each walked vertex we use
        # [position(3), face-normal-avg(3), self-loop-indicator(1)] = 7 dims.
        walk_in = 7
        self.walk_input_proj = nn.Linear(walk_in, feature_dim)
        self.walk_pe = _sinusoidal_pe(walk_len, feature_dim)
        self.walk_transformer = build_small_transformer(
            d_model=feature_dim,
            n_heads=num_heads,
            num_layers=num_layers,
        )

        # Cross-attention from pooled walk token (query) to expert features (keys/values).
        self.expert_proj = nn.Linear(feature_dim, feature_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, batch_first=True
        )
        self.score_head = nn.Linear(feature_dim, 1)

        # A lazy per-input feature projection (in case an expert's features have
        # a different width than feature_dim). Built on first use.
        self._feature_projs: dict = {}

    # ------------------------------------------------------------------
    def parameters(self):
        params = list(self.walk_input_proj.parameters())
        params += list(self.walk_transformer.parameters())
        params += list(self.expert_proj.parameters())
        params += list(self.cross_attn.parameters())
        params += list(self.score_head.parameters())
        for p in self._feature_projs.values():
            params += list(p.parameters())
        return params

    def to(self, device):
        self.walk_input_proj.to(device)
        self.walk_transformer.to(device)
        self.expert_proj.to(device)
        self.cross_attn.to(device)
        self.score_head.to(device)
        self.walk_pe = self.walk_pe.to(device)
        for p in self._feature_projs.values():
            p.to(device)
        self._device = device
        return self

    def train(self, mode: bool = True):
        self.walk_input_proj.train(mode)
        self.walk_transformer.train(mode)
        self.expert_proj.train(mode)
        self.cross_attn.train(mode)
        self.score_head.train(mode)
        return self

    # ------------------------------------------------------------------
    def _project_expert_features(self, feats):
        """Project a (D,) feature vector to (feature_dim,)."""
        import torch.nn as nn

        d = int(feats.shape[-1])
        if d == self.feature_dim:
            return feats
        key = str(d)
        if key not in self._feature_projs:
            proj = nn.Linear(d, self.feature_dim)
            device = getattr(self, "_device", None)
            if device is not None:
                proj.to(device)
            self._feature_projs[key] = proj
        return self._feature_projs[key](feats)

    def _build_walk_features(self, mesh: Mesh, walks: np.ndarray):
        """Return torch tensor of shape (num_walks, walk_len, 7)."""
        import torch

        pos = mesh.vertices[walks]  # (W, L, 3)
        # For each vertex, average of adjacent face normals if available.
        # Cheap approximation: use per-face normals aggregated per vertex.
        vert_normals = np.zeros_like(mesh.vertices)
        counts = np.zeros(mesh.num_vertices, dtype=np.float32)
        if mesh.num_faces > 0:
            fn = mesh.face_normals
            for k in range(3):
                np.add.at(vert_normals, mesh.faces[:, k], fn)
                np.add.at(counts, mesh.faces[:, k], 1.0)
            counts = np.clip(counts, 1.0, None)
            vert_normals = vert_normals / counts[:, None]
        vn = vert_normals[walks]  # (W, L, 3)
        # self-loop indicator: 1 if step repeated (isolated vertex fallback).
        is_repeat = np.zeros(walks.shape, dtype=np.float32)
        is_repeat[:, 1:] = (walks[:, 1:] == walks[:, :-1]).astype(np.float32)
        feats = np.concatenate([pos, vn, is_repeat[..., None]], axis=-1).astype(
            np.float32
        )
        return torch.from_numpy(feats)

    # ------------------------------------------------------------------
    def forward(self, mesh: Mesh, expert_outputs: Sequence[ExpertOutput]):
        """Score experts for one mesh.

        Returns a torch tensor of shape ``[num_experts]`` — unnormalized
        logits (MMEModel applies softmax when combining).
        """
        import torch

        if len(expert_outputs) != self.num_experts:
            raise ValueError(
                f"gate expected {self.num_experts} experts, got {len(expert_outputs)}"
            )

        # Refresh walks each call so we don't overfit to one sample.
        walks = sample_walks(
            mesh, num_walks=self.num_walks, walk_len=self.walk_len, seed=self.seed
        )
        self.seed += 1  # rotate

        walk_feats = self._build_walk_features(mesh, walks)  # (W, L, 7)
        device = getattr(self, "_device", walk_feats.device)
        walk_feats = walk_feats.to(device)

        x = self.walk_input_proj(walk_feats)  # (W, L, d)
        x = x + self.walk_pe.to(device)[None]  # add PE (broadcast over walks)
        x = self.walk_transformer(x)  # (W, L, d)
        walk_tokens = x.mean(dim=1)  # (W, d) — pooled per walk

        # Query = mean walk token; Keys/Values = expert features.
        query = walk_tokens.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, d)

        expert_feats = []
        for o in expert_outputs:
            f = o.features if o.features is not None else o.logits
            f = torch.as_tensor(f).to(device).float().reshape(-1)
            f = self._project_expert_features(f)
            expert_feats.append(f)
        keys = torch.stack(expert_feats, dim=0).unsqueeze(0)  # (1, E, d)
        keys = self.expert_proj(keys)

        attn_out, _ = self.cross_attn(
            query, keys, keys, need_weights=False
        )  # (1, 1, d)
        # Score each expert by dotting projected expert token with the attended query.
        # More precisely: use the additive score head over (attn_out + key_i).
        scores = self.score_head(attn_out + keys).squeeze(-1).squeeze(0)  # (E,)
        return scores
