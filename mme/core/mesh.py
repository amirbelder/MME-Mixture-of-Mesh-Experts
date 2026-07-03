"""Mesh container — a numpy-first representation of a triangle mesh.

Framework-neutral. Experts convert to torch/tf tensors inside their own
``preprocess`` methods; the container itself never imports torch or tf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Mesh:
    """Triangle mesh in numpy.

    Attributes:
        vertices: (V, 3) float32 vertex positions.
        faces: (F, 3) int64 triangle indices into ``vertices``.
        label: Optional integer class label (used by the toy classification example).
        vertex_features: Optional (V, D_v) per-vertex features.
        face_features: Optional (F, D_f) per-face features.
    """

    vertices: np.ndarray
    faces: np.ndarray
    label: Optional[int] = None
    vertex_features: Optional[np.ndarray] = None
    face_features: Optional[np.ndarray] = None
    # Optional stable cross-process identifier for this mesh (usually its file
    # path). Used by PrerenderedExpert to look up precomputed logits.
    source_path: Optional[str] = None

    # Cached derived quantities. Computed lazily; users normally shouldn't set these.
    _edges: Optional[np.ndarray] = field(default=None, repr=False)
    _face_adjacency: Optional[np.ndarray] = field(default=None, repr=False)
    _face_normals: Optional[np.ndarray] = field(default=None, repr=False)
    _vertex_neighbors: Optional[list] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float32)
        self.faces = np.asarray(self.faces, dtype=np.int64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"vertices must be (V, 3); got {self.vertices.shape}")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"faces must be (F, 3); got {self.faces.shape}")
        if self.faces.size and self.faces.max() >= len(self.vertices):
            raise ValueError("face indices reference vertices out of range")

    # ---- Basic properties ----

    @property
    def num_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])

    # ---- Derived quantities (cached) ----

    @property
    def edges(self) -> np.ndarray:
        """(E, 2) int64 array of unique undirected edges."""
        if self._edges is None:
            f = self.faces
            e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
            e = np.sort(e, axis=1)
            self._edges = np.unique(e, axis=0).astype(np.int64)
        return self._edges

    @property
    def face_normals(self) -> np.ndarray:
        """(F, 3) unit face normals."""
        if self._face_normals is None:
            v = self.vertices[self.faces]  # (F, 3, 3)
            n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
            norms = np.linalg.norm(n, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            self._face_normals = (n / norms).astype(np.float32)
        return self._face_normals

    @property
    def face_adjacency(self) -> np.ndarray:
        """(A, 2) int64 array of face-face adjacency (faces sharing an edge)."""
        if self._face_adjacency is None:
            # Map each undirected edge -> list of face indices touching it.
            f = self.faces
            edge_face = {}
            for fi in range(f.shape[0]):
                for a, b in ((0, 1), (1, 2), (2, 0)):
                    key = (int(min(f[fi, a], f[fi, b])), int(max(f[fi, a], f[fi, b])))
                    edge_face.setdefault(key, []).append(fi)
            adj = []
            for faces in edge_face.values():
                if len(faces) == 2:
                    adj.append((faces[0], faces[1]))
            self._face_adjacency = (
                np.asarray(adj, dtype=np.int64)
                if adj
                else np.zeros((0, 2), dtype=np.int64)
            )
        return self._face_adjacency

    @property
    def vertex_neighbors(self) -> list:
        """List of length V; each entry is a numpy int64 array of neighbor vertex indices."""
        if self._vertex_neighbors is None:
            adj = [set() for _ in range(self.num_vertices)]
            for a, b in self.edges:
                adj[int(a)].add(int(b))
                adj[int(b)].add(int(a))
            self._vertex_neighbors = [
                np.fromiter(sorted(s), dtype=np.int64, count=len(s)) for s in adj
            ]
        return self._vertex_neighbors

    # ---- Convenience ----

    def sampled_vertex_features(self, dim: int, seed: int = 0) -> np.ndarray:
        """Deterministic per-mesh feature vector of length ``dim``.

        Concatenates a small set of geometric summaries (centroid, extents,
        vertex/face counts, mean normal) padded/truncated to ``dim``. Used by
        the toy experts so they have a shared, minimal preprocessing path.
        """
        rng = np.random.default_rng(seed)
        centroid = self.vertices.mean(axis=0)
        extents = self.vertices.max(axis=0) - self.vertices.min(axis=0)
        mean_normal = (
            self.face_normals.mean(axis=0)
            if self.num_faces
            else np.zeros(3, np.float32)
        )
        summary = np.concatenate(
            [
                centroid,
                extents,
                mean_normal,
                np.asarray([self.num_vertices, self.num_faces], dtype=np.float32),
            ]
        )
        if summary.size >= dim:
            return summary[:dim].astype(np.float32)
        pad = rng.standard_normal(dim - summary.size).astype(np.float32) * 0.01
        return np.concatenate([summary, pad]).astype(np.float32)
