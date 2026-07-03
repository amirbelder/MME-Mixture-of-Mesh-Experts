"""Vectorized random walks over a mesh (vertex-based).

Deliberately kept in numpy so it's framework-agnostic; the caller wraps the
resulting indices as torch/tf tensors as needed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from mme.core.mesh import Mesh


def sample_walks(
    mesh: Mesh,
    num_walks: int,
    walk_len: int,
    start_indices: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample ``num_walks`` uniform random walks of length ``walk_len`` over the mesh vertex graph.

    At each step we jump to a uniformly-random neighbor of the current vertex.
    Isolated vertices stay put (the walk repeats them).

    Args:
        mesh: input Mesh.
        num_walks: number of walks to sample.
        walk_len: number of vertices per walk (walk includes the starting vertex).
        start_indices: optional (num_walks,) starting vertex ids. If None,
            random vertices are chosen.
        seed: RNG seed for reproducibility.

    Returns:
        Int64 array of shape ``(num_walks, walk_len)`` of vertex indices.
    """
    if walk_len < 1:
        raise ValueError("walk_len must be >= 1")
    if num_walks < 1:
        raise ValueError("num_walks must be >= 1")

    rng = np.random.default_rng(seed)
    V = mesh.num_vertices
    if V == 0:
        raise ValueError("mesh has no vertices")

    neighbors = mesh.vertex_neighbors
    walks = np.zeros((num_walks, walk_len), dtype=np.int64)

    if start_indices is None:
        walks[:, 0] = rng.integers(0, V, size=num_walks)
    else:
        walks[:, 0] = np.asarray(start_indices, dtype=np.int64)

    # Precompute a dense choice function per step. For simplicity/clarity we
    # loop over walks; this is O(num_walks * walk_len) and plenty fast at
    # typical scales (thousands of walks). For big graphs a CSR-based
    # sampler would be faster.
    for w in range(num_walks):
        cur = int(walks[w, 0])
        for t in range(1, walk_len):
            nbrs = neighbors[cur]
            if nbrs.size == 0:
                walks[w, t] = cur  # isolated vertex: repeat
            else:
                cur = int(nbrs[rng.integers(0, nbrs.size)])
                walks[w, t] = cur
    return walks
