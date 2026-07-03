"""Per-vertex feature composition — vendored from the reference dataset.py.

The reference's ``params.net_input`` is a list of strings (e.g. ``['xyz']`` or
``['xyz', 'dxdydz', 'jump_indication']``). Each string is a
``fill_<name>_features`` function that writes ``k`` columns into a preallocated
``(seq_len, total_dim)`` feature buffer. :func:`compose_walk_features` does
exactly that composition and returns the buffer.

We keep the function signatures and slicing conventions bit-for-bit identical
to the reference so behaviour matches ``dataset.mesh_data_to_walk_features``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


# ------------------------------------------------------------------
# Model normalization — matches reference dataset.norm_model.
# ------------------------------------------------------------------


def norm_model(
    vertices: np.ndarray, sub_mean_for_data_augmentation: bool = True
) -> np.ndarray:
    """Center by mean (optional) and scale by max L2 distance from origin.

    Reference does this in-place on the mesh's vertex buffer before feature
    extraction; we return a copy so callers can decide.
    """
    v = np.asarray(vertices, dtype=np.float32).copy()
    if sub_mean_for_data_augmentation:
        v -= v.mean(axis=0, keepdims=True)
    max_dist = float(np.linalg.norm(v, axis=1).max())
    if max_dist > 1e-12:
        v /= max_dist
    return v


# ------------------------------------------------------------------
# Per-vertex feature fillers — signatures match reference for portability.
# Each writes ``k`` columns into ``features[:, f_idx:f_idx+k]`` and returns
# the new ``f_idx`` (start of the next block).
# ------------------------------------------------------------------


def fill_xyz(features, f_idx, vertices, seq, jumps, seq_len):
    walk = vertices[seq[1 : seq_len + 1]]
    features[:, f_idx : f_idx + walk.shape[1]] = walk
    return f_idx + 3


def fill_dxdydz(features, f_idx, vertices, seq, jumps, seq_len):
    walk = np.diff(vertices[seq[: seq_len + 1]], axis=0) * 100.0
    features[:, f_idx : f_idx + walk.shape[1]] = walk
    return f_idx + 3


def fill_jump_indication(features, f_idx, vertices, seq, jumps, seq_len):
    walk = jumps[1 : seq_len + 1][:, None].astype(np.float32)
    features[:, f_idx : f_idx + 1] = walk
    return f_idx + 1


def fill_vertex_indices(features, f_idx, vertices, seq, jumps, seq_len):
    walk = seq[1 : seq_len + 1][:, None].astype(np.float32)
    features[:, f_idx : f_idx + 1] = walk
    return f_idx + 1


def fill_v_normals(
    features,
    f_idx,
    vertices,
    seq,
    jumps,
    seq_len,
    v_normals: Optional[np.ndarray] = None,
):
    if v_normals is None:
        raise ValueError(
            "fill_v_normals needs v_normals (pass via compose_walk_features(v_normals=...))"
        )
    walk = v_normals[seq[1 : seq_len + 1]]
    features[:, f_idx : f_idx + walk.shape[1]] = walk
    return f_idx + 3


# Dimensions per feature name — used to size the output buffer up front.
_FEATURE_DIMS = {
    "xyz": 3,
    "dxdydz": 3,
    "jump_indication": 1,
    "vertex_indices": 1,
    "v_normals": 3,
}


_FEATURE_FILLERS = {
    "xyz": fill_xyz,
    "dxdydz": fill_dxdydz,
    "jump_indication": fill_jump_indication,
    "vertex_indices": fill_vertex_indices,
    "v_normals": fill_v_normals,
}


def net_input_dim(spec: Sequence[str]) -> int:
    """Total feature dim implied by a ``net_input`` spec list."""
    return sum(_FEATURE_DIMS[name] for name in spec)


# ------------------------------------------------------------------
# Composition — the same fill chain as reference dataset.py.
# ------------------------------------------------------------------


def compose_walk_features(
    vertices: np.ndarray,
    seq: np.ndarray,
    jumps: np.ndarray,
    seq_len: int,
    spec: Sequence[str] = ("xyz",),
    *,
    v_normals: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build one walk's ``(seq_len, D)`` feature tensor per ``spec``.

    Args:
        vertices: (V, 3) mesh vertex positions (usually already normalized
            via :func:`norm_model`).
        seq: (seq_len + 1,) int walk indices — the raw output of a sampler
            in :mod:`mme.gate.walk_algorithms`.
        jumps: (seq_len + 1,) bool jump flags from the same sampler.
        seq_len: number of walk steps to emit features for.
        spec: ordered feature names — e.g. ``('xyz',)`` (paper default,
            3-dim), ``('xyz', 'dxdydz')`` (6-dim), or
            ``('dxdydz', 'jump_indication')`` (4-dim).
        v_normals: per-vertex normals (V, 3), required if ``'v_normals'`` is
            in ``spec``.

    Returns:
        ``(seq_len, sum_dims)`` float32 feature buffer, identical shape and
        layout to what the reference's ``fill_features_functions`` chain writes.
    """
    D = net_input_dim(spec)
    features = np.zeros((seq_len, D), dtype=np.float32)
    f_idx = 0
    for name in spec:
        filler = _FEATURE_FILLERS[name]
        if name == "v_normals":
            f_idx = filler(
                features, f_idx, vertices, seq, jumps, seq_len, v_normals=v_normals
            )
        else:
            f_idx = filler(features, f_idx, vertices, seq, jumps, seq_len)
    return features


def batch_compose_walk_features(
    vertices: np.ndarray,
    seqs: np.ndarray,
    jumps: np.ndarray,
    seq_len: int,
    spec: Sequence[str] = ("xyz",),
    *,
    v_normals: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Vectorized helper over ``(num_walks, seq_len+1)`` walk batches.

    Returns ``(num_walks, seq_len, D)`` — the exact shape the vendored
    :class:`mme.gate.walk_hier_transformer_tf.WalkHierTransformer` consumes
    after the reshape in reference ``train_val.py::train_step``.
    """
    num_walks = seqs.shape[0]
    D = net_input_dim(spec)
    out = np.zeros((num_walks, seq_len, D), dtype=np.float32)
    for w in range(num_walks):
        out[w] = compose_walk_features(
            vertices, seqs[w], jumps[w], seq_len, spec=spec, v_normals=v_normals
        )
    return out
