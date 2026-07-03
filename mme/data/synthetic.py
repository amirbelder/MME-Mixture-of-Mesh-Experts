"""Synthetic mesh dataset — deformed platonic solids for the toy example."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from mme.core.mesh import Mesh


SHAPES = ("cube", "sphere", "tetra", "octa")


def _cube() -> Tuple[np.ndarray, np.ndarray]:
    v = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float32,
    )
    f = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int64,
    )
    return v, f


def _tetra() -> Tuple[np.ndarray, np.ndarray]:
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float32)
    f = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=np.int64)
    return v, f


def _octa() -> Tuple[np.ndarray, np.ndarray]:
    v = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float32,
    )
    f = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ],
        dtype=np.int64,
    )
    return v, f


def _sphere(subdiv: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Icosphere via icosahedron subdivision."""
    t = (1.0 + 5.0**0.5) / 2.0
    v = np.array(
        [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1],
        ],
        dtype=np.float32,
    )
    f = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int64,
    )
    for _ in range(subdiv):
        v_list = v.tolist()
        cache: dict = {}
        new_f = []

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key in cache:
                return cache[key]
            m = ((v[a] + v[b]) / 2.0).tolist()
            v_list.append(m)
            idx = len(v_list) - 1
            cache[key] = idx
            return idx

        for tri in f:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_f += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        v = np.asarray(v_list, dtype=np.float32)
        f = np.asarray(new_f, dtype=np.int64)
    # Project to unit sphere
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return v, f


def make_synthetic_mesh(shape: str, seed: int = 0, noise: float = 0.05) -> Mesh:
    """Build a deformed platonic solid labeled by base shape.

    ``shape`` must be one of :data:`SHAPES`. ``noise`` is the std-dev of
    Gaussian jitter added to vertex positions (in unit-sphere space).
    """
    if shape not in SHAPES:
        raise ValueError(f"unknown shape {shape!r}; choose from {SHAPES}")
    if shape == "cube":
        v, f = _cube()
    elif shape == "tetra":
        v, f = _tetra()
    elif shape == "octa":
        v, f = _octa()
    else:
        v, f = _sphere(subdiv=2)
    rng = np.random.default_rng(seed)
    v = v + rng.standard_normal(v.shape).astype(np.float32) * noise
    return Mesh(vertices=v, faces=f, label=SHAPES.index(shape))


class SyntheticShapesDataset:
    """Iterable of deformed platonic solids covering every class."""

    def __init__(
        self, samples_per_class: int = 32, noise: float = 0.05, seed: int = 0
    ) -> None:
        self.samples_per_class = samples_per_class
        self.noise = noise
        self.seed = seed
        self._items: List[Mesh] = []
        rng = np.random.default_rng(seed)
        for si, shape in enumerate(SHAPES):
            for i in range(samples_per_class):
                self._items.append(
                    make_synthetic_mesh(
                        shape=shape, seed=int(rng.integers(1 << 31)), noise=noise
                    )
                )
        # Shuffle deterministically.
        order = rng.permutation(len(self._items))
        self._items = [self._items[i] for i in order]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Mesh:
        return self._items[idx]

    @property
    def num_classes(self) -> int:
        return len(SHAPES)

    @staticmethod
    def class_names() -> Sequence[str]:
        return SHAPES
