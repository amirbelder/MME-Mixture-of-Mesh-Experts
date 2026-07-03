"""Torch Dataset over mesh files in a directory tree.

Expected layout: ``<root>/<class_name>/*.{obj,off,ply}`` — each
subdirectory is one class, class names are sorted lexicographically.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from mme.core.mesh import Mesh
from mme.data.loaders import load_mesh_file


class MeshDataset:
    """A minimal torch-style ``Dataset``. Does NOT depend on torch."""

    def __init__(
        self, root, extensions: Tuple[str, ...] = (".obj", ".off", ".ply")
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"{self.root} does not exist")
        classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        if not classes:
            raise ValueError(f"no class subdirectories under {self.root}")
        self.classes: List[str] = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples: List[Tuple[Path, int]] = []
        for c in classes:
            for p in (self.root / c).iterdir():
                if p.suffix.lower() in extensions:
                    self.samples.append((p, self.class_to_idx[c]))
        if not self.samples:
            raise ValueError(f"no mesh files under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Mesh:
        path, label = self.samples[idx]
        m = load_mesh_file(path)
        m.label = label
        m.source_path = str(path)
        return m

    @property
    def num_classes(self) -> int:
        return len(self.classes)
