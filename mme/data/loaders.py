"""Load .obj / .off / .ply files via trimesh into our Mesh container."""

from __future__ import annotations

from pathlib import Path

from mme.core.mesh import Mesh


def load_mesh_file(path) -> Mesh:
    """Load a mesh file and return an ``mme.core.mesh.Mesh``.

    Requires ``trimesh``. Only vertex positions and triangle faces are read;
    any other attributes are ignored.
    """
    import trimesh

    path = Path(path)
    m = trimesh.load(path, force="mesh", process=False)
    if not hasattr(m, "faces"):
        raise ValueError(f"file {path} does not contain a triangle mesh")
    return Mesh(vertices=m.vertices, faces=m.faces)
