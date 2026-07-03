import numpy as np
from mme.core.mesh import Mesh
from mme.data.synthetic import make_synthetic_mesh


def test_mesh_basic_shapes():
    m = make_synthetic_mesh("sphere", seed=0)
    assert m.num_vertices > 0
    assert m.num_faces > 0
    assert m.face_normals.shape == (m.num_faces, 3)
    # Normals are unit length.
    np.testing.assert_allclose(np.linalg.norm(m.face_normals, axis=1), 1.0, atol=1e-5)


def test_mesh_edges_and_neighbors():
    m = make_synthetic_mesh("tetra", seed=1, noise=0.0)
    assert m.edges.shape[1] == 2
    assert (
        (m.edges.max() < m.num_vertices).item()
        if hasattr(m.edges.max(), "item")
        else True
    )
    nbrs = m.vertex_neighbors
    assert len(nbrs) == m.num_vertices
    for n in nbrs:
        assert n.dtype == np.int64


def test_mesh_validation_rejects_bad_shapes():
    import pytest

    with pytest.raises(ValueError):
        Mesh(vertices=np.zeros((3, 2)), faces=np.zeros((1, 3), dtype=np.int64))
    with pytest.raises(ValueError):
        Mesh(vertices=np.zeros((3, 3)), faces=np.array([[10, 0, 1]], dtype=np.int64))
