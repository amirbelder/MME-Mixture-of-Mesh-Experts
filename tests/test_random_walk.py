from mme.data.synthetic import make_synthetic_mesh
from mme.gate.random_walk import sample_walks


def test_random_walk_shape_and_validity():
    m = make_synthetic_mesh("sphere", seed=0)
    walks = sample_walks(m, num_walks=5, walk_len=10, seed=42)
    assert walks.shape == (5, 10)
    assert int(walks.min()) >= 0
    assert int(walks.max()) < m.num_vertices


def test_random_walk_deterministic_with_seed():
    m = make_synthetic_mesh("cube", seed=0, noise=0.0)
    w1 = sample_walks(m, num_walks=3, walk_len=8, seed=7)
    w2 = sample_walks(m, num_walks=3, walk_len=8, seed=7)
    assert (w1 == w2).all()
