"""Tests for the vendored walk algorithms and feature composition."""

import numpy as np


def test_random_global_jumps_sampler_valid_and_deterministic():
    from mme.data.synthetic import make_synthetic_mesh
    from mme.gate.walk_algorithms import sample_paper_walks

    mesh = make_synthetic_mesh("sphere", seed=0)

    seqs1, jumps1 = sample_paper_walks(
        mesh, num_walks=4, walk_len=20, walk_alg="random_global_jumps", seed=7
    )
    seqs2, jumps2 = sample_paper_walks(
        mesh, num_walks=4, walk_len=20, walk_alg="random_global_jumps", seed=7
    )
    # Same seed → identical seqs and jumps.
    assert np.array_equal(seqs1, seqs2)
    assert np.array_equal(jumps1, jumps2)

    # Shape checks: (num_walks, walk_len + 1).
    assert seqs1.shape == (4, 21)
    assert jumps1.shape == (4, 21)

    # All vertex ids are in range.
    assert int(seqs1.min()) >= 0
    assert int(seqs1.max()) < mesh.num_vertices


def test_walk_algorithm_registry_lists_all_and_raises_on_unknown():
    import pytest
    from mme.gate.walk_algorithms import get_walk_algorithm, WALK_ALGORITHMS

    expected = {
        "no_jumps",
        "random_global_jumps",
        "random_global_jumps_new",
        "constant_global_jumps",
        "local_jumps",
    }
    assert expected.issubset(WALK_ALGORITHMS)
    assert callable(get_walk_algorithm("random_global_jumps"))
    with pytest.raises(KeyError):
        get_walk_algorithm("does_not_exist")


def test_compose_walk_features_matches_reference_shapes():
    from mme.data.synthetic import make_synthetic_mesh
    from mme.gate.walk_algorithms import sample_paper_walks
    from mme.gate.walk_features import (
        batch_compose_walk_features,
        compose_walk_features,
        net_input_dim,
    )

    mesh = make_synthetic_mesh("cube", seed=0)
    seqs, jumps = sample_paper_walks(
        mesh, num_walks=3, walk_len=8, walk_alg="random_global_jumps", seed=1
    )

    # Single-walk composition: xyz alone → 3 dims per step.
    feats = compose_walk_features(mesh.vertices, seqs[0], jumps[0], 8, spec=("xyz",))
    assert feats.shape == (8, 3)
    assert net_input_dim(("xyz",)) == 3

    # Composition matches reference net_input=['xyz', 'dxdydz', 'jump_indication'] → 7 dims.
    spec = ("xyz", "dxdydz", "jump_indication")
    feats = compose_walk_features(mesh.vertices, seqs[0], jumps[0], 8, spec=spec)
    assert feats.shape == (8, 7)
    assert net_input_dim(spec) == 7

    # Batch helper returns (num_walks, walk_len, D) — the exact tensor shape
    # train_val.py::train_step feeds after its reshape.
    batch = batch_compose_walk_features(mesh.vertices, seqs, jumps, 8, spec=spec)
    assert batch.shape == (3, 8, 7)


def test_norm_model_centers_and_scales():
    from mme.gate.walk_features import norm_model

    v = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        dtype=np.float32,
    )
    n = norm_model(v)
    # After centering, mean should be ~0.
    assert np.allclose(n.mean(axis=0), 0.0, atol=1e-6)
    # After scaling, max L2 distance should be ~1.
    assert abs(float(np.linalg.norm(n, axis=1).max()) - 1.0) < 1e-6
