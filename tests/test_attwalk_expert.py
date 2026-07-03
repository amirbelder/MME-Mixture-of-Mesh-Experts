"""Smoke test for AttWalkExpert (vendored TF WalkTransformer)."""

from tests.conftest import needs_tf


@needs_tf
def test_attwalk_expert_forward_returns_num_classes():
    from mme.data.synthetic import make_synthetic_mesh
    from mme.experts.attwalk_tf import AttWalkExpert

    expert = AttWalkExpert(
        num_classes=4,
        walk_len=20,
        num_walks=4,
        d_model=32,
        num_layers=1,
        num_heads=2,
        dff=32,
    )
    mesh = make_synthetic_mesh("cube", seed=0)
    out = expert.forward(expert.preprocess(mesh))
    assert out.logits.shape == (4,)
    assert out.features is not None
