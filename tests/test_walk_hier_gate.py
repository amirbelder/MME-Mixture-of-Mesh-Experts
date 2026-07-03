"""Smoke test for WalkHierGate (vendored TF WalkHierTransformer)."""

from tests.conftest import needs_tf, needs_torch


@needs_torch
@needs_tf
def test_walk_hier_gate_forward_returns_num_experts():
    import torch
    from mme.data.synthetic import make_synthetic_mesh
    from mme.experts.base import ExpertOutput
    from mme.gate.walk_hier_gate import WalkHierGate

    gate = WalkHierGate(
        num_experts=3,
        walk_len=20,
        num_walks=4,
        d_model=32,
        num_layers=1,
        num_heads=2,
        dff=32,
        jump_every_k=5,
        pooling=True,
    )

    mesh = make_synthetic_mesh("sphere", seed=0)
    # WalkHierGate does not read expert outputs, but the MMEModel calling
    # convention passes them in. Empty stubs are fine.
    outputs = [ExpertOutput(logits=torch.zeros(2)) for _ in range(3)]

    scores = gate.forward(mesh, outputs)
    assert scores.shape == (3,)
    # Softmax should sum to 1 (sanity — MMEModel will do this itself).
    assert torch.allclose(
        torch.softmax(scores, dim=0).sum(), torch.tensor(1.0), atol=1e-5
    )
