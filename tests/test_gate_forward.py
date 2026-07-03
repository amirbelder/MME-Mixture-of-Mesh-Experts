import pytest
from tests.conftest import needs_torch


@needs_torch
def test_gate_forward_returns_num_experts_weights():
    import torch
    from mme.data.synthetic import make_synthetic_mesh
    from mme.experts.base import ExpertOutput
    from mme.gate.mme_gate_torch import MMEGateTorch

    gate = MMEGateTorch(num_experts=3, feature_dim=32, walk_len=8, num_walks=2)
    m = make_synthetic_mesh("cube", seed=0)

    # Fake per-expert outputs — features of width 32 so no lazy projection is needed.
    outputs = [
        ExpertOutput(logits=torch.zeros(4), features=torch.randn(32)) for _ in range(3)
    ]
    scores = gate.forward(m, outputs)
    assert scores.shape == (3,)
    weights = torch.softmax(scores, dim=0)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
