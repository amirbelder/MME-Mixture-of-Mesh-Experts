"""Tests for RandomGate + hard-argmax combine."""

from tests.conftest import needs_torch


@needs_torch
def test_random_gate_is_frozen_and_deterministic():
    import torch
    from mme.data.synthetic import make_synthetic_mesh
    from mme.experts.base import ExpertOutput
    from mme.gate.random_gate import RandomGate

    g1 = RandomGate(num_experts=3, feature_dim=32, hidden=64, seed=42)
    g2 = RandomGate(num_experts=3, feature_dim=32, hidden=64, seed=42)

    # Weights identical across two same-seed constructions.
    for (k1, v1), (k2, v2) in zip(g1.state_dict().items(), g2.state_dict().items()):
        assert k1 == k2 and torch.allclose(v1, v2)

    # Different seed => different weights.
    g3 = RandomGate(num_experts=3, feature_dim=32, hidden=64, seed=43)
    any_differ = any(
        not torch.allclose(v1, v3)
        for v1, v3 in zip(g1.state_dict().values(), g3.state_dict().values())
    )
    assert any_differ

    # All params are frozen.
    assert all(not p.requires_grad for p in g1.parameters())

    # Deterministic forward output given a fixed mesh.
    mesh = make_synthetic_mesh("cube", seed=0)
    dummy = [ExpertOutput(logits=torch.zeros(4)) for _ in range(3)]
    a = g1.forward(mesh, dummy)
    b = g1.forward(mesh, dummy)
    assert torch.allclose(a, b)
    assert a.shape == (3,)


@needs_torch
def test_random_gate_not_in_torch_parameters(tmp_path):
    import examples.toy_expert  # noqa: F401  — registers experts
    import torch
    from mme.core.moe import MMEModel
    from mme.experts.registry import _reset_registry_for_tests, get_expert
    from mme.gate.random_gate import RandomGate

    _reset_registry_for_tests()
    import importlib

    import examples.toy_expert as te

    importlib.reload(te)

    experts = [
        get_expert("toy_mlp_small", num_classes=4),
        get_expert("toy_mlp_wide", num_classes=4),
    ]
    gate = RandomGate(num_experts=2, feature_dim=32, seed=7)
    model = MMEModel(experts=experts, gate=gate, combine="hard_argmax")

    gate_param_ids = {id(p) for p in gate.parameters()}
    train_param_ids = {id(p) for p in model.torch_parameters()}
    assert gate_param_ids.isdisjoint(train_param_ids), (
        "frozen RandomGate params must not appear in MMEModel.torch_parameters()"
    )

    # Save/load round-trip.
    path = tmp_path / "gate.pt"
    gate.save(path)
    reloaded = RandomGate.load(path)
    assert reloaded.seed == gate.seed
    for (k1, v1), (k2, v2) in zip(
        gate.state_dict().items(), reloaded.state_dict().items()
    ):
        assert k1 == k2 and torch.allclose(v1, v2)


@needs_torch
def test_hard_argmax_combine_picks_one_expert():
    """With hard_argmax, output logits should exactly equal the picked expert's logits."""
    import examples.toy_expert  # noqa: F401
    import torch
    from mme.core.moe import MMEModel
    from mme.data.synthetic import make_synthetic_mesh
    from mme.experts.registry import _reset_registry_for_tests, get_expert
    from mme.gate.random_gate import RandomGate

    _reset_registry_for_tests()
    import importlib

    import examples.toy_expert as te

    importlib.reload(te)

    experts = [
        get_expert("toy_mlp_wide", num_classes=4),
        get_expert("toy_mlp_deep", num_classes=4),
    ]
    gate = RandomGate(num_experts=2, feature_dim=32, seed=99)
    model = MMEModel(experts=experts, gate=gate, combine="hard_argmax")

    mesh = make_synthetic_mesh("sphere", seed=0)
    with torch.no_grad():
        # Run experts manually and compare.
        e_outs = [model._run_expert(e, mesh) for e in experts]
        expert_logits = torch.stack([o.logits.reshape(-1) for o in e_outs], dim=0)
        raw_scores = gate.forward(mesh, e_outs)
        weights = torch.softmax(raw_scores, dim=0)
        picked = int(torch.argmax(weights))
        expected = expert_logits[picked]

        logits = model.forward([mesh])
    assert torch.allclose(logits.squeeze(0), expected, atol=1e-6)
