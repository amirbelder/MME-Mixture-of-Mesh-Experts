from tests.conftest import needs_torch


@needs_torch
def test_diversity_and_similarity_shapes():
    import torch
    from mme.losses import diversity_loss, similarity_loss, task_ce_loss

    feats = [torch.randn(16) for _ in range(3)]
    d = diversity_loss(feats)
    assert d.ndim == 0

    expert_logits = [torch.randn(4) for _ in range(3)]
    moe = torch.randn(4)
    s = similarity_loss(expert_logits, moe)
    assert s.ndim == 0

    logits = torch.randn(2, 4)
    targets = torch.tensor([0, 3])
    ce = task_ce_loss(logits, targets)
    assert ce.ndim == 0


@needs_torch
def test_dynamic_balance_schedules_return_alpha_beta():
    from mme.losses import cosine_schedule, linear_schedule, step_schedule

    lin = linear_schedule(0.0, 1.0, 0.0, 2.0)
    cos = cosine_schedule(0.0, 1.0, 0.0, 2.0)
    stp = step_schedule([(0, 0.0, 0.0), (5, 0.5, 1.0)])

    for f in (lin, cos):
        a0, b0 = f(0, 10)
        a1, b1 = f(9, 10)
        assert 0.0 <= a0 <= 1.0 and 0.0 <= b0 <= 2.0
        assert abs(a1 - 1.0) < 1e-6 and abs(b1 - 2.0) < 1e-6

    assert stp(0, 10) == (0.0, 0.0)
    assert stp(5, 10) == (0.5, 1.0)
    assert stp(9, 10) == (0.5, 1.0)
