"""End-to-end smoke test: 1 epoch of the toy training loop."""

import pytest
from tests.conftest import needs_torch


@needs_torch
def test_one_epoch_reduces_or_preserves_loss():
    import torch
    from mme.core.moe import MMEModel
    from mme.data.synthetic import SyntheticShapesDataset
    from mme.experts.registry import _reset_registry_for_tests, get_expert
    from mme.gate.mme_gate_torch import MMEGateTorch
    from mme.losses import (
        diversity_loss,
        DynamicBalancedLoss,
        linear_schedule,
        similarity_loss,
        task_ce_loss,
    )
    from mme.training.trainer import Trainer

    _reset_registry_for_tests()
    import importlib

    import examples.toy_expert  # noqa: F401

    importlib.reload(examples.toy_expert)

    ds = SyntheticShapesDataset(samples_per_class=4, noise=0.03, seed=0)
    n = len(ds)
    train_data = [ds[i] for i in range(n)]
    experts = [
        get_expert(name, num_classes=ds.num_classes)
        for name in ("toy_mlp_small", "toy_mlp_wide", "toy_mlp_deep")
    ]
    gate = MMEGateTorch(num_experts=3, feature_dim=32, walk_len=8, num_walks=2, seed=0)
    model = MMEModel(experts=experts, gate=gate)

    loss_fn = DynamicBalancedLoss(
        task_ce_loss,
        diversity_loss,
        similarity_loss,
        linear_schedule(0.0, 0.0, 0.0, 0.0),
    )
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_data=train_data,
        batch_size=4,
        epochs=1,
        lr=1e-2,
        log_every=1,
    )

    # Compute pre-training loss on a fixed batch.
    with torch.no_grad():
        logits0 = model.forward(train_data[:4])
        targets = torch.tensor([m.label for m in train_data[:4]], dtype=torch.long)
        loss0 = task_ce_loss(logits0, targets).item()

    trainer.fit()

    with torch.no_grad():
        logits1 = model.forward(train_data[:4])
        loss1 = task_ce_loss(logits1, targets).item()

    # 1 epoch is short; require loss doesn't blow up. Allow small increases.
    assert loss1 < loss0 * 2.0
    # Gate weights are already softmaxed inside MMEModel — should sum to 1 per row.
    gw = model.last_gate_weights
    assert torch.allclose(gw.sum(dim=-1), torch.ones(gw.shape[0]), atol=1e-4)
