"""End-to-end training on the synthetic platonic-solids dataset."""

from __future__ import annotations

import examples.toy_expert  # noqa: F401 — registers 3 experts
from mme.core.moe import MMEModel
from mme.data.synthetic import SyntheticShapesDataset
from mme.experts.registry import get_expert
from mme.gate.mme_gate_torch import MMEGateTorch
from mme.losses import (
    diversity_loss,
    DynamicBalancedLoss,
    linear_schedule,
    similarity_loss,
    task_ce_loss,
)
from mme.training.trainer import Trainer


def main() -> None:
    ds = SyntheticShapesDataset(samples_per_class=16, noise=0.05)
    num_classes = ds.num_classes
    n = len(ds)
    n_val = max(1, n // 5)
    train_data = [ds[i] for i in range(n - n_val)]
    val_data = [ds[i] for i in range(n - n_val, n)]

    experts = [
        get_expert("toy_mlp_small", num_classes=num_classes),
        get_expert("toy_mlp_wide", num_classes=num_classes),
        get_expert("toy_mlp_deep", num_classes=num_classes),
    ]
    gate = MMEGateTorch(num_experts=3, feature_dim=32, walk_len=16, num_walks=4)
    model = MMEModel(experts=experts, gate=gate)

    loss_fn = DynamicBalancedLoss(
        task_loss_fn=task_ce_loss,
        diversity_loss_fn=diversity_loss,
        similarity_loss_fn=similarity_loss,
        schedule=linear_schedule(0.0, 0.05, 0.0, 0.05),
    )
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_data=train_data,
        val_data=val_data,
        batch_size=8,
        epochs=20,
        lr=1e-3,
    )
    trainer.fit()

    from mme.eval.classify import evaluate_classification

    metrics = evaluate_classification(model, val_data)
    print("validation:", metrics)


if __name__ == "__main__":
    main()
