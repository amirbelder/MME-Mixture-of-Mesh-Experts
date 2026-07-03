"""Two heterogeneous toy experts + a frozen random gate with hard routing.

This is the "does the routing even matter?" ablation from the MoE literature,
and it mirrors the shape of the reference council code (``train_council*.py``)
but with an untrained gate:

- Experts are two of our toy MLP experts with different width/depth/activation
  (heterogeneous — the point of MoE).
- Gate is :class:`mme.gate.random_gate.RandomGate` — a small MLP whose weights
  are drawn from a fixed seed and marked ``requires_grad=False``. The
  optimizer will not see them (see ``MMEModel.torch_parameters()``).
- Combine is ``"hard_argmax"``: per sample, the gate picks one expert; that
  expert's logits are the final output. Loss (task + optional div/sim)
  therefore only backprops through the picked expert on each step.
- At every checkpoint, the *gate's* state is saved separately alongside the
  model checkpoint via :meth:`RandomGate.save` so the exact random init that
  produced any reported number is reproducible via :meth:`RandomGate.load`.

Extending to three experts is a one-line change: append a third
``get_expert("toy_mlp_small", ...)`` to the ``experts`` list and pass
``num_experts=3`` to :class:`RandomGate`.
"""

from __future__ import annotations

import os
from pathlib import Path

import examples.toy_expert  # noqa: F401  — registers toy experts
from mme.core.moe import MMEModel
from mme.data.synthetic import SyntheticShapesDataset
from mme.experts.registry import get_expert
from mme.gate.random_gate import RandomGate
from mme.losses import (
    diversity_loss,
    DynamicBalancedLoss,
    linear_schedule,
    similarity_loss,
    task_ce_loss,
)
from mme.training.trainer import Trainer


CKPT_DIR = Path("runs/two_expert_random_gate")
GATE_SEED = 1234


def build_model(num_classes: int) -> MMEModel:
    experts = [
        get_expert("toy_mlp_wide", num_classes=num_classes),
        get_expert("toy_mlp_deep", num_classes=num_classes),
    ]
    gate = RandomGate(
        num_experts=len(experts), feature_dim=32, hidden=64, seed=GATE_SEED
    )
    return MMEModel(experts=experts, gate=gate, combine="hard_argmax")


def main() -> None:
    ds = SyntheticShapesDataset(samples_per_class=16, noise=0.05)
    n = len(ds)
    n_val = max(1, n // 5)
    train_data = [ds[i] for i in range(n - n_val)]
    val_data = [ds[i] for i in range(n - n_val, n)]

    model = build_model(num_classes=ds.num_classes)

    # Sanity check: the gate must not contribute trainable params.
    n_gate_trainable = sum(1 for p in model.gate.parameters() if p.requires_grad)
    assert n_gate_trainable == 0, "RandomGate should be frozen"
    print(
        f"trainable params: {sum(p.numel() for p in model.torch_parameters())}  "
        f"(gate contributes 0)"
    )

    loss_fn = DynamicBalancedLoss(
        task_loss_fn=task_ce_loss,
        diversity_loss_fn=diversity_loss,
        similarity_loss_fn=similarity_loss,
        # No div/sim pressure by default in this ablation — task loss only.
        schedule=linear_schedule(0.0, 0.0, 0.0, 0.0),
    )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_data=train_data,
        val_data=val_data,
        batch_size=8,
        epochs=20,
        lr=1e-3,
        ckpt_dir=str(CKPT_DIR),
    )
    trainer.fit()

    # Save the exact random gate that produced this run's numbers.
    gate_path = CKPT_DIR / "random_gate.pt"
    model.gate.save(gate_path)
    print(f"saved frozen random gate (seed={GATE_SEED}) to {gate_path}")

    # Round-trip sanity: reload and confirm state matches.
    reloaded = RandomGate.load(gate_path)
    for (k1, v1), (k2, v2) in zip(
        model.gate.state_dict().items(), reloaded.state_dict().items()
    ):
        assert k1 == k2 and (v1 == v2).all(), f"gate reload mismatch on {k1}"
    print("gate save/load round-trip OK")

    from mme.eval.classify import evaluate_classification

    print("validation:", evaluate_classification(model, val_data))


if __name__ == "__main__":
    main()
