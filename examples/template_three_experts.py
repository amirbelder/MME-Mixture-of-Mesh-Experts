"""Template script: three experts + trained gate + train + inference.

Copy this file, rename it, fill in the four slots marked TODO, and you have a
runnable MoE for your own three networks.

Slots:
    1. THREE_EXPERTS — replace with @register_expert-decorated classes of yours.
    2. GATE          — choose MMEGateTorch (trained) or RandomGate (frozen ablation).
    3. DATA          — replace with your dataset (see mme.data.MeshDataset for the
                       standard "<root>/<class>/*.off" layout, or plug in your own).
    4. NUM_CLASSES   — the number of classes for your task (e.g. 30 for SHREC11).

Out of the box this file runs on the synthetic toy dataset with three toy MLP
experts so you can smoke-test the plumbing, then swap in your real experts.
"""

from __future__ import annotations

from pathlib import Path

# --- SLOT 1: register your three experts ---------------------------------
# Replace this import with `import my_experts` (a module you write that uses
# @register_expert to add three classes). The toy import below is only so this
# template file runs end-to-end without any changes.
import examples.toy_expert  # noqa: F401

EXPERT_NAMES = ("toy_mlp_small", "toy_mlp_wide", "toy_mlp_deep")

# from mme.gate.random_gate import RandomGate             # ablation gate

# --- SLOT 3: pick your data ----------------------------------------------
from mme.data.synthetic import SyntheticShapesDataset  # toy default

# --- SLOT 2: pick a gate --------------------------------------------------
from mme.gate.mme_gate_torch import MMEGateTorch  # trained gate

# from mme.data.mesh_dataset import MeshDataset             # real files
# real_dataset = MeshDataset("~/shrec11")

# --- SLOT 4: your num_classes --------------------------------------------
NUM_CLASSES: int | None = None  # None = infer from dataset

# =========================================================================
from mme.core.moe import MMEModel
from mme.experts.registry import get_expert
from mme.losses import (
    diversity_loss,
    DynamicBalancedLoss,
    linear_schedule,
    similarity_loss,
    task_ce_loss,
)
from mme.training.trainer import Trainer


CKPT_DIR = Path("runs/template_three_experts")
COMBINE = "hard_argmax"  # or "weighted_softmax"


def build_gate(num_experts: int):
    # --- SLOT 2 body ---
    return MMEGateTorch(
        num_experts=num_experts,
        feature_dim=32,
        walk_len=16,
        num_walks=4,
    )
    # Ablation baseline:
    # return RandomGate(num_experts=num_experts, feature_dim=32, seed=1234)


def build_dataset():
    # --- SLOT 3 body ---
    return SyntheticShapesDataset(samples_per_class=16, noise=0.05)


def main() -> None:
    ds = build_dataset()
    num_classes = NUM_CLASSES if NUM_CLASSES is not None else ds.num_classes
    n = len(ds)
    n_val = max(1, n // 5)
    train_data = [ds[i] for i in range(n - n_val)]
    val_data = [ds[i] for i in range(n - n_val, n)]

    experts = [get_expert(name, num_classes=num_classes) for name in EXPERT_NAMES]
    if len(experts) != 3:
        raise ValueError(f"template expects three experts; got {len(experts)}")

    gate = build_gate(num_experts=len(experts))
    model = MMEModel(experts=experts, gate=gate, combine=COMBINE)
    print(
        f"experts    : {[e.name for e in experts]}\n"
        f"gate       : {type(gate).__name__}\n"
        f"combine    : {COMBINE}\n"
        f"num_classes: {num_classes}\n"
        f"train / val: {len(train_data)} / {len(val_data)}\n"
        f"trainable  : {sum(p.numel() for p in model.torch_parameters())} params"
    )

    loss_fn = DynamicBalancedLoss(
        task_loss_fn=task_ce_loss,
        diversity_loss_fn=diversity_loss,
        similarity_loss_fn=similarity_loss,
        schedule=linear_schedule(0.0, 0.05, 0.0, 0.05),
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

    # If the gate was a RandomGate, also save its state so the exact init is
    # reproducible (see docs/random_gate.md).
    if hasattr(model.gate, "save") and hasattr(model.gate, "seed"):
        model.gate.save(CKPT_DIR / "random_gate.pt")

    from mme.eval.classify import evaluate_classification

    print("validation:", evaluate_classification(model, val_data))


if __name__ == "__main__":
    main()
