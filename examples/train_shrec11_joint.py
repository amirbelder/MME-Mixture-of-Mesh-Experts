"""SHREC-11 with jointly-trained experts (mode A): reference-council pattern.

All three experts live in the SAME env and every torch parameter (experts + gate)
enters the same optimizer. This is what train_council*.py in your reference
project does. Matches the paper's default when you have the freedom to install
every baseline together.

Requirements:
    - Every expert's model code is importable in this venv (see env/<baseline>/).
    - Baselines that are TF-native (MeshWalker, AttWalk) are supported but
      their weights DO NOT update from MoE gradients — they cross the numpy
      bridge (see docs/mixing_frameworks.md). If you want a TF expert to also
      update, either (a) port it to torch or (b) call its own tf.GradientTape
      training step alongside the MoE step (a hook is shown at the bottom of
      this file).

Replace the three EXPERT_FACTORIES below with imports of your actual expert
classes — one per baseline env you have installed in this venv.

Run:
    python examples/train_shrec11_joint.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List

import numpy as np
import torch
from mme.core.moe import MMEModel
from mme.data.mesh_dataset import MeshDataset
from mme.experts.base import Expert
from mme.gate.mme_gate_torch import MMEGateTorch
from mme.losses import (
    diversity_loss,
    DynamicBalancedLoss,
    linear_schedule,
    similarity_loss,
    task_ce_loss,
)


# ---------- configuration ------------------------------------------------
SHREC11_ROOT = Path("~/shrec11").expanduser()
CKPT_DIR = Path("runs/shrec11_joint")
NUM_CLASSES = 30
EPOCHS = 200
BATCH_SIZE = 8
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- SLOT 1: your three expert factories --------------------------
# Each factory takes num_classes and returns an Expert. Replace with imports
# of the adapter classes you wrote in my_experts.py.
#
# Example — three toy MLP experts, for smoke-testing without SHREC-11:
def _toy_factories() -> List[Callable[[int], Expert]]:
    import examples.toy_expert  # noqa: F401  — registers experts
    from mme.experts.registry import get_expert

    return [
        lambda n: get_expert("toy_mlp_small", num_classes=n),
        lambda n: get_expert("toy_mlp_wide", num_classes=n),
        lambda n: get_expert("toy_mlp_deep", num_classes=n),
    ]


# Real example (uncomment and adapt when you have three real experts):
#
# from my_experts import MeshCNNExpert, SubdivNetExpert, MeshWalkerExpert
# def _real_factories():
#     return [
#         lambda n: MeshCNNExpert(num_classes=n,    ckpt_path="~/checkpoints/meshcnn.pt"),
#         lambda n: SubdivNetExpert(num_classes=n,  ckpt_path="~/checkpoints/subdivnet.jt"),
#         lambda n: MeshWalkerExpert(num_classes=n, ckpt_path="~/checkpoints/meshwalker.ckpt"),
#     ]

EXPERT_FACTORIES = _toy_factories()


# ---------- helpers ------------------------------------------------------
def build_model() -> MMEModel:
    experts = [f(NUM_CLASSES) for f in EXPERT_FACTORIES]
    gate = MMEGateTorch(
        num_experts=len(experts),
        feature_dim=128,
        walk_len=100,
        num_walks=32,
        num_heads=8,
        num_layers=3,
        seed=0,
    )
    return MMEModel(experts=experts, gate=gate, combine="hard_argmax")


def split_train_val(dataset, val_fraction: float = 0.2):
    n = len(dataset)
    n_val = max(1, int(n * val_fraction))
    train = [dataset[i] for i in range(n - n_val)]
    val = [dataset[i] for i in range(n - n_val, n)]
    return train, val


def try_load_dataset():
    """Fall back to synthetic if SHREC-11 root does not exist."""
    if SHREC11_ROOT.exists():
        return MeshDataset(SHREC11_ROOT)
    print(f"warning: {SHREC11_ROOT} not found — using synthetic toy dataset")
    from mme.data.synthetic import SyntheticShapesDataset

    return SyntheticShapesDataset(samples_per_class=16, noise=0.05)


# ---------- the training loop --------------------------------------------
def main() -> None:
    dataset = try_load_dataset()
    train_data, val_data = split_train_val(dataset)
    num_classes = getattr(dataset, "num_classes", NUM_CLASSES)
    print(f"train / val = {len(train_data)} / {len(val_data)}, classes = {num_classes}")

    model = build_model().to(DEVICE)

    # Every expert's trainable torch param + gate params — matches
    # `optim.SGD(model.parameters(), ...)` in the reference council code.
    params = model.torch_parameters()
    n_params = sum(p.numel() for p in params)
    print(f"trainable torch params (experts + gate): {n_params}")

    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60, 120], gamma=0.1
    )

    loss_fn = DynamicBalancedLoss(
        task_loss_fn=task_ce_loss,
        diversity_loss_fn=diversity_loss,
        similarity_loss_fn=similarity_loss,
        schedule=linear_schedule(0.0, 0.05, 0.0, 0.05),
    )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val = 0.0
    total_steps = EPOCHS * ((len(train_data) + BATCH_SIZE - 1) // BATCH_SIZE)
    step = 0

    for epoch in range(EPOCHS):
        # ---------- train ----------
        model.train()
        order = np.random.RandomState(epoch).permutation(len(train_data))
        epoch_loss = 0.0
        epoch_correct = 0
        for start in range(0, len(train_data), BATCH_SIZE):
            batch = [train_data[i] for i in order[start : start + BATCH_SIZE]]
            targets = torch.tensor(
                [m.label for m in batch], dtype=torch.long, device=DEVICE
            )

            # MoE forward: runs every expert, calls gate, combines.
            logits = model.forward(batch)  # (B, num_classes)

            # Gather per-expert quantities for div/sim losses. Re-runs the
            # experts on the same batch — this is what the reference council
            # does too (see train_council_3.py `outputs_1, feas_1 = ...`).
            per_expert_features, per_expert_logits = _per_expert_outputs(
                model, batch, DEVICE
            )

            # Total loss = task_ce + alpha(t)*diversity + beta(t)*similarity
            # (matches the reference's sum(loss_1, loss_2, loss_3, joint_loss)
            #  when alpha/beta are chosen so per-expert CE dominates).
            loss, parts = loss_fn(
                logits,
                targets,
                expert_features=per_expert_features,
                expert_logits=per_expert_logits,
                step=step,
                total_steps=total_steps,
            )

            # Also add per-expert CE, exactly like train_council_3.py:
            per_expert_ce = sum(
                task_ce_loss(el.unsqueeze(0), targets[:1]) for el in per_expert_logits
            )  # summed
            total_loss = loss + 0.1 * per_expert_ce

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Optional hook: run TF-side training step for TF experts.
            _maybe_tf_expert_step(model.experts, batch, targets)

            step += 1
            epoch_loss += float(total_loss)
            epoch_correct += int((logits.argmax(-1) == targets).sum())

        scheduler.step()
        train_acc = epoch_correct / len(train_data)

        # ---------- eval ----------
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for start in range(0, len(val_data), BATCH_SIZE):
                batch = val_data[start : start + BATCH_SIZE]
                targets = torch.tensor(
                    [m.label for m in batch], dtype=torch.long, device=DEVICE
                )
                logits = model.forward(batch)
                correct += int((logits.argmax(-1) == targets).sum())
                total += len(batch)
        val_acc = correct / max(1, total)

        gw = model.last_gate_weights.mean(0).cpu().numpy()
        print(
            f"[epoch {epoch:03d}] loss={epoch_loss / max(1, len(train_data)):.4f}  "
            f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  gate_mean={np.round(gw, 3).tolist()}"
        )

        if val_acc > best_val:
            best_val = val_acc
            _save_checkpoint(model, epoch, val_acc, CKPT_DIR / "best.pt")

    print(f"\nbest val_acc = {best_val:.3f}, saved to {CKPT_DIR / 'best.pt'}")


# ---------- utilities ----------------------------------------------------
def _per_expert_outputs(model, batch, device):
    """Return (list of features per expert, list of logits per expert), each
    a torch tensor of shape (feature_dim,) or (num_classes,) mean-pooled over
    the batch. Used by the diversity + similarity losses.
    """
    from mme.experts.bridge import to_torch

    features, logits_all = [], []
    for e in model.experts:
        fs, ls = [], []
        for m in batch:
            out = e.forward(e.preprocess(m))
            f = out.features if out.features is not None else out.logits
            fs.append(to_torch(f).reshape(-1).float().to(device))
            ls.append(to_torch(out.logits).reshape(-1).float().to(device))
        features.append(torch.stack(fs).mean(0))
        logits_all.append(torch.stack(ls).mean(0))
    return features, logits_all


def _maybe_tf_expert_step(experts, batch, targets):
    """Hook: run a TF training step for each TF expert.

    The MoE optimizer only sees torch params, so a TF expert stays frozen from
    its perspective. If you want the TF expert to *also* update, do it here
    with its own tf.GradientTape — using the same targets, or the picked-gate
    weights as sample weights. Left as a hook because it depends on your
    baseline's API. Example:

        for e in experts:
            if e.framework == "tf" and hasattr(e, "tf_train_step"):
                e.tf_train_step(batch, targets)
    """
    for e in experts:
        if getattr(e, "framework", "torch") == "tf" and hasattr(e, "tf_train_step"):
            e.tf_train_step(batch, targets)


def _save_checkpoint(model, epoch, val_acc, path):
    state = {"epoch": epoch, "val_acc": val_acc}
    for i, e in enumerate(model.experts):
        if getattr(e, "framework", "torch") == "torch" and hasattr(e, "state_dict"):
            state[f"expert_{i}"] = e.state_dict()
    if hasattr(model.gate, "state_dict"):
        state["gate"] = model.gate.state_dict()
    torch.save(state, path)


if __name__ == "__main__":
    main()
