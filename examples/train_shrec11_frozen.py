"""SHREC-11 with frozen experts (mode B): only the gate trains.

The three experts are already trained (each in its own env), their outputs
were dumped to .pt files by env/<baseline>/dump_features.py, and we load them
here with PrerenderedExpert. Zero expert parameters go to the optimizer;
only the gate updates.

Prerequisites:
    ~/dumps/meshcnn_shrec11.pt
    ~/dumps/subdivnet_shrec11.pt
    ~/dumps/meshwalker_shrec11.pt     (or any three of your baselines)

Run:
    python examples/train_shrec11_frozen.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from mme.core.moe import MMEModel
from mme.data.mesh_dataset import MeshDataset
from mme.experts.prerendered import PrerenderedExpert
from mme.gate.mme_gate_torch import MMEGateTorch
from mme.losses import (
    diversity_loss,
    DynamicBalancedLoss,
    linear_schedule,
    similarity_loss,
    task_ce_loss,
)


# ---------- configuration ------------------------------------------------
SHREC11_ROOT = Path("~/shrec11").expanduser()  # <class>/*.obj
DUMPS_DIR = Path("~/dumps").expanduser()
CKPT_DIR = Path("runs/shrec11_frozen")
NUM_CLASSES = 30
EPOCHS = 100
BATCH_SIZE = 8
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model() -> MMEModel:
    experts = [
        PrerenderedExpert(
            name="meshcnn",
            num_classes=NUM_CLASSES,
            dump_path=str(DUMPS_DIR / "meshcnn_shrec11.pt"),
        ),
        PrerenderedExpert(
            name="subdivnet",
            num_classes=NUM_CLASSES,
            dump_path=str(DUMPS_DIR / "subdivnet_shrec11.pt"),
        ),
        PrerenderedExpert(
            name="meshwalker",
            num_classes=NUM_CLASSES,
            dump_path=str(DUMPS_DIR / "meshwalker_shrec11.pt"),
        ),
    ]
    gate = MMEGateTorch(
        num_experts=3,
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


# ------------------------------------------------------------------------
def main() -> None:
    dataset = MeshDataset(SHREC11_ROOT)  # populates mesh.source_path
    train_data, val_data = split_train_val(dataset)
    print(f"train / val = {len(train_data)} / {len(val_data)}, classes = {NUM_CLASSES}")

    model = build_model().to(DEVICE)
    trainable_params = model.torch_parameters()  # only gate params
    print(
        f"trainable params: {sum(p.numel() for p in trainable_params)} (frozen experts contribute 0)"
    )

    optimizer = torch.optim.Adam(trainable_params, lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

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
        for start in range(0, len(train_data), BATCH_SIZE):
            batch = [train_data[i] for i in order[start : start + BATCH_SIZE]]
            targets = torch.tensor(
                [m.label for m in batch], dtype=torch.long, device=DEVICE
            )

            logits = model.forward(batch)  # (B, num_classes)

            # For div/sim losses we need the per-expert quantities. Re-collect
            # (cheap — every PrerenderedExpert forward is a dict lookup).
            per_expert_features = []
            per_expert_logits = []
            for e in model.experts:
                fs, ls = [], []
                for m in batch:
                    out = e.forward(e.preprocess(m))
                    fs.append(
                        (out.features if out.features is not None else out.logits)
                        .reshape(-1)
                        .float()
                    )
                    ls.append(out.logits.reshape(-1).float())
                per_expert_features.append(torch.stack(fs).mean(0))
                per_expert_logits.append(torch.stack(ls).mean(0))

            loss, parts = loss_fn(
                logits,
                targets,
                expert_features=per_expert_features,
                expert_logits=per_expert_logits,
                step=step,
                total_steps=total_steps,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            epoch_loss += float(loss)

        scheduler.step()

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
            f"val_acc={val_acc:.3f}  gate_mean={np.round(gw, 3).tolist()}"
        )

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {"gate": model.gate.state_dict(), "epoch": epoch, "val_acc": val_acc},
                CKPT_DIR / "best.pt",
            )

    print(f"\nbest val_acc = {best_val:.3f}, saved to {CKPT_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
