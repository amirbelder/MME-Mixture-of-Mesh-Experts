"""SHREC-11 Split-16 (16 train / 4 test) with three 100%-accuracy experts.

Experts (all report 100% on SHREC-11 Split-16 in their papers):
    - SubdivNet         (Hu et al., SIGGRAPH 2022)   env/subdivnet/
    - MeshMAE           (Liang et al., ECCV 2022)    env/meshmae/
    - Laplacian2Mesh    (Dong et al., 2023)          env/laplacian2mesh/

These three baselines have MUTUALLY INCOMPATIBLE Python environments
(Jittor vs pytorch3d vs MeshMAE's torch pin). They cannot cohabit one env,
so the only viable workflow is Mode B: train each in its own venv, dump
per-mesh {logits, features} to a .pt file, then load them here via
PrerenderedExpert. See docs/running_the_pipeline.md.

Prerequisites (produced by env/<baseline>/dump_features.py, one per venv):

    ~/dumps/subdivnet_shrec11.pt
    ~/dumps/meshmae_shrec11.pt
    ~/dumps/laplacian2mesh_shrec11.pt

The gate slot below is left as a TODO — plug in your weighted random gate
(see docs/random_gate.md for RandomGate + combine="weighted_softmax", or
write your own).

Run:
    python examples/train_shrec11_100pct.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from mme.core.moe import MMEModel
from mme.data.mesh_dataset import MeshDataset
from mme.experts.prerendered import PrerenderedExpert
from mme.losses import (
    diversity_loss,
    DynamicBalancedLoss,
    linear_schedule,
    similarity_loss,
    task_ce_loss,
)


# ---------- SHREC-11 Split-16 configuration -------------------------------
# Split-16 = 16 train + 4 test meshes per class. 30 classes total.
SHREC11_ROOT = Path("~/shrec11_split16").expanduser()  # <class>/*.obj tree
DUMPS_DIR = Path("~/dumps").expanduser()
CKPT_DIR = Path("runs/shrec11_100pct_split16")
NUM_CLASSES = 30
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed by the split protocol — do NOT change these two.
TRAIN_PER_CLASS = 16
TEST_PER_CLASS = 4


# ---------- three 100%-Split-16 experts -----------------------------------
EXPERT_SPECS = (
    ("subdivnet", DUMPS_DIR / "subdivnet_shrec11.pt"),
    ("meshmae", DUMPS_DIR / "meshmae_shrec11.pt"),
    ("laplacian2mesh", DUMPS_DIR / "laplacian2mesh_shrec11.pt"),
)


# ---------- gate ----------------------------------------------------------
# TODO(amir): replace with your weighted random gate implementation.
# For a starting point that already works, uncomment one of these:
#
# (a) frozen seeded random weights, soft combination:
#   from mme.gate.random_gate import RandomGate
#   def build_gate(): return RandomGate(num_experts=3, feature_dim=32, seed=1234)
#   COMBINE = "weighted_softmax"
#
# (b) paper's trained walk-hier transformer gate (needs TF):
#   from mme.gate.walk_hier_gate import WalkHierGate
#   def build_gate(): return WalkHierGate(num_experts=3, walk_len=100, num_walks=32)
#   COMBINE = "weighted_softmax"
#
# (c) pure-torch re-impl of the paper's gate (approximate, differentiable):
#   from mme.gate.mme_gate_torch import MMEGateTorch
#   def build_gate(): return MMEGateTorch(num_experts=3, feature_dim=128)
#   COMBINE = "weighted_softmax"

from mme.gate.random_gate import RandomGate  # placeholder default


def build_gate():
    return RandomGate(num_experts=3, feature_dim=32, seed=1234)  # ← swap here


COMBINE = "weighted_softmax"


# ==========================================================================
def _check_dumps_or_die() -> None:
    """Fail loudly on startup if any of the three .pt dumps is missing."""
    missing = [str(p) for _, p in EXPERT_SPECS if not p.exists()]
    if missing:
        print(
            "ERROR: missing expert dump(s):\n  - "
            + "\n  - ".join(missing)
            + "\n\nRun each baseline's dump script in its own env first:\n"
            + "  cd env/subdivnet      && python dump_features.py --ckpt ... --data ... --out ~/dumps/subdivnet_shrec11.pt\n"
            + "  cd env/meshmae        && python dump_features.py --ckpt ... --data ... --out ~/dumps/meshmae_shrec11.pt\n"
            + "  cd env/laplacian2mesh && python dump_features.py --ckpt ... --data ... --out ~/dumps/laplacian2mesh_shrec11.pt\n"
            + "See env/README.md and docs/running_the_pipeline.md.",
            file=sys.stderr,
        )
        sys.exit(2)


def _check_dataset_or_die():
    if not SHREC11_ROOT.exists():
        print(
            f"ERROR: SHREC-11 root not found at {SHREC11_ROOT}\n"
            "Expected layout: <root>/<class_name>/*.obj (or .off), 30 classes, "
            "20 meshes per class total (16 train + 4 test after split).\n"
            "MeshCNN's downloader gives you the standard Split-16 layout:\n"
            "  git clone https://github.com/ranahanocka/MeshCNN.git && "
            "bash MeshCNN/scripts/shrec/get_data.sh",
            file=sys.stderr,
        )
        sys.exit(2)


def _split_16_4(dataset):
    """Deterministic 16-train / 4-test partition per class.

    Reference SHREC-11 has exactly 20 meshes per class; we take the first 16
    (sorted by file path) as train, the last 4 as test.
    """
    per_class: dict = {}
    for i in range(len(dataset.samples)):
        path, label = dataset.samples[i]
        per_class.setdefault(label, []).append((str(path), i))

    train_idx, test_idx = [], []
    for label, items in per_class.items():
        items.sort()  # deterministic
        if len(items) < TRAIN_PER_CLASS + TEST_PER_CLASS:
            raise ValueError(
                f"class label={label} has only {len(items)} meshes; "
                f"Split-16 needs at least {TRAIN_PER_CLASS + TEST_PER_CLASS}"
            )
        for _, idx in items[:TRAIN_PER_CLASS]:
            train_idx.append(idx)
        for _, idx in items[TRAIN_PER_CLASS : TRAIN_PER_CLASS + TEST_PER_CLASS]:
            test_idx.append(idx)
    train_data = [dataset[i] for i in train_idx]
    test_data = [dataset[i] for i in test_idx]
    return train_data, test_data


# ==========================================================================
def main() -> None:
    _check_dataset_or_die()
    _check_dumps_or_die()

    dataset = MeshDataset(SHREC11_ROOT)
    if dataset.num_classes != NUM_CLASSES:
        print(
            f"warning: found {dataset.num_classes} classes, expected {NUM_CLASSES}. "
            "Verify your SHREC-11 root has exactly the 30 canonical classes.",
            file=sys.stderr,
        )
    train_data, test_data = _split_16_4(dataset)
    print(
        f"SHREC-11 Split-16 loaded: "
        f"{len(train_data)} train ({TRAIN_PER_CLASS}/class) · "
        f"{len(test_data)} test ({TEST_PER_CLASS}/class) · "
        f"{dataset.num_classes} classes"
    )

    experts = [
        PrerenderedExpert(name=name, num_classes=NUM_CLASSES, dump_path=str(path))
        for name, path in EXPERT_SPECS
    ]
    for e in experts:
        print(f"  expert {e.name!r}: dump {e.dump_path} ({len(e._store)} entries)")

    gate = build_gate()
    model = MMEModel(experts=experts, gate=gate, combine=COMBINE).to(DEVICE)

    trainable = model.torch_parameters()
    print(
        f"gate  : {type(gate).__name__}   combine: {COMBINE}\n"
        f"trainable params: {sum(p.numel() for p in trainable)}"
    )
    if not trainable:
        print(
            "note: no trainable parameters. Both experts and gate are frozen. "
            "This is legit for a pure-inference / random-gate ablation run — "
            "the loop below will just eval a random-gate ensemble."
        )

    # ---- optimizer + schedule + loss ---------------------------------
    if trainable:
        optimizer = torch.optim.Adam(trainable, lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    else:
        optimizer = scheduler = None

    loss_fn = DynamicBalancedLoss(
        task_loss_fn=task_ce_loss,
        diversity_loss_fn=diversity_loss,
        similarity_loss_fn=similarity_loss,
        schedule=linear_schedule(0.0, 0.05, 0.0, 0.05),
    )

    # ---- training loop ----------------------------------------------
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val = 0.0
    total_steps = EPOCHS * ((len(train_data) + BATCH_SIZE - 1) // BATCH_SIZE)
    step = 0

    for epoch in range(EPOCHS):
        # train phase
        model.train()
        order = np.random.RandomState(epoch).permutation(len(train_data))
        epoch_loss = 0.0
        for start in range(0, len(train_data), BATCH_SIZE):
            batch = [train_data[i] for i in order[start : start + BATCH_SIZE]]
            targets = torch.tensor(
                [m.label for m in batch], dtype=torch.long, device=DEVICE
            )
            logits = model.forward(batch)

            # per-expert quantities for div/sim losses
            per_expert_features, per_expert_logits = _per_expert_outputs(
                model, batch, DEVICE
            )
            loss, _ = loss_fn(
                logits,
                targets,
                expert_features=per_expert_features,
                expert_logits=per_expert_logits,
                step=step,
                total_steps=total_steps,
            )

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            step += 1
            epoch_loss += float(loss)

        if scheduler is not None:
            scheduler.step()

        # eval phase
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for start in range(0, len(test_data), BATCH_SIZE):
                batch = test_data[start : start + BATCH_SIZE]
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
            f"test_acc={val_acc:.4f}  gate_mean={np.round(gw, 3).tolist()}"
        )

        if val_acc > best_val:
            best_val = val_acc
            state = {"epoch": epoch, "val_acc": val_acc}
            if hasattr(model.gate, "state_dict"):
                state["gate"] = model.gate.state_dict()
            if hasattr(model.gate, "save"):
                model.gate.save(CKPT_DIR / "gate.pt")  # RandomGate saves seed too
            torch.save(state, CKPT_DIR / "best.pt")

    print(f"\nbest test acc: {best_val:.4f} @ {CKPT_DIR / 'best.pt'}")


# ---------- utilities ----------------------------------------------------
def _per_expert_outputs(model, batch, device):
    """Return per-expert (features, logits), mean-pooled over the batch."""
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


if __name__ == "__main__":
    main()
