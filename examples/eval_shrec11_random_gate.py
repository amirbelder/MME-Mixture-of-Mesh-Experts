"""SHREC-11 Split-16 inference with pretrained experts + UNTRAINED random gate.

Both the experts and the gate are frozen — no training happens. This is the
"how good is the ensemble when the router doesn't know anything?" evaluation:
answers the question you'd expect the paper's trained gate to beat.

- Experts: three PrerenderedExpert instances reading .pt dumps produced by
  env/{subdivnet,meshmae,laplacian2mesh}/dump_features.py, each with its own
  pretrained checkpoint. Zero trainable params.
- Gate:    RandomGate — seeded random MLP, requires_grad=False. Zero
  trainable params. The gate is re-instantiated N_SEEDS times and results
  averaged so the "random" isn't a lucky/unlucky one-off.

Prerequisites:
    ~/shrec11_split16/<class>/*.obj          (30 classes, 20 meshes each)
    ~/dumps/subdivnet_shrec11.pt
    ~/dumps/meshmae_shrec11.pt
    ~/dumps/laplacian2mesh_shrec11.pt

Run:
    python examples/eval_shrec11_random_gate.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

from mme.core.moe import MMEModel
from mme.data.mesh_dataset import MeshDataset
from mme.experts.prerendered import PrerenderedExpert
from mme.gate.random_gate import RandomGate


# ---------- SHREC-11 Split-16 configuration -------------------------------
SHREC11_ROOT    = Path("~/shrec11_split16").expanduser()
DUMPS_DIR       = Path("~/dumps").expanduser()
OUT_DIR         = Path("runs/shrec11_100pct_random_gate_eval")
NUM_CLASSES     = 30
BATCH_SIZE      = 8
TRAIN_PER_CLASS = 16
TEST_PER_CLASS  = 4
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Multiple random-gate seeds → mean ± std over independent random initializations.
# Reproducible: same SEEDS list ⇒ identical numbers on rerun.
SEEDS           = tuple(range(10))          # 10 random gates
COMBINE         = "weighted_softmax"        # or "hard_argmax"
GATE_FEATURE_DIM = 32
GATE_HIDDEN      = 64

# Set EVAL_ON_TRAIN = True to also print train-set accuracy (slow-ish; 480 meshes).
EVAL_ON_TRAIN   = False

EXPERT_SPECS = (
    ("subdivnet",      DUMPS_DIR / "subdivnet_shrec11.pt"),
    ("meshmae",        DUMPS_DIR / "meshmae_shrec11.pt"),
    ("laplacian2mesh", DUMPS_DIR / "laplacian2mesh_shrec11.pt"),
)


# ==========================================================================
def _check_dumps_or_die() -> None:
    missing = [str(p) for _, p in EXPERT_SPECS if not p.exists()]
    if missing:
        print(
            "ERROR: missing expert dump(s):\n  - " + "\n  - ".join(missing)
            + "\n\nRun each baseline's dump script in its own env first:\n"
            + "  cd env/subdivnet      && python dump_features.py --ckpt ... --data ... --out ~/dumps/subdivnet_shrec11.pt\n"
            + "  cd env/meshmae        && python dump_features.py --ckpt ... --data ... --out ~/dumps/meshmae_shrec11.pt\n"
            + "  cd env/laplacian2mesh && python dump_features.py --ckpt ... --data ... --out ~/dumps/laplacian2mesh_shrec11.pt",
            file=sys.stderr,
        )
        sys.exit(2)


def _check_dataset_or_die() -> None:
    if not SHREC11_ROOT.exists():
        print(
            f"ERROR: SHREC-11 root not found at {SHREC11_ROOT}\n"
            "Expected: <root>/<class_name>/*.obj — 30 classes, 20 meshes per class.\n"
            "MeshCNN's downloader gives the Split-16 layout:\n"
            "  git clone https://github.com/ranahanocka/MeshCNN.git && "
            "bash MeshCNN/scripts/shrec/get_data.sh",
            file=sys.stderr,
        )
        sys.exit(2)


def _split_16_4(dataset):
    """Deterministic first-16-train / next-4-test partition per class."""
    per_class: dict = {}
    for i in range(len(dataset.samples)):
        path, label = dataset.samples[i]
        per_class.setdefault(label, []).append((str(path), i))

    train_idx, test_idx = [], []
    for label, items in per_class.items():
        items.sort()                                        # deterministic
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
    test_data  = [dataset[i] for i in test_idx]
    return train_data, test_data


def _build_experts() -> list:
    experts = [
        PrerenderedExpert(name=name, num_classes=NUM_CLASSES, dump_path=str(p))
        for name, p in EXPERT_SPECS
    ]
    for e in experts:
        print(f"  expert {e.name!r}: {e.dump_path} ({len(e._store)} entries)")
    return experts


def _build_gate(seed: int) -> RandomGate:
    return RandomGate(
        num_experts=len(EXPERT_SPECS),
        feature_dim=GATE_FEATURE_DIM,
        hidden=GATE_HIDDEN,
        seed=seed,
    )


# ==========================================================================
def _eval_split(model: MMEModel, data, split_name: str) -> dict:
    """Return {accuracy, count, per_class_correct, per_class_total, gate_hist}.

    ``gate_hist`` is the mean softmaxed gate weight across the whole split —
    tells you how uniformly the random gate distributed mass across experts.
    """
    correct = total = 0
    per_class_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    per_class_total   = np.zeros(NUM_CLASSES, dtype=np.int64)
    gate_sum          = np.zeros(len(EXPERT_SPECS), dtype=np.float64)

    with torch.no_grad():
        for start in range(0, len(data), BATCH_SIZE):
            batch = data[start : start + BATCH_SIZE]
            targets = torch.tensor([m.label for m in batch], dtype=torch.long, device=DEVICE)
            logits = model.forward(batch)
            preds = logits.argmax(-1)

            correct += int((preds == targets).sum())
            total += len(batch)

            for t, p in zip(targets.cpu().numpy(), preds.cpu().numpy()):
                per_class_total[int(t)] += 1
                if int(t) == int(p):
                    per_class_correct[int(t)] += 1

            gate_sum += model.last_gate_weights.mean(0).cpu().numpy()

    n_batches = max(1, (len(data) + BATCH_SIZE - 1) // BATCH_SIZE)
    return {
        "split":             split_name,
        "accuracy":          correct / max(1, total),
        "count":             total,
        "per_class_acc":     (per_class_correct / np.clip(per_class_total, 1, None)).tolist(),
        "gate_mean_weights": (gate_sum / n_batches).tolist(),
    }


# ==========================================================================
def main() -> None:
    _check_dataset_or_die()
    _check_dumps_or_die()

    dataset = MeshDataset(SHREC11_ROOT)
    if dataset.num_classes != NUM_CLASSES:
        print(
            f"warning: found {dataset.num_classes} classes, expected {NUM_CLASSES}",
            file=sys.stderr,
        )
    train_data, test_data = _split_16_4(dataset)
    print(
        f"SHREC-11 Split-16 loaded: "
        f"{len(train_data)} train ({TRAIN_PER_CLASS}/class) · "
        f"{len(test_data)} test ({TEST_PER_CLASS}/class) · "
        f"{dataset.num_classes} classes\n"
    )

    experts = _build_experts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sanity: nothing should be trainable in this run.
    tmp_gate = _build_gate(seed=SEEDS[0])
    tmp_model = MMEModel(experts=experts, gate=tmp_gate, combine=COMBINE).to(DEVICE)
    n_train_params = sum(p.numel() for p in tmp_model.torch_parameters())
    print(
        f"combine    : {COMBINE}\n"
        f"gate       : {type(tmp_gate).__name__}(feature_dim={GATE_FEATURE_DIM}, hidden={GATE_HIDDEN})\n"
        f"seeds      : {list(SEEDS)}   ({len(SEEDS)} random gates)\n"
        f"trainable  : {n_train_params} params (must be 0 for a pure-inference run)\n"
    )
    if n_train_params != 0:
        raise AssertionError(
            "expected zero trainable params — check that both experts and gate "
            "have requires_grad=False on all parameters"
        )

    # Sweep seeds.
    results: list = []
    for seed in SEEDS:
        gate = _build_gate(seed=seed).to(DEVICE)
        model = MMEModel(experts=experts, gate=gate, combine=COMBINE).to(DEVICE)
        model.eval()

        row = {"seed": seed, "splits": {}}
        for split_name, split_data in [
            ("test", test_data),
            *((("train", train_data),) if EVAL_ON_TRAIN else ()),
        ]:
            metrics = _eval_split(model, split_data, split_name)
            row["splits"][split_name] = metrics
            gw = [round(x, 3) for x in metrics["gate_mean_weights"]]
            print(
                f"[seed={seed:03d}] {split_name:5s}  acc={metrics['accuracy']:.4f}  "
                f"gate_mean={gw}"
            )

        # Save this gate's exact state so any single row is reproducible.
        gate.save(OUT_DIR / f"random_gate_seed{seed}.pt")
        results.append(row)

    # -------- aggregate across seeds --------
    test_accs = [r["splits"]["test"]["accuracy"] for r in results]
    mean_test = statistics.mean(test_accs)
    std_test  = statistics.stdev(test_accs) if len(test_accs) > 1 else 0.0
    best_seed = results[int(np.argmax(test_accs))]["seed"]
    worst_seed = results[int(np.argmin(test_accs))]["seed"]

    summary = {
        "combine":     COMBINE,
        "num_seeds":   len(SEEDS),
        "seeds":       list(SEEDS),
        "test":        {
            "mean_accuracy":  mean_test,
            "std_accuracy":   std_test,
            "min_accuracy":   min(test_accs),
            "max_accuracy":   max(test_accs),
            "best_seed":      best_seed,
            "worst_seed":     worst_seed,
            "per_seed_accs":  test_accs,
        },
        "experts":     [name for name, _ in EXPERT_SPECS],
        "expert_dumps": [str(p) for _, p in EXPERT_SPECS],
    }
    if EVAL_ON_TRAIN:
        train_accs = [r["splits"]["train"]["accuracy"] for r in results]
        summary["train"] = {
            "mean_accuracy": statistics.mean(train_accs),
            "std_accuracy":  statistics.stdev(train_accs) if len(train_accs) > 1 else 0.0,
            "per_seed_accs": train_accs,
        }

    summary_path = OUT_DIR / "summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)

    # -------- print report --------
    print("\n" + "=" * 60)
    print(f"SHREC-11 Split-16 · pretrained experts × 3 · untrained random gate × {len(SEEDS)}")
    print("=" * 60)
    print(f"combine        : {COMBINE}")
    print(f"experts        : {', '.join(name for name, _ in EXPERT_SPECS)}")
    print(f"test accuracy  : mean={mean_test:.4f}  std={std_test:.4f}  "
          f"min={min(test_accs):.4f}  max={max(test_accs):.4f}")
    print(f"best  seed     : {best_seed}   acc={max(test_accs):.4f}")
    print(f"worst seed     : {worst_seed}   acc={min(test_accs):.4f}")
    if EVAL_ON_TRAIN:
        print(f"train accuracy : mean={summary['train']['mean_accuracy']:.4f}  "
              f"std={summary['train']['std_accuracy']:.4f}")
    print(f"summary saved  : {summary_path}")
    print(f"per-seed gates : {OUT_DIR}/random_gate_seed<N>.pt")


if __name__ == "__main__":
    main()
