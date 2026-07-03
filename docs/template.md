# The MME template — 3 experts + 1 gate, train + infer

This file is the whole point of the repo. Read it top-to-bottom and copy the last section.

## The template has four slots

```
   ┌──────────────────┐
   │  Expert A        │──►logits_a, features_a
   │  (your net #1)   │
   └──────────────────┘
   ┌──────────────────┐
   │  Expert B        │──►logits_b, features_b     ─┐
   │  (your net #2)   │                              │
   └──────────────────┘                              ▼
   ┌──────────────────┐                       ┌────────────┐
   │  Expert C        │──►logits_c, features_c│    Gate    │─►weights
   │  (your net #3)   │                       └────────────┘
   └──────────────────┘                              │
                                                     ▼
                                           combine (weighted or hard-argmax)
                                                     │
                                                     ▼
                                                  final logits
```

- **Slot 1–3 (experts)** — any `nn.Module` that reads a `Mesh` and returns `ExpertOutput(logits, features)`. Wrap it as a `TorchExpert` and register with `@register_expert("name")`.
- **Slot 4 (gate)** — any object with `forward(mesh, expert_outputs) -> [num_experts]` scores. Two shipped:
  - `MMEGateTorch` — the paper's random-walk-transformer gate (**trained**).
  - `RandomGate` — frozen random-init MLP (**untrained**, for the routing-ablation baseline).

## Step 1 — Register your three experts

```python
# my_experts.py
import torch, torch.nn as nn
from mme.experts.torch_expert import TorchExpert
from mme.experts.base import ExpertOutput
from mme.experts.registry import register_expert

@register_expert("net_a")                    # <-- slot 1
class NetA(TorchExpert, nn.Module):
    def __init__(self, num_classes: int):
        nn.Module.__init__(self); TorchExpert.__init__(self, num_classes=num_classes)
        # ... your architecture (MeshCNN, MeshWalker, DiffusionNet, whatever) ...
        self.backbone = ...      # produces (B, feature_dim)
        self.head = nn.Linear(feature_dim, num_classes)

    def preprocess(self, mesh):
        # Convert mme.core.mesh.Mesh -> whatever your net eats.
        # Return the tensor(s) your forward() expects.
        return ...

    def forward(self, x):
        feats  = self.backbone(x)
        logits = self.head(feats)
        return ExpertOutput(logits=logits, features=feats)
```

Repeat for `net_b`, `net_c`. See `docs/expert_interface.md` for a full walk-through and `examples/toy_expert.py` for a working reference.

If your expert already has a pretrained checkpoint (typical — you trained each expert to 100% on SHREC11 first), load it in `__init__` and freeze:

```python
self.load_state_dict(torch.load("net_a_shrec11.pt"))
for p in self.parameters(): p.requires_grad_(False)
self.eval()
```

`MMEModel.torch_parameters()` filters `requires_grad=False`, so frozen experts don't enter the optimizer — only the gate gets trained. This is exactly the reference council pattern (`train_council*.py`).

## Step 2 — Pick a gate

**Trained gate (paper's random-walk transformer):**

```python
from mme.gate.mme_gate_torch import MMEGateTorch
gate = MMEGateTorch(num_experts=3, feature_dim=32, walk_len=32, num_walks=8)
```

**Random frozen gate (routing-ablation baseline):**

```python
from mme.gate.random_gate import RandomGate
gate = RandomGate(num_experts=3, feature_dim=32, seed=1234)
```

**Your own gate** — anything with `.forward(mesh, expert_outputs) -> torch.Tensor[num_experts]`.

## Step 3 — Assemble the MoE and train

```python
import my_experts  # side-effect registers all three
from mme.experts.registry import get_expert
from mme.core.moe import MMEModel
from mme.losses import DynamicBalancedLoss, task_ce_loss, diversity_loss, similarity_loss, linear_schedule
from mme.training.trainer import Trainer

experts = [get_expert(n, num_classes=30) for n in ("net_a", "net_b", "net_c")]
model = MMEModel(experts=experts, gate=gate, combine="hard_argmax")   # or "weighted_softmax"

loss_fn = DynamicBalancedLoss(
    task_loss_fn=task_ce_loss,
    diversity_loss_fn=diversity_loss,
    similarity_loss_fn=similarity_loss,
    schedule=linear_schedule(0.0, 0.05, 0.0, 0.05),    # α(t), β(t)
)

trainer = Trainer(model=model, loss_fn=loss_fn,
                  train_data=train_meshes, val_data=val_meshes,
                  batch_size=8, epochs=100, lr=1e-3,
                  ckpt_dir="runs/my_council")
trainer.fit()
```

## Step 4 — Inference

```python
import torch
from mme.eval.classify import evaluate_classification

# Reload
state = torch.load("runs/my_council/epoch_099.pt", map_location="cpu")
for i, e in enumerate(model.experts):
    if f"expert_{i}" in state and hasattr(e, "load_state_dict"):
        e.load_state_dict(state[f"expert_{i}"])
if "gate" in state and hasattr(model.gate, "load_state_dict"):
    model.gate.load_state_dict(state["gate"])

print(evaluate_classification(model, val_meshes))
```

For the random-gate ablation, also save the gate separately so the exact random init is reproducible (see `docs/random_gate.md`):

```python
model.gate.save("runs/my_council/random_gate.pt")   # writes seed + state_dict
```

## That's the whole template

- Framework decisions (which architectures to plug in, which gate, whether to freeze experts, what schedule) are yours.
- The framework guarantees: (a) experts and gate compose cleanly, (b) frozen params stay frozen, (c) checkpoints round-trip.

For three concrete architecture suggestions that all report SHREC11 numbers, see [`shrec11_baselines.md`](shrec11_baselines.md).
