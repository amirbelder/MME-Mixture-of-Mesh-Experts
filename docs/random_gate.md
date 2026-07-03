# Random gate ablation

The `RandomGate` is a small MLP router with **frozen, seeded random weights**. It's included so you can answer the classic "does the routing even matter?" question — i.e. compare your learned MME gate against a random baseline.

## Usage

```python
from mme.core.moe import MMEModel
from mme.gate.random_gate import RandomGate

gate = RandomGate(num_experts=3, feature_dim=32, seed=1234)
model = MMEModel(experts=[e1, e2, e3], gate=gate, combine="hard_argmax")
```

- All gate params have `requires_grad=False`, so `MMEModel.torch_parameters()` returns only expert params — the optimizer never touches the gate.
- `combine="hard_argmax"` matches the reference council router (`train_council*.py`): pick the single top-scoring expert per sample; that expert's logits are the final output.

## Save / load the exact gate

```python
gate.save("runs/random_gate.pt")           # writes seed + state_dict
gate = RandomGate.load("runs/random_gate.pt")
```

Any number you report from a run should ship with the saved `random_gate.pt` — that pins down which random init produced it.

## Comparison recipe

1. Train experts with `RandomGate`, note best val accuracy.
2. Same experts, swap in `MMEGateTorch` (the paper's learned gate). Note best val.
3. Delta is the router's contribution.

Do this with `combine="hard_argmax"` on both to isolate the routing choice from the soft-vs-hard combine effect.
