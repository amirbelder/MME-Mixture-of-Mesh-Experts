# Quickstart

## Install

```bash
pip install -e ".[torch,dev]"
```

## Run the toy example

```bash
python examples/train_toy.py
```

Trains 3 tiny torch experts on a synthetic dataset of deformed platonic solids and prints the per-expert gate weights each epoch.

## CLI

```bash
mme list-experts --import-module examples.toy_expert
mme train --config examples/configs/toy_classification.yaml
mme eval  --config examples/configs/toy_classification.yaml --ckpt runs/toy/epoch_019.pt
```

## Programmatic use

```python
import examples.toy_expert  # registers experts
from mme.core.moe import MMEModel
from mme.data.synthetic import make_synthetic_mesh
from mme.experts.registry import get_expert
from mme.gate.mme_gate_torch import MMEGateTorch

experts = [get_expert(name, num_classes=4)
           for name in ("toy_mlp_small", "toy_mlp_wide", "toy_mlp_deep")]
gate = MMEGateTorch(num_experts=3, feature_dim=32, walk_len=16, num_walks=4)
model = MMEModel(experts=experts, gate=gate)

mesh = make_synthetic_mesh("sphere")
logits = model.forward([mesh])
print(logits.shape)  # torch.Size([1, 4])
```
