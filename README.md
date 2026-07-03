# MME — Mixture of Mesh Experts (template)

Reference implementation for the paper **"MME: Mixture of Mesh Experts with Random Walk Transformer Gating"** (Belder & Tal).

**This repo is a template.** Plug in **three (or more) mesh-classification networks**, plug in **a gate** (the paper's trained random-walk transformer, or a random-init frozen baseline), and you get **training + inference out of the box**. Networks can be different architectures and can be frozen (loaded from your own pretrained checkpoints) or trained end-to-end.

## Start here (pick one)

- **Just want the recipe?** → [`docs/template.md`](docs/template.md) — the four slots (3 experts + 1 gate), with runnable code.
- **Multiple envs / need to actually run this?** → [`docs/running_the_pipeline.md`](docs/running_the_pipeline.md) — Mode A (unified env, joint training, matches reference council) vs Mode B (per-env dumps, gate-only training).
- **Per-baseline env recipes** → [`env/README.md`](env/README.md) — one folder per baseline with `requirements.txt` + install + `dump_features.py`.
- **Want a concrete SHREC-11 setup?** → [`docs/shrec11_baselines.md`](docs/shrec11_baselines.md) — three published architectures with public code.
- **Want the paper's gate walkthrough?** → [`docs/gate_architecture.md`](docs/gate_architecture.md).
- **Want the routing ablation?** → [`docs/random_gate.md`](docs/random_gate.md) — frozen random-init gate, saved seed + state_dict.
- **Mixing PyTorch + TensorFlow experts (MeshWalker, AttWalk, …)?** → [`docs/mixing_frameworks.md`](docs/mixing_frameworks.md).

## Install

```bash
# 1. Install the ML framework(s) matching your CUDA first
#    https://pytorch.org/get-started/locally/
# 2. Install this repo
git clone https://github.com/amirbelder/MME-Mixture-of-Mesh-Experts.git
cd MME-Mixture-of-Mesh-Experts
pip install -e ".[torch,dev]"
```

## Smoke test

```bash
python examples/template_three_experts.py
```

Runs the template on synthetic deformed platonic solids with three toy MLP experts + the paper's trained gate. When the loss decreases and per-epoch gate weights specialize, the plumbing works — swap the four slots for your real setup.

## The four slots

| Slot | What goes here | Where |
|---|---|---|
| 1–3 | Three experts (torch **or** TF). Options: your own `TorchExpert`/`TFExpert` classes (fully live in one env — see `examples/train_shrec11_joint.py`), or `PrerenderedExpert` reading pre-dumped logits (multi-env — see `examples/train_shrec11_frozen.py`) | `env/<baseline>/`, `mme/experts/` |
| 4   | Gate: `MMEGateTorch` (trained) or `RandomGate` (frozen ablation) or your own | `mme/gate/` |
| combine | `"hard_argmax"` (matches the reference council router) or `"weighted_softmax"` | `MMEModel(..., combine=...)` |
| data | Synthetic toy, `MeshDataset` directory reader, or your own | `mme.data` |

## Repository layout

```
mme/
├── core/          # Mesh container, MMEModel, protocols, device helpers
├── experts/       # Expert ABC, registry, TorchExpert, TFExpert, PrerenderedExpert, bridge
├── gate/          # random_walk, transformer, MMEGateTorch, MMEGateTF, RandomGate
├── losses/        # task CE, diversity, similarity, dynamic balance
├── data/          # MeshDataset, trimesh loader, synthetic toy, cache
├── training/      # Trainer (used by examples), metrics
├── eval/          # classify, retrieve, segment
├── config.py      # YAML → dataclass config loader
└── cli.py         # mme list-experts | train | eval
env/               # ← one folder per baseline you want to plug in
├── meshcnn/       # torch (Hanocka et al. 2019)
├── subdivnet/     # Jittor (Hu et al. 2022, 100% on SHREC-11 Split-16)
├── meshwalker/    # TensorFlow (Lahav & Tal 2020) ← Amir's paper
└── attwalk/       # TensorFlow (attention walker)
examples/
├── train_shrec11_joint.py              # Mode A — explicit loop, all experts + gate train together
├── train_shrec11_frozen.py             # Mode B — PrerenderedExpert, gate-only training
├── train_shrec11_100pct.py             # Mode B — three 100%-Split-16 experts, gate trains
├── eval_shrec11_random_gate.py         # Inference only — pretrained experts × RandomGate (N seeds)
├── template_three_experts.py           # ← the template you copy to start
├── train_two_expert_random_gate.py     # routing ablation
├── train_toy.py                        # end-to-end synthetic
├── register_and_use.py                 # minimal one-forward demo
├── tf_expert_example.py                # mixed torch+tf MoE
└── toy_expert.py                       # 3 registered toy experts
tests/       # pytest, TF tests auto-skip
docs/        # template, running_the_pipeline, shrec11_baselines, gate_architecture,
             # mixing_frameworks, training, random_gate, expert_interface, quickstart
```

## CLI

```bash
mme list-experts --import-module my_experts
mme train --config configs/my_config.yaml
mme eval  --config configs/my_config.yaml --ckpt runs/my_council/epoch_099.pt
```

## Citation

```bibtex
@article{belder_tal_mme,
  title   = {MME: Mixture of Mesh Experts with Random Walk Transformer Gating},
  author  = {Belder, Amir and Tal, Ayellet},
  year    = {2025},
}
```

## License

MIT. See [`LICENSE`](LICENSE).
