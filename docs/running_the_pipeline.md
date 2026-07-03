# Actually running the pipeline

The three baselines you plug in (see `docs/shrec11_baselines.md`) often have **mutually incompatible Python environments**:

- **SubdivNet** — Jittor.
- **MeshMAE** — its own torch + timm.
- **Laplacian2Mesh** — needs `pytorch3d` (pinned torch/CUDA).
- **MeshWalker**, **AttWalk** — TensorFlow 2.
- **MeshCNN**, **DiffusionNet**, **HodgeNet** — plain PyTorch.

You have **two honest ways** to run the pipeline, depending on which baselines you pick.

## Mode A — Unified env (jointly trained experts + gate)

**When**: all your baselines can share one env. Pure-torch baselines coexist easily. TF baselines can also live here (`mme[all]` installs both torch and TF; TF experts go through the numpy bridge — see `docs/mixing_frameworks.md`). Jittor / pinned-old-pytorch3d **do not** coexist with much — pick baselines accordingly.

**What runs**: `examples/train_shrec11_joint.py` — every torch parameter (experts + gate) enters one optimizer. TF experts are supported but frozen from the MoE's perspective (gradients don't cross the numpy bridge; if you want a TF expert to also update, run its own `tf.GradientTape` step in the hook shown at the bottom of the file).

**Setup**:

```bash
cd MME
python -m venv .venv && source .venv/bin/activate
pip install -e ".[torch,tf,dev]"                # or [all,dev]
# Install each baseline's Python package into this same venv (see env/<baseline>/README.md)
```

**Training loop**: matches the reference council code (`train_council*.py`) — per-expert CE + diversity + similarity + joint CE. Written out explicitly in `examples/train_shrec11_joint.py`; no hidden `Trainer` magic.

## Mode B — Per-env dumps (only gate trains)

**When**: your baselines have incompatible envs (SubdivNet's Jittor + Laplacian2Mesh's pytorch3d cannot cohabit; period). Or you want the fastest possible gate iteration — no expert forward on every step.

**What runs**: `examples/train_shrec11_frozen.py` — uses `PrerenderedExpert` to read per-mesh `{logits, features}` from `.pt` files that were dumped once, in each baseline's own env. Only the **gate** trains.

**Setup** (five phases):

| Phase | Env | What happens |
|---|---|---|
| 1 | per-baseline venv | Train each baseline on your dataset. |
| 2 | per-baseline venv | Run `env/<baseline>/dump_features.py` — iterate the whole dataset, save `{source_path: {logits, features}}` to `~/dumps/<baseline>.pt`. |
| 3 | MME env | `pip install -e ".[torch,dev]"`; sanity-check the dumps. |
| 4 | MME env | `python examples/train_shrec11_frozen.py`. Gate trains. |
| 5 | MME env | Inference from `runs/shrec11_frozen/best.pt`. |

The `env/` folder ships one recipe per baseline (`requirements.txt`, `README.md`, `dump_features.py`) — see `env/README.md` for the full list.

## Which mode to use

| | Mode A (joint) | Mode B (frozen + dumps) |
|---|---|---|
| Envs | One shared env; only compatible baselines | Three separate envs; any baselines |
| Trainable | All experts + gate | Gate only |
| Per-step speed | Slow (all backbones forward+back) | Fast (dict lookups) |
| Ceiling accuracy | Best in theory | Bounded by frozen experts |
| Matches reference council | Yes (this is what `train_council*.py` does) | No — reference council also updates experts |
| Fits paper story | "Full council with joint training" | "Gate + loss balancing on frozen heterogeneous experts" |

Both are legit. If your experts are truly incompatible (Jittor + TF + pytorch3d), Mode B is the only option. If they're all torch or torch+TF, Mode A gives you strictly more capacity.

## Recommended gate: the torch walk gate

For the paper's random-walk transformer gate, the shipped defaults in `mme/gate/mme_gate_torch.py` are a good starting point but I suggest these SHREC-11 settings:

```python
MMEGateTorch(
    num_experts   = 3,
    feature_dim   = 128,   # bumped from the default 32 — 128 fits the paper better
    walk_len      = 100,   # long walks work well on SHREC-11 meshes (~1000 vertices)
    num_walks     = 32,    # 32 walks per mesh gives a stable summary
    num_heads     = 8,     # 8 heads at d_model=128 → head_dim=16 (standard)
    num_layers    = 3,     # 3 encoder layers is plenty; deeper = overfits on SHREC-11
    seed          = 0,     # rotates each forward; keep for repro
)
```

Both training scripts use exactly these settings.

## What about TF experts (MeshWalker, AttWalk)?

Two ways:

1. **Frozen from MoE grads** (default). Wrap as a `TFExpert` and plug into `MMEModel`; outputs cross the numpy bridge and are used forward-only. Gradients from the joint loss do not flow back into TF weights. Fine when the TF expert is already trained to convergence.

2. **Independently trained during MoE loop**. In `examples/train_shrec11_joint.py`, `_maybe_tf_expert_step(...)` is a hook called every step. Implement `tf_train_step(batch, targets)` on your TF expert and run a normal `tf.GradientTape` step in there — the TF expert then updates its own weights using its own optimizer, in parallel with the torch-side MoE training. Simplest way to keep TF experts "learning" without cross-framework autograd.

If you want the TF expert to update **based on gate signal** (e.g. weighted by how confidently the gate routed to it), pass the gate weights into `tf_train_step` as sample weights — the hook has access to `model.last_gate_weights`.

### The TODO on the gate file

`mme/gate/mme_gate_torch.py` is a **paper-based re-implementation, not your reference code**. If you send me `models/attention_gate.py` (`WalkHierTransformer`) from your reference project — it wasn't in the zip — I'll port it verbatim so the gate you publish is bit-for-bit what your paper describes.
