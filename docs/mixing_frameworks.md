# Mixing PyTorch and TensorFlow experts

MME is torch-hosted: `MMEModel` calls the gate in torch and returns torch logits. TensorFlow experts are supported as **forward-only feature producers** — outputs cross the framework boundary as numpy arrays and are rewrapped as torch tensors.

## What this means

- **No cross-framework autograd.** Gradients do NOT flow from the MoE task loss back into TF expert weights. Training a TF expert should be done in a separate TF optimizer step, or the expert should be treated as frozen.
- **Small overhead per boundary crossing** (numpy copy). Fine for the typical MoE forward pass; watch it if you have very high per-mesh throughput.
- **No shared CUDA context guarantees.** If both frameworks are on GPU, they compete for VRAM. See the tips below.

## Recommended setups

- **Torch-only** (simplest): `pip install -e ".[torch]"`. Fully differentiable end-to-end.
- **Torch + frozen TF experts**: `pip install -e ".[all]"`. TF expert weights are trained externally (or pre-trained checkpoints), and the torch gate + torch experts + task loss are jointly optimized.
- **TF-only** (rare): use `mme_gate_tf.MMEGateTF` and write your own TF training loop.

## Memory tips when installing both

- Install `torch` and `tensorflow` matching your CUDA version **from the official channels first**, then `pip install -e ".[all]"`.
- Limit TF's GPU appetite:

  ```python
  import tensorflow as tf
  for g in tf.config.list_physical_devices("GPU"):
      tf.config.experimental.set_memory_growth(g, True)
  ```

- If you don't need TF on the GPU:

  ```bash
  CUDA_VISIBLE_DEVICES="" python examples/tf_expert_example.py
  ```

## Why not a full autograd bridge?

Cross-framework autograd (torch↔tf) via numpy is fragile and adds implicit copies. Making the boundary explicit keeps the codebase small and the behavior predictable. If you need joint training of a TF expert **through** the gate, we recommend porting the expert to torch.
