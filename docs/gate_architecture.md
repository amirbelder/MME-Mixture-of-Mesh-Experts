# Gate architecture

Paper: *"MME: Mixture of Mesh Experts with Random Walk Transformer Gating"* (Belder & Tal).

MME ships **two** implementations of the paper's gate — pick the one that matches your workflow.

## Option A — `WalkHierGate` (vendored TF, authoritative)

`mme/gate/walk_hier_gate.py::WalkHierGate` wraps the paper's real TensorFlow `WalkHierTransformer` (vendored from your reference project at `mme/gate/walk_hier_transformer_tf.py`, bit-for-bit identical to the source in `~/mme_reference/attention_model/attention_model.py`). Use this when:

- You have `.keras` checkpoints from your reference project you want to load.
- You want the exact numerics of the paper.
- Your MME env has TensorFlow installed (`pip install -e ".[all]"`).

```python
from mme.gate.walk_hier_gate import WalkHierGate

gate = WalkHierGate(
    num_experts   = 3,
    walk_len      = 100,
    num_walks     = 32,
    d_model       = 128,
    num_layers    = 3,
    num_heads     = 8,
    dff           = 256,
    jump_every_k  = 20,
    pooling       = True,
)
# after the first forward, weights are built and you can load a .keras ckpt:
# gate.load_weights("attention_gate_walker.keras")
```

Gradients from the torch-side MoE loss do **not** flow into TF weights (numpy bridge). Train the gate with its own `tf.GradientTape` alongside the MoE step — or load a pretrained checkpoint and freeze.

Default `walk_features_fn(mesh, walks)` returns **XYZ position (3-dim)**. Override for the paper's exact per-vertex features.

## Option B — `MMEGateTorch` (paper-based torch re-implementation)

`mme/gate/mme_gate_torch.py::MMEGateTorch` is a pure-torch re-implementation from the paper description. Use this when you don't want TF in your env, or you want the gate to update via the shared torch optimizer.

Three stages:

### 1. Random walks over the mesh
`mme/gate/random_walk.py::sample_walks` samples `num_walks` uniform random walks of length `walk_len` on the vertex adjacency graph. Walks are refreshed on every gate forward.

### 2. Walk transformer
At each walked vertex we compose a 7-dim feature vector:
- vertex position (3)
- averaged adjacent face normal (3)
- self-loop indicator (1) — 1 if the walk stayed put (isolated vertex fallback)

Linearly project to `feature_dim`, add sinusoidal positional encoding along the walk, and run a small `TransformerEncoder` (`mme/gate/transformer.py`). Mean-pool each walk to a single token.

### 3. Cross-attention to expert features
The pooled walk-token is used as the query in a `MultiheadAttention` with the per-expert feature vectors as keys/values. Score each expert as `score = W · (attn_out + key_i)` → `[num_experts]` logit vector.

`MMEModel` applies `softmax` over these logits and combines the per-expert logits accordingly.

## Which to use

| | `WalkHierGate` (TF) | `MMEGateTorch` (torch) |
|---|---|---|
| Loads paper checkpoints | Yes | No (would need weight conversion) |
| Numerics match paper | Yes (bit-for-bit) | Approximate |
| Trains with torch optimizer | No (TF grads only) | Yes |
| Requires TF in the env | Yes | No |
| Env deps | `tensorflow>=2.10`, optionally `tensorflow-addons` | just `torch` |

Both implement the same `MMEModel.gate` interface — they're drop-in swappable.

## Notes / TODOs

- For **segmentation-style outputs**, the gate should return `[num_faces, num_experts]` instead of `[num_experts]`; the `MMEModel` combiner will need a small tweak to broadcast per-face. Currently only classification-style gating is wired.
- For **paper-exact per-vertex features** in `WalkHierGate`, look at how your reference `data/joined_future_data.py::WalkerData` builds the `walks` tensor and pass the same composition via `walk_features_fn`.

## How walks are sent to the gate (matches `train_val.py`)

Both `WalkHierGate` and `AttWalkExpert` build the tensor the TF model consumes **exactly** the way the reference project's `train_val.py::train_step` does:

1. **Optional normalization** — `mme.gate.walk_features.norm_model(vertices)` (center by mean + scale by max L2 distance). Identical to reference `dataset.norm_model`. Default: `normalize=True`.
2. **Walk sampling** — `mme.gate.walk_algorithms.sample_paper_walks(mesh, num_walks, walk_len, walk_alg=...)`. Available samplers (same names as reference `params.walk_alg`):
   - `"random_global_jumps"` — paper default; backtracking + `p=1/100` global jump.
   - `"no_jumps"` — pure backtracking.
   - `"random_global_jumps_new"` — pop-from-stack backtracking variant.
   - `"constant_global_jumps"` — deterministic jump every `k` steps.
   - `"local_jumps"` — KDTree-based (needs precomputed `kdtree_query`).
3. **Feature composition** — `mme.gate.walk_features.compose_walk_features(vertices, seq, jumps, seq_len, spec=...)`. Available filler names (same as reference `params.net_input` list):
   - `"xyz"` (3) — position, `vertices[seq[1:seq_len+1]]`.
   - `"dxdydz"` (3) — `np.diff(vertices[seq]) * 100`.
   - `"jump_indication"` (1) — jump flag.
   - `"v_normals"` (3) — per-vertex normals (needs `v_normals=...`).
   - `"vertex_indices"` (1) — raw ids.
4. **Batching** — `batch_compose_walk_features` returns `(num_walks, walk_len, D)`. That's the same shape as reference `model_ftrs` after its `tf.reshape(model_ftrs_, (-1, sp[-2], sp[-1]))`.

Because the two MME wrappers (`WalkHierGate`, `AttWalkExpert`) share this exact path, walk tensors are byte-compatible across **every option** — Mode A (joint) or Mode B (frozen `PrerenderedExpert`), torch gate or TF gate. If you need to tweak the feature composition, do it in one place (`net_input=(...)` arg) and both flows pick it up automatically.

```python
# Paper default (xyz only, 3-dim):
WalkHierGate(num_experts=3, walk_alg="random_global_jumps", net_input=("xyz",))

# Richer 7-dim features:
WalkHierGate(num_experts=3, net_input=("xyz", "dxdydz", "jump_indication"))

# Full custom callable:
def my_features(mesh, walks):
    # walks: (num_walks, walk_len + 1) int
    return build_something(mesh, walks)   # returns (num_walks, walk_len, D)

WalkHierGate(num_experts=3, walk_features_fn=my_features)
```
