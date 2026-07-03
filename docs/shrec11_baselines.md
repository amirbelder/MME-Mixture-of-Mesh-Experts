# Three SHREC-11 baselines (100% on Split-16) to plug into the template

SHREC-11 is a small (30-class, 20 meshes per class, 600 total) mesh classification benchmark. The three papers below all **hit 100% on the standard Split-16 protocol** (16 train / 4 test per class = 120-mesh test set) and all ship public PyTorch code. Any of them works as an MME expert.

> **Ready-made scripts**:
> - [`examples/train_shrec11_100pct.py`](../examples/train_shrec11_100pct.py) — trains with all three plugged into MMEModel on Split-16.
> - [`examples/eval_shrec11_random_gate.py`](../examples/eval_shrec11_random_gate.py) — **inference only**, no gate training: pretrained experts + `RandomGate` swept over N seeds (mean ± std over independent random routers). Answers "how good is the ensemble when the router doesn't know anything?"

> **Sanity check the URLs and numbers before you commit** — they were correct at the time of writing but repos move and paper tables sometimes get updated.

## Recommended trio (all 100% on Split-16)

| Paper | Repo | Representation | SHREC-11 Split-16 (reported) |
|---|---|---|---|
| **SubdivNet** — Hu et al., SIGGRAPH 2022 | https://github.com/lzhengning/SubdivNet | Loop-subdivision hierarchy | **100.0%** |
| **MeshMAE** — Liang et al., ECCV 2022 | https://github.com/liang3588/MeshMAE | Masked-autoencoder pretrain + fine-tune | **100.0%** |
| **Laplacian2Mesh** — Dong et al., 2023 | https://github.com/QiujieDong/Laplacian2Mesh | Laplacian-spectral CNN | **100.0%** |

They're deliberately heterogeneous: **hierarchical face-based** (SubdivNet), **transformer with masked pretraining** (MeshMAE), and **spectral** (Laplacian2Mesh). Different inductive biases → the MoE gate has a real choice to make.

## Alternatives (strong but below 100% on Split-16)

Use these if you want to swap a slot for a different flavor:

| Paper | Repo | Representation | SHREC-11 Split-16 |
|---|---|---|---|
| PD-MeshNet — Milano et al., ECCV 2020 | https://github.com/MIT-SPARK/PD-MeshNet | Primal–dual graph attention | 99.7% |
| DiffusionNet — Sharp et al., TOG 2022 | https://github.com/nmwsharp/diffusion-net | Intrinsic diffusion, resolution-independent | 99.7% |
| HodgeNet — Smirnov & Solomon, SIGGRAPH 2021 | https://github.com/dmsm/HodgeNet | Hodge-Laplacian spectral | 99.2% |
| MeshWalker — Lahav & Tal, SIGGRAPH Asia 2020 | https://github.com/AlonLahav/MeshWalker | Random walks + GRU | 98.6% |
| MeshCNN — Hanocka et al., SIGGRAPH 2019 | https://github.com/ranahanocka/MeshCNN | Edge-based, learned edge-collapse pooling | 98.6% |
| MeshNet — Feng et al., AAAI 2019 | https://github.com/iMoonLab/MeshNet | Face-based (what your reference council uses) | ModelNet-only in the paper |

## Getting the SHREC-11 data

SubdivNet and MeshMAE both ship a preprocess script that downloads + splits SHREC-11 with the standard 16/4 split. Either works:

```bash
# Option A: SubdivNet's preprocess
git clone https://github.com/lzhengning/SubdivNet.git
cd SubdivNet
# Follow "Datasets" in its README — it produces a train/test tree keyed by class.

# Option B: MeshCNN's downloader (Split-16 by default)
git clone https://github.com/ranahanocka/MeshCNN.git
cd MeshCNN
bash scripts/shrec/get_data.sh
```

Both give you a `<root>/<class_name>/*.obj` (or `.off`) layout that plugs straight into `mme.data.MeshDataset`.

## Environment gotchas

None of the three baselines share a Python environment (different PyTorch versions, different geometry libs). Install each in its own venv and only bring the **trained checkpoint** back into the MME template — the MME framework never runs their training code.

```bash
# One venv per baseline
python -m venv .venv-subdivnet     && source .venv-subdivnet/bin/activate     && pip install -r SubdivNet/requirements.txt     && deactivate
python -m venv .venv-meshmae       && source .venv-meshmae/bin/activate       && pip install -r MeshMAE/requirements.txt         && deactivate
python -m venv .venv-laplacian2mesh && source .venv-laplacian2mesh/bin/activate && pip install -r Laplacian2Mesh/requirements.txt  && deactivate
```

**Known pain points**:
- **SubdivNet** uses [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/) as its default framework in some releases — check the repo's `README` for the PyTorch branch or install Jittor per its docs. There is a PyTorch reimplementation.
- **MeshMAE** requires ImageNet-style pretraining checkpoints that the repo hosts on Google Drive; download the released checkpoint first, then fine-tune on SHREC-11.
- **Laplacian2Mesh** needs `pytorch3d`, which is fussy about CUDA versions. Follow the install instructions in `Laplacian2Mesh/README.md` verbatim.

## Wrapping each as an MME Expert (template)

For each baseline, write ~30 lines like this in your own `my_experts.py`:

```python
# my_experts.py — the "adapter" pattern
import torch, torch.nn as nn
from mme.experts.torch_expert import TorchExpert
from mme.experts.base import ExpertOutput
from mme.experts.registry import register_expert

# 1) import the baseline's model class from its (installed or vendored) source
from subdivnet.models import SubdivNet   # example

@register_expert("subdivnet")
class SubdivNetExpert(TorchExpert, nn.Module):
    def __init__(self, num_classes: int, ckpt_path: str):
        nn.Module.__init__(self); TorchExpert.__init__(self, num_classes=num_classes)
        self.inner = SubdivNet(num_classes=num_classes)         # baseline's ctor
        self.inner.load_state_dict(torch.load(ckpt_path))
        # Freeze so the MoE optimizer only touches the gate:
        for p in self.inner.parameters(): p.requires_grad_(False)
        self.inner.eval()

    def preprocess(self, mesh):
        # Convert mme.core.mesh.Mesh -> whatever SubdivNet expects.
        # SubdivNet wants a subdivision hierarchy; build it here.
        return ...

    def forward(self, x):
        logits, feats = self.inner(x, return_features=True)
        return ExpertOutput(logits=logits, features=feats)
```

Repeat for `MeshMAEExpert` (patch tokens) and `Laplacian2MeshExpert` (Laplacian eigenvalues + intrinsic features). Each adapter is per-baseline glue and does not touch anything else in `mme/`.

## Concrete plan for reproducing "3-expert MME @ 100% on SHREC-11"

1. In three separate venvs, train each baseline to convergence on SHREC-11 Split-16 following its own README. All three should reproduce **100.0%**; if any falls short, the ceiling of the ensemble is the ceiling of that expert.
2. Copy each trained checkpoint into `~/checkpoints/{subdivnet,meshmae,laplacian2mesh}.pt`.
3. In your MME venv (`pip install -e ".[torch,dev]"`), write `my_experts.py` with three adapter classes (as above), each loading its own checkpoint and **freezing** its params.
4. Copy `examples/template_three_experts.py` → `examples/train_shrec11.py`, replace the four slots (import your `my_experts`, use `MeshDataset("~/shrec11")`, `NUM_CLASSES=30`, choose gate).
5. `python examples/train_shrec11.py` — only the gate trains; experts are frozen. Matches the reference council's `CouncilMeshNet` pattern (`load_state_dict(...); .eval()` on experts).
6. Report gate-vs-random-gate delta as your ablation (see `docs/random_gate.md`). When the three experts already hit 100% individually, the random-gate baseline should also hit ~100% (any expert is "correct enough"), which is the honest way to report the story.

### A note on "100% × 3 → what does the gate even do?"

If three experts each hit 100% on the test set, any router — trained, random, coin-flip — will also hit 100%. This is a nice sanity number but not a meaningful ablation of the gate itself. To get a signal on the gate:

- Report on a **harder benchmark** (SHREC-11 Split-10 or ModelNet40), where each expert is <100% and the router's choice actually matters.
- Or **hold-out one class per expert** so no single expert is complete, and the router has to combine complementary competences.

Both fit the same template — swap the dataset / expert checkpoints only.
