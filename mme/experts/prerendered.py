"""PrerenderedExpert — an Expert whose outputs are looked up on disk.

This lets you use baselines whose Python environments are incompatible with
the MME env (SubdivNet on Jittor, MeshMAE on its own torch, Laplacian2Mesh
needs pytorch3d, ...). Instead of importing their code into the MME venv,
you run each baseline **once** in its own venv, dump per-mesh
``{logits, features}`` to a ``.pt`` file keyed by mesh source path, then
consume that file here.

See ``docs/running_the_pipeline.md`` for the end-to-end recipe.

Expected file format (produced by your dump script in each baseline's venv):

    {
        "<abs or repo-relative path to mesh 1>": {
            "logits":   torch.Tensor of shape (num_classes,),
            "features": torch.Tensor of shape (feature_dim,),   # optional
        },
        "<abs or repo-relative path to mesh 2>": { ... },
        ...
    }

You can pass a ``key_fn(mesh) -> str`` if the paths need normalization.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from mme.core.mesh import Mesh
from mme.experts.base import Expert, ExpertOutput


class PrerenderedExpert(Expert):
    """Expert backed by precomputed ``{key: {logits, features}}`` dumps.

    Args:
        name: registered name for this expert instance.
        num_classes: number of classes (must match logits shape).
        dump_path: path to a ``.pt`` file with the mapping described above.
        key_fn: called with each ``Mesh`` to produce the lookup key.
            Defaults to ``mesh.source_path``.

    Notes:
        - This expert has zero trainable parameters. ``MMEModel.torch_parameters``
          returns nothing for it and the optimizer never touches it.
        - ``framework = "torch"`` because we return torch tensors.
        - If a mesh key is missing, raises ``KeyError`` at forward time
          (better to fail loud than silently misroute).
    """

    framework = "torch"

    def __init__(
        self,
        name: str,
        num_classes: int,
        dump_path: str,
        key_fn: Optional[Callable[[Mesh], str]] = None,
    ) -> None:
        super().__init__(num_classes=num_classes)
        self.name = name
        self.dump_path = str(dump_path)
        self._key_fn = key_fn or _default_key_fn

        import torch

        payload = torch.load(self.dump_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(
                f"dump file {self.dump_path!r} must be dict[str, dict]; "
                f"got {type(payload).__name__}"
            )
        self._store: Dict[str, Dict[str, Any]] = payload
        self._device = None

    # ------------------------------------------------------------------
    def parameters(self):
        return []  # no trainable params

    def to(self, device):
        self._device = device
        return self

    def train(self, mode: bool = True):
        return self  # nothing stateful to toggle

    def eval(self):
        return self

    # ------------------------------------------------------------------
    def preprocess(self, mesh: Mesh) -> Mesh:
        return mesh  # lookup happens in forward

    def forward(self, mesh: Mesh) -> ExpertOutput:
        import torch

        key = self._key_fn(mesh)
        if key not in self._store:
            raise KeyError(
                f"PrerenderedExpert({self.name!r}): no precomputed output "
                f"for key {key!r}. Check that the dump was produced against "
                f"the same dataset and that mesh.source_path is populated."
            )
        entry = self._store[key]
        logits = torch.as_tensor(entry["logits"]).float()
        if logits.numel() != self.num_classes:
            raise ValueError(
                f"PrerenderedExpert({self.name!r}) key={key!r}: "
                f"logits have {logits.numel()} elements, expected {self.num_classes}"
            )
        feats = entry.get("features")
        if feats is not None:
            feats = torch.as_tensor(feats).float()
        if self._device is not None:
            logits = logits.to(self._device)
            if feats is not None:
                feats = feats.to(self._device)
        return ExpertOutput(logits=logits, features=feats)


def _default_key_fn(mesh: Mesh) -> str:
    if mesh.source_path is None:
        raise ValueError(
            "PrerenderedExpert default key_fn needs mesh.source_path to be set. "
            "Use MeshDataset (which populates it) or pass a custom key_fn."
        )
    return mesh.source_path
