"""MMEModel — orchestrates experts and gate.

The host framework is PyTorch. Every expert produces an ``ExpertOutput``; TF
expert outputs cross the framework boundary as numpy arrays and are rewrapped
as torch tensors (gradients do NOT flow across the boundary — see
``docs/mixing_frameworks.md``).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from mme.core.mesh import Mesh
from mme.experts.base import Expert, ExpertOutput
from mme.experts.bridge import to_torch


class MMEModel:
    """Runs every expert, gathers outputs, calls the gate, combines.

    Args:
        experts: list of Expert instances (torch and/or tf).
        gate: any object with a ``forward(mesh, expert_outputs)`` method
              returning weights of shape ``[B, num_experts]`` or
              ``[B, num_faces, num_experts]``.
        combine: how to combine per-expert logits with gate weights.
            "weighted_softmax": softmax over experts of gate weights, then
            weighted sum of expert logits. Fully differentiable.
            "hard_argmax": pick the single expert with the largest gate weight
            for each mesh; final logits are that expert's logits. This is the
            router used by the reference council code (train_council*.py).
            Straight-through: the picked expert's logits keep their grad;
            gate weights are used only for selection (no grad through the pick).
    """

    _COMBINE_MODES = ("weighted_softmax", "hard_argmax")

    def __init__(
        self,
        experts: Sequence[Expert],
        gate,
        combine: str = "weighted_softmax",
    ) -> None:
        if len(experts) < 1:
            raise ValueError("MMEModel needs at least one expert")
        if combine not in self._COMBINE_MODES:
            raise ValueError(
                f"unknown combine mode: {combine}; choose from {self._COMBINE_MODES}"
            )
        self.experts: List[Expert] = list(experts)
        self.gate = gate
        self.combine = combine

        num_classes = {e.num_classes for e in self.experts}
        if len(num_classes) != 1:
            raise ValueError(f"all experts must share num_classes, got {num_classes}")
        self.num_classes = num_classes.pop()

    # ------------------------------------------------------------------
    def _run_expert(self, expert: Expert, mesh: Mesh) -> ExpertOutput:
        raw = expert.preprocess(mesh)
        out = expert.forward(raw)
        if not isinstance(out, ExpertOutput):
            raise TypeError(
                f"expert {expert.name!r} must return ExpertOutput; got {type(out).__name__}"
            )
        # Convert TF outputs to torch tensors (no autograd across boundary).
        if expert.framework == "tf":
            out = ExpertOutput(
                logits=to_torch(out.logits),
                features=to_torch(out.features) if out.features is not None else None,
                per_element_attention=to_torch(out.per_element_attention)
                if out.per_element_attention is not None
                else None,
            )
        return out

    # ------------------------------------------------------------------
    def forward(self, meshes: Sequence[Mesh], gate_kwargs: Optional[dict] = None):
        """Run the model on a batch of meshes.

        Returns torch tensor of shape ``[B, num_classes]``.
        """
        import torch

        if len(meshes) == 0:
            raise ValueError("empty batch")

        # Per-mesh per-expert outputs. Loop over meshes; batching within an
        # expert is left to the caller (each expert may consume differently
        # sized meshes, so stacking is not always possible).
        per_mesh_logits: List[torch.Tensor] = []
        per_mesh_weights: List[torch.Tensor] = []

        for mesh in meshes:
            expert_outs = [self._run_expert(e, mesh) for e in self.experts]
            # [num_experts, num_classes]
            expert_logits = torch.stack(
                [o.logits.reshape(-1) for o in expert_outs], dim=0
            )

            gate_kwargs = gate_kwargs or {}
            weights = self.gate.forward(mesh, expert_outs, **gate_kwargs)
            # Expect [num_experts] or [1, num_experts]. Normalize to [num_experts].
            if weights.dim() == 2:
                weights = weights.squeeze(0)
            if weights.shape[0] != len(self.experts):
                raise ValueError(
                    f"gate returned weights of shape {tuple(weights.shape)} "
                    f"for {len(self.experts)} experts"
                )
            weights = torch.softmax(weights, dim=0)

            if self.combine == "weighted_softmax":
                combined = (weights.unsqueeze(-1) * expert_logits).sum(dim=0)
            else:  # hard_argmax — pick one expert; grads flow through only that expert.
                picked = int(torch.argmax(weights))
                combined = expert_logits[picked]
            per_mesh_logits.append(combined)
            per_mesh_weights.append(weights)

        self.last_gate_weights = torch.stack(per_mesh_weights, dim=0)  # [B, E]
        return torch.stack(per_mesh_logits, dim=0)  # [B, num_classes]

    # ------------------------------------------------------------------
    def torch_parameters(self):
        """All *trainable* torch parameters across torch experts and the gate.

        Parameters with ``requires_grad=False`` are filtered out so that frozen
        experts (loaded from a pretrained checkpoint) and frozen gates (e.g.
        :class:`mme.gate.random_gate.RandomGate`) are not fed to the optimizer.
        """
        params = []
        for e in self.experts:
            if e.framework == "torch" and hasattr(e, "parameters"):
                params += [p for p in e.parameters() if p.requires_grad]
        if hasattr(self.gate, "parameters"):
            params += [p for p in self.gate.parameters() if p.requires_grad]
        return params

    def to(self, device):
        for e in self.experts:
            if hasattr(e, "to"):
                e.to(device)
        if hasattr(self.gate, "to"):
            self.gate.to(device)
        return self

    def train(self, mode: bool = True):
        for e in self.experts:
            if hasattr(e, "train"):
                e.train(mode)
        if hasattr(self.gate, "train"):
            self.gate.train(mode)
        return self

    def eval(self):
        return self.train(False)
