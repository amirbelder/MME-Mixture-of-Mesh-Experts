"""RandomGate — a frozen, seeded gate for the "does routing even matter?" ablation.

The gate is a small MLP over ``mesh.sampled_vertex_features``. Its weights are
initialized from a caller-supplied ``seed`` and marked ``requires_grad=False``,
so :meth:`mme.core.moe.MMEModel.torch_parameters` will *not* include them in
the optimizer.

Typical use with hard routing (matches the reference council code):

    from mme.core.moe import MMEModel
    from mme.gate.random_gate import RandomGate

    gate = RandomGate(num_experts=3, feature_dim=32, seed=1234)
    model = MMEModel(experts=[e1, e2, e3], gate=gate, combine="hard_argmax")

Saving / loading the exact random init that produced a reported number:

    gate.save("runs/random_gate.pt")           # writes {"seed":..., "state_dict":...}
    gate2 = RandomGate.load("runs/random_gate.pt")
"""

from __future__ import annotations

from typing import Any, Sequence

from mme.core.mesh import Mesh
from mme.experts.base import ExpertOutput


class RandomGate:
    """Frozen random-init MLP gate over per-mesh feature vectors.

    Args:
        num_experts: number of experts to score.
        feature_dim: input feature width (matches
            ``Mesh.sampled_vertex_features(dim=feature_dim)``).
        hidden: hidden width of the two-layer MLP.
        seed: RNG seed used to initialize weights. Also stored on the gate
            and written to disk by :meth:`save`.
    """

    num_experts: int

    def __init__(
        self,
        num_experts: int,
        feature_dim: int = 32,
        hidden: int = 64,
        seed: int = 0,
    ) -> None:
        import torch
        import torch.nn as nn

        self.num_experts = num_experts
        self.feature_dim = feature_dim
        self.hidden = hidden
        self.seed = int(seed)

        # Initialize weights deterministically from ``seed`` without disturbing
        # the caller's global RNG state.
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        prev_state = torch.random.get_rng_state()
        try:
            torch.random.set_rng_state(gen.get_state())
            self.net = nn.Sequential(
                nn.Linear(feature_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, num_experts),
            )
        finally:
            torch.random.set_rng_state(prev_state)

        for p in self.net.parameters():
            p.requires_grad_(False)

        self._device = None

    # ------------------------------------------------------------------
    def parameters(self):
        # Returned but all have requires_grad=False, so MMEModel.torch_parameters
        # will filter them out. Exposed so save/load/state_dict work.
        return self.net.parameters()

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)

    def to(self, device):
        self.net.to(device)
        self._device = device
        return self

    def train(self, mode: bool = True):
        # Frozen gate has no BN/dropout to toggle, but keep API parity.
        self.net.train(mode)
        return self

    # ------------------------------------------------------------------
    def forward(
        self, mesh: Mesh, expert_outputs: Sequence[ExpertOutput], **_ignored: Any
    ):
        """Score experts for one mesh.

        Returns a torch tensor of shape ``[num_experts]``. ``MMEModel`` applies
        softmax + combine (weighted or hard-argmax).
        """
        import torch

        if len(expert_outputs) != self.num_experts:
            raise ValueError(
                f"gate expected {self.num_experts} experts, got {len(expert_outputs)}"
            )
        x = torch.from_numpy(mesh.sampled_vertex_features(dim=self.feature_dim)).float()
        if self._device is not None:
            x = x.to(self._device)
        with torch.no_grad():
            scores = self.net(x)
        return scores

    # ------------------------------------------------------------------
    def save(self, path) -> None:
        """Save ``{"seed": int, "state_dict": ..., "meta": {...}}`` to ``path``."""
        import torch

        payload = {
            "seed": self.seed,
            "state_dict": self.state_dict(),
            "meta": {
                "num_experts": self.num_experts,
                "feature_dim": self.feature_dim,
                "hidden": self.hidden,
            },
        }
        torch.save(payload, str(path))

    @classmethod
    def load(cls, path) -> "RandomGate":
        """Reconstruct a RandomGate from a file written by :meth:`save`."""
        import torch

        payload = torch.load(str(path), map_location="cpu")
        meta = payload["meta"]
        gate = cls(
            num_experts=meta["num_experts"],
            feature_dim=meta["feature_dim"],
            hidden=meta["hidden"],
            seed=int(payload["seed"]),
        )
        gate.load_state_dict(payload["state_dict"])
        return gate
