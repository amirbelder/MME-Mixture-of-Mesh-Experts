"""Training loop for MMEModel.

Kept small and readable — it's easier for users to fork than to configure
around. If you need distributed training, wrap :class:`MMEModel`'s
``torch_parameters()`` in your own trainer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

from mme.core.mesh import Mesh
from mme.core.moe import MMEModel


class Trainer:
    def __init__(
        self,
        model: MMEModel,
        loss_fn: Callable,
        train_data: Sequence[Mesh],
        val_data: Optional[Sequence[Mesh]] = None,
        batch_size: int = 8,
        epochs: int = 20,
        lr: float = 1e-3,
        device: str = "auto",
        ckpt_dir: Optional[str] = None,
        log_every: int = 1,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.train_data = list(train_data)
        self.val_data = list(val_data) if val_data else None
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.log_every = log_every

    def _resolve_device(self):
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def fit(self) -> None:
        import torch

        device = self._resolve_device()
        self.model.to(device)

        # Warm up: run one forward pass so lazily-built modules (e.g. the
        # gate's per-width feature projections) are materialized BEFORE we
        # snapshot torch parameters into the optimizer.
        import torch as _torch

        with _torch.no_grad():
            self.model.forward(self.train_data[: min(2, len(self.train_data))])

        params = self.model.torch_parameters()
        if not params:
            raise RuntimeError(
                "no torch parameters to optimize (no torch experts + gate?)"
            )
        opt = torch.optim.Adam(params, lr=self.lr)

        n = len(self.train_data)
        total_steps = self.epochs * ((n + self.batch_size - 1) // self.batch_size)
        step = 0

        for epoch in range(self.epochs):
            self.model.train(True)
            # Shuffle indices deterministically per epoch.
            g = torch.Generator().manual_seed(epoch)
            order = torch.randperm(n, generator=g).tolist()

            epoch_loss = 0.0
            epoch_batches = 0

            for start in range(0, n, self.batch_size):
                batch_idx = order[start : start + self.batch_size]
                meshes = [self.train_data[i] for i in batch_idx]
                targets = torch.tensor(
                    [m.label if m.label is not None else 0 for m in meshes],
                    dtype=torch.long,
                    device=device,
                )
                logits = self.model.forward(meshes)

                # Collect expert-side quantities for div/sim losses.
                # For simplicity we recompute expert outputs at the batch level:
                from mme.experts.bridge import to_torch

                expert_features: list = []
                expert_logits: list = []
                for e in self.model.experts:
                    per_mesh_f = []
                    per_mesh_l = []
                    for mesh in meshes:
                        out = e.forward(e.preprocess(mesh))
                        f = out.features if out.features is not None else out.logits
                        per_mesh_f.append(to_torch(f).reshape(-1).float())
                        per_mesh_l.append(to_torch(out.logits).reshape(-1).float())
                    expert_features.append(torch.stack(per_mesh_f, dim=0).mean(dim=0))
                    expert_logits.append(torch.stack(per_mesh_l, dim=0).mean(dim=0))

                loss, parts = self.loss_fn(
                    logits, targets, expert_features, expert_logits, step, total_steps
                )
                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += float(loss)
                epoch_batches += 1
                step += 1

            if epoch % self.log_every == 0:
                gw = self.model.last_gate_weights.mean(dim=0).detach().cpu().tolist()
                gw_str = ", ".join(f"{w:.3f}" for w in gw)
                print(
                    f"[epoch {epoch:03d}] loss={epoch_loss / max(1, epoch_batches):.4f} "
                    f"gate_mean=[{gw_str}]"
                )

            if self.ckpt_dir is not None:
                self.ckpt_dir.mkdir(parents=True, exist_ok=True)
                # Save torch state dicts of each torch expert and the gate.
                state = {"epoch": epoch}
                for i, e in enumerate(self.model.experts):
                    if e.framework == "torch" and hasattr(e, "state_dict"):
                        state[f"expert_{i}"] = e.state_dict()
                if hasattr(self.model.gate, "state_dict"):
                    state["gate"] = self.model.gate.state_dict()
                torch.save(state, self.ckpt_dir / f"epoch_{epoch:03d}.pt")
