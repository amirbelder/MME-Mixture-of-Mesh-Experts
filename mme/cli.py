"""``mme`` CLI: list-experts | train | eval."""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import List, Sequence


def _import_modules(module_names: Sequence[str]) -> None:
    """Import each module so that ``@register_expert`` side-effects run."""
    for name in module_names:
        try:
            importlib.import_module(name)
        except Exception as e:
            print(f"warning: failed to import {name!r}: {e}", file=sys.stderr)


def _build_experts_from_config(cfg) -> list:
    from mme.experts.registry import get_expert

    experts = []
    for ec in cfg.experts:
        kwargs = dict(ec.kwargs)
        kwargs.setdefault("num_classes", cfg.num_classes)
        experts.append(get_expert(ec.name, **kwargs))
    return experts


def _build_gate_from_config(cfg):
    num_experts = len(cfg.experts)
    if cfg.gate.kind == "mme_torch":
        from mme.gate.mme_gate_torch import MMEGateTorch

        return MMEGateTorch(num_experts=num_experts, **cfg.gate.kwargs)
    if cfg.gate.kind == "mme_tf":
        from mme.gate.mme_gate_tf import MMEGateTF

        return MMEGateTF(num_experts=num_experts, **cfg.gate.kwargs)
    raise ValueError(f"unknown gate kind {cfg.gate.kind!r}")


def _build_data_from_config(cfg):
    if cfg.data.kind == "synthetic":
        from mme.data.synthetic import SyntheticShapesDataset

        ds = SyntheticShapesDataset(
            samples_per_class=cfg.data.samples_per_class,
            noise=cfg.data.noise,
        )
        n_val = max(1, int(len(ds) * cfg.data.val_fraction))
        return [ds[i] for i in range(len(ds) - n_val)], [
            ds[i] for i in range(len(ds) - n_val, len(ds))
        ]
    if cfg.data.kind == "directory":
        from mme.data.mesh_dataset import MeshDataset

        ds = MeshDataset(cfg.data.root)
        n_val = max(1, int(len(ds) * cfg.data.val_fraction))
        return [ds[i] for i in range(len(ds) - n_val)], [
            ds[i] for i in range(len(ds) - n_val, len(ds))
        ]
    raise ValueError(f"unknown data kind {cfg.data.kind!r}")


def _build_loss_from_config(cfg):
    from mme.losses import (
        cosine_schedule,
        diversity_loss,
        DynamicBalancedLoss,
        linear_schedule,
        similarity_loss,
        step_schedule,
        task_ce_loss,
    )

    lc = cfg.loss
    if lc.schedule == "linear":
        schedule = linear_schedule(
            lc.alpha_start, lc.alpha_end, lc.beta_start, lc.beta_end
        )
    elif lc.schedule == "cosine":
        schedule = cosine_schedule(
            lc.alpha_start, lc.alpha_end, lc.beta_start, lc.beta_end
        )
    elif lc.schedule == "step":
        pts = [
            (int(x[0]), float(x[1]), float(x[2])) for x in (lc.step_breakpoints or [])
        ]
        schedule = step_schedule(pts)
    else:
        raise ValueError(f"unknown schedule {lc.schedule!r}")
    return DynamicBalancedLoss(task_ce_loss, diversity_loss, similarity_loss, schedule)


def _cmd_list_experts(args: argparse.Namespace) -> int:
    _import_modules(args.import_module or [])
    from mme.experts.registry import list_experts

    names = list_experts()
    if not names:
        print(
            "(no experts registered — try: mme list-experts --import-module examples.toy_expert)"
        )
        return 0
    for n in names:
        print(n)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from mme.config import load_config
    from mme.core.moe import MMEModel
    from mme.training.trainer import Trainer

    cfg = load_config(args.config)
    _import_modules(cfg.register_modules)
    experts = _build_experts_from_config(cfg)
    gate = _build_gate_from_config(cfg)
    model = MMEModel(experts=experts, gate=gate)
    train_data, val_data = _build_data_from_config(cfg)
    loss_fn = _build_loss_from_config(cfg)
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_data=train_data,
        val_data=val_data,
        batch_size=cfg.train.batch_size,
        epochs=cfg.train.epochs,
        lr=cfg.train.lr,
        device=cfg.train.device,
        ckpt_dir=cfg.train.ckpt_dir,
    )
    trainer.fit()
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from mme.config import load_config
    from mme.core.moe import MMEModel
    from mme.eval.classify import evaluate_classification

    cfg = load_config(args.config)
    _import_modules(cfg.register_modules)
    experts = _build_experts_from_config(cfg)
    gate = _build_gate_from_config(cfg)
    model = MMEModel(experts=experts, gate=gate)
    if args.ckpt:
        import torch

        state = torch.load(args.ckpt, map_location="cpu")
        for i, e in enumerate(model.experts):
            key = f"expert_{i}"
            if key in state and hasattr(e, "load_state_dict"):
                e.load_state_dict(state[key])
        if "gate" in state and hasattr(model.gate, "load_state_dict"):
            model.gate.load_state_dict(state["gate"])
    _, val_data = _build_data_from_config(cfg)
    metrics = evaluate_classification(model, val_data, batch_size=cfg.train.batch_size)
    for k, v in metrics.items():
        print(f"{k}: {v}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mme", description="MME command line interface"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list-experts", help="list registered experts")
    p_list.add_argument(
        "--import-module", action="append", help="python module to import first"
    )
    p_list.set_defaults(func=_cmd_list_experts)

    p_train = sub.add_parser("train", help="train from a YAML config")
    p_train.add_argument("--config", required=True)
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("eval", help="evaluate from a YAML config + checkpoint")
    p_eval.add_argument("--config", required=True)
    p_eval.add_argument("--ckpt", required=False)
    p_eval.set_defaults(func=_cmd_eval)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
