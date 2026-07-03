"""YAML → dataclass config loader for the CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ExpertConfig:
    name: str  # registered expert name
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GateConfig:
    kind: str = "mme_torch"  # or "mme_tf"
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    kind: str = "synthetic"  # "synthetic" | "directory"
    root: Optional[str] = None
    samples_per_class: int = 32
    noise: float = 0.05
    val_fraction: float = 0.2


@dataclass
class LossConfig:
    schedule: str = "linear"  # "linear" | "cosine" | "step"
    alpha_start: float = 0.0
    alpha_end: float = 0.1
    beta_start: float = 0.0
    beta_end: float = 0.1
    step_breakpoints: Optional[List[List[float]]] = None


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 8
    lr: float = 1e-3
    device: str = "auto"
    ckpt_dir: Optional[str] = None


@dataclass
class Config:
    num_classes: int
    experts: List[ExpertConfig]
    gate: GateConfig = field(default_factory=GateConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    register_modules: List[str] = field(default_factory=list)


def load_config(path) -> Config:
    with Path(path).open() as fh:
        raw = yaml.safe_load(fh) or {}
    experts = [ExpertConfig(**e) for e in raw.get("experts", [])]
    gate = GateConfig(**raw.get("gate", {}))
    data = DataConfig(**raw.get("data", {}))
    loss = LossConfig(**raw.get("loss", {}))
    train = TrainConfig(**raw.get("train", {}))
    return Config(
        num_classes=int(raw["num_classes"]),
        experts=experts,
        gate=gate,
        data=data,
        loss=loss,
        train=train,
        register_modules=raw.get("register_modules", []),
    )
