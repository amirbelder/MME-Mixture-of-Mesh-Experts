"""Typing protocols for Expert and Gate.

These describe the shape of objects the MoE orchestrator uses. They are
purposely small: the concrete ABCs live in ``mme.experts.base`` and
``mme.gate.*``. Protocols are used for typing / duck-typing so third-party
implementations don't need to inherit from our ABCs.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ExpertProtocol(Protocol):
    name: str
    framework: str  # "torch" or "tf"
    num_classes: int

    def preprocess(self, mesh: Any) -> Any: ...

    def forward(self, inputs: Any) -> Any: ...


@runtime_checkable
class GateProtocol(Protocol):
    num_experts: int

    def forward(self, mesh: Any, expert_outputs: Any) -> Any: ...
