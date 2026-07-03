"""Expert base class + standard output container.

Every expert — torch or tf — subclasses ``Expert`` and implements
``preprocess`` and ``forward``. See ``TorchExpert`` and ``TFExpert`` for the
framework-specific mixins that also expose ``parameters()`` /
``trainable_variables`` and ``to(device)`` used by the trainer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Optional

from mme.core.mesh import Mesh


@dataclass
class ExpertOutput:
    """What every expert returns from ``forward``.

    Attributes:
        logits: (num_classes,) or (num_faces, num_classes) tensor of logits.
        features: Optional (D,) or (num_faces, D) feature vector consumed by the gate.
        per_element_attention: Optional (num_faces,) or (num_vertices,) attention;
            the gate may use it if the expert produces one.
    """

    logits: Any
    features: Optional[Any] = None
    per_element_attention: Optional[Any] = None


class Expert(ABC):
    """Abstract base class every plugin must extend (or duck-type)."""

    #: Unique registered name. Set by ``@register_expert("...")`` or by the subclass.
    name: str = ""
    #: "torch" or "tf".
    framework: Literal["torch", "tf"] = "torch"
    #: Number of output classes.
    num_classes: int = 0

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    @abstractmethod
    def preprocess(self, mesh: Mesh) -> Any:
        """Convert a ``Mesh`` to the tensor(s) this expert consumes."""

    @abstractmethod
    def forward(self, inputs: Any) -> ExpertOutput:
        """Run the expert on preprocessed inputs and return an ``ExpertOutput``."""

    # Framework-specific hooks (``to``, ``train``, ``eval``, ``parameters``,
    # ``state_dict``) are provided by the mixed-in framework class (e.g.
    # ``torch.nn.Module``). We intentionally do NOT define no-op versions
    # here because they would shadow the real implementations under
    # Python's MRO when subclasses inherit ``(TorchExpert, nn.Module)``.
    # Trainer / MMEModel call these hooks defensively via ``hasattr``.
