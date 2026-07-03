"""Expert plugin system."""

from mme.experts.base import Expert, ExpertOutput
from mme.experts.prerendered import PrerenderedExpert
from mme.experts.registry import get_expert, list_experts, register_expert
from mme.experts.tf_expert import TFExpert
from mme.experts.torch_expert import TorchExpert

__all__ = [
    "Expert",
    "ExpertOutput",
    "PrerenderedExpert",
    "TorchExpert",
    "TFExpert",
    "register_expert",
    "get_expert",
    "list_experts",
]

# AttWalkExpert is TF-only — import lazily via `from mme.experts.attwalk_tf
# import AttWalkExpert`. We deliberately do NOT import it at package load so
# users without TensorFlow installed don't hit an ImportError.
