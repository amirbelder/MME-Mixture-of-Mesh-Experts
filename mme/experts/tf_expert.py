"""TensorFlow expert base class.

TF experts are forward-only from the MoE's perspective — gradients do NOT
flow across the framework boundary (outputs are pulled to numpy and rewrapped
as torch tensors). If you want to *train* a TF expert, do so in its own TF
optimizer step outside the MoE loop, then let the gate weight its frozen
outputs. See ``docs/mixing_frameworks.md``.
"""

from __future__ import annotations

from mme.experts.base import Expert


class TFExpert(Expert):
    framework = "tf"

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes=num_classes)

    def trainable_variables(self):
        # Fall back for pure-python experts.
        return []
