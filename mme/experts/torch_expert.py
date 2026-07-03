"""Torch expert base class."""

from __future__ import annotations

from mme.experts.base import Expert


class TorchExpert(Expert):
    """Expert wrapping a ``torch.nn.Module``.

    Subclasses that also mix in ``torch.nn.Module`` inherit
    ``parameters()``, ``to()``, ``train()``, and ``state_dict()`` from it —
    we deliberately do NOT override those here (previous versions did,
    which shadowed the ``nn.Module`` implementations under Python's MRO and
    caused the optimizer to see an empty parameter list).

    Pure-Python experts without ``nn.Module`` should override ``parameters``
    themselves if the trainer needs to see their tunable state.
    """

    framework = "torch"

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes=num_classes)
