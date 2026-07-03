import mme.experts.base as expert_base
import pytest
from mme.experts.registry import (
    _reset_registry_for_tests,
    get_expert,
    list_experts,
    register_expert,
)


def setup_function(_):
    _reset_registry_for_tests()


def test_register_and_get():
    @register_expert("dummy")
    class Dummy(expert_base.Expert):
        framework = "torch"

        def __init__(self, num_classes: int) -> None:
            super().__init__(num_classes=num_classes)

        def preprocess(self, mesh):
            return None

        def forward(self, x):
            return expert_base.ExpertOutput(logits=[0.0])

    assert "dummy" in list_experts()
    inst = get_expert("dummy", num_classes=3)
    assert isinstance(inst, Dummy)
    assert inst.num_classes == 3


def test_duplicate_registration_raises():
    @register_expert("dup")
    class A(expert_base.Expert):
        def preprocess(self, m):
            return None

        def forward(self, x):
            return expert_base.ExpertOutput(logits=[0.0])

    with pytest.raises(ValueError):

        @register_expert("dup")
        class B(expert_base.Expert):
            def preprocess(self, m):
                return None

            def forward(self, x):
                return expert_base.ExpertOutput(logits=[0.0])


def test_unknown_expert_raises():
    with pytest.raises(KeyError):
        get_expert("nope")
