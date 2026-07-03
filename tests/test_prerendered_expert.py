"""Tests for PrerenderedExpert."""

from tests.conftest import needs_torch


@needs_torch
def test_prerendered_expert_looks_up_by_source_path(tmp_path):
    import torch
    from mme.core.mesh import Mesh
    from mme.experts.prerendered import PrerenderedExpert

    # Dump two fake mesh outputs.
    dump = {
        "/data/mesh_a.obj": {
            "logits": torch.tensor([0.1, 2.0, 0.3, 0.4]),
            "features": torch.tensor([1.0, 2.0, 3.0]),
        },
        "/data/mesh_b.obj": {
            "logits": torch.tensor([3.0, 0.1, 0.1, 0.1]),
        },
    }
    dump_path = tmp_path / "expert_a.pt"
    torch.save(dump, dump_path)

    expert = PrerenderedExpert(
        name="baseline_a", num_classes=4, dump_path=str(dump_path)
    )

    # Zero trainable params.
    assert expert.parameters() == []

    # Lookup by source_path.
    import numpy as np

    m = Mesh(
        vertices=np.zeros((3, 3), dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        source_path="/data/mesh_a.obj",
    )
    out = expert.forward(expert.preprocess(m))
    assert torch.allclose(out.logits, dump["/data/mesh_a.obj"]["logits"])
    assert torch.allclose(out.features, dump["/data/mesh_a.obj"]["features"])

    # Missing features -> None.
    m2 = Mesh(
        vertices=np.zeros((3, 3), dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        source_path="/data/mesh_b.obj",
    )
    out2 = expert.forward(expert.preprocess(m2))
    assert out2.features is None

    # Missing key raises.
    m3 = Mesh(
        vertices=np.zeros((3, 3), dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        source_path="/data/mesh_c_missing.obj",
    )
    import pytest

    with pytest.raises(KeyError):
        expert.forward(expert.preprocess(m3))


@needs_torch
def test_prerendered_expert_wrong_num_classes_raises(tmp_path):
    import numpy as np
    import pytest
    import torch
    from mme.core.mesh import Mesh
    from mme.experts.prerendered import PrerenderedExpert

    dump = {"/x.obj": {"logits": torch.zeros(5)}}
    dump_path = tmp_path / "e.pt"
    torch.save(dump, dump_path)

    expert = PrerenderedExpert(name="e", num_classes=4, dump_path=str(dump_path))
    m = Mesh(
        vertices=np.zeros((3, 3), dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        source_path="/x.obj",
    )
    with pytest.raises(ValueError):
        expert.forward(expert.preprocess(m))
