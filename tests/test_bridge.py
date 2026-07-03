import numpy as np
from tests.conftest import needs_tf, needs_torch


@needs_torch
def test_bridge_torch_numpy_roundtrip():
    import torch
    from mme.experts.bridge import to_numpy, to_torch

    x = torch.randn(4, 5)
    y = to_torch(to_numpy(x))
    assert y.shape == x.shape
    np.testing.assert_allclose(y.numpy(), x.numpy(), atol=1e-6)


@needs_torch
@needs_tf
def test_bridge_torch_tf_torch_roundtrip():
    import tensorflow as tf
    import torch
    from mme.experts.bridge import to_numpy, to_tf, to_torch

    x = torch.randn(2, 3)
    y = to_tf(x)
    assert isinstance(y, tf.Tensor)
    z = to_torch(to_numpy(y))
    np.testing.assert_allclose(z.numpy(), x.numpy(), atol=1e-5)
