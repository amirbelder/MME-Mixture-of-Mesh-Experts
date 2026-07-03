# Writing an expert

Every expert implements a small ABC. Two options — pick one:

## Option 1: PyTorch expert

Mix `TorchExpert` **and** `torch.nn.Module` so `parameters()`, `to()`, and `state_dict()` work.

```python
import torch, torch.nn as nn

from mme.experts.base import ExpertOutput
from mme.experts.registry import register_expert
from mme.experts.torch_expert import TorchExpert

@register_expert("my_expert")
class MyExpert(TorchExpert, nn.Module):
    def __init__(self, num_classes: int, hidden: int = 64):
        nn.Module.__init__(self)
        TorchExpert.__init__(self, num_classes=num_classes)
        self.net = nn.Sequential(nn.Linear(32, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, num_classes)
        self.feature_dim = hidden

    def preprocess(self, mesh):
        return torch.from_numpy(mesh.sampled_vertex_features(dim=32)).float()

    def forward(self, x):
        feats = self.net(x)
        logits = self.head(feats)
        return ExpertOutput(logits=logits, features=feats)
```

## Option 2: TensorFlow expert

Mix `TFExpert` with a Keras model. Remember: gradients do **not** cross the framework boundary inside the MoE — see [`mixing_frameworks.md`](mixing_frameworks.md).

```python
import tensorflow as tf
from mme.experts.base import ExpertOutput
from mme.experts.registry import register_expert
from mme.experts.tf_expert import TFExpert

@register_expert("tf_expert")
class TFMy(TFExpert):
    def __init__(self, num_classes: int):
        super().__init__(num_classes=num_classes)
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(32,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_classes),
        ])

    def preprocess(self, mesh):
        return tf.convert_to_tensor(mesh.sampled_vertex_features(dim=32)[None])

    def forward(self, x):
        logits = self.model(x)
        return ExpertOutput(logits=tf.squeeze(logits, axis=0))

    def trainable_variables(self):
        return self.model.trainable_variables
```

## Distributing your expert as a plugin

In your package's `pyproject.toml`:

```toml
[project.entry-points."mme.experts"]
my_expert = "my_package.experts:MyExpert"
```

Any environment with both `mme` and `my_package` installed will see `my_expert` in `mme list-experts`.

## ExpertOutput

```python
ExpertOutput(
    logits,                          # required — (C,) or (N, C)
    features=None,                   # optional — used by the gate
    per_element_attention=None,      # optional — used by segmentation gate variants
)
```
