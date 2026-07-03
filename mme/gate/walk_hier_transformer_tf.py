"""Vendored TensorFlow classes from the reference project's attention_model.py.

Source: ~/mme_reference/attention_model/attention_model.py (Belder & Tal).

Preserves ALL layer definitions bit-for-bit so a .keras checkpoint from the
reference project loads unchanged. The only edits vs. the source file:

- Removed dependency on `EasyDict params` — every model class now takes
  explicit keyword arguments (num_classes, seq_len, net_input_dim,
  one_label_per_model, last_layer_activation).
- Removed dependency on `utils` and checkpoint / logdir logic from __init__
  (add save_weights / load_weights as thin wrappers if you need them).
- `tensorflow_addons.InstanceNormalization` falls back to
  `tf.keras.layers.LayerNormalization` with a warning if tfa is missing.

If you install tensorflow_addons (`pip install tensorflow-addons`) the
behavior is byte-identical to the reference. Without tfa you get a slight
architectural difference (LayerNorm instead of InstanceNorm) — do NOT expect
paper checkpoints to load in that mode.

All classes here are TF-native. Adapters that expose them via the MME
interface live in ``mme/gate/walk_hier_gate.py`` (gate) and
``mme/experts/attwalk_tf.py`` (expert).
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np


def _tf():
    """Import tensorflow lazily so `mme` still imports without TF installed."""
    import tensorflow as tf

    return tf


def _instance_norm(axis: int = -1):
    """Return tfa.layers.InstanceNormalization if available, else LayerNormalization."""
    tf = _tf()
    try:
        import tensorflow_addons as tfa

        return tfa.layers.InstanceNormalization(axis=axis)
    except ImportError:
        warnings.warn(
            "tensorflow_addons not installed; falling back to LayerNormalization. "
            "Paper checkpoints trained with InstanceNormalization will NOT load.",
            RuntimeWarning,
            stacklevel=2,
        )
        return tf.keras.layers.LayerNormalization(epsilon=1e-6)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def dense_layer(
    size,
    activation=None,
    use_bias=True,
    initializer=None,
    regulizer=None,
):
    tf = _tf()
    if initializer is None:
        initializer = tf.initializers.Orthogonal(1)
    if regulizer is None:
        regulizer = tf.keras.regularizers.l2(0.0001)
    return tf.keras.layers.Dense(
        size,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=initializer,
        kernel_regularizer=regulizer,
        bias_regularizer=regulizer,
    )


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    tf = _tf()
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model,
    )
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def coordinate_encoding(position, max_time_step, d_model):
    pos_encoding = positional_encoding(position, d_model)
    step_signal = positional_encoding(max_time_step, d_model)
    return pos_encoding, step_signal


def point_wise_feed_forward_network(d_model, dff):
    tf = _tf()
    return tf.keras.Sequential(
        [
            dense_layer(dff, activation="elu"),
            dense_layer(d_model),
        ]
    )


def scaled_dot_product_attention(q, k, v, mask):
    tf = _tf()
    qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention = qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention += mask * -1e9
    weights = tf.nn.softmax(scaled_attention, axis=-1)
    output = tf.matmul(weights, v)
    return output, weights


# ------------------------------------------------------------------
# Layer classes — instantiated lazily so this module imports without TF
# ------------------------------------------------------------------


def _build_classes():
    """Build all the Layer / Model classes. Called lazily on first instantiation."""
    tf = _tf()
    keras = tf.keras
    layers = tf.keras.layers

    class MultiHeadAttention(keras.layers.Layer):
        def __init__(self, num_neurons, num_heads):
            super().__init__()
            self.num_heads = num_heads
            self.num_neurons = num_neurons
            self.depth = num_neurons // self.num_heads
            self.attention_layer = scaled_dot_product_attention
            self.q_layer = dense_layer(num_neurons, regulizer=None)
            self.k_layer = dense_layer(num_neurons, regulizer=None)
            self.v_layer = dense_layer(num_neurons, regulizer=None)
            self.linear_layer = dense_layer(num_neurons)

        def split(self, x, batch_size):
            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])

        def call(self, v, k, q, mask):
            batch_size = tf.shape(q)[0]
            q = self.q_layer(q)
            k = self.k_layer(k)
            v = self.v_layer(v)
            q = self.split(q, batch_size)
            k = self.split(k, batch_size)
            v = self.split(v, batch_size)
            attention_output, weights = self.attention_layer(q, k, v, mask)
            output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
            concat_attention = tf.reshape(output, (batch_size, -1, self.num_neurons))
            output = self.linear_layer(concat_attention)
            return output, weights

    class LocalGlobalAttention(keras.layers.Layer):
        def __init__(self, num_neurons, num_heads, local_window_len):
            super().__init__()
            self.local_attention_layer = MultiHeadAttention(num_neurons, num_heads)
            self.global_attention_layer = MultiHeadAttention(num_neurons, num_heads)
            self.local_window_len = local_window_len

        def call(self, x, k, q, mask=None):
            xs = x.shape
            x_global_attn, global_weights = self.global_attention_layer(x, k, q, None)
            x = tf.reshape(x, (-1, self.local_window_len, xs[-1]))
            x_local_attn, local_weights = self.local_attention_layer(x, k, q, None)
            x_local_attn = tf.reshape(x_local_attn, (xs[0], -1, xs[-1]))
            return x_global_attn + x_local_attn, tf.stack(
                [global_weights, local_weights]
            )

    class EncoderLayer(tf.keras.layers.Layer):
        def __init__(self, d_model, num_heads, dff, rate=0.1, window_len=None):
            super().__init__()
            if window_len is None:
                self.mha = MultiHeadAttention(d_model, num_heads)
            else:
                self.mha = LocalGlobalAttention(d_model, num_heads, window_len)
            self.ffn = point_wise_feed_forward_network(d_model, dff)
            self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = tf.keras.layers.Dropout(rate)
            self.dropout2 = tf.keras.layers.Dropout(rate)
            self.gate = tf.keras.layers.Dense(
                d_model,
                kernel_initializer=tf.initializers.Identity(),
                use_bias=False,
                activation="sigmoid",
            )

        def call(self, x, training, mask):
            attn_output, attn_map = self.mha(x, x, x, mask)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(self.gate(x) * x + attn_output, training=training)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            out2 = self.layernorm2(out1 + ffn_output, training=training)
            return out2, attn_map

    class DecoderLayer(tf.keras.layers.Layer):
        def __init__(self, d_model, num_heads, dff, rate=0.1):
            super().__init__()
            self.mha1 = MultiHeadAttention(d_model, num_heads)
            self.mha2 = MultiHeadAttention(d_model, num_heads)
            self.ffn = point_wise_feed_forward_network(d_model, dff)
            self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-7)
            self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-7)
            self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-7)
            self.dropout1 = tf.keras.layers.Dropout(rate)
            self.dropout2 = tf.keras.layers.Dropout(rate)
            self.dropout3 = tf.keras.layers.Dropout(rate)

        def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
            attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
            attn1 = self.dropout1(attn1, training=training)
            out1 = self.layernorm1(attn1 + x)
            attn2, attn_weights_block2 = self.mha2(
                enc_output, enc_output, out1, padding_mask
            )
            attn2 = self.dropout2(attn2, training=training)
            out2 = self.layernorm2(attn2 + out1)
            ffn_output = self.ffn(out2)
            ffn_output = self.dropout3(ffn_output, training=training)
            out3 = self.layernorm3(ffn_output + out2)
            return out3, attn_weights_block1, attn_weights_block2

    class FC_embedder(tf.keras.layers.Layer):
        def __init__(self, fc1_size, fc2_size):
            super().__init__()
            k_reg = tf.keras.regularizers.l2(0.0001)
            initializer = tf.initializers.Orthogonal(1)
            self.fc1 = layers.Dense(
                fc1_size,
                kernel_regularizer=k_reg,
                bias_regularizer=k_reg,
                kernel_initializer=initializer,
            )
            self.norm1 = _instance_norm(axis=-1)
            self.fc2 = layers.Dense(
                fc2_size,
                kernel_regularizer=k_reg,
                bias_regularizer=k_reg,
                kernel_initializer=initializer,
            )
            self.norm2 = _instance_norm(axis=-1)

        def call(self, x, training):
            x = self.fc1(x)
            x = self.norm1(x, training=training)
            x = tf.nn.relu(x)
            x = self.fc2(x)
            x = self.norm2(x, training=training)
            x = tf.nn.relu(x)
            return x

    class Encoder(tf.keras.layers.Layer):
        def __init__(
            self,
            num_layers,
            d_model,
            num_heads,
            dff,
            input_vocab_size,
            maximum_position_encoding,
            rate=0.1,
        ):
            super().__init__()
            self.d_model = d_model
            self.num_layers = num_layers
            self.embedding = FC_embedder(d_model // 2, d_model)
            self.pos_encoding = positional_encoding(
                maximum_position_encoding, self.d_model
            )
            self.enc_layers = [
                EncoderLayer(d_model, num_heads, dff, rate, maximum_position_encoding)
                for _ in range(num_layers)
            ]
            self.dropout = tf.keras.layers.Dropout(rate)

        def call(self, x, training, mask):
            seq_len = tf.shape(x)[1]
            x = self.embedding(x, training=training)
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            x += self.pos_encoding[:, :seq_len, :]
            x = self.dropout(x, training=training)
            for i in range(self.num_layers - 1):
                x, attn_map = self.enc_layers[i](x, training, mask)
                x += self.pos_encoding[:, :seq_len, :]
            x, attn_map = self.enc_layers[-1](x, training, mask)
            return x

    class RecurrentEncoder(tf.keras.layers.Layer):
        def __init__(
            self,
            num_timesteps,
            d_model,
            num_heads,
            dff,
            input_vocab_size,
            maximum_position_encoding,
            rate=0.0,
        ):
            super().__init__()
            self.d_model = d_model
            self.n_ts = num_timesteps
            self.embedding = FC_embedder(d_model // 2, d_model)
            self.pos_encoding, self.time_encoding = coordinate_encoding(
                maximum_position_encoding, num_timesteps, self.d_model
            )
            self.recurrent_layer = EncoderLayer(d_model, num_heads, dff, rate)
            self.dropout = tf.keras.layers.Dropout(rate)

        def call(self, x, training, mask):
            seq_len = tf.shape(x)[1]
            x = self.embedding(x, training=training)
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            for i in range(self.n_ts - 1):
                x += self.pos_encoding[:, :seq_len, :]
                x += tf.expand_dims(self.time_encoding[:, i, :], axis=1)
                x = self.dropout(x, training=training)
                x, attn_map = self.recurrent_layer(x, training, mask)
            return x

    class Decoder(tf.keras.layers.Layer):
        def __init__(
            self,
            num_layers,
            d_model,
            num_heads,
            dff,
            target_vocab_size,
            K,
            maximum_position_encoding,
            rate=0.1,
            pooling=False,
        ):
            super().__init__()
            self.pooling = pooling
            self.d_model = d_model
            self.num_layers = num_layers
            self.K = K
            self.embedding = FC_embedder(d_model // 2, d_model)
            self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
            self.dec_layers = [
                DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
            ]
            self.dropout = tf.keras.layers.Dropout(rate)

        def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
            attention_weights = {}
            x = self.embedding(x)
            if self.pooling:
                x = tf.reshape(
                    tf.reduce_mean(tf.reshape(x, (-1, self.K, x.shape[-1])), axis=1),
                    (-1, x.shape[-2] // self.K, x.shape[-1]),
                )
            seq_len = tf.shape(x)[1]
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            x += self.pos_encoding[:, :seq_len, :]
            x = self.dropout(x, training=training)
            for i in range(self.num_layers):
                x, block1, block2 = self.dec_layers[i](
                    x, enc_output, training, look_ahead_mask, padding_mask
                )
                attention_weights[f"decoder_layer{i + 1}_block1"] = block1
                attention_weights[f"decoder_layer{i + 1}_block2"] = block2
            return x, attention_weights

    class WalkTransformer(tf.keras.Model):
        """AttWalk classifier — attention-based walker (line 605 in the source).

        Args (previously bundled in ``params``, now explicit):
            num_classes, net_input_dim, last_layer_activation, one_label_per_model.
        """

        def __init__(
            self,
            num_layers,
            d_model,
            num_heads,
            dff,
            input_vocab_size,
            out_features,
            pe_input,
            pe_target,
            *,
            num_classes: int,
            net_input_dim: int,
            last_layer_activation: Optional[str] = None,
            one_label_per_model: bool = True,
            rate: float = 0.25,
            num_scales: Optional[int] = None,
        ):
            super().__init__()
            self.num_classes = num_classes
            self.net_input_dim = net_input_dim
            self.last_layer_activation = last_layer_activation
            self.one_label_per_model = one_label_per_model

            self.encoder = Encoder(
                num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate
            )
            self.final_layer = tf.keras.layers.Dense(
                out_features,
                activation=last_layer_activation,
                kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                bias_regularizer=tf.keras.regularizers.l2(0.0001),
                kernel_initializer=tf.initializers.Orthogonal(3),
            )
            self._fc_last = layers.Dense(
                num_classes,
                activation=last_layer_activation,
                kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                bias_regularizer=tf.keras.regularizers.l2(0.0001),
                kernel_initializer=tf.initializers.Orthogonal(3),
            )
            # Build variables at construction time using a dummy input so
            # weights exist before load_weights (matches the reference file).
            s_in = (200, self.net_input_dim)
            build_s_in = (
                8,
                4,
            ) + s_in
            _ = tf.keras.layers.Input(shape=s_in)
            self.build(input_shape=build_s_in)

        def call(
            self, inp, enc_padding_mask=None, training=True, both=False, classify=True
        ):
            xs = inp.shape
            inp = tf.reshape(inp, (-1, xs[-2], xs[-1]))
            enc_output = self.encoder(inp, training, enc_padding_mask)
            final_features = self.final_layer(enc_output)
            final_output = self._fc_last(final_features)
            if both:
                return final_output, final_output[:, -1, :]
            elif classify:
                return tf.reduce_mean(final_output, axis=1)
            else:
                return final_features

    class WalkHierTransformer(tf.keras.Model):
        """Hierarchical walk transformer — the MME gate (line 696 in source).

        Args (previously bundled in ``params``, now explicit):
            num_classes, net_input_dim, seq_len, last_layer_activation,
            one_label_per_model.
        """

        def __init__(
            self,
            num_layers,
            d_model,
            num_heads,
            dff,
            input_vocab_size,
            out_features,
            pe_input,
            pe_target,
            *,
            num_classes: int,
            net_input_dim: int,
            seq_len: int,
            last_layer_activation: Optional[str] = None,
            one_label_per_model: bool = True,
            rate: float = 0.0,
            jump_every_k: int = 10,
            pooling: bool = False,
            concat_xyz: bool = False,
            num_scales: Optional[int] = None,
            global_dim_mult: int = 1,
            recurrent: bool = False,
        ):
            super().__init__()
            self.num_classes = num_classes
            self.net_input_dim = net_input_dim
            self.seq_len = seq_len
            self.last_layer_activation = last_layer_activation
            self.one_label_per_model = one_label_per_model
            self.dropout = tf.keras.layers.Dropout(rate)
            self.K = jump_every_k
            self.pooling = pooling if one_label_per_model else False
            self.concat_xyz = concat_xyz

            if recurrent:
                self.local_encoder = RecurrentEncoder(
                    num_layers,
                    d_model,
                    num_heads,
                    dff,
                    input_vocab_size,
                    jump_every_k,
                    rate,
                )
            else:
                self.local_encoder = Encoder(
                    num_layers,
                    d_model,
                    num_heads,
                    dff,
                    input_vocab_size,
                    jump_every_k,
                    rate,
                )
            self.decoder = Decoder(
                num_layers,
                d_model,
                num_heads,
                dff,
                input_vocab_size,
                jump_every_k,
                self.seq_len + 80,
                rate,
                self.pooling,
            )
            self._fc_last = layers.Dense(
                num_classes,
                activation=last_layer_activation,
                kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                bias_regularizer=tf.keras.regularizers.l2(0.0001),
                kernel_initializer=tf.initializers.Orthogonal(3),
            )

        def single_hier_block(self, block_layer, x, k, training):
            xs = x.shape
            x = tf.reshape(x, (-1, k, xs[-1]))
            enc_output = block_layer(x, training, None)
            if self.pooling:
                enc_output = tf.reduce_mean(enc_output, axis=1)
                rs_size = xs[-2] // k
            else:
                rs_size = xs[-2]
            return tf.reshape(enc_output, (-1, rs_size, enc_output.shape[-1]))

        def call(
            self, inp, enc_padding_mask=None, training=True, both=False, classify=True
        ):
            enc_output = self.single_hier_block(
                self.local_encoder, inp, self.K, training
            )
            dec_output, attention_weights = self.decoder(
                inp, enc_output, training, None, None
            )
            final_output = self._fc_last(dec_output)
            if self.one_label_per_model:
                classification = tf.reduce_mean(final_output, axis=1)
            else:
                classification = final_output
            if classify == "both":
                return classification, final_output
            elif classify:
                return classification
            else:
                return (
                    attention_weights,
                    final_output,
                    tf.reduce_mean(final_output, axis=1),
                )

    return {
        "MultiHeadAttention": MultiHeadAttention,
        "LocalGlobalAttention": LocalGlobalAttention,
        "EncoderLayer": EncoderLayer,
        "DecoderLayer": DecoderLayer,
        "FC_embedder": FC_embedder,
        "Encoder": Encoder,
        "RecurrentEncoder": RecurrentEncoder,
        "Decoder": Decoder,
        "WalkTransformer": WalkTransformer,
        "WalkHierTransformer": WalkHierTransformer,
    }


# ------------------------------------------------------------------
# Public accessors — call these to get the built classes lazily.
# ------------------------------------------------------------------


_CACHE = None


def _get(name: str):
    global _CACHE
    if _CACHE is None:
        _CACHE = _build_classes()
    return _CACHE[name]


def MultiHeadAttention(*args, **kwargs):
    return _get("MultiHeadAttention")(*args, **kwargs)


def LocalGlobalAttention(*args, **kwargs):
    return _get("LocalGlobalAttention")(*args, **kwargs)


def EncoderLayer(*args, **kwargs):
    return _get("EncoderLayer")(*args, **kwargs)


def DecoderLayer(*args, **kwargs):
    return _get("DecoderLayer")(*args, **kwargs)


def FC_embedder(*args, **kwargs):
    return _get("FC_embedder")(*args, **kwargs)


def Encoder(*args, **kwargs):
    return _get("Encoder")(*args, **kwargs)


def RecurrentEncoder(*args, **kwargs):
    return _get("RecurrentEncoder")(*args, **kwargs)


def Decoder(*args, **kwargs):
    return _get("Decoder")(*args, **kwargs)


def WalkTransformer(*args, **kwargs):
    return _get("WalkTransformer")(*args, **kwargs)


def WalkHierTransformer(*args, **kwargs):
    return _get("WalkHierTransformer")(*args, **kwargs)
