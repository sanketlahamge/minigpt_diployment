"""
model.py — the GPT itself, written from scratch.

Everything below is built out of Dense layers, matrix multiplies and softmax.
No pre-made attention layer is used, so every line is inspectable in class.

The architecture, top to bottom:

    token ids ──▶ token embedding  +  position embedding
                          │
                          ▼
                 ┌──────────────────┐
                 │ Transformer block│  × n_layer
                 │  ├ self-attention│   "look at earlier words"
                 │  └ feed-forward  │   "think about what you saw"
                 └──────────────────┘
                          │
                          ▼
                   LayerNorm ──▶ Dense(vocab_size) ──▶ logits
                                  (a score for every possible next word)
"""

import tensorflow as tf
# Import Keras directly, NOT via tensorflow.keras. The retraining project
# imports `transformers`, which redirects tensorflow.keras to the legacy
# Keras 2 package - and mixing Keras 2 and Keras 3 classes in one process
# breaks. Importing `keras` always gives Keras 3, so compare.py can load
# this model and a Hugging Face model side by side.
import keras
from keras import layers

from .config import GPTConfig


class CausalSelfAttention(layers.Layer):
    """
    The heart of GPT.

    Each token creates three vectors:
        Q (query) "what am I looking for?"
        K (key)   "what do I offer?"
        V (value) "what do I pass along?"

    Token i scores every token j by Q_i . K_j, turns those scores into
    percentages with softmax, and mixes the V vectors accordingly.

    CAUSAL = a token may only look BACKWARDS. We enforce that with a mask,
    otherwise the model could cheat by peeking at the answer.
    """

    def __init__(self, cfg: GPTConfig, **kwargs):
        super().__init__(**kwargs)
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim

        # One Dense produces Q, K and V together (cheaper than three layers).
        self.qkv = layers.Dense(3 * cfg.n_embd, use_bias=False, name="qkv")
        self.proj = layers.Dense(cfg.n_embd, name="proj")
        self.attn_dropout = layers.Dropout(cfg.dropout)
        self.resid_dropout = layers.Dropout(cfg.dropout)

    def _split_heads(self, x, B, T):
        """(B, T, n_embd) -> (B, n_head, T, head_dim) so heads work in parallel."""
        x = tf.reshape(x, (B, T, self.n_head, self.head_dim))
        return tf.transpose(x, (0, 2, 1, 3))

    def call(self, x, training=False):
        B, T = tf.shape(x)[0], tf.shape(x)[1]

        q, k, v = tf.split(self.qkv(x), 3, axis=-1)          # each (B, T, C)
        q = self._split_heads(q, B, T)                       # (B, h, T, hd)
        k = self._split_heads(k, B, T)
        v = self._split_heads(v, B, T)

        # Attention scores: how much should token i care about token j?
        scale = tf.math.sqrt(tf.cast(self.head_dim, x.dtype))
        att = tf.matmul(q, k, transpose_b=True) / scale      # (B, h, T, T)

        # Causal mask: lower-triangular ones. Future positions get -1e9,
        # which softmax turns into ~0 probability.
        mask = tf.linalg.band_part(tf.ones((T, T), dtype=att.dtype), -1, 0)
        att = att - 1e9 * (1.0 - mask)

        att = tf.nn.softmax(att, axis=-1)                    # rows sum to 1
        att = self.attn_dropout(att, training=training)

        y = tf.matmul(att, v)                                # (B, h, T, hd)
        y = tf.transpose(y, (0, 2, 1, 3))                    # (B, T, h, hd)
        y = tf.reshape(y, (B, T, self.n_head * self.head_dim))
        return self.resid_dropout(self.proj(y), training=training)


class TransformerBlock(layers.Layer):
    """
    Attention (mix information between tokens)
      + feed-forward (process each token on its own)
      + residual connections (x + f(x), so gradients flow through deep stacks)
    """

    def __init__(self, cfg: GPTConfig, **kwargs):
        super().__init__(**kwargs)
        self.ln1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp = keras.Sequential([
            layers.Dense(4 * cfg.n_embd, activation="gelu"),
            layers.Dense(cfg.n_embd),
            layers.Dropout(cfg.dropout),
        ], name="mlp")

    def call(self, x, training=False):
        x = x + self.attn(self.ln1(x), training=training)
        x = x + self.mlp(self.ln2(x), training=training)
        return x


class MiniGPT(keras.Model):
    def __init__(self, cfg: GPTConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        # WHAT the word is:
        self.token_emb = layers.Embedding(cfg.vocab_size, cfg.n_embd, name="token_emb")
        # WHERE the word sits (attention itself has no sense of order):
        self.pos_emb = layers.Embedding(cfg.block_size, cfg.n_embd, name="pos_emb")
        self.drop = layers.Dropout(cfg.dropout)
        self.blocks = [TransformerBlock(cfg, name=f"block_{i}") for i in range(cfg.n_layer)]
        self.ln_f = layers.LayerNormalization(epsilon=1e-5)
        self.head = layers.Dense(cfg.vocab_size, name="lm_head")

    def call(self, idx, training=False):
        T = tf.shape(idx)[1]
        x = self.token_emb(idx) + self.pos_emb(tf.range(T))[tf.newaxis, :, :]
        x = self.drop(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        return self.head(self.ln_f(x))        # (B, T, vocab_size) logits


def build_model(cfg: GPTConfig) -> MiniGPT:
    """Create the model and run one dummy batch so all weights exist."""
    model = MiniGPT(cfg)
    model(tf.zeros((1, cfg.block_size), dtype=tf.int32), training=False)
    return model
