"""
config.py — every knob of our mini GPT lives in ONE place.

Teaching note:
    A real GPT is the *same* code as this, just with bigger numbers.
    GPT-2 small = 12 layers, 12 heads, 768 embedding, 1024 context, 50257 vocab.
    Ours       =  4 layers,  4 heads, 256 embedding,  128 context,  8000 vocab.
"""

from dataclasses import dataclass, asdict
import json
import os


@dataclass
class GPTConfig:
    # ---- data ----
    data_dir: str = "data"          # folder with the .txt transcripts
    vocab_size: int = 8000          # how many distinct words the model knows
    max_train_tokens: int = 400_000  # "minimal data": stop streaming after this many tokens

    # ---- model shape ----
    block_size: int = 128           # context window: how many past tokens it can see
    n_layer: int = 4                # number of transformer blocks stacked
    n_head: int = 4                 # attention heads per block
    n_embd: int = 256               # embedding / hidden size  (must divide by n_head)
    dropout: float = 0.1

    # ---- training ----
    batch_size: int = 32
    steps_per_epoch: int = 200
    epochs: int = 15
    learning_rate: float = 3e-4
    shuffle_buffer: int = 2000
    stride: int = 64                # how far the sliding window moves between samples

    # ---- where things are saved ----
    out_dir: str = "checkpoints"

    # -------------------------------------------------------------- helpers
    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GPTConfig":
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))
