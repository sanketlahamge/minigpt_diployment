"""
train.py — train the mini GPT on the transcripts in data/.

    python train.py                      # sensible defaults
    python train.py --epochs 30          # train longer = better text
    python train.py --max-train-tokens 1000000 --n-layer 6

What happens:
    1. build the vocabulary by streaming the data once
    2. build a streaming tf.data pipeline (nothing loaded into RAM)
    3. build the GPT and train it to predict the next word
    4. after every epoch, print a sample so the class can watch it learn
    5. save weights + config + tokenizer into checkpoints/
"""
import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)

import argparse
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TF's startup noise

import tensorflow as tf
import keras            # Keras 3 directly — see the note in minigpt/model.py

from minigpt.config import GPTConfig
from minigpt.dataset import make_dataset, stream_text
from minigpt.generate import generate
from minigpt.model import build_model
from minigpt.tokenizer import WordTokenizer

SAMPLE_PROMPT = "so today we are going to"


class SampleCallback(keras.callbacks.Callback):
    """Print a generated sample after each epoch — the fun part of the demo."""

    def __init__(self, tok, prompt=SAMPLE_PROMPT):
        super().__init__()
        self.tok = tok
        self.prompt = prompt

    def on_epoch_end(self, epoch, logs=None):
        text = generate(self.model, self.tok, self.prompt,
                        max_new_tokens=40, temperature=0.8, top_k=40)
        print(f"\n  sample> {self.prompt} \033[36m{text}\033[0m\n")


def parse_args() -> GPTConfig:
    cfg = GPTConfig()
    p = argparse.ArgumentParser(description="Train a mini GPT from scratch.")
    # Every config field becomes a --flag automatically.
    for name, value in vars(cfg).items():
        p.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    args = p.parse_args()
    return GPTConfig(**vars(args))


def main():
    cfg = parse_args()
    os.makedirs(cfg.out_dir, exist_ok=True)
    tok_path = os.path.join(cfg.out_dir, "tokenizer.json")
    cfg_path = os.path.join(cfg.out_dir, "config.json")
    w_path = os.path.join(cfg.out_dir, "minigpt.weights.h5")

    # ---------------------------------------------------------- 1. vocabulary
    print("=" * 64)
    print("STEP 1/4  building the vocabulary (streaming through the data)")
    # ~6 chars per word, so this reads roughly max_train_tokens words.
    tok = WordTokenizer.build(
        stream_text(cfg.data_dir, max_chars=cfg.max_train_tokens * 6),
        cfg.vocab_size,
    )
    tok.save(tok_path)
    cfg.vocab_size = tok.vocab_size          # in case the data had fewer words
    cfg.save(cfg_path)
    print(f"          vocab_size = {cfg.vocab_size}  ->  {tok_path}")

    # ------------------------------------------------------------- 2. dataset
    print("\nSTEP 2/4  building the streaming dataset")
    ds = make_dataset(cfg, tok)
    x, y = next(iter(ds.take(1)))
    print(f"          one batch: x={tuple(x.shape)}  y={tuple(y.shape)}")
    print(f"          x[0] decoded: {tok.decode(x[0].numpy()[:12])} ...")
    print(f"          y[0] decoded: {tok.decode(y[0].numpy()[:12])} ...   <- shifted by one!")

    # --------------------------------------------------------------- 3. model
    print("\nSTEP 3/4  building the model")
    model = build_model(cfg)
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    n_params = sum(int(tf.size(w)) for w in model.trainable_weights)
    print(f"          {cfg.n_layer} layers, {cfg.n_head} heads, {cfg.n_embd} dim, "
          f"context {cfg.block_size}")
    print(f"          {n_params:,} trainable parameters "
          f"(GPT-3 had 175,000,000,000)")

    # ------------------------------------------------------------ 4. training
    print("\nSTEP 4/4  training\n")
    t0 = time.time()
    model.fit(
        ds,
        steps_per_epoch=cfg.steps_per_epoch,
        epochs=cfg.epochs,
        callbacks=[
            SampleCallback(tok),
            keras.callbacks.ModelCheckpoint(w_path, save_weights_only=True),
        ],
    )
    model.save_weights(w_path)

    print("=" * 64)
    print(f"done in {time.time() - t0:.0f}s   weights -> {w_path}")
    print("now run:  python chat.py")


if __name__ == "__main__":
    main()
