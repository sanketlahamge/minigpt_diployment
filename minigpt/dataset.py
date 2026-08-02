import glob
import os
import re
from typing import Iterator, List

import tensorflow as tf

from .config import GPTConfig
from .tokenizer import WordTokenizer

# Lines we throw away from the .vtt style transcripts.
_TIMESTAMP = re.compile(r"^\d{2}:\d{2}:\d{2}[.,]\d+\s*-->")
_CUE_ID = re.compile(r"^\d+$")


def list_files(data_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {data_dir!r}")
    return files


def stream_lines(data_dir: str) -> Iterator[str]:
    """Yield cleaned lines of spoken text, one at a time, file after file."""
    for path in list_files(data_dir):
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("WEBVTT"):
                    continue
                if _TIMESTAMP.match(line) or _CUE_ID.match(line):
                    continue
                yield line


def stream_text(data_dir: str, max_chars: int | None = None) -> Iterator[str]:
    """Same as stream_lines but capped — used to build the vocabulary."""
    total = 0
    for line in stream_lines(data_dir):
        yield line + " "
        total += len(line) + 1
        if max_chars is not None and total >= max_chars:
            return


def stream_tokens(cfg: GPTConfig, tok: WordTokenizer) -> Iterator[int]:
    """Yield token ids forever-ish, stopping after cfg.max_train_tokens."""
    n = 0
    for line in stream_lines(cfg.data_dir):
        for tid in tok.encode(line):
            yield tid
            n += 1
            if n >= cfg.max_train_tokens:
                return


def stream_windows(cfg: GPTConfig, tok: WordTokenizer):
    """Slide a window over the token stream, yielding (input, target) pairs."""
    need = cfg.block_size + 1          # +1 because target is shifted by one
    buf: List[int] = []
    for tid in stream_tokens(cfg, tok):
        buf.append(tid)
        if len(buf) == need:
            yield buf[:-1], buf[1:]    # x = t0..t127 , y = t1..t128
            buf = buf[cfg.stride:]     # move the window forward


def make_dataset(cfg: GPTConfig, tok: WordTokenizer) -> tf.data.Dataset:
    """Wrap the generator in a tf.data pipeline that Keras can train on."""
    sig = (
        tf.TensorSpec(shape=(cfg.block_size,), dtype=tf.int32),
        tf.TensorSpec(shape=(cfg.block_size,), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(
        lambda: stream_windows(cfg, tok), output_signature=sig
    )
    return (
        ds.shuffle(cfg.shuffle_buffer)
          .repeat()                       # restart the stream for every epoch
          .batch(cfg.batch_size, drop_remainder=True)
          .prefetch(tf.data.AUTOTUNE)     # load the next batch while training
    )
