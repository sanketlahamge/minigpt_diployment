"""
tokenizer.py — turn text into numbers, and numbers back into text.

A neural network cannot read letters, only numbers. So step #1 of every GPT
is a *tokenizer*: a fixed dictionary  word -> id.

Real GPTs use "BPE" sub-word tokens ("unbelievable" -> "un" + "believ" + "able").
We use plain WORDS, because for a classroom that is the easiest thing to see:
    "so let us start"  ->  [42, 17, 88, 301]

Words outside our vocabulary become <unk> (unknown).
"""

import json
import os
import re
from collections import Counter
from typing import Iterable, List

# Split on words/numbers/apostrophes, and keep punctuation as its own token.
_TOKEN_RE = re.compile(r"[a-z0-9']+|[^\sa-z0-9']")

UNK = "<unk>"


def tokenize(text: str) -> List[str]:
    """'Hello, world' -> ['hello', ',', 'world']"""
    return _TOKEN_RE.findall(text.lower())


class WordTokenizer:
    def __init__(self, itos: List[str]):
        self.itos = itos                                    # id   -> word
        self.stoi = {w: i for i, w in enumerate(itos)}      # word -> id
        self.unk_id = self.stoi[UNK]

    # ------------------------------------------------------------ building
    @classmethod
    def build(cls, text_chunks: Iterable[str], vocab_size: int) -> "WordTokenizer":
        """Count every word in the stream, keep the `vocab_size` most common."""
        counts = Counter()
        for chunk in text_chunks:
            counts.update(tokenize(chunk))

        # <unk> always gets id 0, then the most frequent words.
        itos = [UNK] + [w for w, _ in counts.most_common(vocab_size - 1)]
        print(f"[tokenizer] saw {sum(counts.values()):,} words, "
              f"{len(counts):,} unique -> keeping {len(itos):,}")
        return cls(itos)

    # ------------------------------------------------------------- using it
    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(w, self.unk_id) for w in tokenize(text)]

    def decode(self, ids: Iterable[int]) -> str:
        words = [self.itos[i] for i in ids]
        # Join with spaces, but glue punctuation onto the previous word.
        out = ""
        for w in words:
            if re.fullmatch(r"[^\sa-z0-9']", w) or not out:
                out += w
            else:
                out += " " + w
        return out

    # ------------------------------------------------------------ save/load
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos, f)

    @classmethod
    def load(cls, path: str) -> "WordTokenizer":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))
