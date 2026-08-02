
import numpy as np
import tensorflow as tf

from .tokenizer import WordTokenizer


def generate(model, tok: WordTokenizer, prompt: str,
             max_new_tokens: int = 60,
             temperature: float = 0.9,
             top_k: int = 40,
             seed: int | None = None) -> str:
    rng = np.random.default_rng(seed)
    block_size = model.cfg.block_size

    ids = tok.encode(prompt)
    if not ids:                       # empty prompt -> start from <unk>
        ids = [tok.unk_id]
    generated = []

    for _ in range(max_new_tokens):
        # The model can only see the last `block_size` tokens.
        context = ids[-block_size:]
        x = tf.constant([context], dtype=tf.int32)

        logits = model(x, training=False).numpy()[0, -1]     # scores for next token

        logits = logits / max(temperature, 1e-6)

        # top-k: keep only the k best candidates, kill the rest.
        if top_k and top_k < len(logits):
            cutoff = np.partition(logits, -top_k)[-top_k]
            logits = np.where(logits < cutoff, -np.inf, logits)

        # softmax -> probabilities -> sample one
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        next_id = int(rng.choice(len(probs), p=probs))

        ids.append(next_id)
        generated.append(next_id)

    return tok.decode(generated)
