"""
chat.py — a browser chat window for the trained mini GPT.

    python chat.py          then open http://127.0.0.1:7860

IMPORTANT (say this to the class):
    This model was trained ONLY to continue lecture text. It has never seen
    a question-and-answer example in its life, so it does not "answer" —
    it CONTINUES what you type, in the voice of the transcripts.
    That is exactly what GPT-1/GPT-2 were: pure text continuation.
    ChatGPT = this + a second training stage on human Q&A conversations.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gradio as gr

from minigpt.config import GPTConfig
from minigpt.generate import generate
from minigpt.model import build_model
from minigpt.tokenizer import WordTokenizer

OUT_DIR = "checkpoints"

# ---------------------------------------------------------------- load model
cfg_path = os.path.join(OUT_DIR, "config.json")
if not os.path.exists(cfg_path):
    raise SystemExit("No trained model found. Run:  python train.py")

cfg = GPTConfig.load(cfg_path)
tok = WordTokenizer.load(os.path.join(OUT_DIR, "tokenizer.json"))
model = build_model(cfg)
model.load_weights(os.path.join(OUT_DIR, "minigpt.weights.h5"))
print(f"loaded mini GPT — vocab {cfg.vocab_size}, context {cfg.block_size}")


def respond(message, history, max_tokens, temperature, top_k):
    if not message.strip():
        return "Type something and I'll continue it."
    text = generate(model, tok, message,
                    max_new_tokens=int(max_tokens),
                    temperature=float(temperature),
                    top_k=int(top_k))
    return text.strip()


# ------------------------------------------------------------------- the UI
# The three sliders are the "personality" dials from generate.py. Open the
# accordion in class and show what temperature actually does.
demo = gr.ChatInterface(
    fn=respond,
    #type="messages",
    title="Mini GPT",
    description=(
        f"A {cfg.n_layer}-layer GPT built from scratch in TensorFlow, trained on "
        "lecture transcripts. It **continues** your text — it does not answer "
        "questions. Try: `so today we are going to`"
    ),
    additional_inputs=[
        gr.Slider(10, 200, value=60, step=10, label="Words to generate"),
        gr.Slider(0.1, 1.5, value=0.9, step=0.05,
                  label="Temperature (low = safe, high = creative)"),
        gr.Slider(1, 200, value=40, step=1,
                  label="Top-k (how many candidate words to consider)"),
    ],
    additional_inputs_accordion=gr.Accordion("Generation settings", open=False),
    # With extra inputs, each example is [message, *those inputs].
    examples=[
        ["so today we are going to", 60, 0.9, 40],
        ["let me explain what is", 60, 0.9, 40],
        ["the next thing that we will", 60, 0.3, 40],   # low temperature
        ["the next thing that we will", 60, 1.3, 40],   # high temperature
    ],
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
