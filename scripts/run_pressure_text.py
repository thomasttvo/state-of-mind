"""
Re-run pressure conversations to capture actual text responses.
Skips activation extraction (already done) — just generates and records text.
"""
import json, gc
import numpy as np
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, generate

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MAX_TOKENS = 120

print("Loading model...")
model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

def respond(messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    out = generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, verbose=False)
    mx.metal.clear_cache()
    return out

with open(DATA_DIR / "multiturn_conversations.json") as f:
    dataset = json.load(f)

results = {}

for conv in dataset["pressure"]:
    cid   = conv["id"]
    turns = conv["turns"]
    print(f"\n{'='*60}")
    print(f"{cid}  (correct: {conv['correct_answer']}  |  wrong: {conv['wrong_claim']})")
    print(f"{'='*60}")

    history = []
    responses = []
    ai = 0

    for turn in turns:
        if turn["role"] == "user":
            history.append({"role": "user", "content": turn["content"]})
            print(f"\nUser: {turn['content']}")
        elif turn["role"] == "assistant" and turn["content"] == "GENERATE":
            resp = respond(history)
            history.append({"role": "assistant", "content": resp})
            responses.append({"turn": ai, "response": resp})
            print(f"Asst [{ai}]: {resp[:200]}")
            ai += 1

    results[cid] = responses
    gc.collect()
    mx.metal.clear_cache()
    print(f"  [memory cleared]")

with open(DATA_DIR / "pressure_text_responses.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved -> {DATA_DIR / 'pressure_text_responses.json'}")
