"""
Extract activations and responses for prompts_v2.json (held-out test set).

Same infrastructure as extract_activations.py but for the v2 held-out prompts.
These were never seen during direction fitting — they form the honest test set.

Outputs:
  activations/cc_v2/<idx>.npy            — shape (n_layers, hidden_dim)
  activations/type_b_v2/<idx>.npy
  activations/type_a_hard/<idx>.npy
  data/responses_v2.json                 — model responses
"""

import json
import numpy as np
from pathlib import Path
import sys

import mlx.core as mx
from mlx_lm import load, generate

BASE_DIR        = Path(__file__).parent.parent
DATA_DIR        = BASE_DIR / "data"
ACTIVATIONS_DIR = BASE_DIR / "activations"

sys.path.insert(0, str(BASE_DIR))
from som.extract import extract_all_layers

MODEL_ID   = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
MAX_TOKENS = 200

for pop in ("cc_v2", "type_b_v2", "type_a_hard"):
    (ACTIVATIONS_DIR / pop).mkdir(parents=True, exist_ok=True)

print(f"Loading {MODEL_ID} …")
model, tokenizer = load(MODEL_ID)
inner    = model.model
n_layers = len(inner.layers)
print(f"  {n_layers} transformer layers loaded")

with open(DATA_DIR / "prompts_v2.json") as f:
    dataset = json.load(f)

responses: dict = {}
total = sum(len(v) for v in dataset.values())
done  = 0

for population, items in dataset.items():
    print(f"\n{'─'*60}")
    print(f"Population: {population}  ({len(items)} prompts)")
    print(f"{'─'*60}")
    responses[population] = []

    for idx, item in enumerate(items):
        prompt = item["prompt"]
        done  += 1
        print(f"  [{done:02d}/{total}] ", end="", flush=True)

        # Skip if already extracted
        npy_path = ACTIVATIONS_DIR / population / f"{idx:04d}.npy"
        if npy_path.exists():
            print(f"(cached) ", end="", flush=True)
            acts = np.load(npy_path)
        else:
            print("activations … ", end="", flush=True)
            input_ids = mx.array(tokenizer.encode(prompt))[None]
            acts = extract_all_layers(inner, input_ids)
            np.save(npy_path, acts)
        print(f"shape={acts.shape}  ", end="", flush=True)

        # Generate response (skip if exists in partial run)
        print("generating … ", end="", flush=True)
        response = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            verbose=False,
        )
        responses[population].append({
            "idx":      idx,
            "prompt":   prompt,
            "response": response,
            **{k: v for k, v in item.items() if k != "prompt"},
        })
        print("done")
        print(f"     {response[:140].strip()!r}")

with open(DATA_DIR / "responses_v2.json", "w") as f:
    json.dump(responses, f, indent=2)

print(f"\n{'='*60}")
print(f"Activations → {ACTIVATIONS_DIR}/")
print(f"Responses   → {DATA_DIR / 'responses_v2.json'}")
for pop in dataset:
    s = np.load(ACTIVATIONS_DIR / pop / "0000.npy")
    print(f"  {pop}: {s.shape}")
