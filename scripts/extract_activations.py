"""
Step 2: Extract residual stream activations from model.

For each prompt:
  - Tokenize and run a manual layer-by-layer forward pass (prefix only)
  - Capture last-token hidden state after every transformer layer
  - Generate a short response (separate call) for downstream labeling

Outputs:
  activations/<population>/<idx>.npy  — shape (n_layers, hidden_dim), float32
  data/responses.json                 — model responses keyed by population+idx

No monkey-patching: prefix extraction is done by manually iterating layers,
which avoids the performance cost of materialising activations during generation.
"""

import json
import numpy as np
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.llama import create_attention_mask

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID   = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
MAX_TOKENS = 200

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
DATA_DIR        = BASE_DIR / "data"
ACTIVATIONS_DIR = BASE_DIR / "activations"
RESPONSES_PATH  = DATA_DIR / "responses.json"

for pop in ("correct_confident", "type_b_overconfident", "type_a_sycophantic"):
    (ACTIVATIONS_DIR / pop).mkdir(parents=True, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_ID} …")
model, tokenizer = load(MODEL_ID)
inner   = model.model           # LlamaModel
n_layers = len(inner.layers)
print(f"  {n_layers} transformer layers loaded")

# ── Core: manual layer-by-layer prefix forward pass ──────────────────────────

def extract_activations(prompt: str) -> np.ndarray:
    """
    Runs a forward pass over `prompt` layer-by-layer and captures
    the last-token residual stream after each transformer block.

    Returns shape (n_layers, hidden_dim), float32.
    """
    input_ids = mx.array(tokenizer.encode(prompt))[None]  # (1, seq_len)

    h = inner.embed_tokens(input_ids)

    # Fresh KV cache (None = no prior context)
    cache = [None] * n_layers

    # Attention masks
    fa_mask  = create_attention_mask(h, cache[inner.fa_idx])
    swa_mask = None
    if inner.swa_idx is not None:
        swa_mask = create_attention_mask(
            h, cache[inner.swa_idx], window_size=inner.sliding_window
        )

    captured = []
    for i, (layer, c) in enumerate(zip(inner.layers, cache)):
        mask = swa_mask if getattr(layer, "use_sliding", False) else fa_mask
        h = layer(h, mask, cache=c)
        # Force MLX to materialise h before we numpy-copy the last token.
        mx.eval(h)
        captured.append(np.array(h[0, -1, :], dtype=np.float32))

    return np.stack(captured, axis=0)  # (n_layers, hidden_dim)


# ── Main loop ─────────────────────────────────────────────────────────────────
with open(DATA_DIR / "prompts.json") as f:
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
        done += 1
        print(f"  [{done:02d}/{total}] ", end="", flush=True)

        # 1. Activation extraction
        print("activations … ", end="", flush=True)
        acts = extract_activations(prompt)
        np.save(ACTIVATIONS_DIR / population / f"{idx:04d}.npy", acts)
        print(f"shape={acts.shape}  ", end="", flush=True)

        # 2. Generate response for labeling
        print("generating … ", end="", flush=True)
        response = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            verbose=False,
        )
        responses[population].append({
            "idx": idx,
            "prompt": prompt,
            "response": response,
            **{k: v for k, v in item.items() if k != "prompt"},
        })
        print("done")
        print(f"     {response[:140].strip()!r}")

# ── Save responses ────────────────────────────────────────────────────────────
with open(RESPONSES_PATH, "w") as f:
    json.dump(responses, f, indent=2)

print(f"\n{'='*60}")
print(f"Activations → {ACTIVATIONS_DIR}/")
print(f"Responses   → {RESPONSES_PATH}")

# Quick shape sanity check
for pop in dataset:
    s = np.load(ACTIVATIONS_DIR / pop / "0000.npy")
    print(f"  {pop}: {s.shape}")
