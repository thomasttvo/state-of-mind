"""
Step 4: Gram-Schmidt null direction control.

Pre-registered falsification test: inject a random direction orthogonal to d_fail
(same unit norm) into CC responses at the same alpha used for d_fail patching.

If d_fail injection degrades behavior but d_null injection does not,
the specific direction matters — not perturbation magnitude.

Outputs:
  results/null_injection.json
"""

import json
import numpy as np
from pathlib import Path
import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.llama import create_attention_mask

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

TARGET_LAYER = 6
ALPHA        = 3.0   # same alpha that induced failure in patch_direction.py
MAX_TOKENS   = 120
SEED         = 42    # pre-registered random seed for d_null construction

print("Loading model...")
model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
inner = model.model

# ── Load fitted direction ─────────────────────────────────────────────────────
sys.path.insert(0, str(BASE_DIR))
from som.direction import FailureDirection

fd = FailureDirection.load(RESULTS_DIR / "direction.npz")
d_fail = fd.direction  # (4096,) unit norm
print(f"Loaded d_fail: layer={fd.layer}, CC={fd.cc_mean_proj:.3f}, fail={fd.fail_mean_proj:.3f}")

# ── Construct d_null via Gram-Schmidt ─────────────────────────────────────────
rng      = np.random.RandomState(SEED)
rand_vec = rng.randn(d_fail.shape[0]).astype(np.float32)
rand_vec -= (rand_vec @ d_fail) * d_fail     # project out d_fail component
d_null   = rand_vec / np.linalg.norm(rand_vec)   # normalize to unit norm

cosine_check = float(d_null @ d_fail)
norm_check   = float(np.linalg.norm(d_null))
print(f"d_null · d_fail = {cosine_check:.2e}  (expect ≈ 0)")
print(f"||d_null||      = {norm_check:.6f}  (expect 1.0)")
assert abs(cosine_check) < 1e-5, f"d_null not orthogonal: cosine = {cosine_check}"

d_fail_mx = mx.array(d_fail)
d_null_mx = mx.array(d_null)


# ── Steered generation ────────────────────────────────────────────────────────

def steered_generate(prompt: str, direction_mx, alpha: float) -> str:
    """Generate with `alpha * direction` injected at TARGET_LAYER every step."""
    tokens_in = tokenizer.encode(prompt)

    def one_step(token_ids, cache):
        x  = mx.array(token_ids)[None]
        h  = inner.embed_tokens(x)
        fa = create_attention_mask(h, cache[inner.fa_idx])
        sw = None
        if inner.swa_idx is not None:
            sw = create_attention_mask(
                h, cache[inner.swa_idx], window_size=inner.sliding_window
            )
        for i, (layer, c) in enumerate(zip(inner.layers, cache)):
            mask = sw if getattr(layer, "use_sliding", False) else fa
            h    = layer(h, mask, cache=c)
            if i == TARGET_LAYER:
                h = h + alpha * direction_mx[None, None, :]
        h      = inner.norm(h)
        logits = model.lm_head(h)
        return logits

    cache    = model.make_cache()
    logits   = one_step(tokens_in, cache)
    mx.eval(logits)
    next_tok = int(mx.argmax(logits[0, -1, :]).item())
    out_toks = [next_tok]

    for _ in range(MAX_TOKENS - 1):
        logits   = one_step([next_tok], cache)
        mx.eval(logits)
        next_tok = int(mx.argmax(logits[0, -1, :]).item())
        out_toks.append(next_tok)
        if next_tok == tokenizer.eos_token_id:
            break

    return tokenizer.decode(out_toks)


# ── Load CC prompts ───────────────────────────────────────────────────────────
with open(DATA_DIR / "responses.json") as f:
    responses = json.load(f)

CC_PROMPTS = responses["correct_confident"][:5]

print("\n" + "="*70)
print(f"NULL DIRECTION CONTROL  (alpha={ALPHA}, layer {TARGET_LAYER})")
print("="*70)
print("Prediction: +d_fail degrades behavior; +d_null does NOT.\n")

records = []

for item in CC_PROMPTS:
    prompt   = item["prompt"]
    baseline = item["response"]

    print(f"Prompt: {prompt[:80].strip()}")
    print(f"  Baseline:        {baseline[:100].strip()}")

    out_dfail = steered_generate(prompt, d_fail_mx, +ALPHA)
    print(f"  +d_fail (α={ALPHA}): {out_dfail[:100].strip()}")

    out_dnull = steered_generate(prompt, d_null_mx, +ALPHA)
    print(f"  +d_null (α={ALPHA}): {out_dnull[:100].strip()}")
    print()

    records.append({
        "prompt":          prompt,
        "ground_truth":    item.get("ground_truth", ""),
        "baseline":        baseline,
        "dfail_injection": out_dfail,
        "dnull_injection": out_dnull,
    })

# ── Summary ───────────────────────────────────────────────────────────────────
print("="*70)
print("SUMMARY")
print("="*70)

dfail_degrades = 0
dnull_degrades = 0
for i, (r, item) in enumerate(zip(records, CC_PROMPTS)):
    gt = item.get("ground_truth", "").lower()
    dfail_ok = gt in r["dfail_injection"].lower() if gt else None
    dnull_ok = gt in r["dnull_injection"].lower() if gt else None
    baseline_ok = gt in r["baseline"].lower() if gt else None

    if baseline_ok and not dfail_ok:
        dfail_degrades += 1
    if baseline_ok and not dnull_ok:
        dnull_degrades += 1

    status_dfail = "DEGRADED" if (baseline_ok and not dfail_ok) else "OK"
    status_dnull = "OK"      if (not baseline_ok or dnull_ok)   else "DEGRADED"
    print(f"  [{i}] d_fail: {status_dfail}  |  d_null: {status_dnull}")

print(f"\nd_fail degraded {dfail_degrades}/5 correct answers")
print(f"d_null degraded {dnull_degrades}/5 correct answers")
print("(Expect: d_fail > 0, d_null = 0)")

results = {
    "alpha":                         ALPHA,
    "target_layer":                  TARGET_LAYER,
    "seed":                          SEED,
    "d_null_cosine_with_d_fail":     cosine_check,
    "d_null_norm":                   norm_check,
    "n_prompts":                     len(records),
    "dfail_degraded":                dfail_degrades,
    "dnull_degraded":                dnull_degrades,
    "records":                       records,
}

with open(RESULTS_DIR / "null_injection.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults -> {RESULTS_DIR / 'null_injection.json'}")
