"""
Step 4: Bidirectional direction patching (causal test).

Injection scale: alpha must shift the projection from the CC region (~-0.6)
into the failing region (~+0.7), a gap of ~1.3. We inject at every generation
step (not just prefill) to maintain the steering signal throughout output.

Sweep: test alpha in [1, 2, 3, 5] to find the sweet spot between
behavioral change and incoherence.
"""

import json
import numpy as np
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.llama import create_attention_mask

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print("Loading model...")
model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
inner    = model.model
n_layers = len(inner.layers)

TARGET_LAYER = 6
MAX_TOKENS   = 120

# ── Load direction ────────────────────────────────────────────────────────────
with open(DATA_DIR / "labeled_responses.json") as f:
    labeled = json.load(f)

act_dir = BASE_DIR / "activations"
cc_acts   = np.stack([np.load(act_dir / "correct_confident" / f"{i:04d}.npy") for i in range(20)])
fail_recs = [(pop, item["idx"]) for pop, items in labeled.items()
             for item in items if item["label"] in ("sycophantic","overconfident_wrong")]
fail_acts = np.stack([np.load(act_dir / pop / f"{idx:04d}.npy") for pop, idx in fail_recs])

d_raw  = fail_acts[:, TARGET_LAYER, :].mean(0) - cc_acts[:, TARGET_LAYER, :].mean(0)
d_fail = (d_raw / np.linalg.norm(d_raw)).astype(np.float32)
d_mx   = mx.array(d_fail)

# Sanity: what is the projection gap?
cc_projs   = cc_acts[:, TARGET_LAYER, :] @ d_fail
fail_projs = fail_acts[:, TARGET_LAYER, :] @ d_fail
print(f"CC projections:   {cc_projs.mean():.3f} ± {cc_projs.std():.3f}")
print(f"Fail projections: {fail_projs.mean():.3f} ± {fail_projs.std():.3f}")
print(f"Gap: {fail_projs.mean() - cc_projs.mean():.3f}  (alpha needed to bridge: ~{fail_projs.mean() - cc_projs.mean():.1f})")

# ── Patched generation ────────────────────────────────────────────────────────

def steered_generate(prompt: str, alpha: float) -> str:
    """Generate with direction continuously injected at TARGET_LAYER."""
    tokens_in = tokenizer.encode(prompt)

    def one_step(token_ids, cache):
        x  = mx.array(token_ids)[None]
        h  = inner.embed_tokens(x)
        fa = create_attention_mask(h, cache[inner.fa_idx])
        sw = None
        if inner.swa_idx is not None:
            sw = create_attention_mask(h, cache[inner.swa_idx], window_size=inner.sliding_window)
        for i, (layer, c) in enumerate(zip(inner.layers, cache)):
            mask = sw if getattr(layer, "use_sliding", False) else fa
            h    = layer(h, mask, cache=c)
            if i == TARGET_LAYER:
                h = h + alpha * d_mx[None, None, :]
        h = inner.norm(h)
        logits = model.lm_head(h)
        return logits

    cache      = model.make_cache()
    logits     = one_step(tokens_in, cache)
    mx.eval(logits)
    next_tok   = int(mx.argmax(logits[0, -1, :]).item())
    out_tokens = [next_tok]

    for _ in range(MAX_TOKENS - 1):
        logits   = one_step([next_tok], cache)
        mx.eval(logits)
        next_tok = int(mx.argmax(logits[0, -1, :]).item())
        out_tokens.append(next_tok)
        if next_tok == tokenizer.eos_token_id:
            break

    return tokenizer.decode(out_tokens)


# ── Alpha sweep on one CC and one failing prompt ──────────────────────────────
with open(DATA_DIR / "responses.json") as f:
    responses = json.load(f)

# Pick France/Paris (simplest, ground truth unambiguous)
cc_probe      = responses["correct_confident"][0]
# Pick francium (type_b idx=1: "23.6K" was the failure)
fail_probe_b  = next(x for x in responses["type_b_overconfident"] if x["idx"]==1)
# Pick tongue zones (type_a idx=11: sycophantic)
fail_probe_a  = next(x for x in responses["type_a_sycophantic"]   if x["idx"]==11)

print("\n" + "="*60)
print("ALPHA SWEEP — CC prompt (expect failure as alpha increases)")
print("="*60)
print(f"Prompt: {cc_probe['prompt']}")
print(f"Original: {responses['correct_confident'][0]['response'][:100].strip()}")
for alpha in [1.0, 2.0, 3.0, 5.0]:
    out = steered_generate(cc_probe["prompt"], alpha)
    print(f"  alpha={alpha:.1f}: {out[:140].strip()}")

print("\n" + "="*60)
print("ALPHA SWEEP — type_b failing (expect recovery as alpha decreases)")
print("="*60)
print(f"Prompt: {fail_probe_b['prompt'][:80]}")
print(f"Original (failing): {fail_probe_b['response'][:100].strip()}")
for alpha in [-1.0, -2.0, -3.0, -5.0]:
    out = steered_generate(fail_probe_b["prompt"], alpha)
    print(f"  alpha={alpha:.1f}: {out[:140].strip()}")

print("\n" + "="*60)
print("ALPHA SWEEP — type_a sycophantic (expect recovery as alpha decreases)")
print("="*60)
print(f"Prompt: {fail_probe_a['prompt'][:80]}")
print(f"Original (sycophantic): {fail_probe_a['response'][:100].strip()}")
for alpha in [-1.0, -2.0, -3.0, -5.0]:
    out = steered_generate(fail_probe_a["prompt"], alpha)
    print(f"  alpha={alpha:.1f}: {out[:140].strip()}")

# ── Full run at best alpha (determined from sweep) ────────────────────────────
# After sweep, set BEST_ALPHA manually and re-run all 7 pairs
BEST_ALPHA = 3.0   # adjust after seeing sweep results

print(f"\n{'='*60}")
print(f"FULL RUN at alpha=+/-{BEST_ALPHA}")
print(f"{'='*60}")

results = {"alpha": BEST_ALPHA, "target_layer": TARGET_LAYER, "injections": []}

# 7 CC → failure
cc_test = [(item["prompt"], item.get("ground_truth",""), item["response"])
           for item in responses["correct_confident"][:7]]
print("\n--- CC → failure (+d) ---")
for i, (prompt, truth, original) in enumerate(cc_test):
    out = steered_generate(prompt, +BEST_ALPHA)
    print(f"[{i}] original: {original[:80].strip()}")
    print(f"    patched:  {out[:80].strip()}")
    results["injections"].append({
        "type": "cc_to_fail", "prompt": prompt, "ground_truth": truth,
        "original": original[:300], "patched": out[:300],
    })

# 7 failing → recovery
print("\n--- failing → recovery (-d) ---")
for pop, idx in fail_recs:
    prompt   = next(x["prompt"]   for x in responses[pop] if x["idx"]==idx)
    original = next(x["response"] for x in responses[pop] if x["idx"]==idx)
    label    = next(x["label"]    for x in labeled[pop]   if x["idx"]==idx)
    out = steered_generate(prompt, -BEST_ALPHA)
    print(f"[{pop} {idx}] original: {original[:80].strip()}")
    print(f"              patched:  {out[:80].strip()}")
    results["injections"].append({
        "type": "fail_to_cc", "pop": pop, "idx": idx, "label": label,
        "prompt": prompt, "original": original[:300], "patched": out[:300],
    })

with open(RESULTS_DIR / "patching_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults -> {RESULTS_DIR / 'patching_results.json'}")
