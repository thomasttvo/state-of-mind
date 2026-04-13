"""
Step 5: Cross-model transfer test.

Two distinct transfer claims:

A) SAME-DIM transfer (Mistral base, pre-RLHF):
   Apply Mistral-Instruct direction zero-shot to Mistral base.
   Same architecture + dimensions, different training.
   Tests: does the direction generalize across RLHF fine-tuning?

B) CROSS-ARCHITECTURE (Gemma 2 2B):
   Re-extract direction on Gemma 2 using Mistral's labeled examples.
   Tests: does the PHENOMENON exist in a different architecture?
   (Vector can't transfer directly — different hidden dim)
"""

import json, gc
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from mlx_lm import load
from sklearn.metrics import roc_auc_score

from som.extract import extract_all_layers
from som.direction import FailureDirection

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
ACT_DIR     = BASE_DIR / "activations"

# ── Load Mistral direction + labels ──────────────────────────────────────────
fd_mistral = FailureDirection.load(RESULTS_DIR / "direction.npz")
print(f"Mistral direction: layer={fd_mistral.layer}, dim={fd_mistral.direction.shape[0]}")

with open(DATA_DIR / "labeled_responses.json") as f:
    labeled = json.load(f)

with open(DATA_DIR / "prompts.json") as f:
    dataset = json.load(f)

fail_recs = [
    (pop, item["idx"], item["label"])
    for pop, items in labeled.items()
    for item in items
    if item["label"] in ("sycophantic", "overconfident_wrong")
]
cc_idxs = list(range(20))

all_results = {}


def extract_and_cache(model_key, inner, tokenizer, dataset, act_dir):
    """Extract activations for all 60 prompts if not already cached."""
    for pop, items in dataset.items():
        (act_dir / pop).mkdir(parents=True, exist_ok=True)
        print(f"  Extracting {pop}...")
        for idx, item in enumerate(items):
            out_path = act_dir / pop / f"{idx:04d}.npy"
            if out_path.exists():
                continue
            prompt = item["prompt"]
            try:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": item.get("question", prompt)}],
                    tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
            input_ids = mx.array(tokenizer.encode(prompt))[None]
            acts = extract_all_layers(inner, input_ids)
            np.save(out_path, acts)
            print(f"    [{idx:02d}] {acts.shape}")
    print("  Extraction complete.")


def loo_auroc(cc_projs, fail_projs):
    aucs = [float((fp > cc_projs).mean()) for fp in fail_projs]
    return float(np.mean(aucs)), float(np.std(aucs)), aucs


# ═══════════════════════════════════════════════════════════════════════════════
# A) Same-dim transfer: Mistral 7B BASE (pre-RLHF)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("A) Same-dim transfer: Mistral 7B base (pre-RLHF)")
print(f"{'='*60}")

model_id  = "mlx-community/Mistral-7B-v0.3-4bit"
model_key = "mistral_7b_base"
act_dir   = ACT_DIR / model_key

print(f"Loading {model_id}...")
model, tokenizer = load(model_id)
inner = model.model

extract_and_cache(model_key, inner, tokenizer, dataset, act_dir)

n_layers = np.load(act_dir / "correct_confident" / "0000.npy").shape[0]
transfer_layer = min(fd_mistral.layer, n_layers - 1)
print(f"\n  Layer {transfer_layer} (same as Mistral instruct)")

# Zero-shot: project onto Mistral-Instruct direction
cc_projs_zs = np.array([
    float(np.load(act_dir / "correct_confident" / f"{i:04d}.npy")[transfer_layer] @ fd_mistral.direction)
    for i in cc_idxs
])
fail_projs_zs = np.array([
    float(np.load(act_dir / pop / f"{idx:04d}.npy")[transfer_layer] @ fd_mistral.direction)
    for pop, idx, _ in fail_recs
])

labels = np.array([0]*len(cc_projs_zs) + [1]*len(fail_projs_zs))
auroc_zs = roc_auc_score(labels, np.concatenate([cc_projs_zs, fail_projs_zs]))
loo_mean_zs, loo_std_zs, loo_list_zs = loo_auroc(cc_projs_zs, fail_projs_zs)

print(f"\n  Zero-shot (Mistral-Instruct direction → Mistral base):")
print(f"    CC mean proj:   {cc_projs_zs.mean():.3f}")
print(f"    Fail mean proj: {fail_projs_zs.mean():.3f}")
print(f"    In-sample AUROC: {auroc_zs:.3f}")
print(f"    LOO-AUROC:       {loo_mean_zs:.3f} ± {loo_std_zs:.3f}")
print(f"    Passes 0.70:     {loo_mean_zs >= 0.70}")

# Also: refit on base model itself (upper bound)
cc_acts_base = np.stack([np.load(act_dir / "correct_confident" / f"{i:04d}.npy") for i in cc_idxs])
fail_acts_base = np.stack([np.load(act_dir / pop / f"{idx:04d}.npy") for pop, idx, _ in fail_recs])
fd_base = FailureDirection.fit(cc_acts_base, fail_acts_base, layer=transfer_layer)
fd_base.save(RESULTS_DIR / f"direction_{model_key}.npz")

cc_projs_fit = cc_acts_base[:, transfer_layer, :] @ fd_base.direction
fail_projs_fit = fail_acts_base[:, transfer_layer, :] @ fd_base.direction
labels_fit = np.array([0]*len(cc_projs_fit) + [1]*len(fail_projs_fit))
auroc_fit = roc_auc_score(labels_fit, np.concatenate([cc_projs_fit, fail_projs_fit]))
loo_mean_fit, loo_std_fit, _ = loo_auroc(cc_projs_fit, fail_projs_fit)

print(f"\n  Refitted (upper bound — direction from base model itself):")
print(f"    In-sample AUROC: {auroc_fit:.3f}")
print(f"    LOO-AUROC:       {loo_mean_fit:.3f} ± {loo_std_fit:.3f}")

all_results["mistral_7b_base"] = {
    "model_id": model_id,
    "transfer_type": "same_dim_zero_shot",
    "layer": transfer_layer,
    "zero_shot": {
        "cc_mean_proj": float(cc_projs_zs.mean()),
        "fail_mean_proj": float(fail_projs_zs.mean()),
        "in_sample_auroc": float(auroc_zs),
        "loo_auroc_mean": loo_mean_zs,
        "loo_auroc_std":  loo_std_zs,
        "passes_threshold": loo_mean_zs >= 0.70,
    },
    "refitted_upper_bound": {
        "in_sample_auroc": float(auroc_fit),
        "loo_auroc_mean":  loo_mean_fit,
        "loo_auroc_std":   loo_std_fit,
    },
}

del model, tokenizer, inner
gc.collect()
mx.metal.clear_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# B) Cross-architecture: Gemma 2 2B (different dim — refit only)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("B) Cross-architecture: Gemma 2 2B IT")
print(f"{'='*60}")

model_id  = "mlx-community/gemma-2-2b-it-4bit"
model_key = "gemma2_2b_it"
act_dir   = ACT_DIR / model_key

print(f"Loading {model_id}...")
model, tokenizer = load(model_id)
inner = model.model

extract_and_cache(model_key, inner, tokenizer, dataset, act_dir)

n_layers = np.load(act_dir / "correct_confident" / "0000.npy").shape[0]
hidden_dim = np.load(act_dir / "correct_confident" / "0000.npy").shape[1]
# Scale layer proportionally: Mistral has 32 layers, use same fraction
target_layer_gemma = min(fd_mistral.layer, n_layers - 1)
print(f"\n  Gemma: {n_layers} layers, hidden_dim={hidden_dim}")
print(f"  Using layer {target_layer_gemma} (proportional to Mistral layer {fd_mistral.layer}/32)")

cc_acts_g   = np.stack([np.load(act_dir / "correct_confident" / f"{i:04d}.npy") for i in cc_idxs])
fail_acts_g = np.stack([np.load(act_dir / pop / f"{idx:04d}.npy") for pop, idx, _ in fail_recs])
fd_gemma = FailureDirection.fit(cc_acts_g, fail_acts_g, layer=target_layer_gemma)
fd_gemma.save(RESULTS_DIR / f"direction_{model_key}.npz")

cc_projs_g   = cc_acts_g[:,   target_layer_gemma, :] @ fd_gemma.direction
fail_projs_g = fail_acts_g[:, target_layer_gemma, :] @ fd_gemma.direction
labels_g = np.array([0]*len(cc_projs_g) + [1]*len(fail_projs_g))
auroc_g = roc_auc_score(labels_g, np.concatenate([cc_projs_g, fail_projs_g]))
loo_mean_g, loo_std_g, loo_list_g = loo_auroc(cc_projs_g, fail_projs_g)

print(f"\n  Refitted direction on Gemma 2 (same labeled examples):")
print(f"    CC mean proj:    {cc_projs_g.mean():.3f}")
print(f"    Fail mean proj:  {fail_projs_g.mean():.3f}")
print(f"    In-sample AUROC: {auroc_g:.3f}")
print(f"    LOO-AUROC:       {loo_mean_g:.3f} ± {loo_std_g:.3f}")
print(f"    Passes 0.70:     {loo_mean_g >= 0.70}")

all_results["gemma2_2b_it"] = {
    "model_id": model_id,
    "transfer_type": "cross_architecture_refit",
    "layer": target_layer_gemma,
    "hidden_dim": hidden_dim,
    "refitted": {
        "cc_mean_proj": float(cc_projs_g.mean()),
        "fail_mean_proj": float(fail_projs_g.mean()),
        "in_sample_auroc": float(auroc_g),
        "loo_auroc_mean": loo_mean_g,
        "loo_auroc_std":  loo_std_g,
        "passes_threshold": loo_mean_g >= 0.70,
        "loo_per_example": [
            {"pop": p, "idx": i, "label": l, "loo_auc": a}
            for (p, i, l), a in zip(fail_recs, loo_list_g)
        ],
    },
}

del model, tokenizer, inner
gc.collect()
mx.metal.clear_cache()

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = RESULTS_DIR / "cross_model_transfer.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*60}")
print("TRANSFER SUMMARY")
print(f"{'='*60}")
r = all_results["mistral_7b_base"]
print(f"  Mistral base (zero-shot):  LOO={r['zero_shot']['loo_auroc_mean']:.3f}  "
      f"pass={r['zero_shot']['passes_threshold']}")
print(f"  Mistral base (upper bound): LOO={r['refitted_upper_bound']['loo_auroc_mean']:.3f}")
r2 = all_results["gemma2_2b_it"]
print(f"  Gemma 2 2B (refit):         LOO={r2['refitted']['loo_auroc_mean']:.3f}  "
      f"pass={r2['refitted']['passes_threshold']}")
print(f"\nResults → {out_path}")
