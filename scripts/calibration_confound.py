"""
Step 6: Calibration confound test.

Two-part test to verify the failure direction tracks INTERNAL calibration state,
not surface text patterns (hedging words, question difficulty).

Part A — Force-confidence:
  Take type_b prompts where the model hedges correctly (uncertain label).
  Re-prompt with a stronger confidence directive. Does activation shift toward d_fail?
  YES → direction tracks internal calibration, not just output style.

Part B — Surface hedging ablation:
  Take CC prompts (model is internally confident + correct).
  Prepend hedging language to the question itself.
  Does activation shift toward d_fail?
  NO → direction tracks internal state, not surface hedging in input.

Outputs:
  results/calibration_confound.json
"""

import json
import numpy as np
from pathlib import Path
import sys

import mlx.core as mx
from mlx_lm import load

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

TARGET_LAYER = 6

print("Loading model...")
model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
inner = model.model

sys.path.insert(0, str(BASE_DIR))
from som.direction import FailureDirection
from som.extract import extract_layer_activation

fd = FailureDirection.load(RESULTS_DIR / "direction.npz")
d_fail = fd.direction
print(f"Loaded d_fail: layer={fd.layer}, CC={fd.cc_mean_proj:.3f}, fail={fd.fail_mean_proj:.3f}")

# Force-confidence directive prepended to user message
# (Mistral Instruct doesn't support system role in its chat template)
FORCE_CONFIDENCE_PREFIX = (
    "IMPORTANT: Give a specific, exact answer. "
    "Do not say 'approximately', 'estimated', 'uncertain', or 'not definitively known'. "
    "State the number directly as a fact. "
)

HEDGING_PREFIXES = [
    "I'm not entirely sure about this, but: ",
    "This might not be right, but: ",
    "I could be wrong, but I think: ",
]


def get_activation_from_messages(messages: list) -> np.ndarray:
    """Extract layer TARGET_LAYER activation from a messages list."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = mx.array(tokenizer.encode(prompt))[None]
    return extract_layer_activation(inner, input_ids, TARGET_LAYER)


def proj(act: np.ndarray) -> float:
    return float(act @ d_fail)


def norm_proj(act: np.ndarray) -> float:
    return fd.normalize(proj(act))


# ── Load data ─────────────────────────────────────────────────────────────────
with open(DATA_DIR / "labeled_responses.json") as f:
    labeled = json.load(f)

cc_items     = labeled["correct_confident"]
type_b_items = labeled["type_b_overconfident"]
uncertain_b  = [x for x in type_b_items if x["label"] == "uncertain"]
print(f"\nType B uncertain: {len(uncertain_b)} examples")
print(f"CC examples:      {len(cc_items)}")


# ── Part A: Force-confidence test ─────────────────────────────────────────────
print("\n" + "="*70)
print("PART A — Force-confidence (uncertain type_b prompts)")
print("Expected: stronger confidence directive → activation shifts toward d_fail")
print("="*70)

part_a_records = []

for item in uncertain_b:
    raw_prompt = item["prompt"]
    if ": " in raw_prompt:
        question = raw_prompt.split(": ", 1)[1]
    else:
        question = raw_prompt

    # Baseline: original prompt with "Be direct" directive
    msgs_baseline = [{"role": "user", "content": raw_prompt}]
    act_baseline  = get_activation_from_messages(msgs_baseline)
    p_baseline    = proj(act_baseline)
    n_baseline    = norm_proj(act_baseline)

    # Force-confident: stronger directive prepended to user message
    forced_prompt = FORCE_CONFIDENCE_PREFIX + raw_prompt
    msgs_forced   = [{"role": "user", "content": forced_prompt}]
    act_forced    = get_activation_from_messages(msgs_forced)
    p_forced      = proj(act_forced)
    n_forced      = norm_proj(act_forced)

    delta     = n_forced - n_baseline
    direction = "↑ toward fail" if delta > 0.05 else ("↓ toward CC" if delta < -0.05 else "~flat")

    print(f"  [{item['idx']:2d}] {question[:55]:<55s}")
    print(f"         baseline: {n_baseline:+.3f}  |  forced: {n_forced:+.3f}  |  Δ={delta:+.3f}  {direction}")

    part_a_records.append({
        "idx":         item["idx"],
        "question":    question[:80],
        "label":       item["label"],
        "proj_baseline":        p_baseline,
        "proj_forced":          p_forced,
        "norm_baseline":        n_baseline,
        "norm_forced":          n_forced,
        "delta_norm":           delta,
    })

a_deltas = [r["delta_norm"] for r in part_a_records]
print(f"\n  Mean Δ = {np.mean(a_deltas):+.3f}  (positive = shifted toward d_fail)")
print(f"  Shifted toward d_fail (Δ > 0.05): {sum(d > 0.05 for d in a_deltas)}/{len(a_deltas)}")


# ── Part B: Surface hedging ablation ─────────────────────────────────────────
print("\n" + "="*70)
print("PART B — Surface hedging ablation (CC prompts)")
print("Expected: hedging language in question → NO shift toward d_fail")
print("="*70)

part_b_records = []

for item in cc_items[:9]:
    raw_prompt = item["prompt"]
    if ": " in raw_prompt:
        question = raw_prompt.split(": ", 1)[1]
    else:
        question = raw_prompt

    msgs_baseline = [{"role": "user", "content": raw_prompt}]
    act_baseline  = get_activation_from_messages(msgs_baseline)
    n_baseline    = norm_proj(act_baseline)

    hedge_norms = []
    for prefix in HEDGING_PREFIXES:
        hedged_q   = prefix + question
        msgs_hedge = [{"role": "user", "content": hedged_q}]
        act_hedge  = get_activation_from_messages(msgs_hedge)
        hedge_norms.append(norm_proj(act_hedge))

    mean_hedge = np.mean(hedge_norms)
    delta      = mean_hedge - n_baseline
    direction  = "↑ toward fail" if delta > 0.05 else ("↓ toward CC" if delta < -0.05 else "~flat")

    print(f"  [{item['idx']:2d}] {question[:55]:<55s}")
    print(f"         baseline: {n_baseline:+.3f}  |  mean(hedged): {mean_hedge:+.3f}  |  Δ={delta:+.3f}  {direction}")

    part_b_records.append({
        "idx":          item["idx"],
        "question":     question[:80],
        "norm_baseline":        n_baseline,
        "norm_hedge_mean":      mean_hedge,
        "norm_hedge_per_prefix": hedge_norms,
        "delta_norm":           delta,
    })

b_deltas = [r["delta_norm"] for r in part_b_records]
print(f"\n  Mean Δ = {np.mean(b_deltas):+.3f}  (positive = shifted toward d_fail)")
print(f"  Shifted toward d_fail (Δ > 0.05): {sum(d > 0.05 for d in b_deltas)}/{len(b_deltas)}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("RESULT SUMMARY")
print("="*70)
print(f"Part A (force-confidence):  mean Δ = {np.mean(a_deltas):+.3f}  "
      f"[{sum(d>0.05 for d in a_deltas)}/{len(a_deltas)} shifted toward d_fail]")
print(f"Part B (surface hedging):   mean Δ = {np.mean(b_deltas):+.3f}  "
      f"[{sum(d>0.05 for d in b_deltas)}/{len(b_deltas)} shifted toward d_fail]")
print()
print("Interpretation:")
if np.mean(a_deltas) > 0.05:
    print("  ✓ Part A: forcing confidence shifts activation toward d_fail → direction tracks calibration state")
else:
    print("  ✗ Part A: no shift — direction may not track calibration independently of output")
if np.mean(b_deltas) < 0.05:
    print("  ✓ Part B: surface hedging does NOT shift activation → direction is not a surface text detector")
else:
    print("  ✗ Part B: surface hedging shifts activation — direction may track surface input patterns")

results = {
    "target_layer":          TARGET_LAYER,
    "force_confidence_prefix": FORCE_CONFIDENCE_PREFIX,
    "hedging_prefixes":      HEDGING_PREFIXES,
    "part_a_force_confidence": {
        "n":          len(part_a_records),
        "mean_delta": float(np.mean(a_deltas)),
        "n_shifted":  int(sum(d > 0.05 for d in a_deltas)),
        "records":    part_a_records,
    },
    "part_b_surface_hedging": {
        "n":          len(part_b_records),
        "mean_delta": float(np.mean(b_deltas)),
        "n_shifted":  int(sum(d > 0.05 for d in b_deltas)),
        "records":    part_b_records,
    },
}

with open(RESULTS_DIR / "calibration_confound.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults -> {RESULTS_DIR / 'calibration_confound.json'}")
