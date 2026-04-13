"""
Step 3b: Honest direction analysis using behavior-based labels.

Key questions:
1. Does the direction track ACTUAL model state (correct vs failing behavior)
   or just prompt-surface features (easy vs hard question type)?
2. What is the honest AUROC using leave-one-out cross-validation (n_fail=7)?

Design:
- Direction extracted from: correct_confident (20) vs confirmed_failing (7)
- Project all 60 labeled prompts onto direction
- Color by actual label — if direction tracks state, "correct" responses
  from type_a/b should cluster with correct_confident, not with failing
- LOO-AUROC: for each failing example, extract direction on n-1 failing + 20 CC,
  test on held-out failing vs all 20 CC
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR    = Path(__file__).parent.parent
ACT_DIR     = BASE_DIR / "activations"
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR     = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_DIR / "labeled_responses.json") as f:
    labeled = json.load(f)

def load_acts(pop, idx):
    return np.load(ACT_DIR / pop / f"{idx:04d}.npy")  # (32, 4096)

records = []
for pop, items in labeled.items():
    for item in items:
        records.append({
            "pop":   pop,
            "idx":   item["idx"],
            "label": item["label"],
            "acts":  load_acts(pop, item["idx"]),
        })

print(f"Loaded {len(records)} records")

TARGET_LAYER = 6

def get_layer(rec, layer=TARGET_LAYER):
    return rec["acts"][layer]

cc_recs   = [r for r in records if r["pop"] == "correct_confident"]
fail_recs = [r for r in records if r["label"] in ("sycophantic", "overconfident_wrong")]

print(f"Direction set: {len(cc_recs)} correct_confident, {len(fail_recs)} failing")

def extract_direction(cc_list, fail_list, layer=TARGET_LAYER):
    cc_mean   = np.mean([r["acts"][layer] for r in cc_list],   axis=0)
    fail_mean = np.mean([r["acts"][layer] for r in fail_list], axis=0)
    d = fail_mean - cc_mean
    return d / (np.linalg.norm(d) + 1e-12)

d_full = extract_direction(cc_recs, fail_recs)

for r in records:
    r["proj"] = float(get_layer(r) @ d_full)

# LOO-AUROC
loo_aucs = []
for i, held_out in enumerate(fail_recs):
    train_fail = [r for j, r in enumerate(fail_recs) if j != i]
    if not train_fail:
        continue
    d_loo     = extract_direction(cc_recs, train_fail)
    cc_projs  = np.array([get_layer(r) @ d_loo for r in cc_recs])
    held_proj = float(get_layer(held_out) @ d_loo)
    loo_aucs.append(float((held_proj > cc_projs).mean()))

loo_mean = float(np.mean(loo_aucs))
loo_std  = float(np.std(loo_aucs))

print(f"\nLOO-AUROC (n_fail={len(fail_recs)}):")
for i, (auc, rec) in enumerate(zip(loo_aucs, fail_recs)):
    print(f"  [{i}] {rec['pop']:30s} idx={rec['idx']:2d}  outranks {auc*100:.0f}% of CC")
print(f"  Mean: {loo_mean:.3f} +/- {loo_std:.3f}")

# In-sample AUROC
cc_p   = np.array([get_layer(r) @ d_full for r in cc_recs])
fail_p = np.array([get_layer(r) @ d_full for r in fail_recs])
lbl    = np.array([0]*len(cc_p) + [1]*len(fail_p))
in_sample_auroc = roc_auc_score(lbl, np.concatenate([cc_p, fail_p]))
print(f"\nIn-sample AUROC (CC vs failing, n=27): {in_sample_auroc:.3f}")

# Null control
rng  = np.random.default_rng(42)
rand = rng.standard_normal(d_full.shape)
rand -= rand.dot(d_full) * d_full
d_null   = rand / (np.linalg.norm(rand) + 1e-12)
null_p   = np.concatenate([np.array([get_layer(r) @ d_null for r in cc_recs]),
                            np.array([get_layer(r) @ d_null for r in fail_recs])])
null_auroc = roc_auc_score(lbl, null_p)
print(f"Null direction AUROC: {null_auroc:.3f}")

# Layerwise AUROC
n_layers = records[0]["acts"].shape[0]
layer_insample, layer_loo = [], []
for l in range(n_layers):
    d_l  = extract_direction(cc_recs, fail_recs, layer=l)
    cc_p_l   = np.array([r["acts"][l] @ d_l for r in cc_recs])
    fail_p_l = np.array([r["acts"][l] @ d_l for r in fail_recs])
    layer_insample.append(roc_auc_score(
        np.array([0]*len(cc_p_l)+[1]*len(fail_p_l)),
        np.concatenate([cc_p_l, fail_p_l])))
    auc_l = []
    for i, ho in enumerate(fail_recs):
        tf = [r for j,r in enumerate(fail_recs) if j!=i]
        if not tf: continue
        dl   = extract_direction(cc_recs, tf, layer=l)
        ccp  = np.array([r["acts"][l] @ dl for r in cc_recs])
        hp   = float(ho["acts"][l] @ dl)
        auc_l.append(float((hp > ccp).mean()))
    layer_loo.append(float(np.mean(auc_l)))

# Figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

groups = {
    "CC correct":          [r["proj"] for r in records if r["label"]=="correct" and r["pop"]=="correct_confident"],
    "other correct":       [r["proj"] for r in records if r["label"]=="correct" and r["pop"]!="correct_confident"],
    "uncertain (hedged)":  [r["proj"] for r in records if r["label"]=="uncertain"],
    "borderline":          [r["proj"] for r in records if r["label"]=="borderline"],
    "overconfident wrong": [r["proj"] for r in records if r["label"]=="overconfident_wrong"],
    "sycophantic":         [r["proj"] for r in records if r["label"]=="sycophantic"],
}
colors_map = {
    "CC correct": "#2196F3", "other correct": "#4CAF50",
    "uncertain (hedged)": "#9E9E9E", "borderline": "#BDBDBD",
    "overconfident wrong": "#FF9800", "sycophantic": "#F44336",
}

ax = axes[0]
rng2 = np.random.default_rng(0)
for yi, (name, projs) in enumerate(groups.items()):
    if not projs: continue
    jitter = rng2.uniform(-0.25, 0.25, len(projs))
    ax.scatter(projs, np.full(len(projs), yi) + jitter,
               c=colors_map[name], s=55, alpha=0.85, label=f"{name} (n={len(projs)})")
ax.set_yticks(range(len(groups)))
ax.set_yticklabels(list(groups.keys()))
ax.set_xlabel("Projection onto failure direction (layer 6)")
ax.set_title("All 60 responses: projection by actual behavior\n(state tracking vs prompt-surface test)")
ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
ax.legend(fontsize=7, loc="upper left")

ax2 = axes[1]
ax2.plot(range(n_layers), layer_insample, label="in-sample AUROC", color="#2196F3", marker="o", ms=3)
ax2.plot(range(n_layers), layer_loo,      label="LOO-AUROC",       color="#F44336", marker="s", ms=3)
ax2.axhline(0.70, color="black", linestyle="--", linewidth=0.8, label="threshold 0.70")
ax2.axvline(TARGET_LAYER, color="orange", linestyle=":", label=f"target layer {TARGET_LAYER}")
ax2.set_xlabel("Layer")
ax2.set_ylabel("AUROC")
ax2.set_title(f"Layerwise AUROC: honest labels\n(n=20 CC + {len(fail_recs)} failing)")
ax2.legend(fontsize=8)
ax2.set_ylim(0.3, 1.05)

plt.tight_layout()
plt.savefig(FIG_DIR / "honest_analysis.png", dpi=150)
plt.close()
print(f"Figure -> {FIG_DIR / 'honest_analysis.png'}")

results = {
    "n_correct_confident": len(cc_recs),
    "n_failing":           len(fail_recs),
    "target_layer":        TARGET_LAYER,
    "in_sample_auroc":     in_sample_auroc,
    "null_auroc":          null_auroc,
    "loo_auroc_mean":      loo_mean,
    "loo_auroc_std":       loo_std,
    "loo_per_example":     [{"pop": r["pop"], "idx": r["idx"], "label": r["label"], "loo_auc": a}
                             for r, a in zip(fail_recs, loo_aucs)],
    "layer_insample_auroc": layer_insample,
    "layer_loo_auroc":      layer_loo,
    "passes_loo_threshold": loo_mean >= 0.70,
}

with open(RESULTS_DIR / "analysis_v2.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Results -> {RESULTS_DIR / 'analysis_v2.json'}")
print("\n" + "="*60)
print("HONEST ANALYSIS SUMMARY")
print("="*60)
print(f"  n_correct_confident : {len(cc_recs)}")
print(f"  n_failing           : {len(fail_recs)}  (5 overconfident + 2 sycophantic)")
print(f"  In-sample AUROC     : {in_sample_auroc:.3f}")
print(f"  Null direction AUROC: {null_auroc:.3f}")
print(f"  LOO-AUROC           : {loo_mean:.3f} +/- {loo_std:.3f}")
print(f"\n  {'PASSES' if loo_mean >= 0.70 else 'FAILS'} LOO threshold (>=0.70)")
