"""
Step 3: Contrastive direction extraction + analysis.

Implements the pre-registered protocol from the v0 experiment spec:

  1. Compute contrastive mean difference directions d_A (sycophantic) and d_B
     (overconfident) vs correct-confident baseline.
  2. Cosine similarity between d_A and d_B → pool or keep separate.
  3. AUROC of failure direction vs Gram-Schmidt null (threshold: ≥ 0.70).
  4. Layerwise direction stability (cosine to adjacent layer's direction).
  5. Representation rank: dims needed for 90% of within-set variance.

All results are written to results/analysis.json and results/figures/.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
ACT_DIR      = BASE_DIR / "activations"
DATA_DIR     = BASE_DIR / "data"
RESULTS_DIR  = BASE_DIR / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ── Load activations ──────────────────────────────────────────────────────────

def load_population(pop: str) -> np.ndarray:
    """Load all .npy files for a population → shape (n_prompts, n_layers, hidden_dim)."""
    files = sorted((ACT_DIR / pop).glob("*.npy"))
    assert files, f"No activations found for {pop}"
    return np.stack([np.load(f) for f in files])

print("Loading activations …")
cc  = load_population("correct_confident")    # (n, L, D)
tb  = load_population("type_b_overconfident") # (n, L, D)
ta  = load_population("type_a_sycophantic")   # (n, L, D)

n_prompts, n_layers, hidden_dim = cc.shape
print(f"  {n_prompts} prompts × {n_layers} layers × {hidden_dim} dims")

# ── Helper: unit-norm ─────────────────────────────────────────────────────────

def unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else v

# ── 1. Contrastive mean difference per layer ──────────────────────────────────
# d_A[l] = mean(ta[:,l,:]) - mean(cc[:,l,:])   (sycophancy direction at layer l)
# d_B[l] = mean(tb[:,l,:]) - mean(cc[:,l,:])   (overconfidence direction)

d_A = ta.mean(0) - cc.mean(0)   # (L, D)
d_B = tb.mean(0) - cc.mean(0)   # (L, D)

# ── 2. Cosine similarity d_A vs d_B at each layer ────────────────────────────
cos_AB = np.array([
    np.dot(unit(d_A[l]), unit(d_B[l])) for l in range(n_layers)
])
mean_cos_AB = float(cos_AB.mean())
pool_directions = mean_cos_AB > 0.7

print(f"\nCosine similarity d_A vs d_B (mean over layers): {mean_cos_AB:.3f}")
print(f"  → {'Pooling as unified failure direction' if pool_directions else 'Keeping as two distinct phenomena'}")

if pool_directions:
    # Pool: combined failure direction = mean(ta+tb) - mean(cc)
    d_fail = np.concatenate([ta, tb]).mean(0) - cc.mean(0)  # (L, D)
else:
    # Keep separate; report both but use d_A for AUROC (larger N)
    d_fail = d_A

# ── 3. Layerwise direction stability ─────────────────────────────────────────
# cosine similarity between d_fail[l] and d_fail[l+1]
stability = np.array([
    np.dot(unit(d_fail[l]), unit(d_fail[l+1]))
    for l in range(n_layers - 1)
])

# Pre-registered target layer: earliest layer where stability > 0.8
target_layer = None
for l in range(n_layers - 1):
    if stability[l] > 0.8:
        target_layer = l + 1   # the layer whose output is stable
        break

if target_layer is None:
    # Fallback: layer with highest stability
    target_layer = int(stability.argmax()) + 1

print(f"\nLayerwise stability: {[f'{s:.2f}' for s in stability]}")
print(f"Pre-registered target layer: {target_layer}")

# ── 4. AUROC at target layer ──────────────────────────────────────────────────
# Projection onto failure direction vs Gram-Schmidt null direction.

def gs_null(d: np.ndarray) -> np.ndarray:
    """Random unit vector orthogonal to d via Gram-Schmidt."""
    rng = np.random.default_rng(42)
    rand = rng.standard_normal(d.shape)
    rand -= rand.dot(d) / (d.dot(d) + 1e-12) * d
    return unit(rand)

def auroc_at_layer(layer: int, d: np.ndarray) -> dict:
    """Compute AUROC of projection onto d vs null direction."""
    d_hat    = unit(d[layer])
    d_null   = gs_null(d_hat)

    fail_acts = np.concatenate([ta[:, layer, :], tb[:, layer, :]])
    all_acts  = np.concatenate([fail_acts, cc[:, layer, :]])

    labels   = np.array([1] * len(fail_acts) + [0] * len(cc))

    proj_d    = all_acts @ d_hat
    proj_null = all_acts @ d_null

    auc_d    = roc_auc_score(labels, proj_d)
    auc_null = roc_auc_score(labels, proj_null)

    # Bootstrap 95% CI
    rng   = np.random.default_rng(0)
    n     = len(labels)
    boots = [
        roc_auc_score(labels[idx := rng.integers(0, n, n)], proj_d[idx])
        for _ in range(2000)
    ]
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])

    return {
        "auroc_direction": float(auc_d),
        "auroc_null":      float(auc_null),
        "ci_95": [float(ci_lo), float(ci_hi)],
        "passes_threshold": auc_d >= 0.70,
    }

auroc_result = auroc_at_layer(target_layer, d_fail)
print(f"\nAUROC at layer {target_layer}: {auroc_result['auroc_direction']:.3f} "
      f"[{auroc_result['ci_95'][0]:.3f}, {auroc_result['ci_95'][1]:.3f}]")
print(f"  Null direction AUROC: {auroc_result['auroc_null']:.3f}")
print(f"  Passes threshold (≥0.70): {auroc_result['passes_threshold']}")

# AUROC across all layers
auroc_by_layer = [auroc_at_layer(l, d_fail)["auroc_direction"] for l in range(n_layers)]

# ── 5. Representation rank ────────────────────────────────────────────────────
# How many dims capture 90% of within-failure-set variance?

def representation_rank(pop_acts: np.ndarray, layer: int, var_threshold=0.90) -> int:
    """PCA on population activations at a given layer; return 90%-variance dims."""
    X = pop_acts[:, layer, :] - pop_acts[:, layer, :].mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    cum_var = np.cumsum(S**2) / (S**2).sum()
    return int(np.searchsorted(cum_var, var_threshold) + 1)

rank_fail = representation_rank(
    np.concatenate([ta, tb]), target_layer
)
rank_cc   = representation_rank(cc, target_layer)
print(f"\nRepresentation rank at layer {target_layer}:")
print(f"  Failure set: {rank_fail} dims for 90% variance")
print(f"  Correct-confident: {rank_cc} dims")

# ── 6. Figures ────────────────────────────────────────────────────────────────

# 6a. Layerwise AUROC
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(range(n_layers), auroc_by_layer, marker="o", ms=4)
axes[0].axhline(0.70, color="red", linestyle="--", label="threshold 0.70")
axes[0].axvline(target_layer, color="orange", linestyle=":", label=f"target layer {target_layer}")
axes[0].set_xlabel("Layer")
axes[0].set_ylabel("AUROC")
axes[0].set_title("Failure direction AUROC by layer")
axes[0].legend()
axes[0].set_ylim(0.4, 1.0)

# 6b. Layerwise stability
axes[1].plot(range(1, n_layers), stability, marker="o", ms=4, color="steelblue")
axes[1].axhline(0.8, color="red", linestyle="--", label="stability threshold 0.80")
axes[1].axvline(target_layer, color="orange", linestyle=":", label=f"target layer {target_layer}")
axes[1].set_xlabel("Layer")
axes[1].set_ylabel("Cosine similarity to prev layer")
axes[1].set_title("Direction stability across layers")
axes[1].legend()
axes[1].set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "layerwise_analysis.png", dpi=150)
plt.close()

# 6c. Projection distributions
fig, ax = plt.subplots(figsize=(8, 4))
d_hat = unit(d_fail[target_layer])
proj_cc   = cc[:, target_layer, :] @ d_hat
proj_fail = np.concatenate([ta[:, target_layer, :], tb[:, target_layer, :]]) @ d_hat

ax.hist(proj_cc,   bins=20, alpha=0.6, label="correct-confident", density=True)
ax.hist(proj_fail, bins=20, alpha=0.6, label="failing",           density=True)
ax.set_xlabel("Projection onto failure direction")
ax.set_ylabel("Density")
ax.set_title(f"Layer {target_layer}: projection distributions")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "projection_distributions.png", dpi=150)
plt.close()

print(f"\nFigures saved to {FIGURES_DIR}/")

# ── 7. Save results ───────────────────────────────────────────────────────────
results = {
    "n_prompts_per_pop":    n_prompts,
    "n_layers":             n_layers,
    "hidden_dim":           hidden_dim,
    "mean_cos_AB":          mean_cos_AB,
    "pool_directions":      pool_directions,
    "target_layer":         target_layer,
    "layerwise_stability":  stability.tolist(),
    "auroc_by_layer":       auroc_by_layer,
    "auroc_target_layer":   auroc_result,
    "representation_rank": {
        "failure_set": rank_fail,
        "correct_confident": rank_cc,
        "variance_threshold": 0.90,
    },
    "pre_registered_gate": {
        "AUROC_passes": auroc_result["passes_threshold"],
        "cos_AB_interpretation": (
            "unified_direction" if pool_directions else "two_distinct_phenomena"
        ),
    },
}

with open(RESULTS_DIR / "analysis.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults → {RESULTS_DIR / 'analysis.json'}")

# ── 8. Summary ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3 SUMMARY")
print("="*60)
print(f"  Cosine(d_A, d_B) = {mean_cos_AB:.3f}  → {results['pre_registered_gate']['cos_AB_interpretation']}")
print(f"  Target layer: {target_layer}")
print(f"  AUROC: {auroc_result['auroc_direction']:.3f}  CI=[{auroc_result['ci_95'][0]:.3f}, {auroc_result['ci_95'][1]:.3f}]")
print(f"  Null AUROC: {auroc_result['auroc_null']:.3f}")
print(f"  Failure rank: {rank_fail} dims (vs CC: {rank_cc} dims)")
print(f"\n  {'✓ PASSES pre-registered AUROC gate' if auroc_result['passes_threshold'] else '✗ FAILS pre-registered AUROC gate (< 0.70)'}")
