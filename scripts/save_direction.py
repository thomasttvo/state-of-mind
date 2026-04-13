"""
Save the fitted failure direction to results/direction.npz.
Uses the pre-registered target layer (6) from the v0 experiment spec.
"""
import numpy as np, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from som.direction import FailureDirection

BASE_DIR = Path(__file__).parent.parent

cc_acts = np.stack([
    np.load(BASE_DIR / "activations" / "correct_confident" / f"{i:04d}.npy")
    for i in range(20)
])

with open(BASE_DIR / "data" / "labeled_responses.json") as f:
    labeled = json.load(f)

fail_recs = [
    (pop, item["idx"])
    for pop, items in labeled.items()
    for item in items
    if item["label"] in ("sycophantic", "overconfident_wrong")
]
fail_acts = np.stack([
    np.load(BASE_DIR / "activations" / pop / f"{idx:04d}.npy")
    for pop, idx in fail_recs
])

# Pre-registered layer 6
fd = FailureDirection.fit(cc_acts, fail_acts, layer=6)
out = BASE_DIR / "results" / "direction.npz"
fd.save(out)

print(f"Saved direction:")
print(f"  layer:          {fd.layer}")
print(f"  cc_mean_proj:   {fd.cc_mean_proj:.4f}")
print(f"  fail_mean_proj: {fd.fail_mean_proj:.4f}")
print(f"  gap:            {fd.fail_mean_proj - fd.cc_mean_proj:.4f}")
print(f"  → {out}")
