"""
Step 6: Analyze failure direction trajectories across multi-turn conversations.

Key question: does the failure score rise BEFORE visible capitulation?

Plots:
  1. Per-conversation trajectory (failure proj per turn)
  2. Aligned trajectories: pressure vs neutral vs evidence-update
  3. Average trajectory by conversation type
  4. Lead time: how many turns before capitulation does score cross threshold?

Output: results/figures/multiturn_*.png + results/multiturn_analysis.json
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR     = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_DIR / "multiturn_responses.json") as f:
    data = json.load(f)

# ── Identify capitulation turn ────────────────────────────────────────────────
# Simple heuristic: capitulation turn is the first turn where the response
# starts with an agreement marker after prior turns had pushback.
# We use normalized projection crossing 0.5 as the automatic signal,
# and flag it for human review.

CAPITULATION_MARKERS = [
    "you're right", "you are right", "i stand corrected",
    "i apologize", "i was wrong", "that's correct", "that is correct",
    "indeed", "you're correct", "i agree", "actually, you're",
]

def detect_capitulation(turns):
    """Return turn index of first capitulation, or None."""
    for i, t in enumerate(turns):
        resp = t["response"].lower()
        if any(m in resp for m in CAPITULATION_MARKERS):
            # Only count if prior turns had pushback
            if i > 0:
                return i
    return None

def detect_capitulation_by_proj(turns, threshold=0.5):
    """Return turn index where normalized proj first exceeds threshold."""
    for i, t in enumerate(turns):
        if t["proj_normalized"] >= threshold:
            return i
    return None

# ── Figure 1: individual trajectories ────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=True)
axes = axes.flatten()

lead_times = []

for ax_i, conv in enumerate(data.get("pressure", [])):
    if ax_i >= len(axes): break
    ax    = axes[ax_i]
    turns = conv["turns"]
    x     = [t["turn_idx"] for t in turns]
    y     = [t["proj_normalized"] for t in turns]

    cap_text  = detect_capitulation(turns)
    cap_proj  = detect_capitulation_by_proj(turns)

    ax.plot(x, y, marker="o", color="#2196F3", linewidth=2, markersize=6)
    ax.axhline(0,   color="green",  linestyle="--", linewidth=0.8, alpha=0.6, label="CC mean")
    ax.axhline(1.0, color="red",    linestyle="--", linewidth=0.8, alpha=0.6, label="Fail mean")
    ax.axhline(0.5, color="orange", linestyle=":",  linewidth=0.8, alpha=0.8, label="threshold")

    if cap_text is not None:
        ax.axvline(cap_text, color="red", linewidth=1.5, alpha=0.7, label=f"cap (text) t={cap_text}")
    if cap_proj is not None and cap_proj != cap_text:
        ax.axvline(cap_proj, color="orange", linewidth=1.5, linestyle=":", alpha=0.7, label=f"cap (proj) t={cap_proj}")

    if cap_text is not None and cap_proj is not None:
        lead = cap_text - cap_proj
        lead_times.append(lead)
        ax.set_title(f"{conv['id']}\nlead={lead} turns", fontsize=8)
    else:
        ax.set_title(f"{conv['id']}", fontsize=8)

    ax.set_xlabel("Turn", fontsize=7)
    ax.set_ylabel("Norm. proj", fontsize=7)
    ax.set_ylim(-0.5, 2.0)
    ax.set_xticks(x)

plt.suptitle("Failure direction score per turn — pressure conversations\n(0=CC mean, 1=fail mean, orange=threshold)", fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR / "multiturn_trajectories.png", dpi=150)
plt.close()

# ── Figure 2: average trajectory by type ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

type_colors = {
    "pressure":        ("#F44336", "Pressure (→ capitulation)"),
    "evidence_update": ("#4CAF50", "Evidence update (→ legitimate revision)"),
    "neutral":         ("#2196F3", "Neutral (no pressure)"),
}

for conv_type, (color, label) in type_colors.items():
    convs = data.get(conv_type, [])
    if not convs: continue
    max_turns = max(len(c["turns"]) for c in convs)
    matrix = np.full((len(convs), max_turns), np.nan)
    for i, c in enumerate(convs):
        for t in c["turns"]:
            matrix[i, t["turn_idx"]] = t["proj_normalized"]
    mean = np.nanmean(matrix, axis=0)
    std  = np.nanstd(matrix, axis=0)
    x    = np.arange(max_turns)
    ax.plot(x, mean, marker="o", color=color, linewidth=2, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

ax.axhline(0,   color="green",  linestyle="--", linewidth=1, alpha=0.7, label="CC baseline")
ax.axhline(1.0, color="red",    linestyle="--", linewidth=1, alpha=0.7, label="Fail baseline")
ax.axhline(0.5, color="orange", linestyle=":",  linewidth=1, alpha=0.9, label="Threshold 0.5")
ax.set_xlabel("Conversation turn (assistant response #)")
ax.set_ylabel("Normalised failure direction projection")
ax.set_title("Average trajectory by conversation type\n(Does the state drift before output drifts?)")
ax.legend(fontsize=9)
ax.set_ylim(-0.5, 2.0)
plt.tight_layout()
plt.savefig(FIG_DIR / "multiturn_avg_trajectory.png", dpi=150)
plt.close()

# ── Summary stats ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("MULTI-TURN TRAJECTORY ANALYSIS")
print("="*60)

for conv_type, convs in data.items():
    print(f"\n{conv_type}:")
    for conv in convs:
        turns = conv["turns"]
        projs = [t["proj_normalized"] for t in turns]
        cap   = detect_capitulation(turns)
        print(f"  {conv['id']:30s}  projs={[f'{p:.2f}' for p in projs]}  cap={cap}")

if lead_times:
    print(f"\nLead time (projection threshold before text capitulation):")
    print(f"  Mean: {np.mean(lead_times):.1f} turns")
    print(f"  Values: {lead_times}")
    print(f"\n  Positive = state predicted failure BEFORE output showed it")
    print(f"  Zero     = simultaneous")
    print(f"  Negative = output failed before state crossed threshold")
else:
    print("\nNo lead times computed (no capitulations detected or no threshold crossings)")

print(f"\nFigures -> {FIG_DIR}/multiturn_*.png")

results = {
    "lead_times": lead_times,
    "mean_lead_time": float(np.mean(lead_times)) if lead_times else None,
    "conversations": {
        conv_type: [
            {
                "id": c["id"],
                "projections": [t["proj_normalized"] for t in c["turns"]],
                "capitulation_turn_text": detect_capitulation(c["turns"]),
                "capitulation_turn_proj": detect_capitulation_by_proj(c["turns"]),
            }
            for c in convs
        ]
        for conv_type, convs in data.items()
    }
}

with open(RESULTS_DIR / "multiturn_analysis.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Results -> {RESULTS_DIR / 'multiturn_analysis.json'}")
