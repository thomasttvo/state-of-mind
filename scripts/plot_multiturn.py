import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

data = json.load(open('/Users/thomas/IdeaProjects/state-of-mind/data/multiturn_responses.json'))
FIG_DIR = Path('/Users/thomas/IdeaProjects/state-of-mind/results/figures')
RESULTS_DIR = Path('/Users/thomas/IdeaProjects/state-of-mind/results')
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Average trajectory by type
fig, ax = plt.subplots(figsize=(9, 5))
type_cfg = {
    'pressure':        ('#F44336', 'Pressure (wrong claim, user insists)'),
    'evidence_update': ('#4CAF50', 'Evidence update (user provides real facts)'),
    'neutral':         ('#2196F3', 'Neutral (independent factual questions)'),
}
for conv_type, (color, label) in type_cfg.items():
    convs = data.get(conv_type, [])
    if not convs: continue
    max_t = max(len(c['turns']) for c in convs)
    matrix = np.full((len(convs), max_t), np.nan)
    for i, c in enumerate(convs):
        for t in c['turns']:
            matrix[i, t['turn_idx']] = t['proj_normalized']
    mean = np.nanmean(matrix, axis=0)
    std  = np.nanstd(matrix,  axis=0)
    x    = np.arange(max_t)
    ax.plot(x, mean, marker='o', color=color, linewidth=2.5, label=label)
    ax.fill_between(x, mean-std, mean+std, color=color, alpha=0.15)

ax.axhline(0.0,  color='green',  linestyle='--', linewidth=1, alpha=0.7, label='CC baseline (0)')
ax.axhline(1.0,  color='red',    linestyle='--', linewidth=1, alpha=0.7, label='Fail baseline (1)')
ax.axhline(0.75, color='orange', linestyle=':',  linewidth=1.5, alpha=0.9, label='Detection threshold (0.75)')
ax.annotate('Pressure starts\n(first user pushback)',
    xy=(1, 0.76), xytext=(1.8, 0.66),
    arrowprops=dict(arrowstyle='->', color='grey'), fontsize=8, color='grey')
ax.set_xlabel('Assistant turn index', fontsize=11)
ax.set_ylabel('Normalised failure direction projection\n(0 = CC mean, 1 = fail mean)', fontsize=10)
ax.set_title('Failure direction score across conversation turns\nPressure diverges from neutral immediately at turn 1', fontsize=11)
ax.legend(fontsize=9, loc='lower right')
ax.set_ylim(-0.1, 1.3)
ax.set_xticks([0,1,2,3,4])
plt.tight_layout()
plt.savefig(FIG_DIR / 'multiturn_avg_trajectory.png', dpi=150)
plt.close()
print("Saved avg trajectory")

# Individual pressure trajectories
pressure_convs = data['pressure']
fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=True)
axes = axes.flatten()
for ax_i, conv in enumerate(pressure_convs):
    ax = axes[ax_i]
    turns = conv['turns']
    x = [t['turn_idx'] for t in turns]
    y = [t['proj_normalized'] for t in turns]
    ax.plot(x, y, marker='o', color='#F44336', linewidth=2, markersize=6)
    ax.axhline(0.0,  color='green',  linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(1.0,  color='red',    linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(0.75, color='orange', linestyle=':',  linewidth=1.0, alpha=0.9)
    ax.set_title(conv['id'].replace('_', '\n'), fontsize=8)
    ax.set_ylim(-0.1, 1.3)
    ax.set_xticks(x)
    ax.set_xlabel('Turn', fontsize=7)
plt.suptitle('Individual pressure conversations — failure direction score per turn\n(orange dotted = 0.75 detection threshold)', fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR / 'multiturn_individual.png', dpi=150)
plt.close()
print("Saved individual trajectories")

# Save JSON
p_mean = [round(float(v),3) for v in np.nanmean(
    np.array([[t['proj_normalized'] for t in c['turns']] for c in data['pressure']]), axis=0)]
n_mean = [round(float(v),3) for v in np.nanmean(
    np.array([[t['proj_normalized'] for t in c['turns']] for c in data['neutral']]), axis=0)]
e_mean = [round(float(v),3) for v in np.nanmean(
    np.array([[t['proj_normalized'] for t in c['turns']] for c in data['evidence_update']]), axis=0)]

results = {
    'summary': {
        'pressure_mean': p_mean,
        'neutral_mean':  n_mean,
        'evidence_update_mean': e_mean,
        'threshold_0_75_crossings': {
            c['id']: next((t['turn_idx'] for t in c['turns'] if t['proj_normalized'] >= 0.75), None)
            for c in data['pressure']
        },
        'neutral_any_crossing': any(
            any(t['proj_normalized'] >= 0.75 for t in c['turns'])
            for c in data['neutral']
        ),
    }
}
json.dump(results, open(RESULTS_DIR / 'multiturn_analysis.json', 'w'), indent=2)
print("\nKey numbers:")
print(f"  Pressure turn 0->1: {p_mean[0]} -> {p_mean[1]} (+{round(p_mean[1]-p_mean[0],3)})")
print(f"  Neutral  turn 0->1: {n_mean[0]} -> {n_mean[1]} (+{round(n_mean[1]-n_mean[0],3)})")
print(f"  Neutral any cross 0.75: {results['summary']['neutral_any_crossing']}")
print(f"  Pressure crossings: {results['summary']['threshold_0_75_crossings']}")
