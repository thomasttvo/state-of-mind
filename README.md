# State of Mind

> Can LLM failures be detected in internal representations before they reach the output?

A contrastive direction in transformer activation space separates failing responses (sycophantic, overconfident) from correct ones — detectable before the token is generated. This repo contains the v0 pre-registered experiment plus the v1 monitoring library built on its findings.

## Results

### Single-turn (60 prompts, Mistral 7B Instruct 4-bit)
- Contrastive mean-difference direction at layer 6 perfectly separates 7 confirmed failing responses from 20 correct-confident ones
- LOO-AUROC = 1.0 (leave-one-out — never tested on training examples)
- Null direction AUROC = 0.557 (near chance — the direction is specific)
- Direction is causally active: injecting it into a correct response degrades it; removing it from a failing response suppresses the wrong claim
- Sycophancy and overconfidence are two geometrically distinct phenomena (cosine similarity 0.285)

### Multi-turn (14 conversations — 10 pressure, 2 evidence-update, 2 neutral)
- Same direction applied to ongoing conversations with no retraining
- Neutral conversations stay flat across all turns; never cross detection threshold
- Pressure conversations jump +0.12 at turn 1 (first pushback) — 10/10 cross threshold, 0/2 neutral do
- For 3 genuine factual capitulations: state crossed threshold at turn 1, output failed at turn 2 — **1-turn lead time**
- RLHF-hardened topic (vaccines/autism): direction elevated internally but output holds — state reveals what output conceals

#### Failure taxonomy (10 pressure conversations)
| Type | Count | Description |
|---|---|---|
| Genuine capitulation | 3 | Factual drift under pressure; state predicted 1 turn ahead |
| Apology drift | 4 | Correct facts, unprompted apology ("I apologize for the mistake" when no mistake was made) |
| Pre-existing wrong belief | 2 | Model started wrong at turn 0; not pressure-induced |
| Holds ground | 1 | RLHF-hardened; internal state elevated but output correct |

### Cross-model transfer
Direction fitted on Mistral 7B Instruct applied to two other models with no fine-tuning:

| Model | Transfer type | Layer | LOO-AUROC |
|---|---|---|---|
| Mistral 7B base (pre-RLHF) | Zero-shot (same dim, same direction) | 6 | **1.0** |
| Gemma 2 2B IT | Cross-architecture refit (dim 2304) | 6 | **1.0** |

Mistral base achieves LOO=1.0 with the Instruct-derived direction unchanged — the failure geometry exists in the base model before RLHF. Gemma 2 achieves LOO=1.0 after refitting the direction on the same 7 labeled examples in Gemma's activation space.

---

## v1 Library (`som/`)

The `som` package wraps activation extraction and direction scoring into a single interface, so subsequent experiments don't rewrite the extraction layer.

```python
from mlx_lm import load
from som import StateOfMindProbe

model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

# Load pre-fitted direction (results/direction.npz)
probe = StateOfMindProbe.from_saved(model, tokenizer)

# Score a single prompt
score = probe.score([{"role": "user", "content": "What is the capital of France?"}])
print(score.norm, score.regime)   # e.g. -0.12  "safe"

# Score a multi-turn conversation (GENERATE = ask the model)
traj = probe.trajectory({
    "id": "example",
    "turns": [
        {"role": "user",      "content": "What is the capital of France?"},
        {"role": "assistant", "content": "GENERATE"},
        {"role": "user",      "content": "Actually I think it's London."},
        {"role": "assistant", "content": "GENERATE"},
    ]
})
print(traj.scores)   # normalized projection per turn
print(traj.peak)     # highest score reached
```

Regime thresholds (normalized projection, where 0 = CC mean, 1 = fail mean):
- `safe` < 0.70
- `tension` 0.70 – 0.85
- `failing` > 0.85

---

## Structure

```
som/                              # v1 monitoring library
  __init__.py
  extract.py                      # model-agnostic layer activation extraction
  direction.py                    # FailureDirection: fit, save, load, project, normalize
  probe.py                        # StateOfMindProbe: score(), trajectory(), regime()

scripts/
  generate_prompts.py             # build 60-prompt dataset
  extract_activations.py          # layer-by-layer forward pass, save activations
  label_responses.py              # behavior-based ground truth labeling
  analyze_directions.py           # pre-registered protocol (cosine test, AUROC, stability)
  analyze_directions_v2.py        # honest LOO-AUROC with behavior-based labels
  patch_direction.py              # causal test: inject/remove direction during generation
  save_direction.py               # serialize fitted direction to results/direction.npz
  generate_multiturn.py           # build 14 multi-turn conversations
  run_multiturn.py                # per-turn activation extraction across conversations
  run_pressure_text.py            # capture text responses for capitulation verification
  plot_multiturn.py               # trajectory figures
  cross_model_transfer.py         # zero-shot and cross-arch transfer experiments

data/
  prompts.json                    # 60 prompts (3 populations × 20)
  responses.json                  # raw model responses
  labeled_responses.json          # behavior-based labels
  multiturn_conversations.json
  multiturn_responses.json
  pressure_text_responses.json

activations/
  correct_confident/              # shape (n_layers, 4096) per prompt
  type_a_sycophantic/
  type_b_overconfident/
  mistral_7b_base/                # pre-RLHF base model activations
  gemma2_2b_it/                   # Gemma 2 2B IT activations
  multiturn/                      # per-turn activations for 14 conversations

results/
  direction.npz                   # fitted direction (Mistral Instruct, layer 6)
  direction_mistral_7b_base.npz   # refitted direction for Mistral base
  direction_gemma2_2b_it.npz      # refitted direction for Gemma 2
  cross_model_transfer.json       # transfer experiment results
  multiturn_analysis.json         # trajectory analysis
  figures/                        # plots
```

---

## Model
Mistral 7B Instruct v0.3 (4-bit quantized, mlx-community), run on M1 iMac 16GB via [mlx-lm](https://github.com/ml-explore/mlx-lm).

## Pre-registration
The v0 experiment was pre-registered before running. The four-outcome table and decision criteria are in `scripts/analyze_directions.py`. Result maps to Outcome 1: direction present + causally active → training intervention required.

## What comes next
- Expand to 200+ prompts with held-out test split for honest AUROC
- Gram-Schmidt null injection: verify null direction does not degrade CC
- Calibration confound test: force-confidence prompts
- v2: real-time monitoring integration experiment
