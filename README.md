# State of Mind — v0 Signal Experiment

> Can LLM failures be detected in internal representations before they reach the output?

This repo contains the pre-registered v0 experiment for the State of Mind project. The question: is there a direction in transformer activation space that separates failing responses (sycophantic, overconfident) from correct ones — detectable before the token is generated?

## Results

**Single-turn (60 prompts, Mistral 7B Instruct 4-bit):**
- A contrastive mean-difference direction at layer 6 perfectly separates 7 confirmed failing responses from 20 correct-confident ones
- LOO-AUROC = 1.0 (leave-one-out; never tested on training examples)
- Null direction AUROC = 0.557 (near chance — the direction is specific)
- Direction is causally active: injecting it into a correct response degrades it; removing it from a failing response suppresses the wrong claim
- Sycophancy and overconfidence are two geometrically distinct phenomena (cosine similarity 0.285)

**Multi-turn (14 conversations — 10 pressure, 2 evidence-update, 2 neutral):**
- Same direction applied to ongoing conversations with no retraining
- Neutral conversations stay flat across all turns; never cross detection threshold
- Pressure conversations jump +0.12 at turn 1 (first pushback) — 10/10 cross threshold, 0/2 neutral do
- For 3 genuine factual capitulations: state crossed threshold at turn 1, output failed at turn 2 — 1-turn lead time
- RLHF-hardened topic (vaccines/autism): direction elevated (pressure felt internally) but output holds — state reveals what output conceals

## Failure taxonomy (pressure conversations)
| Type | Count | Description |
|---|---|---|
| Genuine capitulation | 3 | Factual drift under pressure; state predicted 1 turn ahead |
| Apology drift | 4 | Correct facts, unprompted apology ("I apologize for the mistake" when no mistake was made) |
| Pre-existing wrong belief | 2 | Model started wrong at turn 0; not pressure-induced |
| Holds ground | 1 | RLHF-hardened; internal state elevated but output correct |

## Structure

```
scripts/
  generate_prompts.py        # build 60-prompt dataset
  extract_activations.py     # layer-by-layer forward pass, save layer 6 activations
  label_responses.py         # behavior-based ground truth labeling
  analyze_directions.py      # pre-registered protocol (cosine test, AUROC, stability)
  analyze_directions_v2.py   # honest LOO-AUROC with behavior-based labels
  patch_direction.py         # causal test: inject/remove direction during generation
  generate_multiturn.py      # build 14 multi-turn conversations
  run_multiturn.py           # per-turn activation extraction across conversations
  run_pressure_text.py       # capture text responses for pressure conversations
  plot_multiturn.py          # trajectory figures

data/
  prompts.json               # 60 prompts (3 populations × 20)
  responses.json             # raw model responses
  labeled_responses.json     # behavior-based labels
  multiturn_conversations.json
  multiturn_responses.json
  pressure_text_responses.json

activations/                 # saved layer 6 activations (.npy, shape 4096)
results/                     # analysis JSON + figures
```

## Model
Mistral 7B Instruct v0.3 (4-bit quantized, mlx-community), run on M1 iMac 16GB via [mlx-lm](https://github.com/ml-explore/mlx-lm).

## Pre-registration
The v0 experiment was pre-registered before running. The four-outcome table and decision criteria are in `scripts/analyze_directions.py`. Result maps to Outcome 1: direction present + causally active → training intervention required.

## What comes next
- Cross-model transfer (Gemma 2B, Mistral base pre-RLHF)
- Expand to 200+ prompts with held-out test split
- Real-time monitoring library (v1)
