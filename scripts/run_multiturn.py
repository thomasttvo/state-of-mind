"""
Step 5: Run multi-turn conversations and extract per-turn activations.

For each conversation:
  - At every assistant turn, feed the conversation history up to that point
  - Extract layer 6 last-token activation (model's state just before responding)
  - Generate the assistant response
  - Record whether the response is a capitulation

The hypothesis: for pressure conversations, the failure direction score
rises 2-3 turns BEFORE the model visibly capitulates.

Outputs:
  data/multiturn_responses.json   — full conversation transcripts
  activations/multiturn/<id>_turn<N>.npy  — (hidden_dim,) per turn
"""

import json
import numpy as np
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.llama import create_attention_mask

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
ACT_DIR     = BASE_DIR / "activations" / "multiturn"
ACT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LAYER = 6
MAX_TOKENS   = 150

print("Loading model...")
model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
inner = model.model

def extract_last_token_activation(messages: list) -> np.ndarray:
    """
    Given conversation history as [{role, content}, ...],
    format as a prompt, run forward pass, return layer TARGET_LAYER
    last-token hidden state. Shape: (hidden_dim,)
    """
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = mx.array(tokenizer.encode(prompt))[None]

    h     = inner.embed_tokens(input_ids)
    cache = [None] * len(inner.layers)
    fa    = create_attention_mask(h, cache[inner.fa_idx])
    sw    = None
    if inner.swa_idx is not None:
        sw = create_attention_mask(h, cache[inner.swa_idx], window_size=inner.sliding_window)

    captured = None
    for i, (layer, c) in enumerate(zip(inner.layers, cache)):
        mask = sw if getattr(layer, "use_sliding", False) else fa
        h    = layer(h, mask, cache=c)
        if i == TARGET_LAYER:
            mx.eval(h)
            captured = np.array(h[0, -1, :], dtype=np.float32)

    return captured


def generate_response(messages: list) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    from mlx_lm import generate
    return generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, verbose=False)


# ── Load direction ────────────────────────────────────────────────────────────
with open(DATA_DIR / "labeled_responses.json") as f:
    labeled = json.load(f)

act_base = BASE_DIR / "activations"
cc_acts   = np.stack([np.load(act_base / "correct_confident" / f"{i:04d}.npy")
                      for i in range(20)])
fail_recs = [(pop, item["idx"]) for pop, items in labeled.items()
             for item in items if item["label"] in ("sycophantic","overconfident_wrong")]
fail_acts = np.stack([np.load(act_base / pop / f"{idx:04d}.npy")
                      for pop, idx in fail_recs])

d_raw  = fail_acts[:, TARGET_LAYER, :].mean(0) - cc_acts[:, TARGET_LAYER, :].mean(0)
d_fail = (d_raw / np.linalg.norm(d_raw)).astype(np.float32)

cc_mean_proj   = float((cc_acts[:, TARGET_LAYER, :] @ d_fail).mean())
fail_mean_proj = float((fail_acts[:, TARGET_LAYER, :] @ d_fail).mean())
print(f"Direction loaded. CC mean proj: {cc_mean_proj:.3f}, Fail mean proj: {fail_mean_proj:.3f}")

# ── Run conversations ─────────────────────────────────────────────────────────
with open(DATA_DIR / "multiturn_conversations.json") as f:
    dataset = json.load(f)

results = {}

for conv_type, conversations in dataset.items():
    results[conv_type] = []
    print(f"\n{'='*60}")
    print(f"Type: {conv_type}  ({len(conversations)} conversations)")
    print(f"{'='*60}")

    for conv in conversations:
        cid    = conv["id"]
        turns  = conv["turns"]
        print(f"\n  [{cid}]")

        # Skip if all activation files already exist
        n_assistant = sum(1 for t in turns if t["role"] == "assistant" and t["content"] == "GENERATE")
        all_exist = all((ACT_DIR / f"{cid}_turn{i:02d}.npy").exists() for i in range(n_assistant))
        if all_exist:
            print(f"    Skipping (all {n_assistant} turns cached)")
            cached_turns = []
            for i in range(n_assistant):
                act = np.load(ACT_DIR / f"{cid}_turn{i:02d}.npy")
                proj = float(act @ d_fail)
                cached_turns.append({
                    "turn_idx": i, "user_message": "(cached)", "response": "(cached)",
                    "failure_proj": proj,
                    "proj_normalized": (proj - cc_mean_proj) / (fail_mean_proj - cc_mean_proj),
                })
            results[conv_type].append({
                "id": cid, "topic": conv.get("topic", ""),
                "correct_answer": conv.get("correct_answer", ""),
                "wrong_claim": conv.get("wrong_claim", ""),
                "turns": cached_turns,
            })
            continue

        history      = []   # [{role, content}] built up turn by turn
        turn_records = []   # per-assistant-turn data

        assistant_turn_idx = 0

        for t, turn in enumerate(turns):
            if turn["role"] == "user":
                history.append({"role": "user", "content": turn["content"]})

            elif turn["role"] == "assistant" and turn["content"] == "GENERATE":
                # Extract activation BEFORE generating response
                activation = extract_last_token_activation(history)
                proj = float(activation @ d_fail)

                # Generate response
                response = generate_response(history)
                history.append({"role": "assistant", "content": response})

                # Save activation
                npy_path = ACT_DIR / f"{cid}_turn{assistant_turn_idx:02d}.npy"
                np.save(npy_path, activation)

                turn_records.append({
                    "turn_idx":        assistant_turn_idx,
                    "user_message":    history[-2]["content"],
                    "response":        response,
                    "failure_proj":    proj,
                    "proj_normalized": (proj - cc_mean_proj) / (fail_mean_proj - cc_mean_proj),
                })

                bar = "█" * int(max(0, (proj - cc_mean_proj) / (fail_mean_proj - cc_mean_proj) * 20))
                print(f"    Turn {assistant_turn_idx}: proj={proj:+.3f} norm={turn_records[-1]['proj_normalized']:+.2f} {bar}")
                print(f"      Response: {response[:100].strip()!r}")

                assistant_turn_idx += 1

        results[conv_type].append({
            "id":           cid,
            "topic":        conv.get("topic", ""),
            "correct_answer": conv.get("correct_answer", ""),
            "wrong_claim":  conv.get("wrong_claim", ""),
            "turns":        turn_records,
        })

with open(DATA_DIR / "multiturn_responses.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResponses -> {DATA_DIR / 'multiturn_responses.json'}")
print(f"Activations -> {ACT_DIR}/")
