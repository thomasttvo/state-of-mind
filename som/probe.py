"""
StateOfMindProbe — main interface for the v1 monitoring library.

Wraps a loaded mlx-lm model + a fitted FailureDirection into a single object
that can score any prompt or conversation in real time.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

import mlx.core as mx

from .extract import extract_layer_activation
from .direction import FailureDirection

# ── Regime thresholds (normalized projection) ────────────────────────────────
THRESHOLD_TENSION  = 0.70   # above this = tension state
THRESHOLD_FAILING  = 0.85   # above this = high failure risk


@dataclass
class TurnScore:
    turn_idx:    int
    proj:        float   # raw projection
    norm:        float   # normalized (0=CC, 1=fail)
    regime:      str     # "safe" | "tension" | "failing"
    user_msg:    str = ""
    response:    str = ""


@dataclass
class Trajectory:
    conversation_id: str = ""
    turns: List[TurnScore] = field(default_factory=list)

    @property
    def scores(self) -> list[float]:
        return [t.norm for t in self.turns]

    @property
    def peak(self) -> float:
        return max(self.scores) if self.scores else 0.0

    @property
    def drift(self) -> float:
        """Total normalized movement from turn 0 to last turn."""
        if len(self.scores) < 2:
            return 0.0
        return self.scores[-1] - self.scores[0]


class StateOfMindProbe:
    """
    Real-time failure direction monitor for any mlx-lm model.

    Typical usage:
        probe = StateOfMindProbe.from_saved(model, tokenizer)
        score = probe.score([{"role": "user", "content": "..."}])
        print(score.norm, score.regime)

    Cross-model usage (zero-shot transfer):
        probe = StateOfMindProbe.from_saved(gemma_model, gemma_tokenizer,
                                            direction_path="results/direction.npz")
    """

    def __init__(self, model, tokenizer, fd: FailureDirection):
        self.model     = model
        self.tokenizer = tokenizer
        self.inner     = model.model
        self.fd        = fd

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_saved(
        cls,
        model,
        tokenizer,
        direction_path: str | Path | None = None,
        activations_dir: str | Path | None = None,
    ) -> "StateOfMindProbe":
        """
        Build probe from either:
          (a) a pre-saved direction file (.npz), or
          (b) raw activation directories (fits direction from scratch)

        If direction_path is None, looks for results/direction.npz relative
        to the repo root (two levels up from this file).
        """
        if direction_path is None:
            direction_path = Path(__file__).parent.parent / "results" / "direction.npz"

        if Path(direction_path).exists():
            fd = FailureDirection.load(direction_path)
        else:
            if activations_dir is None:
                activations_dir = Path(__file__).parent.parent / "activations"
            fd = cls._fit_from_activations(activations_dir)
            fd.save(direction_path)
            print(f"Direction fitted and saved → {direction_path}")

        print(f"Probe ready — layer {fd.layer}, "
              f"CC={fd.cc_mean_proj:.3f}, fail={fd.fail_mean_proj:.3f}")
        return cls(model, tokenizer, fd)

    @classmethod
    def fit(
        cls,
        model,
        tokenizer,
        cc_acts: np.ndarray,
        fail_acts: np.ndarray,
        layer: int | None = None,
    ) -> "StateOfMindProbe":
        """Fit probe directly from activation arrays."""
        fd = FailureDirection.fit(cc_acts, fail_acts, layer=layer)
        return cls(model, tokenizer, fd)

    # ── Core scoring ──────────────────────────────────────────────────────────

    def _encode(self, messages: list) -> mx.array:
        """Format messages and tokenize."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return mx.array(self.tokenizer.encode(prompt))[None]

    def _activation(self, messages: list) -> np.ndarray:
        input_ids = self._encode(messages)
        return extract_layer_activation(self.inner, input_ids, self.fd.layer)

    def score(self, messages: list) -> TurnScore:
        """
        Score the model's state given conversation history.

        messages: list of {role, content} dicts (same format as chat template)
        Returns a TurnScore with raw projection, normalized score, and regime.
        """
        act  = self._activation(messages)
        proj = self.fd.project(act)
        norm = self.fd.normalize(proj)
        return TurnScore(
            turn_idx=len([m for m in messages if m["role"] == "assistant"]),
            proj=proj,
            norm=norm,
            regime=self.regime(norm),
        )

    def regime(self, norm_score: float) -> str:
        if norm_score >= THRESHOLD_FAILING:
            return "failing"
        if norm_score >= THRESHOLD_TENSION:
            return "tension"
        return "safe"

    # ── Trajectory ────────────────────────────────────────────────────────────

    def trajectory(
        self,
        conversation: dict,
        generate_fn=None,
        max_tokens: int = 150,
        verbose: bool = True,
    ) -> Trajectory:
        """
        Run a multi-turn conversation and score each assistant turn.

        conversation: dict with keys:
            id:    conversation identifier
            turns: list of {role, content} where content=="GENERATE" for
                   assistant turns to be generated
        generate_fn: optional callable(messages) -> str
                     if None and turns contain GENERATE, uses mlx_lm.generate

        Returns a Trajectory with per-turn TurnScore objects.
        """
        from mlx_lm import generate as mlx_generate

        traj    = Trajectory(conversation_id=conversation.get("id", ""))
        history = []
        ai_idx  = 0

        for turn in conversation["turns"]:
            if turn["role"] == "user":
                history.append({"role": "user", "content": turn["content"]})

            elif turn["role"] == "assistant":
                content = turn["content"]

                # Score BEFORE generating
                ts = self.score(history)
                ts.turn_idx  = ai_idx
                ts.user_msg  = history[-1]["content"] if history else ""

                if content == "GENERATE":
                    if generate_fn is not None:
                        response = generate_fn(history)
                    else:
                        prompt = self.tokenizer.apply_chat_template(
                            history, tokenize=False, add_generation_prompt=True
                        )
                        response = mlx_generate(
                            self.model, self.tokenizer,
                            prompt=prompt,
                            max_tokens=max_tokens,
                            verbose=False,
                        )
                    history.append({"role": "assistant", "content": response})
                    ts.response = response
                else:
                    history.append({"role": "assistant", "content": content})
                    ts.response = content

                if verbose:
                    bar = "█" * int(max(0, ts.norm * 20))
                    print(f"  turn {ai_idx}: norm={ts.norm:+.2f} [{ts.regime:8s}] {bar}")

                traj.turns.append(ts)
                ai_idx += 1

        return traj

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fit_from_activations(activations_dir: Path) -> FailureDirection:
        """Load saved activations and fit direction."""
        import json

        activations_dir = Path(activations_dir)
        data_dir = activations_dir.parent / "data"

        with open(data_dir / "labeled_responses.json") as f:
            labeled = json.load(f)

        cc_acts = np.stack([
            np.load(activations_dir / "correct_confident" / f"{i:04d}.npy")
            for i in range(20)
        ])
        fail_recs = [
            (pop, item["idx"])
            for pop, items in labeled.items()
            for item in items
            if item["label"] in ("sycophantic", "overconfident_wrong")
        ]
        fail_acts = np.stack([
            np.load(activations_dir / pop / f"{idx:04d}.npy")
            for pop, idx in fail_recs
        ])

        return FailureDirection.fit(cc_acts, fail_acts)
