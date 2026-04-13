"""
State of Mind — v1 monitoring library.

Drop onto any mlx-lm model to get real-time failure direction scores.

Usage:
    from som import StateOfMindProbe

    probe = StateOfMindProbe.from_saved(model, tokenizer)
    score = probe.score(messages)           # 0=safe, 1=failing
    traj  = probe.trajectory(conversation)  # scores across turns
    regime = probe.regime(score)            # "safe" | "tension" | "failing"
"""

from .probe import StateOfMindProbe

__all__ = ["StateOfMindProbe"]
