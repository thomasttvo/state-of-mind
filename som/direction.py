"""
Contrastive mean-difference direction computation and fitting.

The failure direction is the unit vector from the mean of correct-confident
activations to the mean of confirmed-failing activations at a given layer.

Target layer selection: earliest layer where direction stability > 0.8
(cosine similarity to the next layer's direction).
"""

import numpy as np
from pathlib import Path


def unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else v


def compute_direction(
    cc_acts: np.ndarray,    # (n_cc,   n_layers, hidden_dim)
    fail_acts: np.ndarray,  # (n_fail, n_layers, hidden_dim)
    layer: int,
) -> np.ndarray:
    """Unit-norm contrastive mean difference at a given layer."""
    d = fail_acts[:, layer, :].mean(0) - cc_acts[:, layer, :].mean(0)
    return unit(d)


def find_target_layer(
    cc_acts: np.ndarray,
    fail_acts: np.ndarray,
    stability_threshold: float = 0.8,
) -> int:
    """
    Pre-registered rule: earliest layer l where
    cosine(direction[l], direction[l+1]) > stability_threshold.
    Falls back to the most stable layer if threshold is never crossed.
    """
    n_layers = cc_acts.shape[1]
    directions = [compute_direction(cc_acts, fail_acts, l) for l in range(n_layers)]

    stability = np.array([
        np.dot(directions[l], directions[l + 1])
        for l in range(n_layers - 1)
    ])

    for l in range(n_layers - 1):
        if stability[l] > stability_threshold:
            return l + 1  # the layer whose output is stable

    return int(stability.argmax()) + 1


class FailureDirection:
    """
    Fitted failure direction for a specific model layer.

    Attributes:
        layer:          which transformer layer
        direction:      unit-norm direction vector (hidden_dim,)
        cc_mean_proj:   mean projection of correct-confident examples
        fail_mean_proj: mean projection of confirmed-failing examples
    """

    def __init__(
        self,
        direction: np.ndarray,
        layer: int,
        cc_mean_proj: float,
        fail_mean_proj: float,
    ):
        self.direction = direction.astype(np.float32)
        self.layer = layer
        self.cc_mean_proj = cc_mean_proj
        self.fail_mean_proj = fail_mean_proj

    @classmethod
    def fit(
        cls,
        cc_acts: np.ndarray,    # (n_cc,   n_layers, hidden_dim)
        fail_acts: np.ndarray,  # (n_fail, n_layers, hidden_dim)
        layer: int | None = None,
    ) -> "FailureDirection":
        """
        Fit a FailureDirection from activation arrays.
        If layer is None, auto-selects the earliest stable layer.
        """
        if layer is None:
            layer = find_target_layer(cc_acts, fail_acts)

        direction = compute_direction(cc_acts, fail_acts, layer)
        cc_projs   = cc_acts[:,   layer, :] @ direction
        fail_projs = fail_acts[:, layer, :] @ direction

        return cls(
            direction=direction,
            layer=layer,
            cc_mean_proj=float(cc_projs.mean()),
            fail_mean_proj=float(fail_projs.mean()),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FailureDirection":
        """Load from .npz file saved with .save()."""
        d = np.load(path)
        return cls(
            direction=d["direction"],
            layer=int(d["layer"]),
            cc_mean_proj=float(d["cc_mean_proj"]),
            fail_mean_proj=float(d["fail_mean_proj"]),
        )

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            direction=self.direction,
            layer=np.array(self.layer),
            cc_mean_proj=np.array(self.cc_mean_proj),
            fail_mean_proj=np.array(self.fail_mean_proj),
        )

    def project(self, activation: np.ndarray) -> float:
        """Raw dot product projection."""
        return float(activation @ self.direction)

    def normalize(self, proj: float) -> float:
        """Normalize: 0 = CC mean, 1 = fail mean."""
        gap = self.fail_mean_proj - self.cc_mean_proj
        if abs(gap) < 1e-12:
            return 0.0
        return (proj - self.cc_mean_proj) / gap
