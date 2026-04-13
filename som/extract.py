"""
Model-agnostic activation extraction for mlx-lm models.

Supports any model following the standard mlx-lm architecture:
  model.model.embed_tokens
  model.model.layers[i](h, mask, cache=c)

Handles Mistral (sliding window), Gemma 2 (requires array mask, not sentinel),
and standard LLaMA layouts.
"""

import numpy as np
import mlx.core as mx


def _make_masks(inner, h, cache):
    """
    Build attention masks for any mlx-lm inner model.
    Returns (fa_mask, sw_mask) where sw_mask is None for non-sliding models.

    Forces return_array=True so that all models receive a real mx.array
    rather than the "causal" string sentinel (which Gemma 2 doesn't handle).
    """
    try:
        from mlx_lm.models.llama import create_attention_mask
    except ImportError:
        from mlx_lm.models.base import create_attention_mask

    fa_idx = getattr(inner, "fa_idx", 0)
    cache_entry = cache[fa_idx] if cache else None

    # Always request a real array mask — some models (e.g. Gemma 2) can't
    # handle the "causal" string sentinel returned by default.
    try:
        fa_mask = create_attention_mask(h, cache_entry, return_array=True)
    except TypeError:
        # Older mlx-lm doesn't have return_array param
        fa_mask = create_attention_mask(h, cache_entry)
        if isinstance(fa_mask, str):
            # Build a causal mask manually
            N = h.shape[1]
            fa_mask = mx.tril(mx.ones((N, N), dtype=mx.bool_))

    sw_mask = None
    swa_idx = getattr(inner, "swa_idx", None)
    if swa_idx is not None:
        try:
            sw_mask = create_attention_mask(
                h, cache[swa_idx], window_size=inner.sliding_window, return_array=True
            )
        except TypeError:
            sw_mask = create_attention_mask(h, cache[swa_idx], window_size=inner.sliding_window)

    return fa_mask, sw_mask


def extract_layer_activation(
    inner,
    input_ids: mx.array,
    target_layer: int,
) -> np.ndarray:
    """
    Run a forward pass and return the last-token hidden state at target_layer.

    Args:
        inner:        model.model  (LlamaModel / GemmaModel / etc.)
        input_ids:    shape (1, seq_len)
        target_layer: which transformer layer to capture (0-indexed)

    Returns:
        np.ndarray shape (hidden_dim,), float32
    """
    h = inner.embed_tokens(input_ids)

    # Gemma scales embeddings by sqrt(hidden_size)
    if hasattr(inner, "embed_scale"):
        h = h * inner.embed_scale

    cache = [None] * len(inner.layers)
    fa_mask, sw_mask = _make_masks(inner, h, cache)

    captured = None
    for i, (layer, c) in enumerate(zip(inner.layers, cache)):
        mask = sw_mask if getattr(layer, "use_sliding", False) else fa_mask
        h = layer(h, mask, cache=c)
        if i == target_layer:
            mx.eval(h)
            captured = np.array(h[0, -1, :], dtype=np.float32)
            break

    return captured


def extract_all_layers(
    inner,
    input_ids: mx.array,
) -> np.ndarray:
    """
    Run a full forward pass and return last-token hidden state at every layer.

    Returns:
        np.ndarray shape (n_layers, hidden_dim), float32
    """
    h = inner.embed_tokens(input_ids)

    if hasattr(inner, "embed_scale"):
        h = h * inner.embed_scale

    cache = [None] * len(inner.layers)
    fa_mask, sw_mask = _make_masks(inner, h, cache)

    captured = []
    for i, (layer, c) in enumerate(zip(inner.layers, cache)):
        mask = sw_mask if getattr(layer, "use_sliding", False) else fa_mask
        h = layer(h, mask, cache=c)
        mx.eval(h)
        captured.append(np.array(h[0, -1, :], dtype=np.float32))

    return np.stack(captured, axis=0)
