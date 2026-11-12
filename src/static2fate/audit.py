"""Small helpers for checking the fate label against the live movie (scripts/46_label_audit.py)."""

from __future__ import annotations

import numpy as np


def background_share(n_label: float, n_at_risk: float, base_rate: float) -> float:
    """Share of a label's cells that a background rate alone would produce, clipped to [0, 1].

    n_label cells carry the label at this dose, n_at_risk cells could carry it, and base_rate is
    the label's rate among untreated cells. NaN when no cell carries the label.
    """
    if n_label <= 0:
        return float("nan")
    return float(min(1.0, base_rate * n_at_risk / n_label))


def sustained_onset(mask: np.ndarray) -> np.ndarray:
    """First frame index from which a (cells, frames) boolean mask stays True to the last frame.

    NaN for a cell whose mask is False at the last frame, so a nucleus that rounds up for one
    frame and recovers (a division) has no onset.
    """
    m = np.asarray(mask, bool)
    run = np.flip(np.cumprod(np.flip(m, axis=1), axis=1), axis=1).astype(bool)  # True where all later frames are True
    onset = run.argmax(axis=1).astype(float)
    onset[~run.any(axis=1)] = np.nan
    return onset


def cumulative_incidence(onset: np.ndarray, n_frames: int) -> np.ndarray:
    """Fraction of cells whose onset is at or before each frame (onset counted from 1); NaN = never."""
    o = np.asarray(onset, float)
    return np.array([(o <= f).sum() / len(o) for f in range(1, n_frames + 1)]) if len(o) else np.full(n_frames, np.nan)

