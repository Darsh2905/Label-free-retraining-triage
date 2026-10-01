"""Predicted-class distribution shift (spec §5.2).

Note: scipy's `jensenshannon` returns the JS *distance* (square root of the
divergence). We square it so the reported quantity is the divergence, as named.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon


def predicted_histogram(P: np.ndarray, n_classes: int) -> np.ndarray:
    h = np.bincount(P.argmax(1), minlength=n_classes).astype(np.float64)
    return h / max(h.sum(), 1.0)


def js_divergence(P: np.ndarray, ref_hist: np.ndarray) -> float:
    h = predicted_histogram(P, len(ref_hist))
    return float(jensenshannon(h, ref_hist, base=2) ** 2)
