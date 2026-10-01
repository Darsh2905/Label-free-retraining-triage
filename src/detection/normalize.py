"""Normalization before change-point detection (spec §6.2).

`rolling` (spec): x_t -> (x_t - median(trailing W)) / (1.4826 * MAD(trailing W) + eps).
The trailing window excludes x_t itself. Known weakness (reason this is not the
default): subtracting a trailing median is a high-pass filter. A persistent step is
"normalized away" after about W/2 days, and a slow ramp never stands out, so gradual
degradation cannot produce a drift event.

`anchored` (default): x_t -> (x_t - loc_calm) / scale_calm with loc/scale frozen from
the calm sigma window. D is already re-baselined per model generation (see
observer/composite.py), so a fixed anchor keeps both persistent and gradual shifts
visible while still putting the PELT penalty in noise units.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9


def rolling_mad_normalize(x: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    for i in range(len(x)):
        w = x[max(0, i - window):i]
        if len(w) < min_periods:
            continue                    # not enough history: report 0 (no evidence)
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        out[i] = (x[i] - med) / (1.4826 * mad + EPS)
    return out


def calm_stats(x_calm: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x_calm, float)
    med = float(np.median(x))
    return med, float(1.4826 * np.median(np.abs(x - med)) + EPS)


def anchored_normalize(x: np.ndarray, loc: float, scale: float) -> np.ndarray:
    return (np.asarray(x, float) - loc) / scale
