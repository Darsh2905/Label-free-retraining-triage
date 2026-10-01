"""Reject rate: fraction of flows whose max-prob falls below a cutoff tau (spec §5.4).

tau is fixed once per deployed generation, as a percentile of that generation's
max-prob distribution on its own reference window (W0 for the initial model). With the
default 10th percentile, the in-distribution reject rate is ~0.10 by construction.
"""
from __future__ import annotations

import numpy as np


def fit_tau(P_ref: np.ndarray, percentile: float) -> float:
    return float(np.percentile(P_ref.max(1), percentile))


def reject_rate(P: np.ndarray, tau: float) -> float:
    return float((P.max(1) < tau).mean()) if len(P) else float("nan")
