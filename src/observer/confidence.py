"""Confidence statistics of the classifier's probability output (spec §5.1)."""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def confidence_stats(P: np.ndarray) -> dict[str, float]:
    """P: (n_flows, n_classes). Returns mean/p10 max-prob, mean margin, mean entropy."""
    if len(P) == 0:
        return {"max_prob_mean": np.nan, "max_prob_p10": np.nan, "margin_mean": np.nan, "entropy_mean": np.nan}
    top2 = np.partition(P, -2, axis=1)[:, -2:]
    max_prob = top2[:, 1]
    margin = top2[:, 1] - top2[:, 0]
    entropy = -(P * np.log(P + EPS)).sum(1)
    return {
        "max_prob_mean": float(max_prob.mean()),
        "max_prob_p10": float(np.percentile(max_prob, 10)),
        "margin_mean": float(margin.mean()),
        "entropy_mean": float(entropy.mean()),
    }
