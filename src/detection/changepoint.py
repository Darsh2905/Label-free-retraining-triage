"""PELT change-point detection (spec §6.3) and label-free penalty selection.

Penalty selection follows the spec: sweep a grid on a known-calm stretch (the sigma
window of the initial model) and take the smallest penalty that yields zero change
points there. This uses only D, never labels. The full sweep is logged, because
Smirnov (REEPE 2023) shows the penalty is a genuine trade-off, not a free constant.
"""
from __future__ import annotations

import numpy as np
import ruptures as rpt


def pelt_changepoints(x: np.ndarray, penalty: float, model: str = "l2", min_size: int = 3,
                      jump: int = 1) -> list[int]:
    """Indices where a new segment starts (excluding 0 and len(x))."""
    x = np.asarray(x, float)
    if len(x) < 2 * min_size:
        return []
    bkps = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(x.reshape(-1, 1)).predict(pen=penalty)
    return [b for b in bkps if 0 < b < len(x)]


def select_penalty(calm: np.ndarray, grid: list[float], **kw) -> tuple[float, dict[float, int]]:
    sweep = {float(p): len(pelt_changepoints(calm, p, **kw)) for p in sorted(grid)}
    for p, n in sweep.items():
        if n == 0:
            return p, sweep
    return max(sweep), sweep
