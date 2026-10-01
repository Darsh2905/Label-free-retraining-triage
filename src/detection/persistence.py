"""k-period persistence gate (spec §6.4).

A change point n becomes a candidate drift event only if (a) it is an upward shift,
i.e. towards degradation, (b) the k values right after n all stay above the floor, where
the floor is the mean of the segment before n, and (c) the shift is still active, i.e.
the mean of the most recent k values is still above the floor. The most recent qualifying
change point is returned. Condition (c) is an addition to the spec: without it a
transient that has already reverted keeps triggering the cost check forever.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class DriftEvent:
    week: int            # decision period in which the change point falls
    magnitude: float     # post-change mean minus floor, in normalized units
    day_index: int = -1  # index of the change point within the analysed segment


def persistence_gate(x: np.ndarray, change_points: list[int], k: int) -> Optional[tuple[int, float]]:
    x = np.asarray(x, float)
    bounds = [0] + list(change_points)
    for i in range(len(change_points) - 1, -1, -1):
        n = change_points[i]
        pre = x[bounds[i]:n]
        post = x[n:]
        if len(pre) == 0 or len(post) < k:
            continue
        floor = pre.mean()
        if post.mean() > floor and np.all(post[:k] > floor) and x[-k:].mean() > floor:
            return n, float(post.mean() - floor)
    return None
