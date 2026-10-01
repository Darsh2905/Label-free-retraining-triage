"""STL seasonal removal (spec §6.1).

Applied to the DAILY composite series with period=7, the day-of-week cycle documented
for this dataset (Jančička et al., NOMS 2024). The spec applied STL(period=7) to a
weekly series, where it would model a meaningless seven-week cycle; weekly aggregation
already averages out the day-of-week cycle, so that step did nothing useful.
Returns trend + resid (= x - seasonal).
"""
from __future__ import annotations

import numpy as np
from statsmodels.tsa.seasonal import STL


def deseasonalize(x: np.ndarray, period: int = 7, robust: bool = True) -> np.ndarray:
    x = np.asarray(x, float)
    if len(x) < 2 * period + 1:
        return x.copy()          # too short to estimate a seasonal component
    res = STL(x, period=period, robust=robust).fit()
    return x - res.seasonal
