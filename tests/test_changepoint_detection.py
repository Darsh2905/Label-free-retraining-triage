import numpy as np
import pytest

from src.detection.changepoint import pelt_changepoints, select_penalty
from src.detection.detector import OnlineDriftDetector
from src.detection.normalize import rolling_mad_normalize
from src.detection.persistence import persistence_gate
from src.detection.seasonal import deseasonalize

rng = np.random.default_rng(1)


def test_pelt_finds_step():
    x = np.r_[rng.normal(0, 1, 60), rng.normal(4, 1, 40)]
    cps = pelt_changepoints(x, penalty=10)
    assert any(abs(c - 60) <= 2 for c in cps)


def test_penalty_selection_smallest_zero_on_calm():
    calm = rng.normal(0, 1, 42)
    pen, sweep = select_penalty(calm, [0.5, 1, 3, 10, 30, 100])
    assert sweep[pen] == 0
    assert all(sweep[p] > 0 for p in sweep if p < pen)


@pytest.mark.parametrize("case,expect", [
    ("sustained_up", True), ("transient_spike", False), ("down_step", False), ("reverted", False)])
def test_persistence_gate(case, expect):
    base = rng.normal(0, 0.2, 30)
    if case == "sustained_up":
        x = np.r_[base, rng.normal(3, 0.2, 10)]
    elif case == "transient_spike":
        x = np.r_[base, [3, 3], rng.normal(0, 0.2, 10)]
    elif case == "down_step":
        x = np.r_[base, rng.normal(-3, 0.2, 10)]
    else:
        x = np.r_[base, rng.normal(3, 0.2, 6), rng.normal(-0.5, 0.2, 8)]
    cps = pelt_changepoints(x, penalty=5, min_size=2)
    assert (persistence_gate(x, cps, k=3) is not None) == expect


def test_stl_removes_weekly_cycle():
    t = np.arange(84)
    x = 2 * (t % 7 >= 5) + rng.normal(0, 0.2, 84)
    y = deseasonalize(x, 7)
    assert np.var(y) < 0.2 * np.var(x)


def test_rolling_normalization_forgets_persistent_steps():
    """Documents why `anchored` is the default: after a persistent step the rolling
    median catches up and the normalized signal returns to ~0 within the window."""
    x = np.r_[rng.normal(0, 1, 60), rng.normal(6, 1, 60)]
    z = rolling_mad_normalize(x, window=28, min_periods=7)
    assert z[62] > 3                       # visible right after the step
    assert abs(np.median(z[100:])) < 1     # normalized away a few weeks later


def _dates(n, start="20220103"):
    import datetime as dt
    d0 = dt.date(int(start[:4]), int(start[4:6]), int(start[6:]))
    return [(d0 + dt.timedelta(days=i)).strftime("%Y%m%d") for i in range(n)]


def test_detector_is_causal():
    """The detection at day t must not depend on anything observed after t."""
    cfg = {"stl": {"enabled": True, "period": 7, "robust": True},
           "normalize": {"mode": "anchored", "window_days": 56, "min_periods": 7},
           "pelt": {"model": "l2", "min_size": 3, "jump": 1, "penalty": "auto", "penalty_grid": [1, 3, 10, 30, 100]},
           "persistence": {"k_days": 3}}
    dates = _dates(120)
    x = np.r_[rng.normal(0, 1, 70), rng.normal(5, 1, 50)]

    def run(values, upto):
        det = OnlineDriftDetector(cfg, lambda d: int(d[-2:]))
        for d, v in zip(dates[:42], values[:42]):
            det.observe(d, v)
        det.calibrate()
        out = []
        for d, v in zip(dates[42:upto], values[42:upto]):
            det.observe(d, v)
            out.append(det.detect())
        return out

    a = run(x, 90)
    y = x.copy()
    y[90:] += rng.normal(20, 5, 30)        # wild future
    b = run(y, 120)[: len(a)]
    assert a == b
    assert any(e is not None for e in a)   # and the step at day 70 is detected


def test_detector_skips_missing_days_by_interpolation():
    cfg = {"stl": {"enabled": True, "period": 7, "robust": True},
           "normalize": {"mode": "anchored", "window_days": 56, "min_periods": 7},
           "pelt": {"model": "l2", "min_size": 3, "jump": 1, "penalty": 10, "penalty_grid": []},
           "persistence": {"k_days": 3}}
    det = OnlineDriftDetector(cfg, lambda d: 1)
    dates = _dates(40)
    for i, d in enumerate(dates):
        if i in (20, 21, 22):
            continue                        # three missing days, like 28-30 Jan 2022
        det.observe(d, float(i % 7 == 6))
    det.calibrate()
    grid, x = det.conditioned()
    assert len(grid) == 40
