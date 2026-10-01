"""Online, causal composition of Stage 4: STL -> normalize -> PELT -> persistence.

Every call to `detect()` re-runs the chain on the history observed so far and nothing
else. Running STL/PELT once over the full year and reading off change points would
use future data; tests/test_changepoint_detection.py asserts that perturbing the future
leaves past detections unchanged.

Missing calendar days (the dataset has 8) are linearly interpolated so that the
seasonal period stays aligned to the day of the week.
PELT runs only on the current model generation's segment: a retrain starts a new regime.
"""
from __future__ import annotations

import datetime as dt
from typing import Callable, Optional

import numpy as np

from src.detection.changepoint import pelt_changepoints, select_penalty
from src.detection.normalize import anchored_normalize, calm_stats, rolling_mad_normalize
from src.detection.persistence import DriftEvent, persistence_gate
from src.detection.seasonal import deseasonalize


def _ordinal(date: str) -> int:
    return dt.date(int(date[:4]), int(date[4:6]), int(date[6:])).toordinal()


class OnlineDriftDetector:
    def __init__(self, dcfg: dict, period_of_date: Callable[[str], int]):
        self.cfg = dcfg
        self.period_of_date = period_of_date
        self.dates: list[str] = []
        self.values: list[float] = []
        self.gen_start_ordinal: Optional[int] = None
        self.anchor: Optional[tuple[float, float]] = None
        pen = dcfg["pelt"]["penalty"]
        self.penalty: Optional[float] = None if pen == "auto" else float(pen)
        self.penalty_sweep: dict[float, int] = {}

    # --- state -----------------------------------------------------------------
    def observe(self, date: str, value: float) -> None:
        if self.dates and date <= self.dates[-1]:
            raise ValueError("observations must be strictly chronological")
        self.dates.append(date)
        self.values.append(float(value))
        if self.gen_start_ordinal is None:
            self.gen_start_ordinal = _ordinal(date)

    def new_generation(self, first_date_ordinal: int) -> None:
        self.gen_start_ordinal = first_date_ordinal

    # --- conditioning -------------------------------------------------------------
    def _grid(self) -> tuple[np.ndarray, np.ndarray]:
        o = np.array([_ordinal(d) for d in self.dates])
        grid = np.arange(o[0], o[-1] + 1)
        return grid, np.interp(grid, o, np.array(self.values))

    def _deseason(self, x: np.ndarray) -> np.ndarray:
        s = self.cfg["stl"]
        return deseasonalize(x, s["period"], s["robust"]) if s["enabled"] else x

    def conditioned(self) -> tuple[np.ndarray, np.ndarray]:
        grid, x = self._grid()
        x = self._deseason(x)
        n = self.cfg["normalize"]
        if n["mode"] == "rolling":
            return grid, rolling_mad_normalize(x, n["window_days"], n["min_periods"])
        if self.anchor is None:
            raise RuntimeError("anchored normalization needs calibrate() on the calm window first")
        return grid, anchored_normalize(x, *self.anchor)

    def calibrate(self) -> dict:
        """Call once, at the end of the calm (sigma) window, with only calm data observed."""
        grid, x = self._grid()
        x = self._deseason(x)
        n = self.cfg["normalize"]
        if n["mode"] == "anchored":
            self.anchor = calm_stats(x)
            calm = anchored_normalize(x, *self.anchor)
        else:
            calm = rolling_mad_normalize(x, n["window_days"], n["min_periods"])[n["min_periods"]:]
        p = self.cfg["pelt"]
        if self.penalty is None:
            self.penalty, self.penalty_sweep = select_penalty(
                calm, p["penalty_grid"], model=p["model"], min_size=p["min_size"], jump=p["jump"])
        return {"penalty": self.penalty, "sweep": self.penalty_sweep, "anchor": self.anchor}

    # --- detection ---------------------------------------------------------------
    def detect(self) -> Optional[DriftEvent]:
        if not self.dates or self.penalty is None:
            return None
        grid, x = self.conditioned()
        start = int(np.searchsorted(grid, self.gen_start_ordinal))
        seg = x[start:]
        p = self.cfg["pelt"]
        cps = pelt_changepoints(seg, self.penalty, p["model"], p["min_size"], p["jump"])
        hit = persistence_gate(seg, cps, self.cfg["persistence"]["k_days"])
        if hit is None:
            return None
        n, mag = hit
        date = dt.date.fromordinal(int(grid[start + n])).strftime("%Y%m%d")
        return DriftEvent(week=self.period_of_date(date), magnitude=mag, day_index=n)
