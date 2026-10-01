"""Sealed label store: the only object that holds true labels (spec §8).

Three independent layers enforce the label-free claim:
1. Static: tests/test_no_label_leakage.py walks the import graph and fails if any module
   outside src/evaluation/ and src/pipeline.py imports this module, the label-bearing
   loader, or the evaluation package, directly or transitively and in any import form.
2. Runtime caller check: every accessor inspects its caller's module and raises unless
   it is inside src.evaluation, src.pipeline, or scripts (tests register explicitly).
3. Runtime clock: the harness advances `clock` to the current decision period. No
   accessor returns labels for a period after the clock, so even a permitted caller
   cannot read the future. Calibration labels are only served for the configured
   calibration window, and only once the clock has passed its end.

Accessors return labels per date, aligned with the replay's per-day flow order,
because both come from the same loader output.
"""
from __future__ import annotations

import sys

import numpy as np

_ALLOWED_PREFIXES = ("src.evaluation", "src.pipeline", "__main__", "scripts")


class LabelAccessError(PermissionError):
    pass


def _check_caller(depth: int = 2) -> None:
    frame = sys._getframe(depth)
    mod = frame.f_globals.get("__name__", "")
    if mod.startswith(_ALLOWED_PREFIXES) or mod in SealedLabelStore._extra_allowed:
        return
    raise LabelAccessError(f"module '{mod}' may not read labels")


def macro_f1_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> float:
    present = (tp + fn + fp) > 0      # classes present in y_true or y_pred (sklearn default)
    if not present.any():
        return float("nan")
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return float(f1[present].mean())


class SealedLabelStore:
    _extra_allowed: set[str] = set()

    def __init__(self, labels_by_date: dict[str, np.ndarray], period_of_date: dict[str, int],
                 n_classes: int, calibration_window: tuple[int, int]):
        self._labels = {d: np.asarray(v).copy() for d, v in labels_by_date.items()}
        for v in self._labels.values():
            v.setflags(write=False)
        self._period = dict(period_of_date)
        self.n_classes = n_classes
        self.calibration_window = (int(calibration_window[0]), int(calibration_window[1]))
        self.clock = -10**9
        self.access_log: list[tuple[str, int, int]] = []

    @classmethod
    def allow_module(cls, name: str) -> None:
        """For tests only: permit an additional caller module."""
        cls._extra_allowed.add(name)

    def advance_to(self, period: int) -> None:
        _check_caller()
        if period < self.clock:
            raise LabelAccessError("the clock only moves forward")
        self.clock = period

    def reset_clock(self) -> None:
        """Start a new, independent policy timeline (harness only). Within a timeline
        the clock is monotone; each timeline starts again before W0."""
        _check_caller()
        self.clock = -10**9

    def _dates(self, week_range: range) -> list[str]:
        ws = set(week_range)
        return sorted(d for d, p in self._period.items() if p in ws and d in self._labels)

    def get_calibration_labels(self, week_range: range) -> dict[str, np.ndarray]:
        """Permitted call site #1: the one-time calibration of g."""
        _check_caller()
        lo, hi = self.calibration_window
        if min(week_range) < lo or max(week_range) > hi:
            raise LabelAccessError(f"calibration labels only for weeks {lo}-{hi}")
        if max(week_range) > self.clock:
            raise LabelAccessError("calibration window has not elapsed yet")
        self.access_log.append(("calibration", min(week_range), max(week_range)))
        return {d: self._labels[d] for d in self._dates(week_range)}

    def get_retrain_labels(self, week_range: range) -> dict[str, np.ndarray]:
        """Permitted call site #2: a policy has just decided to retrain."""
        _check_caller()
        if max(week_range) > self.clock:
            raise LabelAccessError(f"retrain labels requested for week {max(week_range)} > clock {self.clock}")
        self.access_log.append(("retrain", min(week_range), max(week_range)))
        return {d: self._labels[d] for d in self._dates(week_range)}

    def grade_counts(self, date: str, predictions: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluation only: per-class tp/fp/fn for one day's predictions. Never feeds
        the live decision path (only P5, the declared non-competing oracle)."""
        _check_caller()
        y = self._labels[date]
        if len(y) != len(predictions):
            raise ValueError("predictions are not aligned with the day's flows")
        k = self.n_classes
        known = y >= 0
        tp = np.bincount(y[known][predictions[known] == y[known]], minlength=k)
        pred_counts = np.bincount(predictions, minlength=k)
        true_counts = np.bincount(y[known], minlength=k)
        return {"tp": tp, "fp": pred_counts - tp, "fn": true_counts - tp,
                "n_known": np.array(int(known.sum())), "n_correct": np.array(int(tp.sum()))}

    def grade(self, week: int, predictions_by_date: dict[str, np.ndarray]) -> float:
        """Spec §8 signature: true macro-F1 of one week's predictions (evaluation only)."""
        _check_caller()
        tot = None
        for d, p in predictions_by_date.items():
            if self._period[d] != week:
                raise ValueError(f"{d} is not in week {week}")
            c = self.grade_counts(d, p)
            tot = c if tot is None else {k: tot[k] + c[k] for k in c}
        return macro_f1_from_counts(tot["tp"], tot["fp"], tot["fn"])
