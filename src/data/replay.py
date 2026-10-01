"""Label-free chronological replay (spec §3.2 rule 3, §3.3).

`Replay` is built from label-free `StreamDay`s (features only). It exposes exactly one
way forward, `next_week()`; there is no indexing, no rewind and no random access, so
out-of-order consumption is structurally awkward. `WeeklyBatch.true_labels` is always
None here -- only the sealed store holds labels.

"Week" means one decision period: an ISO week for TLS-Year22, one day for QUIC22
(configs/quic.yaml), because QUIC22 spans only four weeks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StreamDay:
    date: str
    period: int
    raw: np.ndarray
    volume: float


@dataclass
class WeeklyBatch:
    week_index: int
    features: np.ndarray            # (n_flows, n_features) standardized
    raw_features: np.ndarray        # (n_flows, n_features) pre-transform, for drift tests
    day_of_flow: np.ndarray         # (n_flows,) index into `dates`
    dates: list[str]
    volume: float                   # estimated true flow count in the stream this week
    true_labels: np.ndarray | None = None  # ONLY the sealed store holds labels; always None here

    def day(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        m = self.day_of_flow == i
        return self.features[m], self.raw_features[m]


class FrozenScaler:
    """log1p on heavy-tailed count columns, then standardization; fit ONCE on W0.

    Deviation from spec: counts (bytes, packets, durations, histogram bins) are log1p'd
    before standardizing. Without it the Gaussian-kernel MMD is dominated by a few
    elephant flows. LightGBM is invariant to this monotone transform and KS is too, so
    only MMD is affected. The scaler is never refit (leakage guard); a second `fit`
    raises and the fitted arrays are read-only.
    """

    def __init__(self, log_mask: np.ndarray):
        self.log_mask = np.asarray(log_mask, bool)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def _t(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64, copy=True)
        X[:, self.log_mask] = np.log1p(np.clip(X[:, self.log_mask], 0, None))
        return X

    def fit(self, X_raw: np.ndarray) -> "FrozenScaler":
        if self.mean_ is not None:
            raise RuntimeError("FrozenScaler is fit once on W0 and must never be refit")
        T = self._t(X_raw)
        mean, scale = T.mean(0), T.std(0)
        scale[scale < 1e-9] = 1.0
        mean.setflags(write=False)
        scale.setflags(write=False)
        self.mean_, self.scale_ = mean, scale
        return self

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("scaler not fitted")
        return ((self._t(X_raw) - self.mean_) / self.scale_).astype(np.float32)


class Replay:
    """Chronological iterator over decision periods [start, end]. Forward-only."""

    def __init__(self, days: list[StreamDay], scaler: FrozenScaler, start: int, end: int):
        for d in days:
            if not isinstance(d, StreamDay):
                raise TypeError("Replay accepts label-free StreamDay objects only")
        self._days = sorted((d for d in days if start <= d.period <= end), key=lambda d: d.date)
        self._periods = sorted({d.period for d in self._days})
        self._scaler = scaler
        self._pos = 0

    def has_next(self) -> bool:
        return self._pos < len(self._periods)

    def next_week(self) -> WeeklyBatch:
        if not self.has_next():
            raise StopIteration
        p = self._periods[self._pos]
        self._pos += 1
        days = [d for d in self._days if d.period == p]
        raw = np.concatenate([d.raw for d in days])
        day_of = np.concatenate([np.full(len(d.raw), i, np.int16) for i, d in enumerate(days)])
        return WeeklyBatch(p, self._scaler.transform(raw), raw, day_of,
                           [d.date for d in days], float(sum(d.volume for d in days)))
