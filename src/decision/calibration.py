"""g: D -> expected accuracy loss, fit once by isotonic regression (spec §7.1).

This module is a pure function of (D, loss) pairs. It never touches labels itself: the
evaluation harness computes the calibration-window losses through the sealed store's
calibration accessor and passes them in. So `decision/` needs no label access at all,
which is stricter than the spec's allowance for calibration.py.

Known limitation of isotonic regression, recorded in config as `extrapolation`: with
`clip` (spec), g is flat beyond the largest D seen during calibration. A drift larger
than anything in the calibration window is then priced at the calibration maximum,
possibly far too low. `linear` continues from the last isotonic value with the
non-negative OLS slope of the calibration data. Default: clip (spec). The calibration
window here contains the week-10 exporter change, so its D range is unusually wide.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression


class LossCalibrator:
    def __init__(self, extrapolation: str = "clip"):
        self.extrapolation = extrapolation
        self._iso: IsotonicRegression | None = None
        self._dmax = self._gmax = self._slope = 0.0

    def fit(self, D: np.ndarray, loss: np.ndarray) -> "LossCalibrator":
        if self._iso is not None:
            raise RuntimeError("g is frozen after the one-time calibration fit")
        D, loss = np.asarray(D, float), np.asarray(loss, float)
        ok = np.isfinite(D) & np.isfinite(loss)
        D, loss = D[ok], loss[ok]
        self._iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(D, loss)
        self._dmax = float(D.max())
        self._gmax = float(self._iso.predict([self._dmax])[0])
        self._slope = max(0.0, float(np.polyfit(D, loss, 1)[0])) if len(D) > 2 else 0.0
        return self

    def predict(self, D) -> np.ndarray:
        if self._iso is None:
            raise RuntimeError("g not fitted")
        D = np.atleast_1d(np.asarray(D, float))
        out = self._iso.predict(D)
        if self.extrapolation == "linear":
            hi = D > self._dmax
            out[hi] = self._gmax + self._slope * (D[hi] - self._dmax)
        return out


def transfer_report(g: LossCalibrator, D: np.ndarray, loss: np.ndarray) -> dict[str, float]:
    """Compare g's predicted loss with realised loss outside the calibration window (§7.4)."""
    D, loss = np.asarray(D, float), np.asarray(loss, float)
    pred = g.predict(D)
    rho = stats.spearmanr(D, loss).statistic if len(D) > 2 else np.nan
    return {"n": int(len(D)), "mae": float(np.mean(np.abs(pred - loss))),
            "bias_pred_minus_true": float(np.mean(pred - loss)),
            "spearman_D_loss": float(rho), "mean_true_loss": float(loss.mean()),
            "mean_pred_loss": float(pred.mean())}
