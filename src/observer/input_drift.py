"""Input-feature drift, independent of the classifier (spec §5.3).

- Mean per-feature two-sample KS statistic on raw features (KS is invariant to the
  monotone log/standardize transform, so raw vs. transformed makes no difference).
  Computed exactly with sorted arrays; equivalent to `scipy.stats.ks_2samp(...).statistic`
  (asserted in tests) but vectorised, since it runs for every (model, day) pair.
- Unbiased Gaussian-kernel MMD^2 on standardized features against a fixed reference
  sample; bandwidth by the median heuristic on the reference sample.

Both statistics carry finite-sample bias that depends on sample size, so the current
sample size is fixed (config) and the per-generation baseline is measured at the same size.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel


def ks_statistics(cur: np.ndarray, ref_sorted: list[np.ndarray]) -> np.ndarray:
    out = np.empty(len(ref_sorted))
    for j, a in enumerate(ref_sorted):
        b = np.sort(cur[:, j])
        grid = np.concatenate([a, b])
        cdf_a = np.searchsorted(a, grid, side="right") / len(a)
        cdf_b = np.searchsorted(b, grid, side="right") / len(b)
        out[j] = np.abs(cdf_a - cdf_b).max()
    return out


def ks_mean(cur_raw: np.ndarray, ref_sorted: list[np.ndarray]) -> float:
    return float(ks_statistics(cur_raw, ref_sorted).mean())


def median_heuristic_gamma(X: np.ndarray) -> float:
    d = euclidean_distances(X, X, squared=True)
    med = np.median(d[np.triu_indices_from(d, k=1)])
    return 1.0 / max(med, 1e-9)


def _offdiag_mean(K: np.ndarray) -> float:
    n = len(K)
    return float((K.sum() - np.trace(K)) / (n * (n - 1)))


class MMDReference:
    def __init__(self, ref_std: np.ndarray):
        self.X = np.asarray(ref_std, np.float64)
        self.gamma = median_heuristic_gamma(self.X)
        self.kxx = _offdiag_mean(rbf_kernel(self.X, self.X, gamma=self.gamma))

    def mmd2(self, cur_std: np.ndarray) -> float:
        Y = np.asarray(cur_std, np.float64)
        kyy = _offdiag_mean(rbf_kernel(Y, Y, gamma=self.gamma))
        kxy = float(rbf_kernel(self.X, Y, gamma=self.gamma).mean())
        return self.kxx + kyy - 2 * kxy
