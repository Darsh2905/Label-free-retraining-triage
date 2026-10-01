"""Composite drift score D (spec §5.5), with three corrections to the spec.

1. Sign orientation. max-prob, its p10, and margin FALL under drift, while entropy, JS,
   KS, MMD and reject rate RISE. Summing raw z-scores with positive weights, as the
   spec literally describes, partly cancels the confidence family. Each z-score is
   therefore multiplied by the signal's degradation sign.
2. Family-balanced weights (default). Four of the eight sub-signals are near-collinear
   confidence statistics, so equal per-signal weights give "confidence" half the vote.
   `weighting: family` gives each of the four families (confidence, prediction
   divergence, input drift, reject rate) 1/4, split equally within the family;
   `weighting: equal` is the spec's literal choice. Weights are hand-set, not learned.
3. Per-generation location. The spec z-scores against weeks 3-8 of the initial model
   and never says what happens after a retrain. A retrained model has different
   confidence levels, a different predicted-class histogram and different reference
   inputs, so every generation carries its own label-free reference. The location of
   each z-score is the signal's value on held-out data from that generation's own
   training window, measured at the daily sample size. The SCALE sigma_i is estimated
   once, from the initial model's natural day-to-day spread over the sigma window
   (weeks 3-8, robust MAD), and reused. D therefore means the same thing in every
   generation: "how far this model's inputs and outputs have moved from where they
   were when it was trained, in units of normal daily variation".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.observer.confidence import confidence_stats
from src.observer.divergence import js_divergence, predicted_histogram
from src.observer.input_drift import MMDReference, ks_mean
from src.observer.reject_rate import fit_tau, reject_rate

# name -> (family, sign): sign = +1 if the signal rises under degradation
SIGNALS: dict[str, tuple[str, int]] = {
    "max_prob_mean": ("confidence", -1),
    "max_prob_p10": ("confidence", -1),
    "margin_mean": ("confidence", -1),
    "entropy_mean": ("confidence", +1),
    "js_div": ("divergence", +1),
    "ks_mean": ("input_drift", +1),
    "mmd": ("input_drift", +1),
    "reject_rate": ("reject", +1),
}


@dataclass
class ObserverReference:
    ref_hist: np.ndarray
    ks_ref_sorted: list[np.ndarray]
    mmd_ref: MMDReference
    tau: float
    baseline: dict[str, float] = field(default_factory=dict)


def _sub(rng: np.random.Generator, n_total: int, n: int) -> np.ndarray:
    return np.arange(n_total) if n_total <= n else np.sort(rng.choice(n_total, n, replace=False))


def compute_signals(P: np.ndarray, raw: np.ndarray, std: np.ndarray, ref: ObserverReference,
                    ocfg: dict, rng: np.random.Generator) -> dict[str, float]:
    s = confidence_stats(P)
    s["js_div"] = js_divergence(P, ref.ref_hist)
    s["ks_mean"] = ks_mean(raw[_sub(rng, len(raw), ocfg["ks"]["n_cur"])], ref.ks_ref_sorted)
    s["mmd"] = ref.mmd_ref.mmd2(std[_sub(rng, len(std), ocfg["mmd"]["n_cur"])])
    s["reject_rate"] = reject_rate(P, ref.tau)
    return s


def build_reference(predict_proba, raw_ref: np.ndarray, std_ref: np.ndarray, ocfg: dict,
                    day_size: int, rng: np.random.Generator) -> ObserverReference:
    """Label-free reference for one generation, from UNLABELED held-out features of its
    training window. Half A defines the reference distributions; half B, subsampled to
    the daily sample size, measures the in-distribution level of every signal."""
    perm = rng.permutation(len(raw_ref))
    a, b = perm[: len(perm) // 2], perm[len(perm) // 2:]
    P_a = predict_proba(std_ref[a])
    ia = _sub(rng, len(a), ocfg["ks"]["n_ref"])
    ks_sorted = [np.sort(raw_ref[a][ia, j]) for j in range(raw_ref.shape[1])]
    im = _sub(rng, len(a), ocfg["mmd"]["n_ref"])
    ref = ObserverReference(
        ref_hist=predicted_histogram(P_a, P_a.shape[1]),
        ks_ref_sorted=ks_sorted,
        mmd_ref=MMDReference(std_ref[a][im]),
        tau=fit_tau(P_a, ocfg["reject_percentile"]),
    )
    ib = b[_sub(rng, len(b), day_size)]
    ref.baseline = compute_signals(predict_proba(std_ref[ib]), raw_ref[ib], std_ref[ib], ref, ocfg, rng)
    return ref


def signal_weights(ocfg: dict) -> dict[str, float]:
    if ocfg.get("weights"):
        w = {k: float(ocfg["weights"].get(k, 0.0)) for k in SIGNALS}
    elif ocfg["weighting"] == "equal":
        w = {k: 1.0 for k in SIGNALS}
    elif ocfg["weighting"] == "family":
        fam_sizes: dict[str, int] = {}
        for f, _ in SIGNALS.values():
            fam_sizes[f] = fam_sizes.get(f, 0) + 1
        w = {k: 1.0 / fam_sizes[f] for k, (f, _) in SIGNALS.items()}
    else:
        raise ValueError(ocfg["weighting"])
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()}


def robust_scale(values: np.ndarray, estimator: str = "mad") -> float:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if estimator == "std":
        s = v.std(ddof=1)
    else:
        s = 1.4826 * np.median(np.abs(v - np.median(v)))
    return float(max(s, 1e-9))


class CompositeScorer:
    """z_i = sign_i * (s_i - baseline_i(gen)) / sigma_i ;  D = sum_i w_i z_i."""

    def __init__(self, ocfg: dict):
        self.weights = signal_weights(ocfg)
        self.estimator = ocfg.get("scale_estimator", "mad")
        self.sigma: dict[str, float] | None = None

    def fit_scale(self, daily_signals: list[dict[str, float]], baselines: list[dict[str, float]]) -> None:
        """sigma from the natural spread of (signal - baseline) over the sigma window."""
        if self.sigma is not None:
            raise RuntimeError("signal scales are fit once (sigma window) and frozen")
        self.sigma = {k: robust_scale(np.array([s[k] - b[k] for s, b in zip(daily_signals, baselines)]),
                                      self.estimator) for k in SIGNALS}

    def zscores(self, s: dict[str, float], ref: ObserverReference) -> dict[str, float]:
        assert self.sigma is not None, "fit_scale first"
        return {k: sign * (s[k] - ref.baseline[k]) / self.sigma[k] for k, (_, sign) in SIGNALS.items()}

    def score(self, s: dict[str, float], ref: ObserverReference) -> float:
        z = self.zscores(s, ref)
        return float(sum(self.weights[k] * z[k] for k in SIGNALS))
