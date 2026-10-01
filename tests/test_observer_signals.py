import numpy as np
import pytest
from scipy.stats import ks_2samp

from src.data.replay import FrozenScaler
from src.observer.composite import SIGNALS, CompositeScorer, ObserverReference, signal_weights
from src.observer.confidence import confidence_stats
from src.observer.divergence import js_divergence, predicted_histogram
from src.observer.input_drift import MMDReference, ks_statistics
from src.observer.reject_rate import fit_tau, reject_rate

rng = np.random.default_rng(0)


def _softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def test_ks_matches_scipy():
    a, b = rng.normal(0, 1, (700, 3)), rng.normal(0.3, 1.2, (400, 3))
    b[:, 2] = np.round(b[:, 2])                               # ties/discrete feature
    a[:, 2] = np.round(a[:, 2])
    ours = ks_statistics(b, [np.sort(a[:, j]) for j in range(3)])
    ref = [ks_2samp(a[:, j], b[:, j]).statistic for j in range(3)]
    np.testing.assert_allclose(ours, ref, atol=1e-12)


def test_confidence_stats_known_values():
    P = np.array([[0.7, 0.2, 0.1], [0.5, 0.5, 0.0]])
    s = confidence_stats(P)
    assert s["max_prob_mean"] == pytest.approx(0.6)
    assert s["margin_mean"] == pytest.approx((0.5 + 0.0) / 2)
    ent = -(0.7 * np.log(0.7) + 0.2 * np.log(0.2) + 0.1 * np.log(0.1)) - 2 * 0.5 * np.log(0.5)
    assert s["entropy_mean"] == pytest.approx(ent / 2, rel=1e-6)


def test_js_divergence_bounds():
    P = _softmax(rng.normal(0, 2, (2000, 10)))
    h = predicted_histogram(P, 10)
    assert js_divergence(P, h) == pytest.approx(0.0, abs=1e-12)
    Q = np.zeros_like(P)
    Q[:, 0] = 1
    onehot = np.zeros(10)
    onehot[9] = 1
    assert js_divergence(Q, onehot) == pytest.approx(1.0)       # disjoint support, base 2


def test_mmd_detects_shift_and_is_unbiased_near_zero():
    ref = MMDReference(rng.normal(0, 1, (400, 5)))
    same = np.mean([ref.mmd2(rng.normal(0, 1, (400, 5))) for _ in range(10)])
    shifted = ref.mmd2(rng.normal(0.5, 1, (400, 5)))
    assert abs(same) < 0.01
    assert shifted > 10 * max(abs(same), 1e-4)


def test_reject_rate_at_reference_equals_percentile():
    P = _softmax(rng.normal(0, 2, (5000, 10)))
    tau = fit_tau(P, 10)
    assert reject_rate(P, tau) == pytest.approx(0.10, abs=0.002)


def test_sign_orientation_makes_degradation_positive():
    """A flatter (less confident) prediction matrix must raise every confidence z-score:
    the reason SIGNALS carries a sign per signal."""
    sharp = _softmax(rng.normal(0, 4, (3000, 10)))
    flat = _softmax(rng.normal(0, 1, (3000, 10)))
    s_sharp, s_flat = confidence_stats(sharp), confidence_stats(flat)
    for k in ("max_prob_mean", "max_prob_p10", "margin_mean", "entropy_mean"):
        _, sign = SIGNALS[k]
        assert sign * (s_flat[k] - s_sharp[k]) > 0, k


def test_family_weights():
    w = signal_weights({"weighting": "family"})
    assert sum(w.values()) == pytest.approx(1.0)
    fam = {}
    for k, (f, _) in SIGNALS.items():
        fam[f] = fam.get(f, 0) + w[k]
    assert all(v == pytest.approx(0.25) for v in fam.values())
    we = signal_weights({"weighting": "equal"})
    assert all(v == pytest.approx(1 / len(SIGNALS)) for v in we.values())


def test_composite_scale_is_frozen_and_baseline_is_per_generation():
    sc = CompositeScorer({"weighting": "family"})
    base = {k: 0.0 for k in SIGNALS}
    days = [{k: rng.normal() for k in SIGNALS} for _ in range(30)]
    sc.fit_scale(days, [base] * 30)
    with pytest.raises(RuntimeError):
        sc.fit_scale(days, [base] * 30)
    ref_a = ObserverReference(None, [], None, 0.0, baseline=dict(base))
    ref_b = ObserverReference(None, [], None, 0.0, baseline={k: 5.0 for k in SIGNALS})
    s = {k: 5.0 for k in SIGNALS}
    assert sc.score(s, ref_b) == pytest.approx(0.0)            # at its own baseline -> 0
    assert sc.score(s, ref_a) != pytest.approx(0.0)


def test_scaler_fit_once_and_never_changes():
    """Spec §3.2 rule 6: the W0 scaler's mean_/scale_ never change after the initial fit."""
    X = np.abs(rng.normal(0, 100, (500, 4))).astype(np.float32)
    sc = FrozenScaler(np.array([True, True, False, False])).fit(X)
    m, s = sc.mean_.copy(), sc.scale_.copy()
    for _ in range(3):
        sc.transform(np.abs(rng.normal(50, 300, (200, 4))))
    np.testing.assert_array_equal(sc.mean_, m)
    np.testing.assert_array_equal(sc.scale_, s)
    with pytest.raises(RuntimeError):
        sc.fit(X)
    with pytest.raises(ValueError):
        sc.mean_[0] = 1.0
