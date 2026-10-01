import numpy as np
import pytest

from src.decision.calibration import LossCalibrator
from src.decision.cost_model import CumulativeCostRule, c_err_for_ratio, should_retrain


class StubG:
    def __init__(self, f):
        self.f = f

    def predict(self, D):
        return np.array([self.f(d) for d in D])


def test_should_retrain_is_the_spec_formula():
    g = StubG(lambda d: 0.1 * d)
    # cost_of_inaction = 0.1*D * V * C_err ; cost_of_fixing = C_retrain + C_label
    assert should_retrain(2.0, g, V=1000, C_err=3.0, C_retrain=500, C_label=99) is True     # 600 > 599
    assert should_retrain(2.0, g, V=1000, C_err=3.0, C_retrain=500, C_label=100) is False   # 600 > 600 false


def test_ratio_helper_roundtrip():
    C_err = c_err_for_ratio(7.0, V_ref=2e6, C_retrain=100, C_label=400)
    assert C_err * 2e6 / 500 == pytest.approx(7.0)


def test_myopic_decisions_monotone_in_ratio():
    g = StubG(lambda d: max(0.0, 0.02 * d))
    Ds = np.linspace(0, 20, 41)
    prev = None
    for ratio in [0.3, 1, 3, 10, 30, 100]:
        C_err = c_err_for_ratio(ratio, 1e6, 100, 400)
        fired = {D for D in Ds if should_retrain(D, g, 1e6, C_err, 100, 400)}
        if prev is not None:
            assert prev <= fired
        prev = fired


def test_cumulative_rule_accumulates_and_resets():
    g = StubG(lambda d: 0.05)
    r = CumulativeCostRule(g, C_err=1.0, C_retrain=100, C_label=0, floor=0.01)
    fires = [r.update(1.0, V=1000) for _ in range(3)]          # +40 per step
    assert fires == [False, False, True]
    r.reset()
    assert r.update(1.0, V=1000) is False


def test_calibrator_monotone_frozen_and_extrapolation():
    D = np.array([0, 1, 2, 3, 4, 5.0])
    loss = np.array([0.0, 0.02, 0.01, 0.05, 0.04, 0.06])
    g = LossCalibrator("clip").fit(D, loss)
    pred = g.predict(np.linspace(-1, 10, 50))
    assert np.all(np.diff(pred) >= -1e-12)
    assert g.predict([10])[0] == pytest.approx(g.predict([5])[0])     # flat beyond calibration max
    with pytest.raises(RuntimeError):
        g.fit(D, loss)
    gl = LossCalibrator("linear").fit(D, loss)
    assert gl.predict([10])[0] > gl.predict([5])[0]                   # keeps rising
    assert np.all(np.diff(gl.predict(np.linspace(-1, 20, 80))) >= -1e-12)
