"""The live path (src/evaluation/live.py, used by the demo) must make exactly the decisions the offline harness makes."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.harness import World
from src.evaluation.live import POLICIES, LiveSession


def _run(live: LiveSession):
    while not live.done:
        live.step()


@pytest.fixture(scope="module")
def world_and_runs(synthetic_world):
    cfg, data = synthetic_world["cfg"], synthetic_world["data"]
    w = World(cfg, data, seed=0)
    runs = {k: w.run_timeline(cfg["policies"][k]) for k in ("p3_threshold", "p4_proposed", "p4c_cumulative", "p1_monthly", "p0_never")}
    return cfg, w, runs


@pytest.mark.parametrize("key", ["p3_threshold", "p4_proposed", "p4c_cumulative", "p1_monthly", "p0_never"])
def test_live_matches_offline_harness(world_and_runs, key):
    cfg, w, runs = world_and_runs
    tl = runs[key]
    live = LiveSession(w)
    live.reset(spec=cfg["policies"][key])
    _run(live)
    assert live.retrain_weeks == tl["retrain_weeks"], f"{key}: live retrains {live.retrain_weeks} != harness {tl['retrain_weeks']}"
    assert set(live.D_by_date) == set(tl["daily_D"])
    np.testing.assert_allclose([live.D_by_date[d] for d in sorted(live.D_by_date)], [tl["daily_D"][d] for d in sorted(live.D_by_date)], rtol=1e-9, atol=1e-9)
    wk = pd.DataFrame(tl["rows"]).set_index("week")
    for r in live.week_rows:
        assert r["f1"] == pytest.approx(wk.loc[r["week"], "macro_f1"], abs=1e-12)
        if r["D_mean"] is not None:
            assert r["D_mean"] == pytest.approx(wk.loc[r["week"], "D"], abs=1e-9)


def test_label_ledger_records_exactly_the_retrains(world_and_runs):
    cfg, w, _ = world_and_runs
    live = LiveSession(w)
    live.reset(spec=cfg["policies"]["p4c_cumulative"])
    _run(live)
    led = live.ledger()
    assert led["monitor_label_reads"] == 0
    assert len(led["retrains"]) == len(live.retrain_weeks) > 0
    for e, t in zip(led["retrains"], live.retrain_weeks):
        assert (e["from"], e["to"]) == (t - live.R + 1, t)
        assert e["kind"] == "retrain"


def test_session_can_restart_and_repeat(world_and_runs):
    cfg, w, runs = world_and_runs
    live = LiveSession(w)
    for _ in range(2):
        live.reset(spec=cfg["policies"]["p3_threshold"])
        _run(live)
        assert live.retrain_weeks == runs["p3_threshold"]["retrain_weeks"]


def test_cost_ratio_can_change_while_running(world_and_runs):
    cfg, w, _ = world_and_runs
    spec = dict(POLICIES["p4c"])
    counts = {}
    for r in (0.3, 1000):
        live = LiveSession(w)
        live.reset(spec=dict(spec, cost_ratio=r), ratio=r)
        live.set_ratio(r)
        _run(live)
        counts[r] = len(live.retrain_weeks)
    assert counts[0.3] <= counts[1000]
    # a mid-run change takes effect: start at r=1000, drop to 0.3 after week 8 -> no more retrains than a pure r=1000 run
    live = LiveSession(w)
    live.reset(spec=dict(spec, cost_ratio=1000), ratio=1000)
    while not live.done and live._plan[live.pos][0] <= 8:
        live.step()
    live.set_ratio(0.3)
    _run(live)
    assert len(live.retrain_weeks) <= counts[1000]


def test_injected_drift_is_seen_and_repaired_with_a_live_trained_model(world_and_runs):
    cfg, w, _ = world_and_runs
    live = LiveSession(w)
    live.reset(spec=cfg["policies"]["p3_threshold"])
    D_before, injected_from = [], None
    while not live.done:
        wk, date = live._plan[live.pos]
        if wk == 4 and injected_from is None:             # early, inside the calm period
            live.set_injection(0.2)          # shrinking size features damages accuracy on the synthetic world (x4 barely does)
            injected_from = live.pos
        rec = live.step()[0]
        if injected_from is None:
            D_before.append(rec["D"])
    inj = [r for r in live.records if r["type"] == "day" and r["injected"]]
    assert inj, "no day was marked injected"
    assert np.mean([r["D"] for r in inj[:5]]) > np.mean(D_before) + 2.0, "injected drift did not raise D"
    live_gens = [r for r in live.week_rows if r.get("fire") and r["gen_after"].endswith("L")]
    assert live_gens, "no live-trained generation was deployed after the injection"
    assert live.live_trains == len(live_gens)
    t_fix = live_gens[0]["week"]
    f1_by_week = {r["week"]: r["f1"] for r in live.week_rows}
    assert f1_by_week[t_fix + 1] > f1_by_week[t_fix] + 0.05, "live-trained model did not recover accuracy"


def test_monitor_never_receives_labels(world_and_runs):
    """Same behavioural test as for the harness: garbage grading must not change any live decision."""
    from src.evaluation.sealed_store import SealedLabelStore
    cfg, w, runs = world_and_runs
    orig = SealedLabelStore.grade_counts
    rng = np.random.default_rng(7)

    def garbage(self, date, predictions):
        k = self.n_classes
        return {"tp": rng.integers(0, 50, k), "fp": rng.integers(0, 50, k), "fn": rng.integers(0, 50, k),
                "n_known": np.array(1000), "n_correct": np.array(100)}
    SealedLabelStore.grade_counts = garbage
    try:
        live = LiveSession(w)
        live.reset(spec=cfg["policies"]["p4c_cumulative"])
        _run(live)
    finally:
        SealedLabelStore.grade_counts = orig
    assert live.retrain_weeks == runs["p4c_cumulative"]["retrain_weeks"]
