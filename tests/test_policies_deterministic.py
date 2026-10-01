import numpy as np
import pandas as pd
import pytest

from src.decision.policies import PeriodicRetrain
from src.evaluation.harness import World
from src.evaluation.metrics import summarize
from src.evaluation.oracle import f1_table, frontier, min_cost

ALL = ["p0_never", "p1_monthly", "p2_weekly", "p3_threshold", "p4_proposed", "p4c_cumulative", "p5_oracle"]


def test_periodic_schedule():
    p = PeriodicRetrain(4, start_week=3)
    assert [w for w in range(3, 20) if p.decide(w, None, None)] == [6, 10, 14, 18]


@pytest.fixture(scope="module")
def runs(synthetic_world):
    cfg, data = synthetic_world["cfg"], synthetic_world["data"]
    w = World(cfg, data, seed=0)
    return cfg, data, w, {k: w.run_timeline(cfg["policies"][k]) for k in ALL}


def test_same_seed_same_results(runs, synthetic_world):
    cfg, data, _, first = runs
    w2 = World(cfg, data, seed=0)
    for k in ["p4c_cumulative", "p3_threshold", "p1_monthly"]:
        second = w2.run_timeline(cfg["policies"][k])
        assert first[k]["retrain_weeks"] == second["retrain_weeks"]
        np.testing.assert_allclose([r["macro_f1"] for r in first[k]["rows"]],
                                   [r["macro_f1"] for r in second["rows"]])


def test_timelines_are_isolated(runs):
    """Running P2 (many retrains) in the same World must not alter P0's timeline."""
    cfg, _, w, first = runs
    w.run_timeline(cfg["policies"]["p2_weekly"])
    again = w.run_timeline(cfg["policies"]["p0_never"])
    assert again["retrain_weeks"] == []
    np.testing.assert_allclose([r["macro_f1"] for r in again["rows"]],
                               [r["macro_f1"] for r in first["p0_never"]["rows"]])


def test_dp_oracle_dominates_every_policy(runs):
    cfg, _, w, tls = runs
    weeks = list(range(cfg["replay_start"], cfg["replay_end"] + 1))
    F = f1_table(w, weeks)
    fr = frontier(F, weeks, max_retrains=len(weeks) - 1)
    best = [f["mean_f1"] for f in fr]
    assert all(b2 >= b1 - 1e-12 for b1, b2 in zip(best, best[1:]))          # monotone in budget
    for k, tl in tls.items():
        rows = pd.DataFrame(tl["rows"])
        n = len(tl["retrain_weeks"])
        assert rows.macro_f1.mean() <= best[n] + 1e-9, f"{k} beats the exact oracle?"
    # cost-optimal schedule is no worse than any policy at the same ratio
    vol = {r["week"]: r["volume"] for r in tls["p0_never"]["rows"]}
    ref = w.generation("w0").info["ref_f1"]
    for ratio in (1, 10, 100):
        C_fix = cfg["cost"]["C_retrain"] + cfg["cost"]["C_label"]
        C_err = ratio * C_fix / w.V_ref
        opt = min_cost(F, weeks, vol, ref, C_err, C_fix)
        for k, tl in tls.items():
            s = summarize(tl, cfg, ref, w.V_ref, ratio)
            assert opt["total_cost"] <= s["total_cost"] + 1e-6, (k, ratio)
