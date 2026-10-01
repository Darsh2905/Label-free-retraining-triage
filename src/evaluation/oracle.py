"""Exact hindsight-optimal retraining schedules (evaluation only; not a competing policy).

P5 (threshold oracle) is what the spec calls the upper bound, but a threshold rule on
true F1 is not optimal: it retrains after damage is visible and ignores what the
next model will achieve. Because a retrained model depends only on its training week
(see harness.py), the macro-F1 of every (generation, week) pair can be tabulated once.
Dynamic programming then yields:

- `frontier`: for each retrain budget N, the schedule maximising mean weekly macro-F1.
  This is the true Pareto frontier that every policy's points lie on or below.
- `min_cost`: the schedule minimising total cost at a given cost ratio, the reference for
  cost-regret in the sensitivity sweep.
"""
from __future__ import annotations

import numpy as np

from src.evaluation.sealed_store import macro_f1_from_counts


def f1_table(world, weeks: list[int]) -> dict[tuple[str, int], float]:
    world.store.reset_clock()
    world.store.advance_to(max(weeks))              # hindsight: evaluation after the year
    keys = ["w0"] + [f"t{t}" for t in weeks[:-1]]
    F = {}
    for i, g in enumerate(keys):
        start = 0 if g == "w0" else weeks.index(int(g[1:])) + 1
        for w in weeks[start:]:
            c = world.weekly_counts(g, w, need_signals=True)   # one predict serves oracle + all timelines
            F[(g, w)] = macro_f1_from_counts(c["tp"], c["fp"], c["fn"])
    world.store.reset_clock()
    return F


def _gen(weeks, j):
    return "w0" if j == 0 else f"t{weeks[j - 1]}"


def frontier(F: dict, weeks: list[int], max_retrains: int) -> list[dict]:
    """best[n][j]: max sum of F1 over weeks[j:] with generation _gen(j) deployed at week j
    and at most n retrains left. Retraining at the end of weeks[i] deploys _gen(i+1)."""
    m = len(weeks)
    pref = {}
    for j in range(m):
        g = _gen(weeks, j)
        vals = np.array([F[(g, w)] for w in weeks[j:]])
        pref[j] = np.concatenate([[0.0], np.cumsum(vals)])      # pref[j][k] = sum weeks[j:j+k]
    seg = lambda j, i: pref[j][i - j + 1]                       # sum over weeks[j..i] with gen j
    best = np.full((max_retrains + 1, m + 1), -np.inf)
    choice = np.full((max_retrains + 1, m + 1), -1, int)
    for n in range(max_retrains + 1):
        for j in range(m - 1, -1, -1):
            b, arg = seg(j, m - 1), -1
            if n > 0:
                for i in range(j, m - 1):
                    v = seg(j, i) + best[n - 1][i + 1]
                    if v > b:
                        b, arg = v, i
            best[n][j], choice[n][j] = b, arg
    out = []
    for n in range(max_retrains + 1):
        sched, j, k = [], 0, n
        while k >= 0 and choice[k][j] >= 0:
            i = choice[k][j]
            sched.append(weeks[i])
            j, k = i + 1, k - 1
        out.append({"budget": n, "n_retrains": len(sched), "mean_f1": best[n][0] / m, "schedule": sched})
    return out


def min_cost(F: dict, weeks: list[int], volume: dict[int, float], ref_f1: float, C_err: float,
             C_fix: float) -> dict:
    """Minimise sum_w (ref_f1 - F1_w) * V_w * C_err + n_retrains * C_fix."""
    m = len(weeks)
    cost = lambda j, w: (ref_f1 - F[(_gen(weeks, j), w)]) * volume[w] * C_err
    best = np.zeros(m + 1)
    choice = np.full(m + 1, -1, int)
    for j in range(m - 1, -1, -1):
        run, b, arg = 0.0, None, -1
        for i in range(j, m):
            run += cost(j, weeks[i])
            if i == m - 1:
                v = run
            else:
                v = run + C_fix + best[i + 1]
            if b is None or v < b:
                b, arg = v, (i if i < m - 1 else -1)
        best[j], choice[j] = b, arg
    sched, j = [], 0
    while choice[j] >= 0:
        sched.append(weeks[choice[j]])
        j = choice[j] + 1
    return {"total_cost": float(best[0]), "schedule": sched, "n_retrains": len(sched)}
