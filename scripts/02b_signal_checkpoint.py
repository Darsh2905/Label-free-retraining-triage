"""Build-order steps 3-4: D(t) over the year and the GO/NO-GO correlation checkpoint.

Evaluation-only diagnostic (spec §11.4): correlates the label-free composite D with the
TRUE loss of the never-retrained model. Labels are used here to check the design, never
by the live system. Nothing here feeds back into defaults: the family weighting was fixed
before this script ever ran, and the equal weighting is reported alongside it.

Reported correlations (Pearson and Spearman):
- level:    weekly D vs weekly loss (ref F1 - F1). This is the relationship g relies on.
- wow:      week-over-week change in D vs change in loss. The spec's literal wording.
            Correlating a level signal with a first difference is a weaker test.
- detrended: level correlation after regressing both on the week index, to guard
            against two unrelated monotone trends looking correlated.
- per-signal level correlations of each oriented z-score, to see which families carry signal.
Verdict: STOP if |Spearman level| < 0.3 (spec threshold).

    python scripts/02b_signal_checkpoint.py [--overlay quic.yaml] [--seed 0]
"""
import argparse
import os

import numpy as np
import pandas as pd
from _common import new_run_dir, parse_sets, snapshot, write_json
from scipy import stats

from src.observer.composite import SIGNALS, CompositeScorer
from src.pipeline import build_world, daily_loss
from src.utils import load_config


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return {"pearson": np.nan, "spearman": np.nan, "n": int(ok.sum())}
    return {"pearson": float(stats.pearsonr(x[ok], y[ok]).statistic),
            "spearman": float(stats.spearmanr(x[ok], y[ok]).statistic), "n": int(ok.sum())}


def detrend(v, t):
    v, t = np.asarray(v, float), np.asarray(t, float)
    b = np.polyfit(t, v, 1)
    return v - np.polyval(b, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", help="config override, e.g. --set replay_end=20")
    a = ap.parse_args()
    cfg = load_config(a.overlay, parse_sets(a.set))
    run = new_run_dir(f"checkpoint_{cfg['dataset']}")
    snapshot(cfg, run, seed=a.seed)
    world = build_world(cfg, a.seed)
    cal = world.calibration()
    tl = world.run_timeline(cfg["policies"]["p0_never"])
    w0 = world.generation("w0")
    dates = sorted(tl["daily_D"])
    daily = pd.DataFrame({"date": dates, "week": [world.period_of_date[d] for d in dates],
                          "D": [tl["daily_D"][d] for d in dates],
                          "loss": [daily_loss(world, "w0", d) for d in dates]})
    # oriented z-scores per signal, and the alternative (equal) weighting
    z = pd.DataFrame([cal["scorer"].zscores(world.day_eval("w0", d).signals, w0.reference) for d in dates])
    alt_cfg = dict(cfg["observer"], weighting="equal" if cfg["observer"]["weighting"] == "family" else "family")
    alt = CompositeScorer(alt_cfg)
    alt.sigma = cal["scorer"].sigma
    daily["D_alt"] = [alt.score(world.day_eval("w0", d).signals, w0.reference) for d in dates]
    daily = pd.concat([daily, z.add_prefix("z_")], axis=1)
    weekly = daily.groupby("week").mean(numeric_only=True).reset_index()
    wk = pd.DataFrame(tl["rows"]).set_index("week")
    weekly["loss_weekly"] = weekly.week.map(wk.loss_vs_gen_ref)       # loss from weekly-pooled counts
    daily.to_csv(os.path.join(run, "p0_daily_D_loss.csv"), index=False)
    weekly.to_csv(os.path.join(run, "p0_weekly_D_loss.csv"), index=False)

    rep = {"weighting": cfg["observer"]["weighting"], "alt_weighting": alt_cfg["weighting"]}
    rep["weekly_level"] = corr(weekly.D, weekly.loss_weekly)
    rep["weekly_level_alt_weighting"] = corr(weekly.D_alt, weekly.loss_weekly)
    rep["weekly_wow"] = corr(np.diff(weekly.D), np.diff(weekly.loss_weekly))
    rep["weekly_detrended"] = corr(detrend(weekly.D, weekly.week), detrend(weekly.loss_weekly, weekly.week))
    rep["daily_level"] = corr(daily.D, daily.loss)
    rep["per_signal_weekly_level"] = {k: corr(weekly[f"z_{k}"], weekly.loss_weekly) for k in SIGNALS}
    r = rep["weekly_level"]["spearman"]
    rep["verdict"] = "GO" if abs(r) >= 0.3 else "STOP: |Spearman(D, loss)| < 0.3 -- report as a finding"
    rep["sigma"] = cal["scorer"].sigma
    rep["pelt_penalty"] = cal["penalty"]
    rep["pelt_penalty_sweep_on_calm"] = cal["penalty_sweep"]
    write_json(rep, os.path.join(run, "checkpoint_report.json"))
    world.save_cache()
    print(weekly[["week", "D", "D_alt", "loss_weekly"]].round(4).to_string(index=False))
    for k in ("weekly_level", "weekly_level_alt_weighting", "weekly_wow", "weekly_detrended", "daily_level"):
        print(f"{k:28s} {rep[k]}")
    for k, v in rep["per_signal_weekly_level"].items():
        print(f"  z_{k:16s} spearman {v['spearman']:+.3f} pearson {v['pearson']:+.3f}")
    print("PELT penalty sweep on calm window:", cal["penalty_sweep"], "-> chosen", cal["penalty"])
    print("VERDICT:", rep["verdict"])
    print("results ->", run)


if __name__ == "__main__":
    main()
