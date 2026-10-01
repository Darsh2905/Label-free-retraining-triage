"""Build-order step 8 (spec §10): the identical pipeline on CESNET-QUIC22.

Required check: does the label-free pipeline react at or near the documented W47 event
(day 22 = 21 Nov 2022)? The check has three parts, reported separately rather than
merged into one yes/no:
  1. Is there a real drop at W47 in this setup? P0's true daily F1 (evaluation only).
     If the drop is absent here, P4 firing (or not) says nothing about the event.
  2. Does D rise, and does the detector raise a persistent drift event near day 22?
  3. Do P3 / P4 / P4c retrain near day 22, and do they avoid retraining in W45-W46?

    python scripts/04_run_quic_validation.py [--seeds 0 1 2 3 4]
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from _common import new_run_dir, parse_sets, snapshot, write_json

from src.evaluation import plots
from src.pipeline import policy_specs, run_year
from src.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", default="quic.yaml")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--set", action="append")
    a = ap.parse_args()
    cfg = load_config(a.overlay, parse_sets(a.set))
    seeds = a.seeds if a.seeds is not None else cfg["seeds"]
    run = new_run_dir(f"quic_{cfg['dataset']}")
    snapshot(cfg, run, seeds=seeds)
    event = list(cfg["evaluation"]["known_events"].values())[0]
    report, summaries, weekly = {"event_day": event, "seeds": {}}, [], []
    for seed in seeds:
        out = run_year(cfg, seed, policy_specs(cfg, sweeps=True), oracle=(seed == seeds[0]))
        world, tls = out["world"], out["timelines"]
        p0 = pd.DataFrame(tls["P0_never"]["rows"])
        pre = p0[p0.week < event].macro_f1.mean()
        post = p0[p0.week >= event].macro_f1.mean()
        ev_days = sorted(set(p0.drift_event_week.dropna().astype(int)))
        rep = {"p0_f1_before_event": pre, "p0_f1_from_event": post, "p0_drop": pre - post,
               "p0_daily_f1": dict(zip(p0.week.astype(int), p0.macro_f1.round(4))),
               "p0_daily_D": dict(zip(p0.week.astype(int), p0.D.round(3))),
               "drift_event_changepoint_days": ev_days, "ref_f1_w0": out["ref_f1_w0"],
               "pelt_penalty": out["calibration"]["penalty"]}
        for name in ("P3_threshold", "P4_proposed", "P4c_cumulative", "P1_monthly", "P5_oracle"):
            rw = tls[name]["retrain_weeks"]
            after = [w for w in rw if w >= event - 1]
            rep[name] = {"retrain_days": rw, "false_alarms_before_event": [w for w in rw if w < event - 1],
                         "first_reaction_delay_days": (after[0] - event) if after else None}
        for r in (cfg["cost"]["default_ratio"], 1, 100):
            rep[f"P4_r{r}_retrains"] = tls.get(f"P4_r{r}", {}).get("retrain_weeks")
            rep[f"P4c_r{r}_retrains"] = tls.get(f"P4c_r{r}", {}).get("retrain_weeks")
        report["seeds"][seed] = rep
        summaries.append(out["summaries"])
        weekly.append(pd.concat([pd.DataFrame(v["rows"]) for v in tls.values()]))
        if seed == seeds[0]:
            plots.timeline(tls, os.path.join(run, "figures", "quic_timeline.png"), cfg["evaluation"]["known_events"],
                           names=("P0_never", "P4_proposed", "P4c_cumulative", "P5_oracle"))
            with open(os.path.join(run, f"timelines_seed{seed}.pkl"), "wb") as f:
                pickle.dump({"timelines": {k: {kk: vv for kk, vv in v.items() if kk != "spec"} for k, v in tls.items()},
                             "g": out["calibration"]["g"]}, f)
            if "oracle_frontier" in out:
                out["oracle_frontier"].to_csv(os.path.join(run, "oracle_frontier.csv"), index=False)
                out["oracle_min_cost"].to_csv(os.path.join(run, "oracle_min_cost.csv"), index=False)
        world.save_cache()
        print(f"seed {seed}: P0 F1 {pre:.4f} -> {post:.4f} (drop {pre - post:+.4f}); events at {ev_days}; "
              + "; ".join(f"{n}: {rep[n]['retrain_days']}" for n in ("P3_threshold", "P4_proposed", "P4c_cumulative", "P5_oracle")),
              flush=True)
    s = pd.concat(summaries)
    s.to_csv(os.path.join(run, "sensitivity.csv"), index=False)
    s[s.cost_ratio == cfg["cost"]["default_ratio"]].to_csv(os.path.join(run, "policy_comparison.csv"), index=False)
    pd.concat(weekly).to_csv(os.path.join(run, "daily_rows.csv"), index=False)
    drops = [v["p0_drop"] for v in report["seeds"].values()]
    report["mean_p0_drop"] = float(np.mean(drops))
    write_json(report, os.path.join(run, "quic_validation.json"))
    print("results ->", run)


if __name__ == "__main__":
    main()
