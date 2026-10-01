"""Build-order step 7: full replay, all policies, parameter sweeps, all seeds.

For every seed: P0-P5 + P4c, each policy family's parameter sweep (Pareto curves), the
cost-ratio sensitivity sweep (§7.3; P4/P4c are re-run at each ratio), the calibration
transfer check (§7.4), and, for --oracle-seeds, the exact DP frontier and
cost-optimal schedules.

Outputs in results/run_<ts>_replay_<dataset>/:
  config_snapshot.yaml, policy_comparison.csv (one row per policy x seed at the default
  ratio), sensitivity.csv (every policy x seed x ratio), weekly_signals.csv (every
  timeline's weekly rows), oracle_frontier.csv, oracle_min_cost.csv,
  calibration.json, calibration_transfer.json, timelines_seed<k>.pkl

    python scripts/03_run_replay.py --config configs/base.yaml [--seeds 0 1 2 3 4] [--workers 3]
"""
import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from _common import new_run_dir, parse_sets, snapshot, write_json

from src.pipeline import calibration_transfer, policy_specs, run_year
from src.utils import load_config


def _one_seed(args):
    cfg, seed, oracle, run = args
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)
    out = run_year(cfg, seed, policy_specs(cfg, sweeps=True), oracle=oracle, log=log)
    world, cal = out["world"], out["calibration"]
    transfer = {n: calibration_transfer(world, out["timelines"][n])
                for n in ("P0_never", "P4_proposed", "P4c_cumulative") if n in out["timelines"]}
    world.save_cache()
    calib = {"seed": seed, "sigma": cal["scorer"].sigma, "weights": cal["scorer"].weights,
             "pelt_penalty": cal["penalty"], "pelt_penalty_sweep_calm": cal["penalty_sweep"],
             "anchor": cal["anchor"], "p3_auto_threshold": cal["p3_auto_threshold"], "g_floor": cal["g_floor"],
             "V_ref": world.V_ref, "ref_f1_w0": out["ref_f1_w0"], "w0_info": world.generation("w0").info,
             "calib_D": cal["calib_D"].tolist(), "calib_loss": cal["calib_loss"].tolist(),
             "calib_periods": cal["calib_periods"].tolist()}
    tls = {k: {kk: vv for kk, vv in v.items() if kk != "spec"} for k, v in out["timelines"].items()}
    with open(os.path.join(run, f"timelines_seed{seed}.pkl"), "wb") as f:
        pickle.dump({"timelines": tls, "g": cal["g"], "class_names": world.class_names}, f)
    log(f"seed {seed} done")
    return {"seed": seed, "summaries": out["summaries"], "calibration": calib, "transfer": transfer,
            "frontier": out.get("oracle_frontier"), "min_cost": out.get("oracle_min_cost"),
            "weekly": pd.concat([pd.DataFrame(v["rows"]).assign(family=v["family"]) for v in tls.values()])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml", help="base config (other configs/*.yaml are merged)")
    ap.add_argument("--overlay", default=None, help="dataset overlay, e.g. quic.yaml")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--oracle-seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--set", action="append")
    a = ap.parse_args()
    cfg = load_config(a.overlay, parse_sets(a.set))
    seeds = a.seeds if a.seeds is not None else cfg["seeds"]
    cfg["classifier"]["num_threads"] = max(1, (os.cpu_count() or 4) // max(1, min(a.workers, len(seeds))))
    run = new_run_dir(f"replay_{cfg['dataset']}")
    snapshot(cfg, run, seeds=seeds, oracle_seeds=a.oracle_seeds)
    print("results ->", run, flush=True)
    jobs = [(cfg, s, s in a.oracle_seeds, run) for s in seeds]
    with ProcessPoolExecutor(min(a.workers, len(seeds))) as ex:
        res = list(ex.map(_one_seed, jobs))
    summ = pd.concat([r["summaries"] for r in res], ignore_index=True)
    summ.to_csv(os.path.join(run, "sensitivity.csv"), index=False)
    summ[summ.cost_ratio == cfg["cost"]["default_ratio"]].to_csv(os.path.join(run, "policy_comparison.csv"), index=False)
    pd.concat([r["weekly"] for r in res]).to_csv(os.path.join(run, "weekly_signals.csv"), index=False)
    fr = [r["frontier"] for r in res if r["frontier"] is not None]
    if fr:
        pd.concat(fr).to_csv(os.path.join(run, "oracle_frontier.csv"), index=False)
        pd.concat([r["min_cost"] for r in res if r["min_cost"] is not None]).to_csv(
            os.path.join(run, "oracle_min_cost.csv"), index=False)
    write_json([r["calibration"] for r in res], os.path.join(run, "calibration.json"))
    write_json({r["seed"]: r["transfer"] for r in res}, os.path.join(run, "calibration_transfer.json"))
    pc = summ[summ.cost_ratio == cfg["cost"]["default_ratio"]]
    main_pol = ["P0_never", "P1_monthly", "P2_weekly", "P3_threshold", "P4_proposed", "P4c_cumulative", "P5_oracle"]
    tab = pc[pc.policy.isin(main_pol)].groupby("policy")[["mean_macro_f1", "n_retrains", "total_cost",
                                                          "false_trigger_rate", "mean_detection_delay"]].agg(["mean", "std"])
    print(tab.round(4).to_string())
    print("results ->", run)


if __name__ == "__main__":
    main()
