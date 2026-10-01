"""Orchestrates one full year-run end to end (spec §1, `src/pipeline.py`).

build_world -> World (splits loader output into label-free replay + sealed store)
policy_specs -> the six spec policies, P4c, and each policy family's parameter sweep
run_year -> every timeline for one seed, plus DP oracle, sensitivity and transfer checks
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.data.loader import load_world
from src.decision.calibration import transfer_report
from src.evaluation.harness import World
from src.evaluation.metrics import summarize
from src.evaluation.oracle import f1_table, frontier, min_cost
from src.evaluation.sealed_store import macro_f1_from_counts
from src.utils import ROOT, set_seed


def build_world(cfg: dict, seed: int) -> World:
    set_seed(seed)
    data = load_world(cfg, seed)
    cache = os.path.join(ROOT, cfg["cache_root"]) if cfg.get("cache_root") else None
    return World(cfg, data, seed, cache)


def policy_specs(cfg: dict, sweeps: bool = True) -> list[dict]:
    specs = [cfg["policies"][k] for k in ("p0_never", "p1_monthly", "p2_weekly", "p3_threshold",
                                          "p4_proposed", "p4c_cumulative", "p5_oracle")]
    if not sweeps:
        return specs
    sw = cfg["sweeps"]
    fam = []
    fam += [{"name": f"periodic_{k}", "family": "periodic", "type": "periodic", "every": k} for k in sw["periodic_every"]]
    fam += [{"name": f"P3_z{z}", "family": "P3", "type": "threshold", "z": z} for z in sw["p3_z"]]
    for r in sw["cost_ratio"]:
        fam.append({"name": f"P4_r{r}", "family": "P4", "type": "proposed", "rule": "myopic",
                    "require_drift_event": True, "cost_ratio": r})
        fam.append({"name": f"P4c_r{r}", "family": "P4c", "type": "proposed", "rule": "cumulative",
                    "require_drift_event": False, "cost_ratio": r})
    fam += [{"name": f"P5_d{d}", "family": "P5", "type": "oracle", "target_drop": d} for d in sw["p5_target_drop"]]
    return specs + fam


def daily_loss(world: World, gen_key: str, date: str) -> float:
    c = world.day_eval(gen_key, date).counts
    return world.generation(gen_key).info["ref_f1"] - macro_f1_from_counts(c["tp"], c["fp"], c["fn"])


def calibration_transfer(world: World, timeline: dict) -> dict:
    """§7.4 (evaluation only): g frozen on the calibration window vs realised loss later."""
    cfg, cal = world.cfg, world.calibration()
    lo, hi = cfg["transfer_window"]
    gen_of = {}
    for r in timeline["rows"]:
        for d in world.dates_by_period.get(r["week"], []):
            gen_of[d] = r["gen"]
    dates = [d for d in sorted(timeline["daily_D"]) if lo <= world.period_of_date[d] <= hi]
    D = np.array([timeline["daily_D"][d] for d in dates])
    loss = np.array([daily_loss(world, gen_of[d], d) for d in dates])
    early = transfer_report(cal["g"], cal["calib_D"], cal["calib_loss"])
    late = transfer_report(cal["g"], D, loss) if len(dates) else {}
    return {"policy": timeline["policy"], "early_in_sample": early, "late": late,
            "late_points": {"D": D.tolist(), "loss": loss.tolist(),
                            "gens": sorted({gen_of[d] for d in dates})},
            "early_points": {"D": cal["calib_D"].tolist(), "loss": cal["calib_loss"].tolist()}}


def run_year(cfg: dict, seed: int, specs: list[dict], world: World | None = None, oracle: bool = True,
             log=print) -> dict:
    world = world or build_world(cfg, seed)
    cal = world.calibration()
    ref0 = world.generation("w0").info["ref_f1"]
    timelines = {}
    for i, s in enumerate(specs):
        timelines[s["name"]] = world.run_timeline(s)
        timelines[s["name"]]["family"] = s.get("family", s["name"])
        log(f"  seed {seed} [{i + 1}/{len(specs)}] {s['name']}: retrains at {timelines[s['name']]['retrain_weeks']}")
        world.save_cache()
    ratios = sorted(set(cfg["sweeps"]["cost_ratio"]) | {cfg["cost"]["default_ratio"]})
    summaries = []
    for name, tl in timelines.items():
        for r in ratios:
            s = summarize(tl, cfg, ref0, world.V_ref, r)
            s.update(seed=seed, family=tl["family"])
            summaries.append(s)
    out = {"world": world, "timelines": timelines, "summaries": pd.DataFrame(summaries),
           "calibration": cal, "ref_f1_w0": ref0}
    if oracle:
        weeks = sorted(p for p in world.dates_by_period if cfg["replay_start"] <= p <= cfg["replay_end"])
        F = f1_table(world, weeks)
        world.save_cache()
        fr = frontier(F, weeks, max_retrains=len(weeks) - 1)
        vol = {r["week"]: r["volume"] for r in timelines[specs[0]["name"]]["rows"]}
        C_fix = cfg["cost"]["C_retrain"] + cfg["cost"]["C_label"]
        opt = [{"cost_ratio": r, **min_cost(F, weeks, vol, ref0, r * C_fix / world.V_ref, C_fix)} for r in ratios]
        out.update(oracle_frontier=pd.DataFrame(fr).assign(seed=seed),
                   oracle_min_cost=pd.DataFrame(opt).assign(seed=seed), f1_table=F)
    return out
