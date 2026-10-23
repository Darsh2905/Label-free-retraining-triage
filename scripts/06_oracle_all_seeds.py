"""Exact hindsight-optimal retraining frontier for every seed (evaluation only).

03_run_replay.py computes the DP frontier for --oracle-seeds only (default: seed 0). The
paper compares every policy with the frontier at the SAME retrain budget and the SAME
seed, so this script tabulates the (generation x week) macro-F1 table F for each seed,
saves it (so cost-optimal schedules under any loss convention can be recomputed without
re-predicting), and writes the per-budget frontier.

Generations and day evaluations come from results/cache/ (same config hash as the
replay), so only pairs no policy happened to visit are computed.

    python scripts/06_oracle_all_seeds.py [--seeds 0 1 2 3 4] [--workers 3]
"""
import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from _common import new_run_dir, parse_sets, snapshot

from src.evaluation.oracle import f1_table, frontier
from src.pipeline import build_world
from src.utils import load_config


def one_seed(args):
    cfg, seed, run = args
    t0 = time.time()
    world = build_world(cfg, seed)
    weeks = sorted(p for p in world.dates_by_period if cfg["replay_start"] <= p <= cfg["replay_end"])
    F = f1_table(world, weeks, need_signals=False)
    world.save_cache()
    fr = frontier(F, weeks, max_retrains=len(weeks) - 1)
    vol = {p: float(sum(world.day[d].volume for d in world.dates_by_period[p])) for p in weeks}
    meta = {"seed": seed, "weeks": weeks, "volume": vol, "V_ref": world.V_ref,
            "ref_f1_w0": world.generation("w0").info["ref_f1"]}
    with open(os.path.join(run, f"F_seed{seed}.pkl"), "wb") as f:
        pickle.dump({"F": F, "meta": meta}, f)
    print(f"seed {seed} done in {time.time() - t0:.0f}s: |F|={len(F)}", flush=True)
    return pd.DataFrame(fr).assign(seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--set", action="append")
    a = ap.parse_args()
    cfg = load_config(None, parse_sets(a.set))
    seeds = a.seeds if a.seeds is not None else cfg["seeds"]
    cfg["classifier"]["num_threads"] = max(1, (os.cpu_count() or 4) // max(1, min(a.workers, len(seeds))))
    run = new_run_dir(f"oracle_{cfg['dataset']}")
    snapshot(cfg, run, seeds=seeds)
    print("results ->", run, flush=True)
    with ProcessPoolExecutor(min(a.workers, len(seeds))) as ex:
        frs = list(ex.map(one_seed, [(cfg, s, run) for s in seeds]))
    pd.concat(frs).to_csv(os.path.join(run, "oracle_frontier.csv"), index=False)
    print("results ->", run, flush=True)


if __name__ == "__main__":
    main()
