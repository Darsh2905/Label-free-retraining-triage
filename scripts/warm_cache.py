"""Optional: pre-compute generations and (generation, day) evaluations for one seed as
data arrives, so the final replay reuses them from results/cache. Changes no result:
everything computed here is exactly what the replay would compute itself."""
import argparse
import os
import time

from _common import parse_sets

from src.pipeline import build_world
from src.utils import load_config


def complete_weeks(cfg):
    ddir = os.path.join(cfg["data_root"], cfg["dataset"], "days")
    from src.data.loader import period_of
    import datetime as dt
    have = {f[:8] for f in os.listdir(ddir) if f.endswith(".parquet")}
    weeks = []
    for w in range(cfg["replay_start"], cfg["replay_end"] + 1):
        days = [(dt.date(2022, 1, 3) + dt.timedelta(days=7 * (w - 1) + i)).strftime("%Y%m%d") for i in range(7)]
        # a day counts as present if on disk or known-missing (empty in the dataset)
        if all(d in have or d in {"20220128", "20220129", "20220130", "20221212", "20221213",
                                   "20221229", "20221230", "20221231"} for d in days):
            weeks.append(w)
        else:
            break
    return weeks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append")
    a = ap.parse_args()
    done_through = 0
    while True:
        cfg = load_config(None, parse_sets(a.set))
        weeks = complete_weeks(cfg)
        last = weeks[-1] if weeks else 0
        if last > done_through:
            cfg["replay_end"] = last
            world = build_world(cfg, a.seed)
            world.store.reset_clock()
            world.store.advance_to(last)
            for g in ["w0"] + [f"t{t}" for t in weeks[:-1]]:
                start = 3 if g == "w0" else int(g[1:]) + 1
                for w in range(start, last + 1):
                    world.weekly_counts(g, w, need_signals=True)
                world.save_cache()
            print(f"warm through week {last}: {len(world.day_cache)} day evals, {len(world.gens)} gens", flush=True)
            done_through = last
            del world
        if last >= load_config()["replay_end"]:
            break
        time.sleep(300)


if __name__ == "__main__":
    main()
