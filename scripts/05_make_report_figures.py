"""Build-order step 9: figures and summary tables from a replay results directory.

    python scripts/05_make_report_figures.py results/run_<ts>_replay_tls-year22 [--per-service results/run_..._ref_.../p0_per_service_f1.csv]
"""
import argparse
import glob
import json
import os
import pickle

import pandas as pd
from _common import write_json

from src.evaluation import plots
from src.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--per-service", default=None, help="p0_per_service_f1.csv from script 02")
    a = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(os.path.join(a.run_dir, "config_snapshot.yaml")))
    fig = plots.ensure(os.path.join(a.run_dir, "figures"))
    summ = pd.read_csv(os.path.join(a.run_dir, "sensitivity.csv"))
    fr_p = os.path.join(a.run_dir, "oracle_frontier.csv")
    fr = pd.read_csv(fr_p) if os.path.exists(fr_p) else None
    ratio = cfg["cost"]["default_ratio"]
    plots.pareto(summ, fr, os.path.join(fig, "pareto.png"), ratio)
    mc_p = os.path.join(a.run_dir, "oracle_min_cost.csv")
    if os.path.exists(mc_p):
        C_fix = cfg["cost"]["C_retrain"] + cfg["cost"]["C_label"]
        reg = plots.cost_regret(summ, pd.read_csv(mc_p), os.path.join(fig, "cost_regret.png"), C_fix)
        reg.to_csv(os.path.join(a.run_dir, "cost_regret.csv"), index=False)
    pk = sorted(glob.glob(os.path.join(a.run_dir, "timelines_seed*.pkl")))[0]
    d = pickle.load(open(pk, "rb"))
    plots.timeline(d["timelines"], os.path.join(fig, "timeline.png"), cfg["evaluation"].get("known_events") or {})
    tr = json.load(open(os.path.join(a.run_dir, "calibration_transfer.json")))
    first = tr[sorted(tr)[0]]
    for name in ("P0_never", "P4c_cumulative"):
        if name in first:
            plots.calibration_transfer(first[name], d["g"], os.path.join(fig, f"calibration_transfer_{name}.png"))
    if a.per_service:
        drop = plots.per_service(pd.read_csv(a.per_service), os.path.join(fig, "per_service_degradation.png"),
                                 late=tuple(cfg.get("per_service_late", (40, cfg["replay_end"]))))
        drop.to_csv(os.path.join(a.run_dir, "per_service_drop.csv"))
    # compact summary table (mean +- std over seeds) at the default ratio
    pc = summ[summ.cost_ratio == ratio]
    agg = pc.groupby(["family", "policy"]).agg(
        f1=("mean_macro_f1", "mean"), f1_sd=("mean_macro_f1", "std"), retrains=("n_retrains", "mean"),
        retrains_sd=("n_retrains", "std"), cost=("total_cost", "mean"), cost_sd=("total_cost", "std"),
        false_trig=("false_trigger_rate", "mean"), delay=("mean_detection_delay", "mean")).reset_index()
    agg.to_csv(os.path.join(a.run_dir, "summary_table.csv"), index=False)
    print(agg.round(4).to_string(index=False))
    print("figures ->", fig)


if __name__ == "__main__":
    main()
