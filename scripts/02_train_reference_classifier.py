"""Build-order step 2: train the W0 reference classifier and trace P0's degradation.

Exit criterion (spec §11.2): P0 macro-F1 degrades over the year (published pattern:
drop within weeks, then partial stabilisation or further decline). Writes
p0_weekly.csv and per-service F1 by week (for the per-service degradation figure).

    python scripts/02_train_reference_classifier.py [--overlay quic.yaml] [--seed 0]
"""
import argparse
import os

import numpy as np
import pandas as pd
from _common import new_run_dir, parse_sets, snapshot, write_json

from src.pipeline import build_world
from src.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", help="config override, e.g. --set replay_end=20")
    a = ap.parse_args()
    cfg = load_config(a.overlay, parse_sets(a.set))
    run = new_run_dir(f"ref_{cfg['dataset']}")
    snapshot(cfg, run, seed=a.seed)
    world = build_world(cfg, a.seed)
    world.store.advance_to(max(cfg["w0_periods"]))
    g = world.generation("w0")
    print(f"W0 model: {len(world.class_names)} classes, {len(world.feature_names)} features, {g.info}")
    tl = world.run_timeline(cfg["policies"]["p0_never"])
    df = pd.DataFrame(tl["rows"])
    df.to_csv(os.path.join(run, "p0_weekly.csv"), index=False)
    # per-service F1 by week (evaluation only)
    per = []
    for r in tl["rows"]:
        c = world.weekly_counts("w0", r["week"])
        denom = 2 * c["tp"] + c["fp"] + c["fn"]
        f1 = np.where(denom > 0, 2 * c["tp"] / np.maximum(denom, 1), np.nan)
        per.append({"week": r["week"], **{world.class_names[k]: f1[k] for k in range(world.n_classes)}})
    pd.DataFrame(per).to_csv(os.path.join(run, "p0_per_service_f1.csv"), index=False)
    write_json({"w0_info": g.info, "classes": world.class_names, "features": world.feature_names,
                "V_ref": world.V_ref}, os.path.join(run, "reference_model.json"))
    world.save_cache()
    print(df[["week", "macro_f1", "accuracy", "loss_vs_gen_ref"]].round(4).to_string(index=False))
    print(f"\nholdout ref F1 {g.info['ref_f1']:.4f}; P0 mean F1 {df.macro_f1.mean():.4f}, "
          f"min {df.macro_f1.min():.4f} (week {int(df.loc[df.macro_f1.idxmin(), 'week'])})")
    print("results ->", run)


if __name__ == "__main__":
    main()
