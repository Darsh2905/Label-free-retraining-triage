"""Per-timeline metrics (spec §9): macro-F1, retrain count, cost, detection delay,
false-trigger rate.

Definitions, chosen so that they are comparable across policies:
- Error cost of week w = (ref_F1_w0 - macroF1_w) * V_w * C_err. Measuring loss against
  one fixed reference (the initial model's held-out F1) instead of the deployed model's
  own reference keeps the cost a policy-invariant function of realised F1. Any constant
  reference gives the same policy ranking.
- A retrain at the end of week t is "justified" if the deployed model's true loss in
  week t (its own held-out reference F1 minus the week's F1) exceeds
  `justified_loss`. False-trigger rate = 1 - justified fraction.
- A degradation episode starts at the first week the deployed model's true loss
  exceeds `justified_loss`. Its detection delay is the number of weeks until the policy
  retrains (0 = retrained at the end of the onset week). Episodes never acted on are
  censored at the end of the replay and counted separately.
- Known-event delay: weeks from a documented event (TLS: week-10 exporter update)
  to the first retrain at or after it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def summarize(tl: dict, cfg: dict, ref_f1_w0: float, V_ref: float, cost_ratio: float,
              clip_loss: bool = False) -> dict:
    ev = cfg["evaluation"]
    rows = pd.DataFrame(tl["rows"])
    rows = rows[(rows.week >= ev["eval_start"]) & (rows.week <= cfg["replay_end"])]
    C_fix = cfg["cost"]["C_retrain"] + cfg["cost"]["C_label"]
    C_err = cost_ratio * C_fix / V_ref
    loss = ref_f1_w0 - rows.macro_f1
    if clip_loss:      # a week above the W0 reference earns no credit (see README, cost-metric caveat)
        loss = loss.clip(lower=0.0)
    err_cost = float((loss * rows.volume * C_err).sum())
    retr = rows[rows.retrain]
    n_ret = int(len(retr))
    thr = ev["justified_loss"]
    justified = int((retr.loss_vs_gen_ref > thr).sum())
    # episodes on the policy's own trajectory
    delays, censored, in_episode, onset = [], 0, False, None
    for _, r in rows.iterrows():
        if not in_episode and r.loss_vs_gen_ref > thr:
            in_episode, onset = True, r.week
        if in_episode and r.retrain:
            delays.append(r.week - onset)
            in_episode = False
        elif in_episode and r.loss_vs_gen_ref <= thr and not r.retrain:
            in_episode = False                     # recovered on its own: not an episode to act on
    if in_episode:
        censored += 1
    known = {}
    for name, wk in (ev.get("known_events") or {}).items():
        after = retr[retr.week >= wk]
        known[f"delay_{name}"] = float(after.week.iloc[0] - wk) if len(after) else np.nan
    return {
        "policy": tl["policy"], "mean_macro_f1": float(rows.macro_f1.mean()),
        "mean_accuracy": float(rows.accuracy.mean()), "min_macro_f1": float(rows.macro_f1.min()),
        "n_retrains": n_ret, "error_cost": err_cost, "retrain_cost": n_ret * C_fix,
        "total_cost": err_cost + n_ret * C_fix, "cost_ratio": cost_ratio,
        "false_trigger_rate": (1 - justified / n_ret) if n_ret else np.nan,
        "mean_detection_delay": float(np.mean(delays)) if delays else np.nan,
        "n_episodes_acted": len(delays), "n_episodes_censored": censored,
        "retrain_weeks": " ".join(map(str, tl["retrain_weeks"])), **known,
    }
