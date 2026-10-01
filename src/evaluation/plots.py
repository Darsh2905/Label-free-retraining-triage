"""Paper figures (spec §9): Pareto, cost regret, timeline, per-service degradation,
calibration transfer.

Static figures for print. Colours follow a fixed entity->slot assignment, validated
for colour-vision deficiency (adjacent CVD dE >= 9.1). The three slots below 3:1
contrast get relief from distinct markers plus direct labels, so identity never
rests on colour alone. One y-scale per panel; related measures go in stacked panels.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
FAMILY_STYLE = {   # entity -> (colour slot, marker, label)
    "periodic": ("#2a78d6", "o", "Periodic (every k weeks)"),
    "P3": ("#eb6834", "s", "P3 threshold on D"),
    "P4": ("#1baf7a", "^", "P4 proposed (myopic, spec)"),
    "P4c": ("#eda100", "D", "P4c cumulative cost"),
    "P5": ("#e87ba4", "v", "P5 threshold oracle"),
}
SPEC_POINTS = {"P0_never": "P0", "P1_monthly": "P1", "P2_weekly": "P2", "P3_threshold": "P3",
               "P4_proposed": "P4", "P4c_cumulative": "P4c", "P5_oracle": "P5"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _fig(w=6.4, h=4.0, rows=1, sharex=False, heights=None):
    fig, axes = plt.subplots(rows, 1, figsize=(w, h), sharex=sharex, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": heights} if heights else None)
    for ax in np.atleast_1d(axes):
        _style(ax)
    return fig, axes


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def pareto(summ: pd.DataFrame, frontier: pd.DataFrame | None, path: str, ratio: float) -> None:
    """x = retrains, y = mean weekly macro-F1; one curve per policy family (mean +- std
    over seeds), spec policies labelled, exact DP frontier as the reference line."""
    s = summ[summ.cost_ratio == ratio]
    fig, ax = _fig(6.6, 4.4)
    if frontier is not None and len(frontier):
        fr = frontier.groupby("n_retrains").mean_f1.mean().reset_index()
        fr = fr[fr.n_retrains <= s.n_retrains.max() + 2]
        ax.plot(fr.n_retrains, fr.mean_f1, color=INK, lw=1.2, ls="--", zorder=2)
        ax.annotate("hindsight-optimal (DP)", (fr.n_retrains.iloc[-1], fr.mean_f1.iloc[-1]), fontsize=7,
                    color=INK, xytext=(-4, 6), textcoords="offset points", ha="right")
    for fam, (col, mk, lab) in FAMILY_STYLE.items():
        f = s[s.family == fam]
        if not len(f):
            continue
        g = f.groupby("policy").agg(x=("n_retrains", "mean"), y=("mean_macro_f1", "mean"),
                                    ey=("mean_macro_f1", "std"), ex=("n_retrains", "std")).sort_values("x")
        ax.errorbar(g.x, g.y, yerr=g.ey.fillna(0), xerr=g.ex.fillna(0), color=col, marker=mk, ms=6, lw=2,
                    elinewidth=0.8, capsize=0, mec=SURFACE, mew=1.2, label=lab, zorder=3)
    p0 = s[s.policy == "P0_never"]
    if len(p0):
        ax.scatter([p0.n_retrains.mean()], [p0.mean_macro_f1.mean()], color=MUTED, marker="X", s=60, zorder=4,
                   label="P0 never")
    pts: dict[tuple, list[str]] = {}
    for name, short in SPEC_POINTS.items():
        p = s[s.policy == name]
        if len(p):
            key = (round(p.n_retrains.mean(), 2), round(p.mean_macro_f1.mean(), 4))
            pts.setdefault(key, []).append(short)
    for (x, y), shorts in pts.items():             # coinciding spec policies share one label
        ax.annotate("=".join(shorts), (x, y), fontsize=7.5, color=INK, xytext=(5, -10),
                    textcoords="offset points", fontweight="bold")
    ax.set_xlabel("retrains over the replay (count)")
    ax.set_ylabel("mean weekly macro-F1")
    ax.legend(fontsize=7, frameon=False, loc="lower right", labelcolor=INK2)
    _save(fig, path)


def cost_regret(summ: pd.DataFrame, opt: pd.DataFrame, path: str, C_fix: float) -> None:
    """Regret vs the hindsight cost-optimal schedule at each cost ratio, in units of one
    retrain's cost. P4/P4c are evaluated at their own (matched) ratio."""
    ratios = sorted(opt.cost_ratio.unique())
    best = opt.groupby(["seed", "cost_ratio"]).total_cost.first()
    rows = []
    for (seed, r), sub in summ.groupby(["seed", "cost_ratio"]):
        if (seed, r) not in best.index:
            continue
        b = best[(seed, r)]
        def reg(mask):
            v = sub[mask].total_cost
            return (v.min() - b) / C_fix if len(v) else np.nan
        rows.append({"seed": seed, "ratio": r,
                     "P4 myopic (matched r)": reg(sub.policy == f"P4_r{r}"),
                     "P4c cumulative (matched r)": reg(sub.policy == f"P4c_r{r}"),
                     "best periodic (tuned in hindsight)": reg(sub.family == "periodic"),
                     "P1 monthly": reg(sub.policy == "P1_monthly"),
                     "P3 threshold (auto)": reg(sub.policy == "P3_threshold"),
                     "P0 never": reg(sub.policy == "P0_never")})
    df = pd.DataFrame(rows)
    fig, ax = _fig(6.6, 4.2)
    styles = {"P4 myopic (matched r)": ("#1baf7a", "^", "-"), "P4c cumulative (matched r)": ("#eda100", "D", "-"),
              "best periodic (tuned in hindsight)": ("#2a78d6", "o", "-"), "P1 monthly": ("#2a78d6", "o", ":"),
              "P3 threshold (auto)": ("#eb6834", "s", "-"), "P0 never": (MUTED, "X", "-")}
    for k, (col, mk, ls) in styles.items():
        g = df.groupby("ratio")[k].agg(["mean", "std"]).reindex(ratios)
        ax.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), color=col, marker=mk, ls=ls, lw=2, ms=6,
                    mec=SURFACE, mew=1.2, elinewidth=0.8, label=k)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("cost ratio r = C_err·V_ref / (C_retrain + C_label)")
    ax.set_ylabel("regret vs hindsight optimum (retrain-cost units)")
    ax.legend(fontsize=7, frameon=False, labelcolor=INK2)
    _save(fig, path)
    return df


def timeline(tls: dict, path: str, events: dict, names=("P0_never", "P4_proposed", "P4c_cumulative", "P2_weekly")):
    fig, (a1, a2) = _fig(8.2, 5.2, rows=2, sharex=True, heights=[3, 2])
    cols = {"P0_never": (MUTED, "X"), "P4_proposed": ("#1baf7a", "^"), "P4c_cumulative": ("#eda100", "D"),
            "P2_weekly": ("#2a78d6", "o"), "P5_oracle": ("#e87ba4", "v")}
    shown = [n for n in names if n in tls]
    for n in shown:
        df = pd.DataFrame(tls[n]["rows"])
        col, mk = cols.get(n, (INK2, "o"))
        ls, lw = ((0, (3, 2)), 1.2) if n == "P0_never" else ("-", 2)
        a1.plot(df.week, df.macro_f1, color=col, lw=lw, ls=ls, marker=mk, ms=3.5, label=n, mec=SURFACE, mew=0.6,
                zorder=4 if n == "P0_never" else 3)
    lo, hi = a1.get_ylim()
    step = (hi - lo) * 0.045
    a1.set_ylim(lo - step * (len(shown) + 1), hi)
    for i, n in enumerate(shown):                  # retrain rug: one row per policy under the curves
        df = pd.DataFrame(tls[n]["rows"])
        col, mk = cols.get(n, (INK2, "o"))
        rw = df[df.retrain]
        y = lo - step * (i + 1)
        a1.scatter(rw.week, [y] * len(rw), color=col, marker=mk, s=16, zorder=5, edgecolor=SURFACE, lw=0.5)
        a1.annotate(f"retrains: {n}", (1.01, y), xycoords=("axes fraction", "data"), fontsize=6, color=INK2,
                    va="center", ha="left", annotation_clip=False)
    p0 = pd.DataFrame(tls["P0_never"]["rows"])
    a2.plot(p0.week, p0.D, color=MUTED, lw=2, label="D (never-retrained model)")
    ev = p0[p0.drift_event_week.notna()]
    a2.scatter(ev.week, ev.D, color=INK, s=14, zorder=4, label="persistent drift event active")
    for name, wk in events.items():
        for ax in (a1, a2):
            ax.axvline(wk, color=INK2, lw=0.8, ls=":")
        a1.annotate(name.replace("_", " "), (wk, a1.get_ylim()[1]), fontsize=7, color=INK2, xytext=(3, -10),
                    textcoords="offset points")
    a1.set_ylabel("weekly macro-F1")
    a2.set_ylabel("composite D (weekly mean)")
    a2.set_xlabel("week of 2022")
    a1.legend(fontsize=7, frameon=False, ncol=2, labelcolor=INK2)
    a2.legend(fontsize=7, frameon=False, labelcolor=INK2)
    _save(fig, path)


def per_service(per: pd.DataFrame, path: str, early=(3, 6), late=(40, 52), top: int = 25):
    e = per[(per.week >= early[0]) & (per.week <= early[1])].drop(columns="week").mean()
    l = per[(per.week >= late[0]) & (per.week <= late[1])].drop(columns="week").mean()
    drop = (e - l).sort_values(ascending=False).head(top)[::-1]
    fig, ax = _fig(6.4, 0.22 * len(drop) + 1.0)
    ax.barh(drop.index, drop.values, color="#2a78d6", height=0.7, edgecolor=SURFACE, lw=2)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel(f"F1 drop, weeks {early[0]}-{early[1]} mean minus weeks {late[0]}-{late[1]} mean (never retrained)")
    ax.grid(axis="y", visible=False)
    _save(fig, path)
    return drop[::-1]


def calibration_transfer(tr: dict, g, path: str):
    fig, ax = _fig(6.0, 4.2)
    e, l = tr["early_points"], tr["late_points"]
    ax.scatter(e["D"], e["loss"], color="#2a78d6", s=18, marker="o", label="calibration window (fit g)",
               edgecolor=SURFACE, lw=0.6, zorder=3)
    if l["D"]:
        ax.scatter(l["D"], l["loss"], color="#eb6834", s=22, marker="s", label="late window (transfer)",
                   edgecolor=SURFACE, lw=0.6, zorder=3)
    xs = np.linspace(min(e["D"] + l["D"]), max(e["D"] + l["D"]), 200)
    ax.plot(xs, g.predict(xs), color=INK, lw=1.5, label="g (frozen isotonic)")
    ax.set_xlabel("daily composite D")
    ax.set_ylabel("true macro-F1 loss vs model reference")
    ax.legend(fontsize=7, frameon=False, labelcolor=INK2)
    _save(fig, path)


def ensure(path):
    os.makedirs(path, exist_ok=True)
    return path
