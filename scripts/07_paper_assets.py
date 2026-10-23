"""Build every table, figure and quoted number in paper/ from result files.

Nothing in paper/main.tex is typed from memory: tables are written to paper/tables/*.tex,
figures to paper/figures/*.pdf (vector, TrueType-embedded, 8 pt Times-compatible), and
every scalar quoted in the text to paper/numbers.json.

Loss convention. results/.../sensitivity.csv was produced with the original cost metric,
which credits a week whose F1 exceeds the fixed W0 reference with *negative* error cost
(and can drive totals below zero at large cost ratios). The paper reports the clipped
variant (a week at or above the reference costs nothing); both are recomputed here from the
stored timelines, and the DP cost-optimum is recomputed with the same convention.

    python scripts/07_paper_assets.py [--replay DIR] [--oracle DIR] [--quic DIR] [--checkpoint DIR]
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
import yaml
from _common import ROOT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluation.metrics import summarize  # noqa: E402
from src.evaluation.oracle import frontier as dp_frontier, min_cost  # noqa: E402

PAPER = os.path.join(ROOT, "paper")
FIG, TAB = os.path.join(PAPER, "figures"), os.path.join(PAPER, "tables")

INK, INK2, MUTED, GRID = "#000000", "#333333", "#7a7a7a", "#dddddd"
STYLE = {  # entity -> (colour, marker, linestyle)
    "P0": (MUTED, "X", (0, (3, 2))), "periodic": ("#2a78d6", "o", "-"), "P2": ("#2a78d6", "o", "-"),
    "P3": ("#eb6834", "s", "-"), "P4": ("#1baf7a", "^", "-"), "P4c": ("#eda100", "D", "-"),
    "P5": ("#e87ba4", "v", "-"),
}


def rc():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.linewidth": 0.6, "grid.linewidth": 0.4, "grid.color": GRID, "axes.edgecolor": INK2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "stix", "figure.dpi": 150,
    })


def latest(pattern: str) -> str:
    c = sorted(glob.glob(os.path.join(ROOT, "results", pattern)))
    if not c:
        raise SystemExit(f"no results dir matches {pattern}")
    return c[-1]


def ax_style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="both")
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.5, colors=INK2)


# ------------------------------------------------------------------------------ data
def load_replay(replay: str, oracle: str = None):
    cfg = yaml.safe_load(open(os.path.join(replay, "config_snapshot.yaml")))
    calib = {c["seed"]: c for c in json.load(open(os.path.join(replay, "calibration.json")))}
    tls = {}
    for p in sorted(glob.glob(os.path.join(replay, "timelines_seed*.pkl"))):
        seed = int(p.split("seed")[-1].split(".")[0])
        tls[seed] = pickle.load(open(p, "rb"))["timelines"]
    Fs = {}
    # F tables may come from several oracle runs (e.g. seeds split across processes); the newest file per seed wins
    files = sorted(glob.glob(os.path.join(ROOT, "results", "run_*_oracle_tls-year22", "F_seed*.pkl")), key=os.path.getmtime)
    for p in files:
        d = pickle.load(open(p, "rb"))
        Fs[d["meta"]["seed"]] = d
    # frontier recomputed from each seed's saved F table (the oracle job writes F_seed*.pkl as each seed finishes)
    frs = []
    for seed, dd in Fs.items():
        weeks = dd["meta"]["weeks"]
        frs.append(pd.DataFrame(dp_frontier(dd["F"], weeks, len(weeks) - 1)).assign(seed=seed))
    fr = pd.concat(frs, ignore_index=True)
    return cfg, calib, tls, Fs, fr


def summaries(cfg, calib, tls, ratios, clip=True) -> pd.DataFrame:
    rows = []
    for seed, tl in tls.items():
        for name, t in tl.items():
            for r in ratios:
                s = summarize(t, cfg, calib[seed]["ref_f1_w0"], calib[seed]["V_ref"], r, clip_loss=clip)
                s.update(seed=seed, family=t["family"])
                rows.append(s)
    return pd.DataFrame(rows)


def dp_gap(df: pd.DataFrame, fr: pd.DataFrame) -> pd.Series:
    """Per-row gap (F1 points) between the DP optimum at the same seed and retrain budget and
    the policy's mean F1. Non-negative by construction (tested in tests/test_policies_deterministic)."""
    best = fr.set_index(["seed", "budget"]).mean_f1
    return pd.Series([best.get((int(r.seed), int(r.n_retrains)), np.nan) - r.mean_macro_f1 for r in df.itertuples()],
                     index=df.index)


def f(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


# ------------------------------------------------------------------------------ tables
def table_main(s: pd.DataFrame, fr: pd.DataFrame, cfg, ratio: float) -> tuple[str, pd.DataFrame]:
    d = s[s.cost_ratio == ratio].copy()
    d["gap"] = dp_gap(d, fr) * 100
    C_fix = cfg["cost"]["C_retrain"] + cfg["cost"]["C_label"]
    d["cost_u"] = d.total_cost / C_fix
    rows = [("P0", "Never", "P0_never"), ("P1", "Periodic, $k{=}4$", "P1_monthly"), ("P2", "Periodic, $k{=}1$", "P2_weekly"),
            ("--", "Periodic, $k{=}8$", "periodic_8"), ("P3", "Threshold on $D$", "P3_threshold"),
            ("P4", "Myopic cost rule", "P4_proposed"), ("P4c", "Cumulative cost rule", "P4c_cumulative"),
            ("--", "Cumulative, $r{=}100$", "P4c_r100"), ("P5", "Threshold oracle", "P5_oracle")]
    out = []
    for tag, label, pol in rows:
        g = d[d.policy == pol]
        out.append(f"{tag} & {label} & {g.n_retrains.mean():.1f} & {g.mean_macro_f1.mean():.3f}"
                   f" $\\pm$ {g.mean_macro_f1.std():.3f} & {f(g.gap.mean(), 2)} & {f(g.cost_u.mean(), 1)}"
                   f" & {f(g.mean_detection_delay.mean(), 1)} & {f(g.false_trigger_rate.mean(), 2)} \\\\")
    return "\n".join(out), d


def table_sweep(s: pd.DataFrame, fr: pd.DataFrame, cfg) -> str:
    out = []
    for r in cfg["sweeps"]["cost_ratio"]:
        cells = []
        for fam in ("P4", "P4c"):
            g = s[(s.cost_ratio == 10) & (s.policy == f"{fam}_r{r:g}")]
            g = g.assign(gap=dp_gap(g, fr) * 100)
            cells += [f"{g.n_retrains.mean():.1f}", f"{g.mean_macro_f1.mean():.3f}", f"{g.gap.mean():.2f}"]
        out.append(f"{r:g} & " + " & ".join(cells) + " \\\\")
    return "\n".join(out)


# ------------------------------------------------------------------------------ figures
def fig_pareto(s, fr, ratio, path):
    d = s[s.cost_ratio == ratio]
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    ax_style(ax)
    f_ = fr.groupby("budget").mean_f1.mean()
    f_ = f_[f_.index <= 30]
    ax.plot(f_.index, f_.values, color=INK, lw=0.9, ls="--", label="Hindsight-optimal (DP)", zorder=2)
    spec = [("periodic", "Periodic, $k\\in\\{1..26\\}$", "periodic"), ("P3", "Threshold on $D$", "P3"),
            ("P4", "Myopic cost rule", "P4"), ("P4c", "Cumulative cost rule", "P4c"), ("P5", "Threshold oracle", "P5")]
    for fam, lab, key in spec:
        g = d[d.family == fam].groupby("policy").agg(x=("n_retrains", "mean"), y=("mean_macro_f1", "mean"),
                                                     sy=("mean_macro_f1", "std")).sort_values("x")
        g = g[g.x <= 30]
        c, mk, _ = STYLE[key]
        ax.errorbar(g.x, g.y, yerr=g.sy, color=c, marker=mk, ms=3.6, lw=1.0, elinewidth=0.5, capsize=0,
                    mec="white", mew=0.4, label=lab, zorder=3)
    p0 = d[d.policy == "P0_never"]
    ax.scatter([0], [p0.mean_macro_f1.mean()], color=MUTED, marker="X", s=22, zorder=4, label="Never (P0)")
    ax.set_xlabel("Retrains over 50 weeks")
    ax.set_ylabel("Mean weekly macro-F1")
    ax.set_xlim(-0.8, 30)
    ax.legend(loc="lower right", frameon=False, handlelength=1.8, borderpad=0.2, labelspacing=0.25)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def fig_timeline(tl, path, event, event_label, names, wide=True, xlabel="Week of 2022", figsize=(7.1, 2.35)):
    cols = {"P0_never": ("P0 never", "P0"), "P2_weekly": ("P2 weekly", "P2"), "P4c_cumulative": ("P4c cumulative", "P4c"),
            "P4_proposed": ("P4 myopic", "P4"), "P5_oracle": ("P5 oracle", "P5")}
    if wide:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.35, 1]})
    else:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
    for a in (a1, a2):
        ax_style(a)
    for n in names:
        df = pd.DataFrame(tl[n]["rows"])
        lab, key = cols[n]
        c, mk, ls = STYLE[key]
        a1.plot(df.week, df.macro_f1, color=c, ls=ls, lw=1.1 if n != "P0_never" else 0.9, marker=mk, ms=2.2,
                mec="white", mew=0.2, label=lab, zorder=3)
    lo, hi = a1.get_ylim()
    rows_ = [m for m in names if m not in ("P0_never", "P2_weekly")]
    rng = hi - lo
    band = 0.06 * len(rows_) + 0.02                      # own band below the data for retrain markers
    a1.set_ylim(lo - band * rng, hi)
    for i, n in enumerate(rows_):
        df = pd.DataFrame(tl[n]["rows"])
        c = STYLE[cols[n][1]][0]
        rw = df[df.retrain]
        a1.scatter(rw.week, [lo - band * rng + rng * (0.075 + 0.07 * i)] * len(rw), color=c, marker="^", s=13, edgecolor="white",
                   linewidth=0.3, zorder=6, clip_on=False)
    p0 = pd.DataFrame(tl["P0_never"]["rows"])
    a2.plot(p0.week, p0.D, color=INK2, lw=1.0, label="$D$ (never-retrained model)")
    ev = p0[p0.drift_event_week.notna()]
    a2.scatter(ev.week, ev.D, s=4, color=INK, zorder=4, label="Persistent drift event active")
    for a in (a1, a2):
        a.axvline(event, color=INK2, lw=0.6, ls=":")
    a1.text(event + 0.4, a1.get_ylim()[1], event_label, va="top", fontsize=6.5, color=INK2)
    a1.set_ylabel("Weekly macro-F1")
    a2.set_ylabel("Composite score $D$")
    a2.set_xlabel(xlabel)
    if wide:
        a1.set_xlabel(xlabel)
    a1.legend(loc="lower left" if wide else "upper left", bbox_to_anchor=(0.0, 0.10) if wide else (0.0, 1.0), frameon=False, ncol=1, handlelength=1.6, borderpad=0.2, labelspacing=0.2)
    if wide:
        a2.legend(loc="lower right", frameon=False, borderpad=0.2, labelspacing=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default=os.path.join(ROOT, "results/run_20261002_204539_replay_tls-year22"))
    ap.add_argument("--oracle", default=None)
    ap.add_argument("--quic", default=os.path.join(ROOT, "results/run_20261009_103414_quic_quic22"))
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "results/run_20261002_204408_checkpoint_tls-year22"))
    ap.add_argument("--ref", default=os.path.join(ROOT, "results/run_20261002_154154_ref_tls-year22"))
    a = ap.parse_args()
    oracle = a.oracle or latest("run_*_oracle_tls-year22")
    for d in (FIG, TAB):
        os.makedirs(d, exist_ok=True)
    rc()
    cfg, calib, tls, Fs, fr = load_replay(a.replay, oracle)
    ratios = sorted(set(cfg["sweeps"]["cost_ratio"]) | {cfg["cost"]["default_ratio"]})
    ratio = cfg["cost"]["default_ratio"]
    s_clip = summaries(cfg, calib, tls, ratios, clip=True)
    s_raw = summaries(cfg, calib, tls, [ratio], clip=False)
    N = {}

    s_clip.to_csv(os.path.join(PAPER, "summary_clipped.csv"), index=False)       # also read by scripts/08_make_deck.py
    fr.to_csv(os.path.join(PAPER, "frontier_all_seeds.csv"), index=False)
    body, d = table_main(s_clip, fr, cfg, ratio)
    main_hdr = ("\\begin{tabular}{@{}llrcrrrr@{}}\n\\toprule\n & Policy & Retrains & Macro-F1 & Gap & Cost & Delay & FT \\\\\n\\midrule\n")
    sweep_hdr = ("\\begin{tabular}{@{}rrccrcc@{}}\n\\toprule\n & \\multicolumn{3}{c}{P4 (myopic)} & \\multicolumn{3}{c}{P4c (cumulative)} \\\\\n"
                 "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\n$r$ & Ret. & F1 & Gap & Ret. & F1 & Gap \\\\\n\\midrule\n")
    open(os.path.join(TAB, "main.tex"), "w").write(main_hdr + body + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(TAB, "sweep.tex"), "w").write(sweep_hdr + table_sweep(s_clip, fr, cfg) + "\n\\bottomrule\n\\end{tabular}\n")
    for pol in ("P0_never", "P1_monthly", "P2_weekly", "periodic_8", "P3_threshold", "P4_proposed", "P4c_cumulative",
                "P4c_r100", "P4c_r300", "P5_oracle"):
        g = d[d.policy == pol]
        N[pol] = {"retrains": g.n_retrains.mean(), "f1": g.mean_macro_f1.mean(), "f1_sd": g.mean_macro_f1.std(),
                  "gap_pt": g.gap.mean(), "gap_pt_sd": g.gap.std(), "cost_units": g.cost_u.mean(),
                  "delay": g.mean_detection_delay.mean(), "false_trig": g.false_trigger_rate.mean(),
                  "delay_exporter_wk10": g.delay_exporter_update.mean()}
    raw = s_raw[s_raw.cost_ratio == ratio].groupby("policy").total_cost.mean() / (cfg["cost"]["C_retrain"] + cfg["cost"]["C_label"])
    N["unclipped_cost_units"] = {k: raw[k] for k in ("P0_never", "P1_monthly", "P2_weekly", "P3_threshold",
                                                       "P4_proposed", "P4c_cumulative", "P5_oracle")}
    # cost ranking identical under both conventions at the default ratio?
    N["cost_rank_same_clip_vs_raw"] = bool(
        list(d.groupby("policy").total_cost.mean().loc[list(raw.index)].rank().astype(int)) == list(raw.rank().astype(int)))

    # cost-optimal regret (clipped convention) from the saved F tables
    C_fix = cfg["cost"]["C_retrain"] + cfg["cost"]["C_label"]
    reg = []
    for seed, dd in Fs.items():
        m = dd["meta"]
        for r in ratios:
            opt = min_cost(dd["F"], m["weeks"], m["volume"], m["ref_f1_w0"], r * C_fix / m["V_ref"], C_fix, clip_loss=True)
            for pol in ("P0_never", "P1_monthly", "P3_threshold", f"P4_r{r:g}", f"P4c_r{r:g}"):
                x = s_clip[(s_clip.seed == seed) & (s_clip.cost_ratio == r) & (s_clip.policy == pol)]
                if len(x):
                    reg.append({"seed": seed, "r": r, "policy": pol.split("_r")[0], "regret_units": (x.total_cost.iloc[0] - opt["total_cost"]) / C_fix})
            per = s_clip[(s_clip.seed == seed) & (s_clip.cost_ratio == r) & (s_clip.family == "periodic")]
            reg.append({"seed": seed, "r": r, "policy": "best_periodic", "regret_units": (per.total_cost.min() - opt["total_cost"]) / C_fix})
    reg = pd.DataFrame(reg)
    reg.to_csv(os.path.join(PAPER, "cost_regret_clipped.csv"), index=False)
    N["regret_units_by_ratio"] = {f"{p}@{r:g}": float(v) for (p, r), v in
                                   reg.groupby(["policy", "r"]).regret_units.mean().items()}

    # DP frontier facts
    fb = fr.groupby("budget").mean_f1.mean()
    N["dp_frontier"] = {int(b): float(fb[b]) for b in (0, 1, 2, 3, 4, 5, 6, 7, 12, 20, 49) if b in fb.index}

    # checkpoint + reference
    ck = json.load(open(os.path.join(a.checkpoint, "checkpoint_report.json")))
    N["checkpoint"] = {k: ck[k] for k in ("weekly_level", "weekly_level_alt_weighting", "weekly_wow", "weekly_detrended",
                                          "daily_level", "per_signal_weekly_level", "pelt_penalty")}
    p0 = pd.read_csv(os.path.join(a.ref, "p0_weekly.csv"))
    N["p0"] = {"ref_f1_w0": calib[0]["ref_f1_w0"], "mean_f1": float(p0.macro_f1.mean()), "min_f1": float(p0.macro_f1.min()),
               "min_week": int(p0.loc[p0.macro_f1.idxmin(), "week"]), "f1_week9": float(p0[p0.week == 9].macro_f1.iloc[0]),
               "f1_week10": float(p0[p0.week == 10].macro_f1.iloc[0]), "f1_week11": float(p0[p0.week == 11].macro_f1.iloc[0]),
               "f1_week40_52": float(p0[p0.week >= 40].macro_f1.mean()), "V_ref_weekly_flows": calib[0]["V_ref"]}
    ps = pd.read_csv(os.path.join(a.ref, "p0_per_service_f1.csv"))
    e = ps[(ps.week >= 3) & (ps.week <= 6)].drop(columns="week").mean()
    l = ps[ps.week >= 40].drop(columns="week").mean()
    N["per_service_drop_top"] = {k: float(v) for k, v in (e - l).sort_values(ascending=False).head(5).items()}
    N["per_service_drop_median"] = float((e - l).median())
    N["calibration"] = {"pelt_penalty": calib[0]["pelt_penalty"], "g_floor": calib[0]["g_floor"],
                        "sigma": calib[0]["sigma"], "n_calib_days": len(calib[0]["calib_D"])}
    # TLS retrain weeks (seed 0) for the text
    N["tls_retrain_weeks_seed0"] = {k: tls[0][k]["retrain_weeks"] for k in ("P4_proposed", "P4c_cumulative", "P5_oracle", "P3_threshold")}

    # QUIC
    q = json.load(open(os.path.join(a.quic, "quic_validation.json")))
    N["quic"] = {"event_day": q["event_day"], "mean_p0_drop": q["mean_p0_drop"],
                 "seeds": {s: {k: v[k] for k in ("p0_f1_before_event", "p0_f1_from_event", "p0_drop", "drift_event_changepoint_days",
                                                  "P3_threshold", "P4_proposed", "P4c_cumulative", "P1_monthly", "P5_oracle")}
                           for s, v in q["seeds"].items()}}
    qs = pd.read_csv(os.path.join(a.quic, "sensitivity.csv"))
    qs = qs[qs.cost_ratio == ratio]
    N["quic"]["retrains_mean"] = {p: float(qs[qs.policy == p].n_retrains.mean()) for p in
                                  ("P1_monthly", "P2_weekly", "P3_threshold", "P4_proposed", "P4c_cumulative", "P5_oracle")}
    N["quic"]["f1_mean"] = {p: float(qs[qs.policy == p].mean_macro_f1.mean()) for p in
                            ("P0_never", "P1_monthly", "P3_threshold", "P4_proposed", "P4c_cumulative", "P5_oracle")}
    with open(os.path.join(a.quic, "timelines_seed0.pkl"), "rb") as fh:
        qtl = pickle.load(fh)["timelines"]

    # ---- LaTeX macros: every number quoted in the prose comes from here ----
    mac = {}
    keys = {"Pzero": "P0_never", "Pone": "P1_monthly", "Ptwo": "P2_weekly", "Peight": "periodic_8", "Pthree": "P3_threshold",
            "Pfour": "P4_proposed", "Pfourc": "P4c_cumulative", "Pfourchundred": "P4c_r100", "Pfourcthreehundred": "P4c_r300",
            "Pfive": "P5_oracle"}
    for k, pol in keys.items():
        g = d[d.policy == pol]
        mac[f"retr{k}"] = f"{g.n_retrains.mean():.1f}".rstrip("0").rstrip(".")
        mac[f"fone{k}"] = f"{g.mean_macro_f1.mean():.3f}"
        mac[f"gap{k}"] = f"{g.gap.mean():.2f}"
        mac[f"delay{k}"] = f"{g.mean_detection_delay.mean():.1f}"
        mac[f"ft{k}"] = f"{g.false_trigger_rate.mean():.2f}"
        mac[f"cost{k}"] = f"{g.cost_u.mean():.1f}"
    mac["gainweekly"] = f"{(N['P2_weekly']['f1'] - N['P4c_r300']['f1']) * 100:.1f}" if "P4c_r300" in N else ""
    rb = reg[reg.r == ratio].groupby("policy").regret_units.mean()
    for k, pol in {"Pfourc": "P4c", "Pfour": "P4", "Pthree": "P3_threshold", "Pone": "P1_monthly", "Pzero": "P0_never",
                   "Periodic": "best_periodic"}.items():
        mac[f"reg{k}"] = f"{rb[pol]:.1f}"
    n_or = len(Fs)
    mac["noracleseeds"] = str(n_or)
    mac["gapscope"] = "seed 0" if n_or == 1 else f"mean over {n_or} seeds"
    # where does P4c have the lowest cost regret among the compared policies? (derived, not asserted)
    rm = reg.groupby(["r", "policy"]).regret_units.mean().unstack()
    cmp_cols = [c for c in ("P0_never", "P1_monthly", "P3_threshold", "P4", "P4c", "best_periodic") if c in rm.columns]
    lowest = [float(r) for r, row in rm[cmp_cols].iterrows() if row.idxmin() == "P4c"]
    N["p4c_lowest_regret_ratios"] = lowest
    mac["regPfourcHigh"] = f"{rm.loc[300, 'P4c']:.0f}"
    mac["regPeriodicHigh"] = f"{rm.loc[300, 'best_periodic']:.0f}"
    mac["regPfourcVeryHigh"] = f"{rm.loc[1000, 'P4c']:.0f}"
    mac["regPeriodicVeryHigh"] = f"{rm.loc[1000, 'best_periodic']:.0f}"
    mac["regRangeLo"], mac["regRangeHi"] = f"{min(lowest):g}", f"{max(lowest):g}"
    assert lowest == [3.0, 10.0, 30.0, 100.0], f"paper text says 3<=r<=100 but P4c is lowest at {lowest}"
    fbm = fr.groupby("budget").mean_f1.mean()
    mac["dpfour"], mac["dptwelve"] = f"{fbm[4]:.3f}", f"{fbm[12]:.3f}"
    p0r = pd.DataFrame(tls[0]["P0_never"]["rows"]).set_index("week")
    mac.update(pzeroref=f"{N['p0']['ref_f1_w0']:.3f}", pzeromean=f"{N['p0']['mean_f1']:.3f}", pzeromin=f"{N['p0']['min_f1']:.3f}",
               pzerowkNine=f"{N['p0']['f1_week9']:.3f}", pzerowkTen=f"{N['p0']['f1_week10']:.3f}",
               pzerowkEleven=f"{N['p0']['f1_week11']:.3f}",
               dweekNine=f"{p0r.loc[9, 'D']:.1f}", dweekTen=f"{p0r.loc[10, 'D']:.1f}", dweekEleven=f"{p0r.loc[11, 'D']:.1f}")
    top = list(N["per_service_drop_top"].items())
    mac.update(psvcone=f"{top[0][1]:.2f}", psvctwo=f"{top[1][1]:.2f}", psvcmed=f"{N['per_service_drop_median']:.2f}")
    N["per_service_names_top2"] = [top[0][0], top[1][0]]
    # g transfer (calibration window vs weeks 40-45), averaged over seeds, never-retrained model
    ct = json.load(open(os.path.join(a.replay, "calibration_transfer.json")))
    late = [v["P0_never"]["late"] for v in ct.values() if "P0_never" in v and v["P0_never"].get("late")]
    mac["gtransrho"] = f"{np.mean([x['spearman_D_loss'] for x in late]):.2f}"
    mac["gtruemean"] = f"{np.mean([x['mean_true_loss'] for x in late]):.3f}"
    mac["gpredmean"] = f"{np.mean([x['mean_pred_loss'] for x in late]):.3f}"
    mac["gtransmae"] = f"{np.mean([x['mae'] for x in late]):.3f}"
    mac["gtransbias"] = f"{np.mean([x['bias_pred_minus_true'] for x in late]):+.3f}"
    N["g_transfer_late"] = {"mae": float(np.mean([x["mae"] for x in late])), "bias": float(np.mean([x["bias_pred_minus_true"] for x in late])),
                            "spearman": float(np.mean([x["spearman_D_loss"] for x in late]))}
    q0 = pd.DataFrame(qtl["P0_never"]["rows"]).set_index("week")
    mac.update(qfdayeight=f"{q0.loc[8, 'macro_f1']:.2f}", qfdaytwentyone=f"{q0.loc[21, 'macro_f1']:.2f}",
               qdrop=f"{q['mean_p0_drop']:.3f}",
               qdroplo=f"{min(v['p0_drop'] for v in q['seeds'].values()):.3f}".replace("-", "$-$"),
               qdrophi=f"{max(v['p0_drop'] for v in q['seeds'].values()):.3f}")
    for k, pol in {"Pfour": "P4_proposed", "Pfourc": "P4c_cumulative", "Pfive": "P5_oracle", "Pzero": "P0_never"}.items():
        mac[f"qfone{k}"] = f"{N['quic']['f1_mean'][pol]:.3f}"
    with open(os.path.join(PAPER, "numbers.tex"), "w") as fh:
        fh.write("% generated by scripts/07_paper_assets.py -- do not edit\n")
        for k, v in mac.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    json.dump(mac, open(os.path.join(PAPER, "macros.json"), "w"), indent=1)


    # ---- Table III: per-signal correlation with true weekly loss; Table IV: QUIC retrain days per seed
    names = {"max_prob_mean": ("Max. probability (mean)", "Confidence"), "max_prob_p10": ("Max. probability (10th pct.)", "Confidence"),
             "margin_mean": ("Top-2 margin (mean)", "Confidence"), "entropy_mean": ("Entropy (mean)", "Confidence"),
             "js_div": ("JS divergence, predicted classes", "Divergence"), "ks_mean": ("KS statistic (mean over features)", "Input drift"),
             "mmd": ("MMD$^2$", "Input drift"), "reject_rate": ("Reject rate", "Reject")}
    rows = []
    for k, (lab, fam) in names.items():
        v = N["checkpoint"]["per_signal_weekly_level"][k]
        rows.append(f"{lab} & {fam} & {v['spearman']:.2f} & {v['pearson']:.2f} \\\\")
    hdr = "\\begin{tabular}{@{}llcc@{}}\n\\toprule\nSignal & Family & Spearman & Pearson \\\\\n\\midrule\n"
    comp = N["checkpoint"]["weekly_level"]
    rows.append("\\midrule\nComposite $D$ & All & %.2f & %.2f \\\\" % (comp["spearman"], comp["pearson"]))
    open(os.path.join(TAB, "signals.tex"), "w").write(hdr + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    qrows = []
    seeds_q = sorted(q["seeds"], key=int)
    def days(v):
        return "--" if not v else ", ".join(str(x) for x in v)
    for lab, key in (("P3 threshold", "P3_threshold"), ("P4 myopic", "P4_proposed"), ("P4c cumulative", "P4c_cumulative"), ("P5 oracle", "P5_oracle")):
        qrows.append(lab + " & " + " & ".join(days(q["seeds"][sd][key]["retrain_days"]) for sd in seeds_q) + " \\\\")
    qrows.append("\\midrule\nChange point in $D$ & " + " & ".join(days(q["seeds"][sd]["drift_event_changepoint_days"]) for sd in seeds_q) + " \\\\")
    qh = "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n & S0 & S1 & S2 & S3 & S4 \\\\\n\\midrule\n"
    open(os.path.join(TAB, "quic.tex"), "w").write(qh + "\n".join(qrows) + "\n\\bottomrule\n\\end{tabular}\n")

    # figures
    fig_pareto(s_clip, fr, ratio, os.path.join(FIG, "pareto.pdf"))
    fig_timeline(tls[0], os.path.join(FIG, "timeline_tls.pdf"), 10, "exporter update", ["P0_never", "P2_weekly", "P4c_cumulative"])
    fig_timeline(qtl, os.path.join(FIG, "timeline_quic.pdf"), 22, "certificate change", ["P0_never", "P4c_cumulative", "P5_oracle"],
                 wide=False, xlabel="Day of replay", figsize=(3.45, 2.7))
    json.dump(N, open(os.path.join(PAPER, "numbers.json"), "w"), indent=2,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: N[k] for k in ("P4_proposed", "P4c_cumulative", "P5_oracle", "periodic_8", "P1_monthly", "cost_rank_same_clip_vs_raw")},
                     indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("paper assets ->", PAPER)


if __name__ == "__main__":
    main()
