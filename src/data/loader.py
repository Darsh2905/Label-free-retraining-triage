"""Load the per-day samples written by scripts/01_fetch_data.py into memory.

This is a LABEL-BEARING module: `load_world` returns labels alongside features.
Only src/pipeline.py and src/evaluation/ may import it; they split its output into a
label-free `Replay` (features only) and a `SealedLabelStore` (labels only). The
static leakage test forbids observer/, detection/, decision/ and classifier/ from
importing this module, directly or transitively.

Preprocessing rules implemented here (spec §3.2, with documented deviations):
1. Class subset: top-N services by flow volume. Default `class_selection: w0` ranks
   by W0 volume only (information available at deployment time). The spec's literal
   rule ranks by whole-year volume, which uses future information and silently
   removes the "a top service disappears" form of drift; available as `year`.
   Volumes come from each day's stats json (full-day counts, not our sample).
2. Closed world (spec): flows outside the top-N are dropped from the stream.
   `closed_world: false` keeps them in the stream (label -1) so monitors see the
   open-world mix; they are never graded or trained on.
3. Per-day subsampling is uniform and label-agnostic (see src/data/fetch.py for why a
   per-service cap is not used). `flows_per_day` caps each day at a fixed size, which
   also keeps the finite-sample bias of KS/JS/MMD constant across days.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.fetch import PHIST_COLS, SCALAR_COLS

LOG1P_PREFIXES = ("BYTES", "PACKETS", "DURATION", "PPI_", "PHIST_")


@dataclass
class DayData:
    date: str
    period: int
    raw: np.ndarray        # (n, F) float32, untransformed flow statistics
    labels: np.ndarray     # (n,) int16 class index, -1 = outside the known class set
    volume: float          # estimated true in-stream flow count for the day


@dataclass
class WorldData:
    dataset: str
    feature_names: list[str]
    class_names: list[str]
    days: list[DayData]
    period_of_date: dict[str, int] = field(default_factory=dict)

    def periods(self) -> list[int]:
        return sorted({d.period for d in self.days})


def period_of(date: str, period: str) -> int:
    """Map YYYYMMDD to a decision-period index. `iso_week`: ISO week number, with the
    days before ISO week 1 mapped to 0 (matches the dataset's WEEK-2022-00 folder).
    `day`: consecutive day index starting at 1 on the first available date."""
    d = dt.date(int(date[:4]), int(date[4:6]), int(date[6:]))
    if period == "iso_week":
        y, w, _ = d.isocalendar()
        return 0 if y < d.year else w
    raise ValueError(period)


def _feature_names(columns) -> list[str]:
    names = [c for c in SCALAR_COLS if c in columns]
    names += [c for c in columns if any(c.startswith(p + "_") for p in PHIST_COLS)]
    return names


def log_transform_mask(feature_names: list[str]) -> np.ndarray:
    return np.array([n.startswith(LOG1P_PREFIXES) for n in feature_names])


def _app_count(c) -> int:
    """TLS-Year22's stats json gives a plain int per app (rows actually saved to file).
    QUIC22's gives {"seen": N, "saved": M} -- N before, M after the dataset's own
    per-app downsampling. Use "saved" so the ranking is consistent across both
    datasets: "how many of this app's flows are actually in the file we read."""
    return int(c["saved"]) if isinstance(c, dict) else int(c)


def select_classes(meta_by_date: dict[str, dict], dates: list[str], n: int,
                   exclude: tuple[str, ...] = ()) -> list[str]:
    counts: dict[str, int] = {}
    for d in dates:
        for app, c in meta_by_date[d]["app_counts_full_day"].items():
            if app not in exclude:
                counts[app] = counts.get(app, 0) + _app_count(c)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [a for a, _ in ranked[:n]]


def load_world(cfg: dict, seed: int) -> WorldData:
    ddir = os.path.join(cfg["data_root"], cfg["dataset"], "days")
    files = sorted(f[:-8] for f in os.listdir(ddir) if f.endswith(".parquet"))
    metas = {d: json.load(open(os.path.join(ddir, f"{d}.json"))) for d in files}
    files = [d for d in files if metas[d]["n_sampled"] > 0]
    if cfg["period"] == "day":
        o0 = min(dt.date(int(d[:4]), int(d[4:6]), int(d[6:])).toordinal() for d in files)
        pmap = {d: dt.date(int(d[:4]), int(d[4:6]), int(d[6:])).toordinal() - o0 + 1 for d in files}
    else:
        pmap = {d: period_of(d, cfg["period"]) for d in files}
    first, last = min(cfg["w0_periods"]), cfg["replay_end"]
    files = [d for d in files if first <= pmap[d] <= last]
    w0_dates = [d for d in files if pmap[d] in cfg["w0_periods"]]
    if not w0_dates:
        raise RuntimeError("no W0 days available on disk")
    sel_dates = w0_dates if cfg["class_selection"] == "w0" else files
    classes = select_classes(metas, sel_dates, cfg["n_classes"], tuple(cfg.get("exclude_classes") or ()))
    cidx = {c: i for i, c in enumerate(classes)}
    cap = int(cfg["flows_per_day"])
    days, feat_names = [], None
    for d in files:
        df = pd.read_parquet(os.path.join(ddir, f"{d}.parquet"))
        if feat_names is None:
            feat_names = _feature_names(df.columns)
        lab = df["APP"].map(cidx).fillna(-1).astype(np.int16).to_numpy()
        # Stream definition: closed world keeps only known-class flows (spec §3.2).
        keep = lab >= 0 if cfg["closed_world"] else np.ones(len(lab), bool)
        rate = metas[d]["sample_rate"]
        volume = float(keep.sum()) / rate
        idx = np.flatnonzero(keep)
        rng = np.random.default_rng([seed, int(d), 7])
        if len(idx) > cap:
            idx = np.sort(rng.choice(idx, cap, replace=False))
        raw = df[feat_names].to_numpy(np.float32)[idx]
        days.append(DayData(d, pmap[d], raw, lab[idx], volume))
    return WorldData(cfg["dataset"], feat_names, classes, days, pmap)
