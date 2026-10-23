"""Live, day-by-day execution of the full pipeline (backs the interactive demo in demo/).

Everything the dashboard shows is computed here, for real, one day at a time:
deployed model -> predict_proba -> 8 label-free signals -> composite D -> online detector ->
cost rule -> (weekly) retrain decision. Nothing is replayed from a cache except the pre-trained
retrained models, which are loaded from disk exactly as the offline experiments load them.

This module lives in src.evaluation, i.e. on the harness side of the label wall, and plays the same role
as World.run_timeline: it is the only place that
  * grades predictions against true labels, for DISPLAY ONLY (the monitor never sees them), and
  * fetches labels for a retrain, through the sealed store (which logs every read).
The monitor components it calls (observer, detector, policies) stay label-free, and the label ledger it
exposes is read straight from the store's access log.

Equivalence with the offline harness is a tested property (tests/test_live_session.py): for the same
world, policy and seed, the live path produces the same daily D, the same drift events and the same
retrain weeks as World.run_timeline.

Injected drift ("what-if"): the operator can scale size-like features of the incoming flows. The monitor
sees only the perturbed features; true labels are unchanged. A retrain whose training window contains
perturbed days cannot use the pre-trained models (they never saw such data), so it trains a model live
on the buffered, perturbed flows with a smaller budget (fewer rounds, fewer flows) and says so.
"""
from __future__ import annotations

import datetime as dt
import time
import zlib
from typing import Optional

import numpy as np

from src.classifier.frozen_state import DeployedModel, Generation, train_generation
from src.decision.cost_model import c_err_for_ratio
from src.detection.detector import _ordinal
from src.evaluation.harness import World
from src.evaluation.sealed_store import macro_f1_from_counts
from src.observer.composite import build_reference, compute_signals

POLICIES = {
    "p4c": {"name": "P4c cumulative cost rule (ours)", "type": "proposed", "rule": "cumulative", "require_drift_event": False},
    "p4": {"name": "P4 myopic cost rule (spec)", "type": "proposed", "rule": "myopic", "require_drift_event": True},
    "p3": {"name": "P3 fixed threshold alarm", "type": "threshold", "threshold": "auto"},
    "p1": {"name": "P1 retrain every 4 weeks", "type": "periodic", "every": 4},
    "p0": {"name": "P0 never retrain", "type": "never"},
}
SIZE_NAMES = ("BYTES", "BYTES_REV", "PACKETS", "PACKETS_REV")
SIGNAL_LABELS = {"max_prob_mean": "Max probability (mean)", "max_prob_p10": "Max probability (10th pct.)", "margin_mean": "Top-2 margin",
                 "entropy_mean": "Entropy", "js_div": "JS divergence (predicted classes)", "ks_mean": "KS statistic (inputs)",
                 "mmd": "Kernel MMD² (inputs)", "reject_rate": "Reject rate"}
DEMO_ROUNDS, DEMO_MAX_TRAIN = 60, 40_000        # budget of a live (injected-drift) retrain


def _fmt(date: str) -> str:
    return f"{date[:4]}-{date[4:6]}-{date[6:]}"


class LiveSession:
    def __init__(self, world: World, p0_reference: Optional[dict[int, float]] = None, live_threads: int = 8):
        self.w = world
        self.cfg = world.cfg
        self.cal = world.calibration()
        self.scorer = self.cal["scorer"]
        self.s_hi = self.cfg["sigma_window"][1]
        self.R = int(self.cfg["retrain_window"])
        self.weeks = sorted(p for p in world.dates_by_period if self.cfg["replay_start"] <= p <= self.cfg["replay_end"])
        self.last = self.cfg["replay_end"]
        self.p0_reference = p0_reference or {}
        self.live_threads = live_threads
        self.size_cols = [i for i, n in enumerate(world.feature_names) if n in SIZE_NAMES] or list(range(min(4, len(world.feature_names))))
        self.cal_log_len = len(world.store.access_log)      # ledger entries made before any session (the calibration read)
        self.reset("p4c", self.cfg["cost"]["default_ratio"])

    # ------------------------------------------------------------------ session control
    def reset(self, policy="p4c", ratio: float | None = None, spec: dict | None = None) -> None:
        w, c = self.w, self.cfg
        self.policy_key = policy if spec is None else "custom"
        self.ratio = float(ratio if ratio is not None else c["cost"]["default_ratio"])
        self.spec = dict(spec) if spec is not None else dict(POLICIES[policy], cost_ratio=self.ratio)
        self.policy, self.active_from = w.make_policy(self.spec, lambda wk: 0.0)
        self.deployed = DeployedModel(w.generation("w0"))
        self.det = w._new_detector()
        w.store.reset_clock()
        w.store.advance_to(max(c["w0_periods"]))
        self.ledger_base = len(w.store.access_log)
        self.records: list[dict] = []
        self.buffer: dict[str, np.ndarray] = {}          # the flows the model actually saw (perturbed copies when injected)
        self.perturbed: set[str] = set()
        self.inject: Optional[dict] = None
        self.fed_gen: Optional[str] = None
        self.D_by_date: dict[str, float] = {}
        self.retrain_weeks: list[int] = []
        self.week_rows: list[dict] = []
        self._plan = [(wk, d) for wk in self.weeks for d in sorted(w.dates_by_period[wk])]
        self.pos = 0
        self.week_counts: list[dict] = []
        self.week_D: list[float] = []
        self.live_trains = 0
        self.busy: Optional[str] = None

    @property
    def total_days(self) -> int:
        return len(self._plan)

    @property
    def done(self) -> bool:
        return self.pos >= len(self._plan)

    def set_ratio(self, ratio: float) -> None:
        """Change the cost ratio r while running (cost rules only)."""
        self.ratio = float(ratio)
        cost = self.cfg["cost"]
        if hasattr(self.policy, "C_err"):
            ce = c_err_for_ratio(self.ratio, self.w.V_ref, cost["C_retrain"], cost["C_label"])
            self.policy.C_err = ce
            if getattr(self.policy, "_cum", None) is not None:
                self.policy._cum.C_err = ce

    def set_injection(self, factor: Optional[float]) -> None:
        """Scale the size-like features of all FUTURE flows by `factor` (None/1 stops the injection)."""
        self.inject = None if factor in (None, 1, 1.0) else {"factor": float(factor)}
        week, date = self._plan[min(self.pos, len(self._plan) - 1)]
        self.records.append({"type": "inject", "factor": None if self.inject is None else self.inject["factor"], "i": self.pos,
                             "week": week, "date": _fmt(date)})

    # ------------------------------------------------------------------ one day
    def step(self) -> list[dict]:
        """Process the next day (and, on the last day of a week, make the weekly decision). Returns new records."""
        if self.done:
            return []
        w = self.w
        week, date = self._plan[self.pos]
        days_of_week = sorted(w.dates_by_period[week])
        last_in_week = date == days_of_week[-1]
        w.store.advance_to(week)
        gen = self.deployed.current

        raw = w.day[date].raw
        if self.inject is not None:
            raw = raw.copy()
            raw[:, self.size_cols] *= self.inject["factor"]
            self.perturbed.add(date)
        self.buffer[date] = raw
        std = w.scaler.transform(raw)
        P = gen.model.predict_proba(std)
        rng = np.random.default_rng([w.seed, int(date), zlib.crc32(gen.gen_id.encode()), 17])
        sig = compute_signals(P, raw, std, gen.reference, self.cfg["observer"], rng)
        z = self.scorer.zscores(sig, gen.reference)
        D = float(self.scorer.score(sig, gen.reference))
        self.D_by_date[date] = D
        if self.fed_gen is not None and gen.gen_id != self.fed_gen:
            self.det.new_generation(_ordinal(date))
        self.fed_gen = gen.gen_id
        self.det.observe(date, D)

        pred = P.argmax(1)
        cnt = w.store.grade_counts(date, pred)                   # DISPLAY ONLY: the monitor never reads this
        f1 = macro_f1_from_counts(cnt["tp"], cnt["fp"], cnt["fn"])
        self.week_counts.append(cnt)
        self.week_D.append(D)

        out = [self._day_record(week, date, gen, P, pred, sig, z, D, f1)]
        self.pos += 1
        if last_in_week:
            out.append(self._end_of_week(week, days_of_week))
        self.records.extend(out)
        return out

    # ------------------------------------------------------------------ weekly decision
    def _end_of_week(self, week: int, days: list[str]) -> dict:
        w, t = self.w, week
        gen = self.deployed.current
        D_t = float(np.mean([self.D_by_date[d] for d in days])) if t >= self.s_hi else None
        event = self.det.detect() if t >= self.s_hi else None
        volume = float(sum(w.day[d].volume for d in days))
        tot = {k: sum(c[k] for c in self.week_counts) for k in ("tp", "fp", "fn")}
        f1_week = macro_f1_from_counts(tot["tp"], tot["fp"], tot["fn"])
        fire = False
        if t >= self.active_from and t < self.last:
            fire = bool(self.policy.decide(t, D_t, event, volume=volume))
        panel = self._panel(D_t, volume, event, final=True)
        rec = {"type": "week", "week": t, "gen": gen.gen_id, "D_mean": D_t, "f1": f1_week, "f1_p0": self.p0_reference.get(t),
               "event": ({"week": event.week, "magnitude": event.magnitude} if event else None), "fire": fire, "panel": panel,
               "volume": volume, "monitoring": t >= self.s_hi}
        if fire:
            new, mode, window = self._deploy(t)
            rec.update({"gen_after": new.gen_id, "mode": mode, "label_window": window})
            self.retrain_weeks.append(t)
            self.policy.notify_retrain(t)
        self.week_counts, self.week_D = [], []
        self.week_rows.append(rec)
        return rec

    def _deploy(self, t: int):
        w = self.w
        window = list(range(t - self.R + 1, t + 1))
        dates = [d for p in window for d in w.dates_by_period.get(p, [])]
        before = len(w.store.access_log)
        if any(d in self.perturbed for d in dates):
            self.busy = f"Retraining live on the labels of weeks {min(window)}–{max(window)} (a few seconds)…"
            try:
                gen = self._train_live(t, window)
            finally:
                self.busy = None
            mode = f"trained live on the last {self.R} weeks of (perturbed) labeled flows: {DEMO_ROUNDS} rounds, ≤{DEMO_MAX_TRAIN // 1000}k flows"
            self.live_trains += 1
        else:
            gen = w.generation(f"t{t}")                    # pre-trained offline with the same procedure (or trained now if missing)
            mode = f"model trained on the last {self.R} weeks of labels (pre-trained offline with the experiment's procedure)"
        if len(w.store.access_log) == before:               # cached model: still read (and log) the labels a retrain consumes
            w.store.get_retrain_labels(range(min(window), max(window) + 1))
        self.deployed.swap(gen, t)
        return gen, mode, [min(window), max(window)]

    def _train_live(self, t: int, window: list[int]) -> Generation:
        w, c = self.w, self.cfg
        labels = w.store.get_retrain_labels(range(min(window), max(window) + 1))
        dates = sorted(labels)
        raw = np.concatenate([self.buffer.get(d, w.day[d].raw) for d in dates])
        y = np.concatenate([labels[d] for d in dates]).astype(np.int64)
        known = np.flatnonzero(y >= 0)
        rng = np.random.default_rng([w.seed, zlib.crc32(f"t{t}L".encode()), 11])
        perm = rng.permutation(known)
        n_hold = int(len(perm) * float(c["holdout_fraction"]))
        hold, train = perm[:n_hold], perm[n_hold:]
        if len(train) > DEMO_MAX_TRAIN:
            train = rng.choice(train, DEMO_MAX_TRAIN, replace=False)
        std = w.scaler.transform(raw)
        params = dict(c["classifier"], n_estimators=DEMO_ROUNDS, num_threads=self.live_threads)
        gen = train_generation(f"t{t}L", max(window), std[train], y[train], w.n_classes, params, w.seed)
        gen.reference = build_reference(gen.model.predict_proba, raw[hold], std[hold], c["observer"], w.day_size,
                                        np.random.default_rng([w.seed, zlib.crc32(f"t{t}L".encode()), 13]))
        gen.info.update(n_train=int(len(train)), live=True)
        return gen

    # ------------------------------------------------------------------ what the cost rule is "thinking"
    def _panel(self, D_t, volume, event, final: bool) -> dict:
        """Decisive quantity of the current policy, normalised so that 1.0 = 'retrain now'."""
        p, spec = self.policy, self.spec
        t = spec["type"]
        if t == "never":
            return {"kind": "never", "value": 0.0, "text": "This policy never retrains."}
        if D_t is None:
            return {"kind": t, "value": 0.0, "text": "Calibrating: signal scales are fixed on weeks 3–8, decisions start after week 8."}
        if t == "periodic":
            return {"kind": t, "value": 0.0, "text": f"Retrains on a calendar: every {spec['every']} weeks, whatever the traffic does."}
        if t == "threshold":
            return {"kind": t, "value": D_t / p.threshold, "text": f"Weekly D = {D_t:.1f}; alarm threshold = {p.threshold:.1f}. Cost is ignored."}
        g_val = float(p.g.predict([D_t])[0])
        if p.rule == "cumulative":
            cum = p._cum
            acc = cum.accumulated
            if not final:
                acc = acc + max(0.0, g_val - cum.floor) * volume * p.C_err
            return {"kind": "cumulative", "value": acc / cum.C_fix, "g": g_val,
                    "text": f"Expected F1 loss g(D) = {g_val:.3f}. Excess loss accumulated since deployment, priced in retrain units: "
                            f"{acc / cum.C_fix:.2f} (retrain at 1.00)."}
        cost_inaction = g_val * volume * p.C_err
        ratio = cost_inaction / (p.C_retrain + p.C_label)
        if not final:
            tail = "A persistent drift event must also be active; that check runs at the end of the week."
        else:
            tail = f"A persistent drift event is {'ACTIVE' if event is not None else 'not active'} (both conditions are needed)."
        return {"kind": "myopic", "value": ratio, "g": g_val, "event": event is not None,
                "text": f"Expected F1 loss g(D) = {g_val:.3f}. This week's cost of inaction = {ratio:.2f} × cost of fixing. {tail}"}

    # ------------------------------------------------------------------ records
    def _day_record(self, week, date, gen, P, pred, sig, z, D, f1) -> dict:
        w = self.w
        names = w.class_names
        share = np.bincount(pred, minlength=len(names)) / len(pred)
        ref = gen.reference.ref_hist
        top = np.argsort(-np.maximum(share, ref))[:8]
        maxp = P.max(1)
        hist = np.histogram(maxp, bins=10, range=(0, 1))[0] / len(maxp)
        rs = np.random.default_rng([int(date), 5])
        idx = rs.choice(len(pred), 6, replace=False)
        fn = w.feature_names
        col = lambda n: fn.index(n) if n in fn else None
        raw = self.buffer[date]
        samples = []
        for i in idx:
            row = {"pred": names[pred[i]], "conf": float(maxp[i]), "uncertain": bool(maxp[i] < gen.reference.tau)}
            for lab, a, b in (("bytes", "BYTES", "BYTES_REV"), ("packets", "PACKETS", "PACKETS_REV")):
                ca, cb = col(a), col(b)
                row[lab] = float((raw[i, ca] if ca is not None else 0) + (raw[i, cb] if cb is not None else 0)) if ca is not None else float(raw[i, 0])
            cd = col("DURATION")
            row["duration"] = float(raw[i, cd]) if cd is not None else None
            samples.append(row)
        dd = dt.date(int(date[:4]), int(date[4:6]), int(date[6:]))
        n_week = len([d for d in self.w.dates_by_period[week] if d <= date])
        run_mean = float(np.mean(self.week_D))
        vol_est = w.V_ref
        panel = self._panel(run_mean if week >= self.s_hi else None, vol_est, None, final=False)
        return {"type": "day", "i": self.pos, "date": _fmt(date), "dow": dd.strftime("%a"), "week": week, "gen": gen.gen_id,
                "injected": date in self.perturbed, "D": D, "f1": f1, "z": {k: float(v) for k, v in z.items()},
                "sig": {k: float(v) for k, v in sig.items()}, "maxprob": float(maxp.mean()), "tau": float(gen.reference.tau),
                "reject": float((maxp < gen.reference.tau).mean()), "n": int(len(pred)),
                "top": [[names[k], float(share[k]), float(ref[k])] for k in top], "conf_hist": [float(x) for x in hist],
                "samples": samples, "monitoring": week >= self.s_hi, "panel": panel, "day_of_week_no": n_week}

    # ------------------------------------------------------------------ ledger / status
    def ledger(self) -> dict:
        log = self.w.store.access_log
        retrains = [{"kind": k, "from": a, "to": b} for k, a, b in log[self.ledger_base:]]
        cal = [{"kind": k, "from": a, "to": b} for k, a, b in log[:self.cal_log_len]]
        return {"calibration": cal, "retrains": retrains, "monitor_label_reads": 0}

    def status(self) -> dict:
        week, date = self._plan[min(self.pos, len(self._plan) - 1)]
        return {"week": week, "date": _fmt(date), "gen": self.deployed.current.gen_id, "pos": self.pos, "total": self.total_days,
                "policy": self.spec["name"], "ratio": self.ratio, "retrains": list(self.retrain_weeks), "done": self.done, "busy": self.busy,
                "injection": self.inject, "live_trains": self.live_trains}

    def config(self) -> dict:
        starts = {}
        for i, (wk, _) in enumerate(self._plan):
            starts.setdefault(wk, i)
        return {"week_start_idx": starts, "total_days": self.total_days, "policies": {k: v["name"] for k, v in POLICIES.items()}, "ratio_default": self.cfg["cost"]["default_ratio"],
                "weeks": [self.weeks[0], self.weeks[-1]], "monitor_from_week": self.s_hi + 1, "n_classes": self.w.n_classes,
                "n_features": len(self.w.feature_names), "retrain_window": self.R, "demo_rounds": DEMO_ROUNDS, "dataset": self.cfg["dataset"],
                "C_fix": self.cfg["cost"]["C_retrain"] + self.cfg["cost"]["C_label"]}
