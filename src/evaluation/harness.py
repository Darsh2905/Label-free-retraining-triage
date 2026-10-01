"""Evaluation harness (spec §9): replays the year once per policy, with isolated state.

Key design point: a model retrained at the end of week t is trained on the labels of
weeks t-R+1..t, whichever policy asked for it. Its predictions on a given day are
likewise policy-independent. The `World` therefore memoizes generations by training
week and (generation, day) evaluations. Every policy timeline is still fully isolated:
it has its own deployed-model pointer, detector, and policy state, and cannot see
another timeline's decisions. Memoization only avoids recomputing identical work.
The same property makes the exact DP oracle (src/evaluation/oracle.py) tractable.

Label flow in this module (the only glue allowed to touch the sealed store):
- generation training            -> store.get_retrain_labels  (clock-guarded)
- one-time calibration of g      -> store.get_calibration_labels
- per-day grading for reporting  -> store.grade_counts (never passed to observer,
                                    detector or policies, except P5's injected oracle fn)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pickle
import zlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.classifier.frozen_state import DeployedModel, Generation, train_generation
from src.data.loader import WorldData, log_transform_mask, period_of
from src.data.replay import FrozenScaler, Replay, StreamDay
from src.decision.calibration import LossCalibrator
from src.decision.cost_model import c_err_for_ratio
from src.decision.policies import (NeverRetrain, OracleRetrain, PeriodicRetrain, ProposedRetrain,
                                   ThresholdRetrain)
from src.detection.detector import OnlineDriftDetector, _ordinal
from src.detection.normalize import calm_stats
from src.evaluation.sealed_store import SealedLabelStore, macro_f1_from_counts
from src.observer.composite import SIGNALS, CompositeScorer, build_reference, compute_signals


@dataclass
class DayEval:
    signals: dict
    counts: dict


def _counts(y: np.ndarray, pred: np.ndarray, k: int) -> dict:
    known = y >= 0
    tp = np.bincount(y[known][pred[known] == y[known]], minlength=k)
    return {"tp": tp, "fp": np.bincount(pred, minlength=k) - tp,
            "fn": np.bincount(y[known], minlength=k) - tp,
            "n_known": np.array(int(known.sum())), "n_correct": np.array(int(tp.sum()))}


def _sum_counts(cs: list[dict]) -> dict:
    out = {k: v.copy() for k, v in cs[0].items()}
    for c in cs[1:]:
        for k in out:
            out[k] = out[k] + c[k]
    return out


def _config_hash(cfg: dict, keys: list[str]) -> str:
    sub = {k: cfg.get(k) for k in keys}
    if isinstance(sub.get("classifier"), dict):     # thread count does not change results (deterministic=True)
        sub["classifier"] = {k: v for k, v in sub["classifier"].items() if k != "num_threads"}
    return hashlib.sha1(json.dumps(sub, sort_keys=True, default=str).encode()).hexdigest()[:10]


class World:
    """Everything that is shared, policy-independent, and memoizable for one seed."""

    def __init__(self, cfg: dict, data: WorldData, seed: int, cache_dir: Optional[str] = None):
        self.cfg, self.seed = cfg, seed
        self.n_classes = len(data.class_names)
        self.class_names = data.class_names
        self.feature_names = data.feature_names
        self.day_size = int(cfg["flows_per_day"])
        pmap = {d.date: d.period for d in data.days}
        self.period_of_date = pmap
        self.store = SealedLabelStore({d.date: d.labels for d in data.days}, pmap, self.n_classes,
                                      tuple(cfg["calibration_window"]))
        # From here on, features and labels travel separately.
        self.stream_days = [StreamDay(d.date, d.period, d.raw, d.volume) for d in data.days]
        self.day = {d.date: d for d in self.stream_days}
        self.dates_by_period: dict[int, list[str]] = {}
        for d in self.stream_days:
            self.dates_by_period.setdefault(d.period, []).append(d.date)
        w0_raw = np.concatenate([self.day[x].raw for p in cfg["w0_periods"] for x in self.dates_by_period.get(p, [])])
        self.scaler = FrozenScaler(log_transform_mask(self.feature_names)).fit(w0_raw)
        w0_vol = [sum(self.day[x].volume for x in self.dates_by_period[p]) for p in cfg["w0_periods"]]
        self.V_ref = float(np.mean(w0_vol))            # nominal weekly volume, label-free
        self.gens: dict[str, Generation] = {}
        self.day_cache: dict[tuple[str, str], DayEval] = {}
        self._cal: Optional[dict] = None
        self.cache_dir = None
        if cache_dir:
            h_model = _config_hash(cfg, ["dataset", "flows_per_day", "max_train_flows", "holdout_fraction",
                                         "retrain_window", "classifier", "n_classes", "class_selection",
                                         "closed_world", "w0_periods", "observer", "exclude_classes"])
            self.cache_dir = os.path.join(cache_dir, f"{cfg['dataset']}_{h_model}_s{seed}")
            os.makedirs(self.cache_dir, exist_ok=True)
            p = os.path.join(self.cache_dir, "day_cache.pkl")
            if os.path.exists(p):
                self.day_cache = pickle.load(open(p, "rb"))

    # ------------------------------------------------------------------ periods
    def period_fn(self, date: str) -> int:
        if date in self.period_of_date:
            return self.period_of_date[date]
        if self.cfg["period"] == "iso_week":
            return period_of(date, "iso_week")
        o = _ordinal(date)
        near = min(self.period_of_date, key=lambda d: abs(_ordinal(d) - o))
        return self.period_of_date[near]

    # ------------------------------------------------------------------ generations
    def _window(self, key: str) -> list[int]:
        if key == "w0":
            return list(self.cfg["w0_periods"])
        t = int(key[1:])
        return list(range(t - int(self.cfg["retrain_window"]) + 1, t + 1))

    def generation(self, key: str) -> Generation:
        if key in self.gens:
            return self.gens[key]
        path = os.path.join(self.cache_dir, f"gen_{key}.pkl") if self.cache_dir else None
        if path and os.path.exists(path):
            self.gens[key] = pickle.load(open(path, "rb"))
            return self.gens[key]
        periods = self._window(key)
        labels = self.store.get_retrain_labels(range(min(periods), max(periods) + 1))
        dates = sorted(labels)
        raw = np.concatenate([self.day[d].raw for d in dates])
        y = np.concatenate([labels[d] for d in dates]).astype(np.int64)
        known = np.flatnonzero(y >= 0)             # never train on out-of-set flows
        rng = np.random.default_rng([self.seed, zlib.crc32(key.encode()), 11])
        perm = rng.permutation(known)
        n_hold = int(len(perm) * float(self.cfg["holdout_fraction"]))
        hold, train = perm[:n_hold], perm[n_hold:]
        if len(train) > int(self.cfg["max_train_flows"]):
            train = rng.choice(train, int(self.cfg["max_train_flows"]), replace=False)
        std = self.scaler.transform(raw)
        gen = train_generation(key, max(periods), std[train], y[train], self.n_classes,
                               self.cfg["classifier"], self.seed)
        # label-free observer reference from the UNLABELED holdout features
        gen.reference = build_reference(gen.model.predict_proba, raw[hold], std[hold], self.cfg["observer"],
                                        self.day_size, np.random.default_rng([self.seed, zlib.crc32(key.encode()), 13]))
        # reference macro-F1 on held-out chunks of daily size (labels: retrain window, permitted)
        chunks = [hold[i:i + self.day_size] for i in range(0, len(hold), self.day_size)][:3]
        f1s, accs = [], []
        for c in chunks:
            if len(c) < self.day_size // 2:
                continue
            pred = gen.model.predict_proba(std[c]).argmax(1)
            cnt = _counts(y[c], pred, self.n_classes)
            f1s.append(macro_f1_from_counts(cnt["tp"], cnt["fp"], cnt["fn"]))
            accs.append(float(cnt["n_correct"] / max(cnt["n_known"], 1)))
        gen.info.update(ref_f1=float(np.mean(f1s)), ref_acc=float(np.mean(accs)),
                        n_train=int(len(train)), n_holdout=int(n_hold), window=periods)
        self.gens[key] = gen
        if path:
            pickle.dump(gen, open(path, "wb"))
        return gen

    # ------------------------------------------------------------------ per-day evaluation
    def day_eval(self, gen_key: str, date: str, need_signals: bool = True) -> DayEval:
        """Predict once per (generation, day); grade always, observer signals on demand."""
        k = (gen_key, date)
        ev = self.day_cache.get(k)
        if ev is not None and (ev.signals is not None or not need_signals):
            return ev
        gen = self.generation(gen_key)
        raw = self.day[date].raw
        std = self.scaler.transform(raw)
        P = gen.model.predict_proba(std)
        sig = None
        if need_signals:
            rng = np.random.default_rng([self.seed, int(date), zlib.crc32(gen_key.encode()), 17])
            sig = compute_signals(P, raw, std, gen.reference, self.cfg["observer"], rng)
        cnt = ev.counts if ev is not None else self.store.grade_counts(date, P.argmax(1))
        ev = DayEval(sig, cnt)
        self.day_cache[k] = ev
        return ev

    def save_cache(self) -> None:
        if self.cache_dir:
            tmp = os.path.join(self.cache_dir, "day_cache.pkl.tmp")
            pickle.dump(self.day_cache, open(tmp, "wb"))
            os.replace(tmp, os.path.join(self.cache_dir, "day_cache.pkl"))

    def weekly_counts(self, gen_key: str, period: int, need_signals: bool = False) -> dict:
        return _sum_counts([self.day_eval(gen_key, d, need_signals).counts for d in self.dates_by_period[period]])

    # ------------------------------------------------------------------ one-time calibration
    def calibration(self) -> dict:
        """Setup phase on the initial model's trajectory (identical to P0's):
        sigma window -> signal scales, detector anchor and PELT penalty, P3 threshold;
        calibration window -> g (via the store's calibration accessor only)."""
        if self._cal is not None:
            return self._cal
        c = self.cfg
        s_lo, s_hi = c["sigma_window"]
        k_lo, k_hi = c["calibration_window"]
        self.store.reset_clock()
        self.store.advance_to(max(c["w0_periods"]))
        w0 = self.generation("w0")
        sig_dates = [d for p in range(s_lo, s_hi + 1) for d in self.dates_by_period.get(p, [])]
        self.store.advance_to(s_hi)
        scorer = CompositeScorer(c["observer"])
        scorer.fit_scale([self.day_eval("w0", d).signals for d in sig_dates], [w0.reference.baseline] * len(sig_dates))
        det = OnlineDriftDetector(c["detection"], self.period_fn)
        for d in sig_dates:
            det.observe(d, scorer.score(self.day_eval("w0", d).signals, w0.reference))
        det_cal = det.calibrate()
        D_calm = np.array(det.values)
        loc, scale = calm_stats(D_calm)
        # g: labels ONLY through the calibration accessor, predictions recomputed here
        self.store.advance_to(k_hi)
        labels = self.store.get_calibration_labels(range(k_lo, k_hi + 1))
        Ds, losses, periods = [], [], []
        for d in sorted(labels):
            std = self.scaler.transform(self.day[d].raw)
            pred = w0.model.predict_proba(std).argmax(1)
            cnt = _counts(labels[d], pred, self.n_classes)
            f1 = macro_f1_from_counts(cnt["tp"], cnt["fp"], cnt["fn"])
            Ds.append(scorer.score(self.day_eval("w0", d).signals, w0.reference))
            losses.append(w0.info["ref_f1"] - f1)
            periods.append(self.period_of_date[d])
        Ds, losses, periods = np.array(Ds), np.array(losses), np.array(periods)
        g = LossCalibrator(c["decision"]["g_extrapolation"]).fit(Ds, losses)
        floor = float(np.mean(losses[periods == k_lo]))
        self.store.reset_clock()
        self._cal = {"scorer": scorer, "anchor": det_cal["anchor"], "penalty": det_cal["penalty"],
                     "penalty_sweep": det_cal["sweep"], "g": g, "g_floor": floor,
                     "calm_loc": loc, "calm_scale": scale,
                     "p3_auto_threshold": loc + float(c["decision"]["p3_auto_z"]) * scale,
                     "calib_D": Ds, "calib_loss": losses, "calib_periods": periods}
        return self._cal

    # ------------------------------------------------------------------ policies
    def make_policy(self, spec: dict, true_loss_fn):
        c, cal = self.cfg, self.calibration()
        t, name = spec["type"], spec.get("name", spec["type"])
        cost = c["cost"]
        if t == "never":
            pol, active = NeverRetrain(name), c["replay_start"]
        elif t == "periodic":
            pol, active = PeriodicRetrain(spec["every"], c["replay_start"], name), c["replay_start"]
        elif t == "threshold":
            if "z" in spec:            # sweep: threshold at calm median + z * calm scale of daily D
                thr = cal["calm_loc"] + float(spec["z"]) * cal["calm_scale"]
            elif spec.get("threshold", "auto") == "auto":
                thr = cal["p3_auto_threshold"]
            else:
                thr = float(spec["threshold"])
            pol, active = ThresholdRetrain(thr, name), c["sigma_window"][1]
        elif t == "proposed":
            ratio = float(spec.get("cost_ratio", cost["default_ratio"]))
            C_err = c_err_for_ratio(ratio, self.V_ref, cost["C_retrain"], cost["C_label"])
            pol = ProposedRetrain(cal["g"], C_err, cost["C_retrain"], cost["C_label"], spec.get("rule", "myopic"),
                                  spec.get("require_drift_event", True),
                                  cal["g_floor"] if spec.get("subtract_floor", spec.get("rule") == "cumulative") else 0.0,
                                  name)
            active = c["calibration_window"][1]
        elif t == "oracle":
            pol, active = OracleRetrain(true_loss_fn, spec["target_drop"], name), c["replay_start"]
        else:
            raise ValueError(t)
        return pol, active

    def _new_detector(self) -> OnlineDriftDetector:
        cal = self.calibration()
        det = OnlineDriftDetector(self.cfg["detection"], self.period_fn)
        det.anchor, det.penalty, det.penalty_sweep = cal["anchor"], cal["penalty"], cal["penalty_sweep"]
        return det

    # ------------------------------------------------------------------ one policy timeline
    def run_timeline(self, spec: dict) -> dict:
        c = self.cfg
        cal = self.calibration()
        scorer = cal["scorer"]
        s_hi = c["sigma_window"][1]
        last = c["replay_end"]
        current_loss: dict[int, float] = {}
        policy, active_from = self.make_policy(spec, lambda w: current_loss[w])
        self.store.reset_clock()
        self.store.advance_to(max(c["w0_periods"]))
        replay = Replay(self.stream_days, self.scaler, c["replay_start"], last)
        deployed = DeployedModel(self.generation("w0"))
        det = self._new_detector()
        pending: list[tuple[str, str]] = []          # (date, gen_key) not yet fed to the detector
        fed_gen: Optional[str] = None
        D_by_date: dict[str, float] = {}
        rows, retrain_weeks = [], []
        while replay.has_next():
            batch = replay.next_week()
            t = batch.week_index
            self.store.advance_to(t)
            gen = deployed.current
            for d in batch.dates:
                self.day_eval(gen.gen_id, d)
                pending.append((d, gen.gen_id))
            if t >= s_hi:                             # D exists from the end of the sigma window on
                for d, gk in pending:
                    g_ = self.generation(gk)
                    D_by_date[d] = scorer.score(self.day_eval(gk, d).signals, g_.reference)
                    if fed_gen is not None and gk != fed_gen:
                        det.new_generation(_ordinal(d))
                    fed_gen = gk
                    det.observe(d, D_by_date[d])
                pending = []
            wk = _sum_counts([self.day_eval(gen.gen_id, d).counts for d in batch.dates])
            f1 = macro_f1_from_counts(wk["tp"], wk["fp"], wk["fn"])
            acc = float(wk["n_correct"] / max(wk["n_known"], 1))
            current_loss[t] = gen.info["ref_f1"] - f1
            D_t = float(np.mean([D_by_date[d] for d in batch.dates])) if t >= s_hi else None
            event = det.detect() if t >= s_hi else None
            fire = False
            if t >= active_from and t < last:
                fire = bool(policy.decide(t, D_t, event, volume=batch.volume))
            sig_mean = {k: float(np.mean([self.day_eval(gen.gen_id, d).signals[k] for d in batch.dates]))
                        for k in SIGNALS}
            rows.append({"policy": policy.name, "seed": self.seed, "week": t, "gen": gen.gen_id,
                         "macro_f1": f1, "accuracy": acc, "ref_f1_gen": gen.info["ref_f1"],
                         "loss_vs_gen_ref": current_loss[t], "D": D_t, "volume": batch.volume,
                         "drift_event_week": event.week if event else None,
                         "drift_event_mag": event.magnitude if event else None,
                         "retrain": fire, **{f"sig_{k}": v for k, v in sig_mean.items()}})
            if fire:
                new = self.generation(f"t{t}")
                deployed.swap(new, t)
                policy.notify_retrain(t)
                retrain_weeks.append(t)
        self.store.reset_clock()
        return {"policy": policy.name, "rows": rows, "retrain_weeks": retrain_weeks,
                "daily_D": D_by_date, "spec": spec}
