"""Retraining policies P0-P5 (spec §9) plus P4c (cumulative cost rule).

Common interface: `decide(week, D_t, drift_event, volume=None) -> bool`, called at the
END of `week`; True means "retrain now on the trailing labeled window and deploy for
week + 1". `D_t` is None until the composite score exists, i.e. until the end of the
sigma window. D-based policies cannot act before then; that is the real cost of a
label-free calibration phase and is charged to them in the evaluation.

P5 (oracle) is the only policy that sees true performance. It gets a `true_loss_fn`
callable injected by the evaluation harness. This module never imports the sealed store
(enforced by tests/test_no_label_leakage.py), which resolves the spec's contradiction:
its test forbade decision/ from importing the store, yet P5 must use grade().
"""
from __future__ import annotations

from typing import Callable, Optional, Protocol

from src.decision.cost_model import CumulativeCostRule, should_retrain
from src.detection.persistence import DriftEvent


class RetrainPolicy(Protocol):
    name: str

    def decide(self, week: int, D_t: Optional[float], drift_event: Optional[DriftEvent],
               volume: Optional[float] = None) -> bool: ...

    def notify_retrain(self, week: int) -> None: ...


class _Base:
    name = "base"

    def notify_retrain(self, week: int) -> None:
        pass


class NeverRetrain(_Base):
    """P0."""

    def __init__(self, name: str = "P0_never"):
        self.name = name

    def decide(self, week, D_t, drift_event, volume=None) -> bool:
        return False


class PeriodicRetrain(_Base):
    """P1 (every=4) / P2 (every=1): retrain at the end of every `every`-th replay week."""

    def __init__(self, every: int, start_week: int, name: str | None = None):
        self.every, self.start = int(every), int(start_week)
        self.name = name or f"periodic_{every}"

    def decide(self, week, D_t, drift_event, volume=None) -> bool:
        return (week - self.start + 1) % self.every == 0


class ThresholdRetrain(_Base):
    """P3: retrain iff the weekly composite D exceeds a fixed threshold. Ignores costs."""

    def __init__(self, threshold: float, name: str = "P3_threshold"):
        self.threshold, self.name = float(threshold), name

    def decide(self, week, D_t, drift_event, volume=None) -> bool:
        return D_t is not None and D_t > self.threshold


class ProposedRetrain(_Base):
    """P4 (rule='myopic', require_drift_event=True): spec §9 verbatim --
    retrain iff a persistent drift event is active AND should_retrain(...) is True.
    P4c (rule='cumulative'): accumulated expected excess loss exceeds the cost of fixing."""

    def __init__(self, g, C_err: float, C_retrain: float, C_label: float, rule: str = "myopic",
                 require_drift_event: bool = True, floor: float = 0.0, name: str = "P4_proposed"):
        self.g, self.C_err, self.C_retrain, self.C_label = g, C_err, C_retrain, C_label
        self.rule, self.require_drift_event, self.name = rule, require_drift_event, name
        self._cum = CumulativeCostRule(g, C_err, C_retrain, C_label, floor) if rule == "cumulative" else None

    def decide(self, week, D_t, drift_event, volume=None) -> bool:
        if D_t is None or self.g is None or volume is None:
            return False
        if self.rule == "cumulative":
            fire = self._cum.update(D_t, volume)
        else:
            fire = should_retrain(D_t, self.g, volume, self.C_err, self.C_retrain, self.C_label)
        if self.require_drift_event and drift_event is None:
            return False
        return fire

    def notify_retrain(self, week: int) -> None:
        if self._cum is not None:
            self._cum.reset()


class OracleRetrain(_Base):
    """P5: non-competing reference. Retrains when the deployed model's TRUE loss (its
    reference macro-F1 minus this week's macro-F1) exceeds `target_drop`."""

    def __init__(self, true_loss_fn: Callable[[int], float], target_drop: float, name: str = "P5_oracle"):
        self.true_loss_fn, self.target_drop, self.name = true_loss_fn, float(target_drop), name

    def decide(self, week, D_t, drift_event, volume=None) -> bool:
        return self.true_loss_fn(week) > self.target_drop
