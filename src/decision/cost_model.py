"""Retrain-vs-hold cost rules (spec §7.2) and cost-ratio helpers (§7.3). Label-free.

`should_retrain` is the spec's formula verbatim (the "myopic" rule): it compares one
period's expected error cost against the one-off cost of fixing.

`CumulativeCostRule` is an addition, evaluated as a separate policy (P4c) rather than
replacing P4. The myopic rule compares a per-period cost stream against a one-off cost,
so a degradation that stays just below the threshold never triggers, even though its
cost accumulates without bound. The cumulative rule retrains once the *accumulated*
expected excess loss since the current model was deployed exceeds the cost of fixing,
the classic accumulated-cost replacement heuristic for deteriorating systems. "Excess"
loss subtracts `floor` = g's expected loss in the first week after deployment, i.e. the
part of the loss a retrain cannot remove because every model is deployed one week
after its training data.

Note on the sensitivity sweep: under the myopic rule the decision reduces to
g(D) > (C_retrain + C_label) / (C_err * V). Sweeping the ratio (C_err * V_ref) /
(C_retrain + C_label) therefore sweeps a threshold on g(D). Because g is monotone, that
is a threshold on D, weighted by the current week's volume. A longer benefit horizon H
would just multiply the ratio, so the sweep already covers it.
"""
from __future__ import annotations


def should_retrain(D_t: float, g, V: float, C_err: float, C_retrain: float, C_label: float) -> bool:
    expected_acc_loss = g.predict([D_t])[0]
    cost_of_inaction = expected_acc_loss * V * C_err
    cost_of_fixing = C_retrain + C_label
    return bool(cost_of_inaction > cost_of_fixing)


def c_err_for_ratio(ratio: float, V_ref: float, C_retrain: float, C_label: float) -> float:
    """C_err such that (C_err * V_ref) / (C_retrain + C_label) == ratio."""
    return ratio * (C_retrain + C_label) / V_ref


class CumulativeCostRule:
    def __init__(self, g, C_err: float, C_retrain: float, C_label: float, floor: float = 0.0):
        self.g, self.C_err, self.C_fix, self.floor = g, C_err, C_retrain + C_label, floor
        self.accumulated = 0.0

    def update(self, D_t: float, V: float) -> bool:
        excess = max(0.0, float(self.g.predict([D_t])[0]) - self.floor)
        self.accumulated += excess * V * self.C_err
        return self.accumulated > self.C_fix

    def reset(self) -> None:
        self.accumulated = 0.0
