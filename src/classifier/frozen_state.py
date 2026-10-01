"""The currently deployed model within one simulated timeline (spec §4).

A `Generation` is one deployed model plus the label-free reference it is monitored
against. A retrain swaps in a new generation without restructuring the replay loop.
Labels for a retrain are fetched by the evaluation harness (the only place allowed to
touch the sealed store) and handed to `train_generation` as plain arrays; this module
never imports the store.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.classifier.model import TrafficClassifier


@dataclass
class Generation:
    gen_id: str                 # e.g. "w0" or "t12" (trained on labels up to period 12)
    trained_through: int        # last period whose labels were used
    model: TrafficClassifier
    reference: Any = None       # observer.composite.ObserverReference (set by the harness)
    info: dict = field(default_factory=dict)


def train_generation(gen_id: str, trained_through: int, X_train: np.ndarray, y_train: np.ndarray,
                     n_classes: int, params: dict, seed: int) -> Generation:
    model = TrafficClassifier(n_classes, params, seed).fit(X_train, y_train)
    return Generation(gen_id, trained_through, model)


class DeployedModel:
    """Mutable holder of the deployed generation for one policy's timeline."""

    def __init__(self, initial: Generation):
        self.current = initial
        self.history: list[tuple[int, str]] = [(initial.trained_through, initial.gen_id)]

    def swap(self, new: Generation, at_period: int) -> None:
        self.current = new
        self.history.append((at_period, new.gen_id))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.current.model.predict_proba(features)
