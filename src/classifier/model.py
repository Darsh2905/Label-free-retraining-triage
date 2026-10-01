"""LightGBM multiclass classifier: the fixed substrate (spec §4). Not a contribution.

Hyperparameters live in configs/base.yaml and are never tuned against replay data.
"""
from __future__ import annotations

import pickle

import lightgbm as lgb
import numpy as np


class TrafficClassifier:
    def __init__(self, n_classes: int, params: dict, seed: int):
        self.n_classes = n_classes
        self.params = dict(params)
        self.seed = seed
        self._booster: lgb.Booster | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TrafficClassifier":
        p = dict(self.params)
        rounds = int(p.pop("n_estimators"))
        p.update(objective="multiclass", num_class=self.n_classes, seed=self.seed,
                 bagging_seed=self.seed, feature_fraction_seed=self.seed,
                 deterministic=True, force_row_wise=True, verbose=-1)
        # num_class is fixed at N even if a class is absent from a retraining window,
        # so the probability matrix always has N columns.
        self._booster = lgb.train(p, lgb.Dataset(X, label=y.astype(np.int32)), num_boost_round=rounds)
        self._check_not_diverged(X, y)
        return self

    def _check_not_diverged(self, X: np.ndarray, y: np.ndarray, n: int = 5000) -> None:
        """Multiclass boosting can blow up: near-certain probabilities with near-chance
        training accuracy. Fail loudly rather than monitor a broken substrate."""
        idx = np.random.default_rng(0).choice(len(X), min(n, len(X)), replace=False)
        P = self.predict_proba(X[idx])
        acc = float((P.argmax(1) == y[idx]).mean())
        majority = float(np.bincount(y[idx].astype(np.int64)).max() / len(idx))
        if P.max(1).mean() > 0.99 and acc < max(0.5, majority + 0.05):
            raise RuntimeError(f"classifier diverged: train acc {acc:.3f} with mean max-prob {P.max(1).mean():.4f}")

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Returns (n_flows, n_classes) probability matrix. This is the ONLY output
        any downstream module is allowed to consume from the classifier."""
        if self._booster is None:
            raise RuntimeError("classifier not trained")
        if len(features) == 0:
            return np.zeros((0, self.n_classes), np.float32)
        return self._booster.predict(features).astype(np.float32)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "TrafficClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)
